import atexit
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import streamlit as st
from skimage.draw import polygon

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("radiomics")

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Radiomics ROI Analyzer", layout="wide")
st.title("Radiomics ROI Analyzer")

st.markdown("""
Puoi caricare:

- **un solo ZIP** con tutti i pazienti
- **più ZIP**, uno per paziente

L'app calcola Mean, STD, Min e Max HU per le ROI selezionate.
""")

# =====================================================
# TEMP DIR — created once per session, cleaned on exit
# =====================================================
if "temp_dir" not in st.session_state:
    td = tempfile.mkdtemp(prefix="radiomics_")
    st.session_state["temp_dir"] = td
    atexit.register(shutil.rmtree, td, ignore_errors=True)

TEMP_DIR: str = st.session_state["temp_dir"]

# =====================================================
# CACHE CT LOADING
# =====================================================
@st.cache_data(show_spinner=False)
def load_ct_series(files: tuple[str, ...]):
    """
    Load a CT series from a sorted tuple of file paths.
    Returns volume in HU, slices, z_positions, spacing, origin.

    Args:
        files: Sorted tuple of DICOM file paths (must be a tuple for cache hashing).

    Returns:
        volume     : np.ndarray of shape (rows, cols, n_slices), float32, in HU
        slices     : list of pydicom Datasets, sorted by Z
        z_positions: np.ndarray of Z coordinates (mm)
        spacing    : tuple (row_spacing_mm, col_spacing_mm, slice_spacing_mm)
        origin     : np.ndarray [X, Y, Z] of the first slice's ImagePositionPatient
    """
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume_list = []
    z_positions = []

    for s in slices:
        img = s.pixel_array.astype(np.float32)
        slope = float(getattr(s, "RescaleSlope", 1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        img = img * slope + intercept
        volume_list.append(img)
        z_positions.append(float(s.ImagePositionPatient[2]))

    volume = np.stack(volume_list, axis=-1)  # shape: (rows, cols, n_slices)
    z_positions = np.array(z_positions)

    # Validate PixelSpacing
    ps = slices[0].PixelSpacing
    if len(ps) < 2:
        raise ValueError(
            f"PixelSpacing has fewer than 2 elements: {ps}. "
            "This series may not be a standard CT."
        )

    # DICOM PixelSpacing: [row_spacing, col_spacing]
    #   row_spacing = distance between rows  → step along Y axis
    #   col_spacing = distance between cols  → step along X axis
    row_spacing = float(ps[0])
    col_spacing = float(ps[1])
    slice_spacing = abs(z_positions[1] - z_positions[0]) if len(z_positions) > 1 else 1.0

    spacing = (row_spacing, col_spacing, slice_spacing)
    origin = np.array([float(v) for v in slices[0].ImagePositionPatient])

    return volume, slices, z_positions, spacing, origin


# =====================================================
# RTSTRUCT HELPERS
# =====================================================
def get_referenced_series_uid(rt) -> str | None:
    """Extract the SeriesInstanceUID that this RTSTRUCT references."""
    try:
        return (
            rt.ReferencedFrameOfReferenceSequence[0]
            .RTReferencedStudySequence[0]
            .RTReferencedSeriesSequence[0]
            .SeriesInstanceUID
        )
    except (AttributeError, IndexError) as exc:
        log.warning("Could not extract referenced SeriesInstanceUID: %s", exc)
        return None


def get_roi_names(rt) -> list[str]:
    return [r.ROIName for r in rt.StructureSetROISequence]


def build_roi_number_map(rt) -> dict[str, int]:
    """Return {ROIName: ROINumber} for all ROIs in the structure set."""
    return {r.ROIName: r.ROINumber for r in rt.StructureSetROISequence}


# =====================================================
# CONTOUR -> MASK
# =====================================================
def contour_to_mask(
    rt,
    roi_name: str,
    volume_shape: tuple,
    z_positions: np.ndarray,
    spacing: tuple,
    origin: np.ndarray,
    z_tolerance_mm: float = 1.0,
) -> np.ndarray:
    """
    Convert RTSTRUCT contours for a single ROI into a boolean voxel mask.

    Coordinate mapping (DICOM convention):
        ImagePositionPatient = [X_origin, Y_origin, Z_origin]
        PixelSpacing          = [row_spacing, col_spacing]
          → row index increases along Y  (pts[:, 1])
          → col index increases along X  (pts[:, 0])

    Args:
        rt            : pydicom Dataset of the RTSTRUCT file
        roi_name      : Name of the ROI to rasterize
        volume_shape  : (rows, cols, n_slices) — must match the loaded CT volume
        z_positions   : 1-D array of Z coordinates for each slice (mm)
        spacing       : (row_spacing, col_spacing, slice_spacing) in mm
        origin        : [X, Y, Z] of the first voxel (ImagePositionPatient)
        z_tolerance_mm: Maximum distance (mm) to accept a slice match;
                        contours farther than this are skipped with a warning.

    Returns:
        Boolean mask of shape (rows, cols, n_slices)
    """
    mask = np.zeros(volume_shape, dtype=bool)

    roi_number_map = build_roi_number_map(rt)
    roi_number = roi_number_map.get(roi_name)
    if roi_number is None:
        log.warning("ROI '%s' not found in StructureSetROISequence.", roi_name)
        return mask

    # Find matching ROIContourSequence entry
    roi_contours = None
    for rc in rt.ROIContourSequence:
        if rc.ReferencedROINumber == roi_number:
            roi_contours = rc
            break

    if roi_contours is None:
        log.warning("No ROIContourSequence entry for ROI '%s' (number %s).", roi_name, roi_number)
        return mask

    # Guard against ROIs that have been approved but not yet contoured
    if not hasattr(roi_contours, "ContourSequence"):
        log.warning("ROI '%s' has no ContourSequence (empty ROI).", roi_name)
        return mask

    row_spacing, col_spacing, _ = spacing
    # origin[0] = X of the first voxel; origin[1] = Y of the first voxel
    x_origin = origin[0]
    y_origin = origin[1]

    skipped = 0
    for contour in roi_contours.ContourSequence:
        pts = np.array(contour.ContourData).reshape(-1, 3)

        z = pts[0, 2]
        dists = np.abs(z_positions - z)
        slice_idx = int(np.argmin(dists))

        if dists[slice_idx] > z_tolerance_mm:
            skipped += 1
            continue

        # Project physical (X, Y) coordinates to pixel (row, col)
        #   col increases with X  →  col = (X - X_origin) / col_spacing
        #   row increases with Y  →  row = (Y - Y_origin) / row_spacing
        rows = (pts[:, 1] - y_origin) / row_spacing
        cols = (pts[:, 0] - x_origin) / col_spacing

        # Warn if contour bounding box falls outside image extent
        if (
            rows.min() < 0 or rows.max() >= volume_shape[0]
            or cols.min() < 0 or cols.max() >= volume_shape[1]
        ):
            log.warning(
                "ROI '%s': contour at Z=%.2f mm partially outside image extent. "
                "Clipping will occur.",
                roi_name, z,
            )

        rr, cc = polygon(rows, cols, shape=volume_shape[:2])
        mask[rr, cc, slice_idx] = True

    if skipped:
        log.warning(
            "ROI '%s': %d contour(s) skipped (Z mismatch > %.1f mm).",
            roi_name, skipped, z_tolerance_mm,
        )

    return mask


# =====================================================
# MULTI ZIP UPLOAD
# =====================================================
uploaded_zips = st.file_uploader(
    "📦 Upload uno o più ZIP",
    type=["zip"],
    accept_multiple_files=True,
)

if uploaded_zips:

    st.info(f"Estrazione di {len(uploaded_zips)} ZIP...")

    # Extract all ZIPs into the persistent session temp dir
    for i, zip_file in enumerate(uploaded_zips):
        zip_path = os.path.join(TEMP_DIR, f"dataset_{i}.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        extract_dir = os.path.join(TEMP_DIR, f"zip_{i}")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    st.success("ZIP estratti ✔")

    # =================================================
    # SCAN DICOM FILES
    # =================================================
    ct_map: dict[str, list[str]] = {}
    rt_map: dict[str, str] = {}
    all_files = [f for f in Path(TEMP_DIR).rglob("*") if f.is_file()]

    scan_progress = st.progress(0, text="Scansione file DICOM...")
    skipped_files = 0

    for i, file in enumerate(all_files):
        try:
            ds = pydicom.dcmread(str(file), stop_before_pixels=True)
            modality = getattr(ds, "Modality", None)

            if modality == "CT":
                uid = ds.SeriesInstanceUID
                ct_map.setdefault(uid, []).append(str(file))

            elif modality == "RTSTRUCT":
                uid = get_referenced_series_uid(ds)
                if uid:
                    rt_map[uid] = str(file)
                else:
                    log.warning("RTSTRUCT file %s: could not resolve referenced SeriesUID.", file)

        except Exception as exc:  # noqa: BLE001
            skipped_files += 1
            log.debug("Skipped file %s: %s", file, exc)

        scan_progress.progress((i + 1) / len(all_files))

    scan_progress.empty()

    if skipped_files:
        st.warning(f"⚠️ {skipped_files} file non DICOM ignorati durante la scansione.")

    series_ids = sorted(set(ct_map) & set(rt_map))

    if not series_ids:
        st.error("❌ Nessun match CT / RTSTRUCT trovato.")
        st.stop()

    st.success(f"✅ Dataset trovati: {len(series_ids)}")

    # =================================================
    # ROI SELECTION — union across all patients
    # =================================================
    all_roi_names: set[str] = set()
    for uid in series_ids:
        try:
            rt = pydicom.dcmread(rt_map[uid])
            all_roi_names.update(get_roi_names(rt))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Impossibile leggere RTSTRUCT per series {uid[:8]}…: {exc}")

    all_roi_names_sorted = sorted(all_roi_names)

    selected_rois = st.multiselect(
        "Seleziona ROI (unione di tutti i pazienti)",
        all_roi_names_sorted,
        default=all_roi_names_sorted,
    )

    # =================================================
    # ANALYSIS
    # =================================================
    if st.button("▶ Run Analysis"):

        if not selected_rois:
            st.warning("Seleziona almeno una ROI prima di eseguire l'analisi.")
            st.stop()

        results = []
        analysis_progress = st.progress(0, text="Analisi in corso...")
        errors = []

        for i, uid in enumerate(series_ids):
            try:
                files_sorted = tuple(sorted(ct_map[uid]))
                volume, slices, z_positions, spacing, origin = load_ct_series(files_sorted)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Series {uid[:8]}…: errore caricamento CT — {exc}")
                analysis_progress.progress((i + 1) / len(series_ids))
                continue

            try:
                rt = pydicom.dcmread(rt_map[uid])
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Series {uid[:8]}…: errore lettura RTSTRUCT — {exc}")
                analysis_progress.progress((i + 1) / len(series_ids))
                continue

            patient_id = getattr(slices[0], "PatientID", None) or f"unknown_{uid[:8]}"

            for roi in selected_rois:
                try:
                    mask = contour_to_mask(
                        rt,
                        roi,
                        volume.shape,
                        z_positions,
                        spacing,
                        origin,
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"Patient {patient_id} / ROI '{roi}': {exc}")
                    continue

                vals = volume[mask]

                if len(vals) == 0:
                    log.info("Patient %s / ROI '%s': mask vuota, skip.", patient_id, roi)
                    continue

                results.append(
                    {
                        "PatientID": patient_id,
                        "SeriesUID": uid,
                        "ROI": roi,
                        "Mean_HU": round(float(np.mean(vals)), 2),
                        "STD_HU": round(float(np.std(vals)), 2),
                        "Min_HU": round(float(np.min(vals)), 2),
                        "Max_HU": round(float(np.max(vals)), 2),
                        "N_voxels": int(len(vals)),
                    }
                )

            analysis_progress.progress((i + 1) / len(series_ids))

        analysis_progress.empty()

        # Surface any per-series errors to the user
        if errors:
            with st.expander(f"⚠️ {len(errors)} errore/i durante l'analisi"):
                for err in errors:
                    st.write(err)

        if not results:
            st.error("❌ Nessun risultato prodotto. Verifica i file caricati e le ROI selezionate.")
            st.stop()

        df = pd.DataFrame(results)

        st.subheader("📊 Results")

        for pid in df["PatientID"].unique():
            sub = df[df["PatientID"] == pid].reset_index(drop=True)
            st.markdown(f"### 👤 Patient {pid}")
            st.dataframe(sub, use_container_width=True)

        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=False),
            "radiomics_results.csv",
            "text/csv",
        )
