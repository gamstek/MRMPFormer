# utils/io_utils.py

import csv
import time
import functools
from pathlib import Path
import pandas as pd
from natsort import natsorted


# ==============================
# 🕒 Decorator
# ==============================

def time_master(func):
    """Decorator to measure execution time of a function."""
    @functools.wraps(func)
    def wrapper_time_master(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper_time_master


# ==============================
# 📁 File & Path Utilities
# ==============================

def get_files(path, suffix):
    """
    Recursively find all files with given suffix in the directory.

    Parameters:
    - path (str): Root directory to search.
    - suffix (str): File extension (e.g., 'mzML', 'mzXML').

    Returns:
    - list[Path]: Sorted list of matching file paths.
    """
    try:
        p = Path(path).resolve()
        if not p.is_dir():
            raise NotADirectoryError(f"{path} is not a valid directory.")
        paths = [path for path in p.rglob(f"*.{suffix}")]
        return natsorted(paths)
    except FileNotFoundError:
        print(f"The directory {path} does not exist.")
        return []
    except NotADirectoryError as e:
        print(e)
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def validate_file_path(path):
    """Ensure the given path points to an existing file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")
    if not p.is_file():
        raise ValueError("The provided path is not a file.")


# ==============================
# 🖼️ Image Loading
# ==============================

def load_images(images_path, suffix='jpeg'):
    """
    Load all ROI images from a directory (supports .jpg, .jpeg, etc.).

    Parameters:
    - images_path (str): Directory containing ROI images.
    - suffix (str): Image file extension (default: 'jpeg').

    Returns:
    - list[str]: Natural-sorted list of full image paths.
    """
    p = Path(images_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image directory not found: {images_path}")
    paths = [str(path) for path in p.rglob(f"*.{suffix}")]
    sorted_paths = natsorted(paths)
    print(f"[INFO] Loaded {len(sorted_paths)} ROI images from {images_path}")
    return sorted_paths


# ==============================
# 🧪 Feature CSV Handling
# ==============================

def replace_special_characters(series):
    """
    Replace special characters in compound names that may cause file/path issues.
    """
    replaced_series = series.copy()
    for index, value in series.items():
        if isinstance(value, str):
            for char in [":", " ", "(", ")", "（", "）"]:
                value = value.replace(char, "_")
            # Optional: remove leading/trailing underscores
            value = value.strip("_")
            replaced_series[index] = value
    return replaced_series


def load_features(csv_path, preserve_order=False):
    """
    Load and clean targeted feature CSV file (same format as main.py / testXIC output).

    Expected columns: 'Compound Name', 'mz', 'RT'
    Optional columns: 'q3' (fragment m/z)；'native_id'（与 mzML chromatogram id 一致，用于区分 transition）

    Parameters:
    - csv_path: Path to feature CSV.
    - preserve_order: If True, do not sort and do not drop rows (only drop where all of
      Compound Name, mz, RT are NaN). Use when CSV row order must match xic_matrix.npy
      (e.g. after testXIC.py). If False, dropna and natsort by Compound Name (traditional mode).

    Returns:
    - pd.DataFrame: Columns at least ['Compound Name', 'mz', 'RT']; plus 'q3' if present in file.
    """
    validate_file_path(csv_path)

    try:
        data = pd.read_csv(csv_path, encoding_errors='ignore')

        if 'Compound Name' not in data.columns:
            raise ValueError("CSV must contain 'Compound Name' column.")
        if 'mz' not in data.columns or 'RT' not in data.columns:
            raise ValueError("CSV must contain 'mz' and 'RT' columns.")

        # Clean compound names
        data['Compound Name'] = replace_special_characters(data['Compound Name'])

        # Ensure numeric types
        data['mz'] = pd.to_numeric(data['mz'], errors='coerce')
        data['RT'] = pd.to_numeric(data['RT'], errors='coerce')
        if 'q3' in data.columns:
            data['q3'] = pd.to_numeric(data['q3'], errors='coerce')

        if preserve_order:
            # Keep row order to match xic_matrix.npy; drop only rows with all key fields NaN
            key_cols = ['Compound Name', 'mz', 'RT']
            out_cols = key_cols + (['q3'] if 'q3' in data.columns else [])
            if 'native_id' in data.columns:
                out_cols.append('native_id')
            features_info = data[out_cols].copy()
            drop_mask = features_info[['Compound Name', 'mz', 'RT']].isna().all(axis=1)
            features_info = features_info.loc[~drop_mask].reset_index(drop=True)
            print(f"[INFO] Loaded {len(features_info)} compounds from {csv_path} (order preserved).")
        else:
            # Traditional: dropna on key columns and sort by Compound Name
            features_info = data[['Compound Name', 'mz', 'RT']].dropna()
            if 'q3' in data.columns:
                features_info['q3'] = data.loc[features_info.index, 'q3']
            if 'native_id' in data.columns:
                features_info['native_id'] = data.loc[features_info.index, 'native_id']
            features_info = features_info.sort_values(
                by='Compound Name',
                key=lambda x: natsorted(x.tolist())
            ).reset_index(drop=True)
            print(f"[INFO] Loaded {len(features_info)} valid compounds from {csv_path}")

        return features_info

    except pd.errors.EmptyDataError:
        raise ValueError("The provided CSV file is empty.")
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV parsing error: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load features: {e}")


# ==============================
# 💾 Export Results
# ==============================

def export_results(area, output_path):
    """
    Export quantification results to CSV.

    Expected `area` format: list of tuples/lists with 11 elements:
    [image_path, compound_name, mz, old_rt, rt_min, rt_max, rt_peak,
     intensity_max, area, score, point_count]

    Parameters:
    - area (list): Quantification results.
    - output_path (str): Output CSV file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Image_Path',
            'Compound Name',
            'M/Z',
            'Old RT',
            'rt_min',
            'rt_max',
            'Retention Time',
            'intensity_max',
            'Area',
            'Score',
            'Point counts'
        ])
        for item in area:
            writer.writerow([
                str(item[0]),
                str(item[1]),
                float(item[2]),
                float(round(item[3], 3)),
                float(item[4]),
                float(item[5]),
                float(item[6]),
                float(round(item[7], 3)),
                float(round(item[8], 3)),
                float(round(item[9], 3)),
                int(item[10])
            ])

    print(f"[INFO] Successfully exported results to {output_path}")