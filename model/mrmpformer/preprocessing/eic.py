import os
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from mrmpformer.inference.visualize import plot_xic, calc_coordinate, smooth_xic
from mrmpformer.util.io import time_master, load_images

# ==== 可选后端开关 ====
USE_PYMZML = False  # 使用 pyopenms 作为默认后端

if USE_PYMZML:
    import pymzml
else:
    from pyopenms import MzMLFile, MSExperiment


def get_closest(mz_array, target_mz):
    pos = np.searchsorted(mz_array, target_mz)
    if pos == len(mz_array):
        return pos - 1
    elif pos == 0:
        return pos
    else:
        return pos if (mz_array[pos] - target_mz) < (target_mz - mz_array[pos - 1]) else pos - 1


def extract_eic_pyopenms(path, df_info, ppm):
    exp = MSExperiment()
    MzMLFile().load(path, exp)

    rt_list = []
    peak_lists = []

    for spec in exp:
        if spec.getMSLevel() == 1:
            rt = spec.getRT() / 60.0  # 转分钟
            mz_array = spec.getMZArray()
            int_array = spec.getIntensityArray()
            if mz_array.size > 0:
                rt_list.append(rt)
                peak_lists.append((mz_array, int_array))

    if not rt_list:
        raise ValueError(f"No MS1 scans found in {path}")

    n_compounds = len(df_info)
    n_scans = len(rt_list)
    matrix = np.zeros((n_compounds + 1, n_scans))
    matrix[0, :] = rt_list

    _ppm = float(ppm) * 1e-6

    for i in range(n_compounds):
        target_mz = float(df_info[i][1])
        tolerance = target_mz * _ppm
        intensities = []
        for mz_array, int_array in peak_lists:
            closest_idx = get_closest(mz_array, target_mz)
            mz_diff = abs(mz_array[closest_idx] - target_mz)
            intensities.append(int_array[closest_idx] if mz_diff <= tolerance else 0.0)
        matrix[i + 1, :] = intensities

    return matrix


def extract_eic(path, df_info, ppm):
    if USE_PYMZML:
        # 如果需要使用 pymzML，请取消注释并实现 extract_eic_pymzml 函数。
        raise NotImplementedError("pymzML backend is not implemented.")
    else:
        return extract_eic_pyopenms(path, df_info, ppm)


@time_master
def build(paths, features, plot, args):
    processes_number = getattr(args, 'processes_number', 1)
    ppm = getattr(args, 'ppm', 10)
    
    info_sorted = features.sort_values(by='mz').reset_index(drop=True)
    df_info_sorted = info_sorted.values

    roi_exists = (
        os.path.exists(args.images_path) 
        and any(file.endswith(".jpeg") for file in os.listdir(args.images_path))
    )
    
    if roi_exists:
        print(f"[INFO] ROI images already exist in {args.images_path}, skipping generation.")
        return load_images(args.images_path)
    else:
        xic_list = Parallel(n_jobs=processes_number)(
            delayed(extract_eic)(str(p), df_info_sorted, ppm) for p in paths
        )

        if plot:
            if processes_number == 1:
                for i in range(len(xic_list)):
                    draw_eic_by_mz_order(i, paths, df_info_sorted, xic_list, args)
            else:
                Parallel(n_jobs=processes_number)(
                    delayed(draw_eic_by_mz_order)(i, paths, df_info_sorted, xic_list, args)
                    for i in range(len(xic_list))
                )

        if hasattr(args, 'images_path') and args.images_path:
            raw_data_dir = Path(args.images_path) / "raw_xic_data"
            raw_data_dir.mkdir(parents=True, exist_ok=True)
            for i, matrix in enumerate(xic_list):
                npy_path = raw_data_dir / f"sample_{i}_xic.npy"
                np.save(npy_path, matrix)
                print(f"[INFO] Saved Raw XIC Data: {npy_path}")

                mz_columns = [f"mz_{row[1]:.4f}" for row in df_info_sorted]
                header = '\t'.join(['Retention_Time_min'] + mz_columns)
                txt_path = raw_data_dir / f"sample_{i}_xic.txt"
                np.savetxt(
                    txt_path,
                    matrix.T,
                    fmt='%.6f',
                    delimiter='\t',
                    header=header,
                    comments=''
                )
                print(f"[INFO] Saved Raw XIC Matrix (TXT): {txt_path}")

        return xic_list


def draw_eic_by_mz_order(index, path, df_info_sorted, xic_list, args):
    sigma = getattr(args, 'smooth_sigma', 0)
    eic_path = args.images_path
    xic = xic_list[index]
    rt = xic[0]
    sample_name = Path(path[index]).stem
    output_folder = Path.cwd() / eic_path / sample_name
    output_folder.mkdir(parents=True, exist_ok=True)

    for k in range(len(df_info_sorted)):
        compound_name = str(df_info_sorted[k][0])
        mz_val = float(df_info_sorted[k][1])
        intensity = xic[k + 1]
        calc_intensity, calc_rt = calc_coordinate(df_info_sorted, intensity, rt, k)
        smooth_rt, smooth_intensity = smooth_xic(calc_intensity, calc_rt, sigma)
        safe_name = (
            compound_name
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace("\"", "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
        )
        filename = f"{safe_name}_mz{mz_val:.4f}"
        plot_xic(smooth_rt, smooth_intensity, filename, str(output_folder))


def load_existing_rois(images_path):
    """
    加载已存在的 ROI 图像路径。
    """
    return load_images(images_path)