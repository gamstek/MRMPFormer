# QuanFormer 批量 mzML 流水线（适配 Agilent 等 native_id 为 bytes、且 Q1/Q3 多在 precursor/product 元数据中的文件）
# 依赖：已修复的 mzml_box_outside_snr_pipeline.py（native_id 解码 + precursor/product 回退）
# 用法：在 PowerShell 中执行
#   Set-Location "D:\pycharm\QuanFormer-main"
#   .\scripts\run_pipeline_20251120-01_1.ps1

$ErrorActionPreference = "Stop"
Set-Location "D:\pycharm\QuanFormer-main"

python .\main.py --mode pipeline_batch_mzml `
  --batch_dir "D:\pycharm\QuanFormer-main\20251120-01_1" `
  --model "D:\pycharm\QuanFormer-main\resources\checkpoint0029.pth" `
  --output_dir "D:\pycharm\QuanFormer-main\results\pipeline_samples_yumi" `
  --standard_refs_csv "D:\pycharm\QuanFormer-main\results\pipeline_standards\standard_mode_out\standard_refs.csv" `
  --threshold 0.90 --smooth_sigma 0.0 `
  --snr_min 3.0 --snr_gaussian_sigma 0.8 --snr_min_noise_points 5 `
  --post_small_peak_rt_tol 0.25 `
  --post_min_secondary_ratio 0.04 `
  --post_noise_barrier_ratio 0.8 `
  --post_edge_noise_percentile 20 `
  --post_edge_max_span_min 0.6 `
  --plot --post_plot_sigma 0.8 --post_plot_dir_name refined_plots
