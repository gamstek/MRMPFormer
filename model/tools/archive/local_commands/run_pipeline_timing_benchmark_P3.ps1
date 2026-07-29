# P3 单样品全流程：重复 20 次并统计平均耗时/资源

# 用法（PowerShell）:

#   Set-Location "D:\pycharm\QuanFormer-main"

#   .\scripts\run_pipeline_timing_benchmark_P3.ps1

# 修改次数: .\scripts\run_pipeline_timing_benchmark_P3.ps1 -Runs 10



param(

    [int]$Runs = 20,

    [string]$SummaryOutputDir = ""

)



$ErrorActionPreference = "Stop"

Set-Location "D:\pycharm\QuanFormer-main"



$benchOut = "D:\pycharm\QuanFormer-main\粮食局数据\results\MULT\timing_benchmark_$Runs"



$summaryArgs = @()

if ($SummaryOutputDir) {

    $summaryArgs += "--summary-output-dir", $SummaryOutputDir

}



python .\scripts\run_pipeline_timing_benchmark.py --runs $Runs --benchmark-dir $benchOut @summaryArgs -- `

  --mode pipeline_mzml `

  --mzml "D:\pycharm\QuanFormer-main\粮食局数据\P3\Data20240523-P3-17TOXINS-2.5P3.mzML" `

  --model "D:\pycharm\QuanFormer-main\resources\checkpoint0029.pth" `

  --output_dir "D:\pycharm\QuanFormer-main\粮食局数据\results\MULT" `

  --threshold 0.90 --smooth_sigma 0.0 `

  --snr_min 3.0 --snr_gaussian_sigma 0.8 --snr_min_noise_points 5 `

  --post_edge_noise_stop_mode roi_bottom_decile_mean `

  --post_edge_max_span_min 0.24 --post_edge_noise_percentile 55 `

  --post_boundary_posterior_lookahead 0 `

  --plot --post_plot_sigma 0.8 --post_plot_dir_name refined_plots



Write-Host ""

$summaryRoot = if ($SummaryOutputDir) { $SummaryOutputDir } else { $benchOut }

Write-Host ""

Write-Host "完成。查看汇总:" -ForegroundColor Green

Write-Host "  $summaryRoot\benchmark_summary.log"

Write-Host "  $summaryRoot\benchmark_key_metrics.csv"

Write-Host "单次 run 明细: $benchOut\run_001\pipeline_timing.log 等"


