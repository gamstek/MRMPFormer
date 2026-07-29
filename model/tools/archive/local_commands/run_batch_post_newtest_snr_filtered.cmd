@echo off
setlocal
REM 仓库根目录（与脚本位置相对；若移动本 bat 请改 REPO）
set "REPO=%~dp0.."
cd /d "%REPO%"

REM ========== 按你的 result 布局修改这两行 ==========
set "SNR_ROOT=D:\pycharm\QuanFormer-main\truedata\TQ8000-Data\1223-硝基酚\20251120-01\result\snr_filtered"
set "RESULT_ROOT=D:\pycharm\QuanFormer-main\truedata\TQ8000-Data\1223-硝基酚\20251120-01\result"

REM SNR 子目录名须与文件夹一致，例如 SNR_box_3、SNR_box_5
set "SNR_SUB=SNR_box_3"

python "%REPO%\scripts\batch_post_newtest_under_snr_filtered.py" ^
  --snr_filtered_dir "%SNR_ROOT%" ^
  --result_root "%RESULT_ROOT%" ^
  --snr_subdir "%SNR_SUB%" ^
  --small_peak_rt_tol 1 ^
  --min_confidence 0.96 ^
  --main_double_split_min_peak_sep_ratio_of_span 0.03 ^
  --plot ^
  --plot_sigma 0.8 ^
  --plot_dir_name refined_plots ^
  --plot_output_parent "%RESULT_ROOT%\plots_bundle"

set "EC=%ERRORLEVEL%"
echo.
echo 退出码 %EC%
exit /b %EC%
