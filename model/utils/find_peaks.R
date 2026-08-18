# ============================================================
# [DISABLED] 本脚本已整体禁用（R/xcms 峰检测不再使用）
# 原功能：调用 MSnbase/xcms CentWave 对 mzML 做全谱峰检测，
#         输出 peak_list 供 getFeature.py / detection_helper.py 使用。
# 禁用原因：按需求注释所有 R 语言相关部分。
# 以下原代码全部注释保留，不再执行。
# ============================================================

# specialized_srm_analysis.R
# library("MSnbase")
# library("xcms")
#
# args = commandArgs(trailingOnly = TRUE)
#
# pathData <- args[1]
# charge   <- args[2]
# ppm      <- as.numeric(args[3])
# minWidth <- as.numeric(args[4])  # 最小峰宽（秒）
# maxWidth <- as.numeric(args[5])  # 最大峰宽（秒）
# noise    <- as.numeric(args[7])  # 噪声水平
# s2n      <- as.numeric(args[8])  # 信噪比阈值
# prefilter<- as.numeric(args[9])  # 预过滤
# mzDiff   <- as.numeric(args[10]) # m/z 差异
# frac     <- as.numeric(args[11]) # 最小分数
# fn       <- args[6]              # 输出文件名
#
# register(SerialParam())
#
# setwd(pathData)
# files <- list.files(pattern = "\\.mzML$", ignore.case = TRUE, full.names = TRUE)
#
# if (length(files) == 0) {
#   stop("错误：未找到 .mzML 文件！")
# }
#
# cat("正在读取", length(files), "个 SRM 文件...\n")
#
# # 读取数据
# xs <- readSRMData(files)
# cat("成功读取", nrow(xs), "个色谱图\n")
#
# # 检查数据
# if (nrow(xs) == 0) {
#   stop("错误：未读取到任何色谱图！")
# }
#
# # 分析色谱图特征
# cat("\n分析色谱图特征...\n")
# rt_lengths <- c()
# intensity_stats <- c()
#
# for (i in 1:min(10, nrow(xs))) {
#   chrom <- xs[i, 1]
#   if (!is.null(chrom)) {
#     rt <- rtime(chrom)
#     int <- intensity(chrom)
#
#     rt_lengths <- c(rt_lengths, length(rt))
#     if (length(int) > 0) {
#       intensity_stats <- c(intensity_stats,
#                           max(int, na.rm = TRUE),
#                           mean(int, na.rm = TRUE),
#                           sd(int, na.rm = TRUE))
#     }
#   }
# }
#
# cat("RT点数范围:", min(rt_lengths), "-", max(rt_lengths), "\n")
# if (length(intensity_stats) > 0) {
#   cat("最大强度:", max(intensity_stats, na.rm = TRUE), "\n")
#   cat("平均强度:", mean(intensity_stats, na.rm = TRUE), "\n")
# }
#
# # 方法1：尝试使用更宽松的参数进行峰检测
# cat("\n=== 方法1：使用宽松参数进行峰检测 ===\n")
#
# # 调整参数以适应无m/z信息的数据
# params <- CentWaveParam(
#   ppm        = 50,           # 放宽 ppm 设置
#   peakwidth  = c(2, 30),     # 调整峰宽范围（秒）
#   noise      = 1000,         # 降低噪声阈值
#   snthresh   = 3,            # 降低信噪比阈值
#   mzdiff     = 0.001,        # 小 m/z 差异
#   prefilter  = c(3, 1000),   # 降低预过滤阈值
#   integrate  = 1,
#   fitgauss   = FALSE,        # 不进行高斯拟合
#   verboseColumns = TRUE
# )
#
# tryCatch({
#   xs2 <- findChromPeaks(xs, param = params)
#   peaks <- chromPeaks(xs2)
#   cat("检测到的峰数量:", nrow(peaks), "\n")
#
#   if (nrow(peaks) > 0) {
#     # 保存结果
#     peaktable <- as.data.frame(peaks)
#     write.table(peaktable, fn, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
#     cat("✅ 方法1成功：保存了", nrow(peaktable), "个峰到", fn, "\n")
#     quit(status = 0)
#   }
# }, error = function(e) {
#   cat("方法1失败:", e$message, "\n")
# })
#
# # 方法2：手动峰检测
# cat("\n=== 方法2：手动峰检测 ===\n")
#
# results <- data.frame()
# peak_counter <- 0
#
# for (chrom_idx in 1:nrow(xs)) {
#   chrom <- xs[chrom_idx, 1]
#   if (!is.null(chrom)) {
#     rt <- rtime(chrom)
#     int <- intensity(chrom)
#
#     if (length(rt) >= 10 && length(int) >= 10) {
#       # 平滑数据
#       smoothed <- smooth.spline(rt, int, spar = 0.5)$y
#
#       # 找局部最大值
#       peaks <- which(diff(sign(diff(smoothed))) == -2) + 1
#
#       if (length(peaks) > 0) {
#         for (peak_idx in peaks) {
#           if (peak_idx > 5 && peak_idx < length(smoothed) - 5) {
#             # 计算峰参数
#             peak_rt <- rt[peak_idx]
#             peak_int <- smoothed[peak_idx]
#
#             # 计算基线（使用峰两侧的均值）
#             left_baseline <- mean(smoothed[max(1, peak_idx-10):max(1, peak_idx-5)], na.rm = TRUE)
#             right_baseline <- mean(smoothed[min(length(smoothed), peak_idx+5):min(length(smoothed), peak_idx+10)], na.rm = TRUE)
#             baseline <- mean(c(left_baseline, right_baseline), na.rm = TRUE)
#
#             # 计算信噪比
#             snr <- ifelse(baseline > 0, (peak_int - baseline) / baseline, peak_int)
#
#             # 计算峰宽（半高峰宽）
#             half_height <- baseline + (peak_int - baseline) / 2
#             left_bound <- peak_idx
#             right_bound <- peak_idx
#
#             # 向左找边界
#             while (left_bound > 1 && smoothed[left_bound] > half_height) {
#               left_bound <- left_bound - 1
#             }
#
#             # 向右找边界
#             while (right_bound < length(smoothed) && smoothed[right_bound] > half_height) {
#               right_bound <- right_bound + 1
#             }
#
#             peak_width <- rt[right_bound] - rt[left_bound]
#
#             # 应用过滤条件
#             if (peak_int > noise &&
#                 snr >= s2n &&
#                 peak_width >= minWidth &&
#                 peak_width <= maxWidth) {
#
#               peak_counter <- peak_counter + 1
#
#               results <- rbind(results, data.frame(
#                 peak_id = peak_counter,
#                 chrom_index = chrom_idx,
#                 rt = round(peak_rt, 3),
#                 rt_min = round(rt[left_bound], 3),
#                 rt_max = round(rt[right_bound], 3),
#                 intensity = round(peak_int, 1),
#                 baseline = round(baseline, 1),
#                 snr = round(snr, 2),
#                 peak_width = round(peak_width, 3),
#                 area = round(sum(smoothed[left_bound:right_bound]), 1)
#               ))
#             }
#           }
#         }
#       }
#     }
#   }
# }
#
# cat("手动检测到的峰数量:", nrow(results), "\n")
#
# if (nrow(results) > 0) {
#   # 按强度排序
#   results <- results[order(-results$intensity), ]
#
#   # 保存结果
#   write.table(results, fn, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
#   cat("✅ 方法2成功：保存了", nrow(results), "个峰到", fn, "\n")
# } else {
#   # 方法3：提取所有色谱图的最高点
#   cat("\n=== 方法3：提取每个色谱图的最高点 ===\n")
#
#   simple_results <- data.frame()
#
#   for (chrom_idx in 1:nrow(xs)) {
#     chrom <- xs[chrom_idx, 1]
#     if (!is.null(chrom)) {
#       rt <- rtime(chrom)
#       int <- intensity(chrom)
#
#       if (length(int) > 0) {
#         max_idx <- which.max(int)
#         if (length(max_idx) > 0) {
#           simple_results <- rbind(simple_results, data.frame(
#             chrom_index = chrom_idx,
#             rt = rt[max_idx],
#             intensity = int[max_idx],
#             rt_points = length(rt)
#           ))
#         }
#       }
#     }
#   }
#
#   cat("提取到的最高点数量:", nrow(simple_results), "\n")
#
#   if (nrow(simple_results) > 0) {
#     write.table(simple_results, fn, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
#     cat("✅ 方法3成功：保存了", nrow(simple_results), "个最高点到", fn, "\n")
#   } else {
#     # 创建包含表头的空文件
#     empty_df <- data.frame(
#       peak_id = integer(),
#       chrom_index = integer(),
#       rt = numeric(),
#       rt_min = numeric(),
#       rt_max = numeric(),
#       intensity = numeric(),
#       baseline = numeric(),
#       snr = numeric(),
#       peak_width = numeric(),
#       area = numeric()
#     )
#     write.table(empty_df, fn, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
#     cat("⚠️ 所有方法都未找到峰，生成带表头的空文件\n")
#   }
# }
#
# cat("\n处理完成！输出文件:", fn, "\n")
