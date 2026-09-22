# test3 逐峰预警对比图

本目录用于人工查看 MassNova 传统算法与 MRMPFormer＋信号算法对同一 XIC 的选峰差异。

- 黑线：推理同口径平滑后的 XIC。
- 绿色阴影和虚线：MassNova 传统算法峰区间、边界和峰顶 RT。
- 蓝色阴影和点划线：MRMPFormer＋信号算法峰区间、边界和峰顶 RT。
- 图下方：传统峰与 AI 峰的 RT、边界、面积、SNR、`score_ai`、匹配状态、RT 差、面积倍数、复合置信度、风险分和预警原因。

本次共生成 118 张代表性图，绘图失败 0 张：

| 文件夹 | 内容 | 数量 |
|---|---|---:|
| `01_ai_score_missing` | 已匹配但 AI 模型分缺失 | 12 |
| `02_trad_peak_missed` | 传统峰没有匹配到 AI 峰 | 12 |
| `03_rt_hard_limit` | RT 差超过 0.30 min | 12 |
| `04_area_hard_limit` | 面积差超过 4 倍 | 12 |
| `05_composite_low` | 没有触发硬规则，但复合置信度低 | 12 |
| `06_trad_low_quality` | 传统参照规则质量低 | 12 |
| `07_ambiguous_match` | 一对一匹配存在歧义 | 12 |
| `08_competitive_extra` | AI 竞争性额外峰 | 12 |
| `09_green_controls` | 绿色正常峰对照 | 16 |
| `10_reference_missing` | 传统字段缺失 | 6 |

打开 `index.html` 可以按缩略图连续浏览；`peak_plot_index.csv` 保存每张图片对应的原始置信度记录。

重新生成命令：

```powershell
cd D:\yinlibo\MRMPFormer\model
python -m postprocessing.plot_confidence_peak_gallery `
  --confidence_csv project_review\confidence_v1_test3\peak_confidence.csv `
  --mzml_dir ..\data\mzml\test3 `
  --output_dir project_review\confidence_v1_test3\peak_plots `
  --per_category 12 `
  --green_controls 16
```
