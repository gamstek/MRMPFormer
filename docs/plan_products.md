# 产物优化方案（任务 4）

> 日期：2026-08-19 ｜ 范围：推理管线全部输出产物
> 本文档三部分：§1 为**已实施**的 prediction.csv 一名三义修复记录；§2 为其余产物优化项（§2.1/§2.2/§2.6 已于 2026-08-20 实施，其余暂缓）；§3 决策表。

---

## 1. 已实施：prediction.csv 一名三义修复

### 1.1 问题回顾

同一文件名 `prediction.csv` 在管线中承担三个语义：

| 位置 | 语义 | 生产者 |
|---|---|---|
| `batch_predictions/<stem>/prediction.csv` | 阶段②模型预测原始输出（逐框积分明细） | predictor |
| `snr_filtered/<stem>/SNR_box_<thr>/prediction.csv` | 阶段③SNR 筛选后保留行 | snr_filter |
| （同名第三义）`predictor --prediction_output` 单目录模式 | 手动指定路径的输出 | predictor |

下游脚本（evaluate_baseline、trace_refined_case、visualize_compare 等）与人工核对均易拿错。

### 1.2 修复方案：SNR 阶段输出改名 `prediction_snr.csv`

**命名规则**：与既有 `prediction_{integration_method}.csv`（如 `prediction_raw.csv`）的 `prediction_<阶段/方法>` 后缀体系一致。改名后三个语义各归其名：

- `prediction.csv` = 阶段②模型原始预测（唯一含义）
- `prediction_snr.csv` = 阶段③SNR 筛选后
- `prediction_refined.csv` = 阶段④精修后（原名不变）

### 1.3 改动清单（已实施，py_compile 全部通过）

**写方（1 处）**

| 文件 | 改动 |
|---|---|
| [snr_filter.py:683](file:///d:/work/MRMPFormer/model/postprocessing/snr_filter.py#L683) | 输出文件名 `prediction.csv` → `prediction_snr.csv`；同步 2 处提示文本 |

**读方（5 处，均带旧名回退兼容）**

| 文件 | 改动 | 兼容策略 |
|---|---|---|
| [cli.py:873-874](file:///d:/work/MRMPFormer/model/inference/cli.py#L873-L874) | post 前置存在性检查改查 `prediction_snr.csv` | —（同进程刚生成，无需回退） |
| [cli.py:958](file:///d:/work/MRMPFormer/model/inference/cli.py#L958) | `[样品]` 结论行 SNR 保留计数 | 同上 |
| [peak_refinement.py:2093-2098](file:///d:/work/MRMPFormer/model/postprocessing/peak_refinement.py#L2093-L2098) | `run_post_newtest` 主读取 | 先找 `prediction_snr.csv`，无则回退旧 `prediction.csv` |
| [peak_refinement.py:2008-2010](file:///d:/work/MRMPFormer/model/postprocessing/peak_refinement.py#L2008-L2010) | 绘图 pred 查找表 | 同上回退 |
| [reprocess.py:190-193](file:///d:/work/MRMPFormer/model/tools/batch/reprocess.py#L190-L193) | 批重处理 post 输入定位 | 同上回退 |
| [trace_refined_case.py:91-93](file:///d:/work/MRMPFormer/model/tools/diagnostics/trace_refined_case.py#L91-L93) | 诊断脚本 SNR 表读取 | 同上回退 |
| [check_box_rt_mapping.py:14](file:///d:/work/MRMPFormer/model/tools/diagnostics/check_box_rt_mapping.py#L14) | docstring 示例路径 | 文档更新 |

**文档（1 处）**：README 输出目录树同步标注新名（含旧名说明）。

### 1.4 兼容性说明

- **新跑管线**：只产出 `prediction_snr.csv`，全程使用新名；
- **读旧产物目录**：peak_refinement / reprocess / trace_refined_case 均有回退逻辑，**旧目录无需重跑**即可继续 post/诊断；
- **无回退的两处**（cli.py 检查与计数）与 snr_filter 同进程串联，不存在旧文件场景；
- `predictor --prediction_output` 单目录模式的第三义保持原样（用户显式指定路径，语义自明）。

### 1.5 建议验证命令（用户 PowerShell 执行）

> 输出目录约定：正式运行统一写到 `../output/`（相对 model/）；**测试运行单独放到 `../output/test/` 子目录**，不与正式产物混放。

```powershell
cd D:\work\MRMPFormer\model
D:\Anaconda3\envs\gamstekpeaking\python.exe -m inference.cli --mode pipeline --config configs/inference_pipeline.json --mzml ..\data\test\20260715_shiyaoyuan_test\20260715_shiyaoyuan_test_1.mzML --output_dir ..\output\test\pipeline_snrv2_check
# 验收点：
# 1) ..\output\test\pipeline_snrv2_check\snr_filtered\<stem>\SNR_box_3\ 下出现 prediction_snr.csv（无 prediction.csv）
# 2) prediction_refined.csv 正常生成（post 读取回退逻辑生效）
# 3) [样品] 结论行 SNR 保留计数非 -1
```

---

## 2. 其余产物优化项（§2.1/§2.2/§2.6 已实施 2026-08-20，其余暂缓）

### 2.1 中文目录名 ASCII 化：`筛选保留/`、`筛选剔除/` ✅ 已实施

- **改动**（py_compile 通过）：
  - [snr_filter.py:507-508](file:///d:/work/MRMPFormer/model/postprocessing/snr_filter.py#L507-L508) 目录名 → `snr_kept/`、`snr_dropped/`；
  - [snr_filter.py:674](file:///d:/work/MRMPFormer/model/postprocessing/snr_filter.py#L674) image 列前缀 `Path("snr_kept")`；
  - 同文件 docstring / 打印提示 / `--no_save_jpeg` help 同步（含 §1 遗漏的 docstring `prediction.csv` → `prediction_snr.csv`）；
  - [cli.py:662](file:///d:/work/MRMPFormer/model/inference/cli.py#L662) `--save_snr_jpeg` help、[inference_pipeline.json:78](file:///d:/work/MRMPFormer/model/configs/inference_pipeline.json#L78) 注释同步；
  - [check_box_rt_mapping.py:12](file:///d:/work/MRMPFormer/model/tools/diagnostics/check_box_rt_mapping.py#L12) docstring 更新（新名 + 旧名说明）。
- **兼容性**：`_shared/artifacts.py` 按 basename 匹配（注释已补新旧名说明），旧产物目录读取不受影响；[test_shared.py:133](file:///d:/work/MRMPFormer/model/tools/tests/test_shared.py#L133) 保留旧名 fixture 作为回退兼容覆盖。
- **验证命令**：同 §1.5，验收点追加：`--save_snr_jpeg` 开启时生成 `snr_kept/`、`snr_dropped/`（无中文目录名）；prediction_snr.csv 的 image 列前缀为 `snr_kept/`。

### 2.2 SNR 产物目录冗余数据文件 ✅ 已按建议补文档（保留现状）

- **现状**：snr_filter 每样品复制生成 `feature.csv`、`roi_windows.csv`、`xic_matrix.npy`（[snr_filter.py:684-688](file:///d:/work/MRMPFormer/model/postprocessing/snr_filter.py#L684-L688)），与 `xic-roi-batch/<stem>/` 同名文件内容重叠（SNR 版为通过行的紧凑子集，但 ROI 阶段文件本就包含全量）。
- **价值与代价**：自包含目录便于单独 post 运行（不依赖 roi 目录）——但 pipeline 链路中 post 用的 xic 优先取 `--xic_dir`（ROI 目录，[peak_refinement.py:2096-2102](file:///d:/work/MRMPFormer/model/postprocessing/peak_refinement.py#L2096-L2102)），SNR 目录内 npy 仅为单目录独立运行兜底。
- **建议**：保留现状（自包含设计有独立运行价值），仅在 README 标注「SNR 目录数据文件为筛选后子集，全量在 xic-roi-batch」。**不建议删**。

### 2.3 `image_path` 跨目录相对引用断链风险

- **现状**：prediction.csv 的 `image_path` 写的是生成时的相对路径（如 `..\data\test\pipeline_v2\xic-roi-batch\...`），SNR 目录下 CSV 同样指回 ROI 目录。
- **问题**：移动/重命名输出根目录或换机器后断链；脚本多按 `image` 列 basename 重新定位，实际影响有限，但人工按图索骥会踩空。
- **建议**（低优先）：image_path 改为「相对输出根目录」或仅存 basename（image 列已有 basename）。改动点 [predictor.py:_integrate_each_predicted_box](file:///d:/work/MRMPFormer/model/inference/predictor.py#L231) 的 `'image_path': img_path` 一行 + 下游引用检查（grep 确认 evaluate_baseline/visualize_compare 用 image 列，不读 image_path）。
- **风险**：低，但需先确认无脚本依赖 image_path 开文件。

### 2.4 死默认值清理

- **现状**：cli.py batch_dir 分支 Namespace 的 `prediction_output="../output/inference/prediction.csv"`（[cli.py:1007](file:///d:/work/MRMPFormer/model/inference/cli.py#L1007)）在批量分支永不写入；pipeline 分支同字段（[cli.py:802](file:///d:/work/MRMPFormer/model/inference/cli.py#L802)）同样只在单目录路径生效。
- **建议**：与 plan_debug.md §2.1（Namespace 工厂化）合并处理——中修方案落地时统一清理；此处仅记录。

### 2.5 pipeline_timing.log 无限追加

- **现状**：[cli.py:437](file:///d:/work/MRMPFormer/model/inference/cli.py#L437) 以 `"a"` 模式追加，多批次后膨胀；JSONL 同理但结构化尚可接受。
- **建议**（低优先）：超过阈值（如 5MB）时轮转为 `pipeline_timing.log.1`；或按日期分段 `pipeline_timing_YYYYMMDD.log`。改动小（写前查 size），属锦上添花。

### 2.6 `--output_dir` 默认值体系统一说明（文档项）✅ 已实施

- **现状**：roi→`xic-roi-batch`、batch_dir→`../output/inference/batch_predictions`、pipeline→`../output/inference/full_pipeline`，各有道理但散落各分支。
- **实施**（2026-08-20）：README「轻量模式」末尾输出目录约定改为三模式默认值表格；§1.5 的 README 目录树更新已含新名标注。

---

## 3. 决策记录（2026-08-20）

| 项 | 建议 | 状态 |
|---|---|---|
| §2.1 中文目录 ASCII 化 | 建议做（2 行 + docstring） | [x] 已实施 |
| §2.2 SNR 冗余数据文件 | 保留现状仅补文档 | [x] README 已标注子集说明 |
| §2.3 image_path 断链 | 低优先，可缓 | [ ] 暂缓 |
| §2.4 死默认值 | 并入 plan_debug §2.1 中修 | [ ] 暂缓（待中修） |
| §2.5 timing 滚动 | 低优先，可缓 | [ ] 暂缓 |
| §2.6 默认值文档表 | 顺带做（纯文档） | [x] 已实施 |
