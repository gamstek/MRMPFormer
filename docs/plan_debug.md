# Bug 复核与修复建议（任务 2）

> 复核日期：2026-08-18 ｜ 基线：commit `8717e4c`（模式 7→3 重构）+ 工作区未提交改动
> 背景：模式精简改名（`roi` / `batch_dir` / `pipeline`）后，对任务 2 原列 6 项 bug 重新审阅，确认存留状态并给出修复建议。

---

## 1. 原六项 bug 复核结论总表

| # | 原 bug | 严重度 | 现状 | 证据 |
|---|---|:--:|---|---|
| 1 | `pipeline_batch_mzml` 文件收集重复（glob `+` 拼接） | 🔴 | **已修复**（随重构消除） | [cli.py:506](file:///d:/work/MRMPFormer/model/inference/cli.py#L506)：`sorted(set(rglob("*.mzml")) \| set(rglob("*.mzML")))`——集合并集去重；Windows 下 `PureWindowsPath` 相等性大小写不敏感，两种模式返回的同一文件 Path 对象相等，必然去重。另 [cli.py:510-518](file:///d:/work/MRMPFormer/model/inference/cli.py#L510-L518) 对同名 stem 自动加路径前缀，顺带修掉了「不同子目录同名样品输出互相覆盖」的隐患 |
| 2 | `batch_json_dir` 每张图重新加载模型 | 🔴 | **已消除**（模式整体删除） | `batch_json_dir`、`single` 模式已删；`process_single_image` / `_generate_single_roi` 从 cli.py 移除，全库无残留调用；`tools/batch/json_batches.py`（唯一 subprocess 调用方）已删除 |
| 3 | 两条 pipeline 代码路径分叉 | 🟡 | **已合并** | 单一 `pipeline` 模式统一循环 [cli.py:779-789](file:///d:/work/MRMPFormer/model/inference/cli.py#L779-L789) 直接调 `extract_xic_with_pyopenms`，不再分单文件/批量两套实现；ROI 统计、异常处理单点维护 |
| 4 | 手搓 `argparse.Namespace` 跨模块传参 | 🟡 | **已修复**（2026-08-20，最小修） | §2.1-a batch_dir 硬编码 `integration_method="linear"` 改为透传 `args.integration_method`；§2.1-c 两处死 `prediction_output` 删除。§2.1-b（Namespace 工厂化）按建议保留为可选项中修，未实施 | 
| 5 | QC 参数在阶段①/③重复下发 | 🟡 | **已修复**（2026-08-20，方案 A） | 阶段③ `snr_pipeline_run` 不再传 `min_chrom_points`/`min_chrom_max_intensity`；snr_filter 独立 CLI 入口参数保留 |
| 6 | `single` 模式 ROI 独立实现副本 | 🟢 | **已消除**（随模式删除） | `_generate_single_roi`（硬编码 window_half_min=1.0、无 QC 的平行实现）已删 |

**结论（2026-08-20 更新）：6 项 bug 全部修复/消除。** 4 项随模式重构修复，另 2 项（#4、#5）与重构新引入问题（§2.1-a）已按下方建议于 2026-08-20 修复，详见各小节末尾「已修复」标注。

---

## 2. 仍存留问题详析与修复建议

### 2.1 手搓 Namespace 传参（原 #4）+ 新引入的参数硬编码

> ✅ **已修复（2026-08-20，最小修）**：batch_dir 分支 `integration_method="linear"` → `args.integration_method` 透传（[cli.py:1015](file:///d:/work/MRMPFormer/model/inference/cli.py#L1015)）；pipeline（[cli.py:800](file:///d:/work/MRMPFormer/model/inference/cli.py#L800)）与 batch_dir（[cli.py:1010](file:///d:/work/MRMPFormer/model/inference/cli.py#L1010)）两处死 `prediction_output` 均改为 `None`（predictor 批量分支以 `batch_output/<子目录>/<pred_basename>` 落盘，不用该字段）。§2.1-b（Namespace 工厂化中修）暂缓，`predict_smooth_sigma`/`keep_smoothed_inputs` 仍走 predictor 内 `getattr` 兜底。

**现状证据（修复前）：**

- pipeline 分支（[cli.py:802-816](file:///d:/work/MRMPFormer/model/inference/cli.py#L802-L816)）：手工构造 15 个字段的 Namespace 传 `newtest_main(a)`；
- batch_dir 分支（[cli.py:1012-1015](file:///d:/work/MRMPFormer/model/inference/cli.py#L1012-L1015)）：手工构造 14 个字段。

**(a) 新问题：batch_dir 模式硬编码 `integration_method="linear"`**
cli 定义了 `--integration_method`（choices: linear/raw/external_baseline，[cli.py:533-535](file:///d:/work/MRMPFormer/model/inference/cli.py#L533-L535)），pipeline 分支正确使用 `args.integration_method`，但 batch_dir 分支 Namespace 里写死 `"linear"`——**用户在 batch_dir 模式下传该参数被静默忽略**。

**(b) 既有问题：字段清单与 predictor 支持的参数不同步**
predictor 实际支持 `predict_smooth_sigma`、`keep_smoothed_inputs`（[predictor.py:856-866](file:///d:/work/MRMPFormer/model/inference/predictor.py#L856-L866)），两个 Namespace 均未传，靠 predictor 内 `getattr(args, "predict_smooth_sigma", 0.0)` 兜底——功能默认关闭且 CLI 无从开启（pipeline 模式下想启用预测输入平滑目前做不到）。

**(c) 死默认值**：batch_dir 的 `prediction_output="../output/inference/prediction.csv"`（[cli.py:1013](file:///d:/work/MRMPFormer/model/inference/cli.py#L1013)）——predictor 批量分支只写每子目录的 csv，该顶层路径永不写入，纯误导。

**影响面：**
- 会动的代码：`cli.py` 两处 Namespace、`predictor.py` 的 `main()` 入口签名（若改为 kwargs）；
- 波及调用方：`postprocessing/peak_refinement.py:3657`（import `run_single`，不受 main 签名影响）、`tools/evaluation/evaluate_baseline.py`、`tools/benchmark/runner.py`（均走 subprocess cli，不受影响）。

**修复建议（按侵入度递增，三选一）：**

| 方案 | 做法 | 侵入度 | 风险 |
|---|---|---|---|
| **最小修**（推荐先行） | 仅改 batch_dir 分支：`integration_method=args.integration_method`（或 `getattr(args,...)`），删除死 `prediction_output` 默认值 | 2 行 | 近零 |
| 中修 | 给 `predictor.main()` 增加一个 `build_namespace(**overrides)` 工厂（predictor 内自持字段默认值），cli 两处改调工厂 | ~40 行 | 低，需回归 pipeline/batch_dir 各跑一次 |
| 大修 | predictor.main 改纯函数签名（显式 kwargs），argparse 仅做薄壳 | ~150 行 | 中，peak_refinement/area_integration 对 predictor 的既有 import 需同步核对 |

**验证方法（修复后）**：`--mode batch_dir --integration_method raw` 后检查输出文件名为 `prediction_raw.csv`（修复前恒为 `prediction.csv`）。

### 2.2 QC 参数双阶段重复下发（原 #5）

> ✅ **已修复（2026-08-20，方案 A）**：cli 阶段③ `snr_pipeline_run` 调用删除 `min_chrom_points`/`min_chrom_max_intensity` 两参（[cli.py:843-855](file:///d:/work/MRMPFormer/model/inference/cli.py#L843-L855)），并加注释说明「QC 仅在阶段①生效，阶段③只做 SNR」；snr_filter 独立 CLI 入口（`python -m postprocessing.snr_filter`）保留两参，单独调用功能不受影响。

**现状证据（修复前）：**

- 阶段① ROI 生成：`min_chrom_points` / `min_max_intensity` → `extract_xic_with_pyopenms`（不达标 → **不生成 ROI 图**，记入 `pipeline_qc_excluded.csv`）；
- 阶段③ SNR 筛选：同两参 → `snr_pipeline_run(min_chrom_points=..., min_chrom_max_intensity=...)`（不达标 → 该框剔除）。

**语义分析：**
阶段①是「通道级」预过滤（低强度/少点数通道不进模型）；阶段③是「框级」复核。由于阶段①已把不达标通道整条排除，阶段③再对**同一对参数**复核时理论上永远通过——**当前是纯冗余**（除非用户中途手改了 prediction.csv 来源，那也不该用 QC 参数兜底）。冗余的实际代价：调参心智负担（「我改了 min_max_intensity，到底在哪层生效？」）+ snr_filter 内多一份判定逻辑。

**影响面：**
- 会动的代码：`cli.py:860-861` 两行（删除传参）、`snr_filter.py` 的 `run()` 签名与内部判定块（若彻底删参数）；
- 波及调用方：`snr_filter.py` 独立 CLI 入口（`python -m postprocessing.snr_filter`）仍保留该参数则兼容无破坏。

**修复建议（推荐方案 A）：**

| 方案 | 做法 | 风险 |
|---|---|---|
| **A（推荐）** | cli 阶段③不再传 QC 两参（保留 snr_filter 独立入口的参数，供单独调用时使用）；在 cli.py 加一行注释说明「QC 仅在阶段①生效，阶段③只做 SNR」 | 近零——阶段③判定本就恒真 |
| B | 彻底从 snr_filter.run 删除两参 | 中——独立 CLI 入口功能收窄，需同步改其 argparse |

**验证方法**：pipeline 跑一次，`snr_filtered/<stem>/SNR_box_3/` 的 prediction.csv 行数与修复前一致（理论必然，因恒真分支）。

### 2.3 附带确认（任务 3 范围）

> ✅ **已处理（fa3836a）**：`--standard_refs_csv` 三处（参数定义、pipeline 分支提示、configs/inference_pipeline.json 注释键）已全部删除，cli.py 与 configs 现无任何残留。

`--standard_refs_csv` 原存在三处：参数定义、pipeline 分支提示、configs/inference_pipeline.json 注释键。按原计划在任务 3 删除。

---

## 3. 修复顺序建议（2026-08-20：前两项已完成）

1. **§2.1-a**（batch_dir 硬编码 integration_method）——✅ 已完成（2026-08-20）；
2. **§2.2 方案 A**（QC 冗余下发）——✅ 已完成（2026-08-20）；
3. **§2.1-b/c**（Namespace 工厂化）——中修，可与任务 4/5 终端输出优化合并提交（暂缓）；
4. 顺带核对：`docs/Bugs.md`（旧问题清单，多处引用已不存在的文件如 `utils/postprocess.py`、`GUI/ms-main.py`）可另行清理，不属本任务。
