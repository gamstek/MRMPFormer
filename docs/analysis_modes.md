# 模式精简分析报告（任务 0：是否删除多余模式）——已结题

> 日期：2026-08-18 ｜ 范围：`model/inference/cli.py` 的 `--mode`
> 状态：**已决策并实施完毕（7 → 3）**。本文前半保留原分析（历史决策依据），第 7 节为最终决策与实施记录。

---

## 1. 原状：7 种模式功能矩阵（重构前）

| # | 模式 | 实际功能（阶段覆盖） | 输入 | 关键依赖 |
|---|---|---|---|---|
| 1 | `single` | 单图：ROI 生成 → 预测 → 积分（阶段①+②，无 SNR/post） | JSON（stdin/文件） | `process_single_image`（cli 内私有实现） |
| 2 | `mzml` | **仅** ROI 生成（阶段①） | 单个 mzML | `extract_xic_with_pyopenms` |
| 3 | `batch_mzml` | **仅** ROI 生成，批量循环（阶段①） | 目录 | `run_batch_mzml` |
| 4 | `batch_dir` | **仅** 预测+积分（阶段②），对已有 ROI 目录 | 目录（子目录=样品） | 直接调 `predictor.main()` |
| 5 | `batch_json_dir` | 逐 JSON：ROI→预测→积分（阶段①+②） | 目录（递归 JSON） | `process_single_image` 逐张 |
| 6 | `pipeline_mzml` | 完整管线（阶段①→②→③→④）+ 计时 | 单个 mzML | 四阶段全套 |
| 7 | `pipeline_batch_mzml` | 完整管线批量 + 计时 | 目录 | 四阶段全套 |

## 2. 调用方盘点（重构前，代码内真实调用）

| 调用方 | 调用方式 | 使用模式 |
|---|---|---|
| tools/evaluation/evaluate_baseline.py | subprocess | `pipeline_mzml` |
| tools/benchmark/runner.py | subprocess | `pipeline_mzml` |
| tools/batch/json_batches.py | subprocess（直接按路径调 cli.py） | `batch_json_dir` |
| `desktop/`（PySide6 桌面端） | **未接入**。`pages/peak_finding.py` 为空壳占位页，无任何 import；原始数据（.msdata/.wiff）经 `workers/converter.py`（msdata2mzml.exe）转 mzML 后进管线 | `single`（仅规划） |
| postprocessing/peak_refinement.py | Python import `predictor.run_single` | （不经 cli 模式） |
| tools/maintenance/regenerate_xic.py | Python import `xic_extraction` | （不经 cli 模式） |

## 3. 功能重叠分析（重构前）

```
阶段①ROI生成          阶段②预测积分        阶段③SNR    阶段④精修
    ├─ mzml ──────────┐
    ├─ batch_mzml ────┤→ batch_dir（消费①的产物）
    │                 │
    ├─ single（①+②，私有ROI实现）
    ├─ batch_json_dir（①+②，逐张）
    └─ pipeline_mzml / pipeline_batch_mzml（①+②+③+④）
```

关键判定：
- `mzml` vs `batch_mzml`：**完全重叠**（后者为前者的目录循环版）→ 已合并。
- `pipeline_mzml` vs `pipeline_batch_mzml`：95% 代码相同，分叉仅 3 处（输入收集 / ROI 阶段 / 预测 Namespace 构造）→ 已合并为统一批量路径。
- `single` / `batch_json_dir`：依赖 `process_single_image`（全仓库零外部调用）；chrom JSON 为历史中间格式，其生产脚本（extract_json）已不在仓库，遗留数据可由 `regenerate_xic.py` + `batch_dir` 覆盖 → 已删除。

## 4. 原候选方案（历史记录）

| 方案 | 内容 | 结果 |
|---|---|---|
| A（保守） | 仅删 `batch_mzml` | 未采纳 |
| B（原推荐） | 7→5，删 `mzml` + `batch_mzml` | 被最终方案取代 |
| C（激进） | 7→4，再删 `batch_dir` | 未采纳（违背 cli 统一入口约定） |

---

## 5. 最终决策：7 → 3（2026-08-18 用户确认）

用户决策要点：
1. **删 `single`**：无真实调用方；desktop 寻峰页为空壳，需要时按 mzML 管线接入。
2. **删 `batch_json_dir`**：仪器原始数据为 .msdata/.wiff，必须先转 mzML 才能进管线（desktop 内嵌 msdata2mzml.exe）；chrom JSON 为历史中间格式。
3. **`mzml` 与 `batch_mzml` 合并** 为 `mzml`，支持单文件与目录递归。
4. **`pipeline_mzml` 与 `pipeline_batch_mzml` 合并** 为 `pipeline`，支持单文件与目录递归（含子目录）。
5. 同名 mzML 冲突处理：**自动加路径前缀**（如 `子目录A__样品1`），避免输出互相覆盖。
6. 连带删除：`tools/batch/json_batches.py`、死代码函数（`process_single_image`/`_generate_single_roi`）、专用参数 `--input/--output/--keep_temp/--batch_plot_dir`。

## 6. 重构后的 3 种模式

| 模式 | 功能 | 输入 | 输出 |
|---|---|---|---|
| `pipeline`（默认） | ①ROI→②预测→③SNR→④精修 + 计时 | `--mzml` 单文件/目录，或 `--batch_dir` 目录（均递归） | `<output_dir>/{xic-roi-batch,batch_predictions,snr_filtered}/<key>/` |
| `roi` | 仅 ROI 生成（①） | 同上 | `<output_dir>/<key>/` |
| `batch_dir` | 对已有 ROI 目录预测+积分（②） | `--batch_dir` ROI 根目录 | `<output_dir>/<子目录>/prediction.csv` |

- `<key>` = mzML 文件名 stem；目录递归下同名 stem 自动改为相对路径展平（`A__s1`），并打印提示。
- 输入收集统一由 `_collect_mzml_inputs` 完成：`--mzml` 与 `--batch_dir` 互斥；单文件不建子目录层级差异（与其他文件一致走 `<key>/`）。
- 合并后的 `pipeline` 单文件同样走 predictor 批量路径（`batch_dir=roi_root`），输出布局与旧批量模式一致 → evaluate_baseline 等调用方产物路径不变。

## 7. 实施记录（2026-08-18）

| 变更 | 文件 |
|---|---|
| cli.py 重写：3 模式 + `_collect_mzml_inputs` 递归收集 + 删除 4 模式及死代码（1448→1021 行） | `model/inference/cli.py` |
| 模式名同步 `pipeline_mzml` → `pipeline` | `model/tools/evaluation/evaluate_baseline.py` |
| 模式名同步 + docstring 更新 | `model/tools/benchmark/runner.py` |
| docstring 更新（产物路径假设不变，仅去旧模式名） | `model/tools/diagnostics/check_chrom_snr_alignment.py` |
| 删除 | `model/tools/batch/json_batches.py` |
| 文档同步（模式表 7→3、示例、参数模板） | `README.md`、`User_Tutorials.md` |
| 配置同步（mode 枚举、清理 input/output/keep_temp/batch_plot_dir 键） | `model/configs/inference_pipeline.json` |

验证：`py_compile` 通过；`--help` 输出确认 3 模式；`_collect_mzml_inputs` 单测（递归收集、同名前缀 `A__s1`/`B__s1`、单文件直通）通过。

### 对后续任务 1-5 的口径更新

1. 任务 1（改名）：**已随本次合并完成**，最终名 `roi` / `batch_dir` / `pipeline`（`roi` 于 2026-08-18 由 `mzml` 更名）。
2. 任务 2（bug 修复）：`batch_json_dir` 模型重复加载 bug、pipeline 单/批双路径问题**已随删除/合并消除**；其余审阅项不变。
3. 任务 3（删 `--standard_refs_csv`）：不受影响，该参数仍在 pipeline 分支（仅打印弃用提示）。
4. 任务 4/5（产物、终端输出）：合并后单文件走批量路径，产物布局统一，无新增差异。
