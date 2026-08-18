# improve.md — QuanFormer 基线补齐清单

> 目标：把 QuanFormer 从「推理可用」补成「可重训、可评测、可对比」的严格实验基线。
> 每项含【问题】（为什么缺）与【要干什么】（大致做法），完成后逐项打勾。

---

## 1. COCO 标注生成脚本 ☑ (2026-08-16)

- **问题**：训练要求数据集为 COCO 格式（`framework/datasets/coco.py` 读取 `coco_path/train/train_coco.json` 与 `coco_path/val/val_coco.json`），但仓库中没有任何脚本产出 bbox 标注，基线无法重训/换数据复现。
- **要干什么**：
  1. 新建转换脚本（放 `preprocessing/` 或 `tools/`），读 `data/testcase_data.xlsx` → 生成 COCO 格式标注文件（`images` + `annotations`（bbox + category_id=1 峰类）+ `categories`）；
  2. 脚本内支持 train/val 划分（按比例或按文件清单），同时输出 `train/`、`val/` 目录结构与对应 json；
  3. 顺带统一路径约定：README 训练文档写的是 `train2017/ + annotations/instances_train2017.json`，与 `coco.py` 实际读的 `train/ + train_coco.json` 不一致，照文档摆数据会报错 → 二选一修齐。
- **完成情况**：
  - 新建 `model/preprocessing/coco_annotation.py`：mzML（复用 `extract_xic_with_pyopenms`，与推理图像完全一致）+ 标注 xlsx（标准库 zipfile+ET 解析，无 openpyxl 依赖）→ `peak_start/peak_end`（分钟）经 `roi_windows.csv` 窗口线性映射为像素 bbox（y 全高 [0,300]）→ 按 mzML 文件分组划分输出 `train/ + train_coco.json`、`val/ + val_coco.json`；TIC 等无标注 ROI 纳入为负样本；
  - 已生成 `data/coco/`：train 61 图 60 框 + 1 负样本（`_1.mzML`），val 61 图 60 框 + 1 负样本（`_2.mzML`）；
  - 验证：pycocotools 官方 API 加载通过；全量 120 框「峰顶落框内 120/120、框边界近基线 120/120」；抽查图见 `data/coco/_inspect/`；
  - README 训练节修齐：删除 `train2017/ + instances_*.json` 旧约定，改为实际读取路径 `train/ + train_coco.json`，并新增「生成 COCO 数据集」小节。

## 2. 参数配置外置1 ☑ (2026-08-16)

- **问题**：超参只散在 `train.py` 的 argparse 默认值与 checkpoint 内嵌 `args` 里，且两者有出入（`num_queries` 默认 10、checkpoint 实际 3；`lr_drop=35 > epochs=30`，StepLR 永不衰减），复现训练全靠手抄参数。
- **要干什么**：
  1. 新建配置文件（如 `model/configs/quanformer_baseline.yaml`），训练/评测以配置文件为主，CLI 参数仅作覆盖；
  2. 从 `quanformer.pth` 的 `args` 提取实际训练参数，固化为基线配置，并在配置中修正 `lr_drop` 等不合理默认值；
  3. 推理阈值（见第 4 条）也纳入同一配置体系。
- **完成情况**：
  - `train.py` 新增 `--config`（JSON）机制：配置文件作为默认参数，CLI 仍可覆盖；环境无 pyyaml 故用标准库 json（非 improve.md 建议的 yaml，等效）；
  - 固化 `configs/quanformer_baseline.json`：从 quanformer.pth 的 args 提取全部超参（batch_size=16、num_queries=3、enc/dec_layers=1 等），修正不可复现项——coco_path 指向 `../data/coco`（原为旧机器绝对路径）、lr_drop 35→20（修正 StepLR 永不衰减）、resume 置空（基线从零训练）；
  - 新增 `configs/quanformer_v2_finetune.json`：微调专用（lr=1e-5/lr_backbone=1e-6/epochs=10/batch_size=4/reset_optimizer=true，resume=quanformer.pth）；
  - 训练实测暴露并修复两处 `--device auto` 设备解析 bug（torch.device('auto') 直接抛错）；
  - 配置加载验证通过（num_queries=3 生效、CLI 覆盖 epochs/batch_size 正常、resume 空值=不加载）。
  - 推理阈值纳入配置：见第 4 条（未完成）。

## 3. 基线评测协议（一键精度评测）☑ (2026-08-16)

- **问题**：`tools/benchmark/` 只测耗时/显存，不测精度；`postprocessing/evaluation/`（R²、standard_curves 等）是散件；当前 `quanformer.pth` 在测试集上的分数无处记录，之后 MRMPFormer v1 没有对比锚点。
- **要干什么**：
  1. 固定测试集清单（mzML 文件列表 + 版本记录）与统一置信度阈值；
  2. 写一个一键评测脚本（可放 `tools/evaluation/`）：跑完整推理管线 → 汇总输出峰检测指标（RT 容差内匹配的 P/R/F1）+ 定量指标（面积 R²、RSD）；
  3. 用该脚本跑 `quanformer.pth`，把基线参考数字记录进文档（README 或 docs/），作为后续 v1 对比的基准线。
- **完成情况**：
  - 新建 `model/tools/evaluation/evaluate_baseline.py`：双口径评测协议——
    - **检测口径**（`--tiou`，默认 0.95）：预测 RT 区间 vs 标注区间 tIoU 超阈值判 TP，算 P/R/F1；
    - **定量口径**（`--quant_tiou`，默认 0.5）：未达严格 TP 但 tIoU 超宽松阈值的对子参与 RT 边界偏差 / 面积 R² / RSD，不影响 P/R/F1 计数；
    - 支持两种运行：`--run_inference 1` 自动跑 pipeline_mzml；`--run_inference 0` 复用已有 prediction.csv（本次实测 5s/两样品）；
    - 输出 `evaluation_report.json`（含协议参数）+ `match_details.csv`（逐条 TP/FP/FN + quant 标记）+ `area_pairs.csv`。
  - 复用：`linear_fit_r2`（standard_curves.py）、`parse_labels_xlsx`/`label_key`（coco_annotation.py）标注对齐、prediction.csv 自带 rt_min/rt_max/area。
  - **quanformer.pth 基线参考分数**（20260715 两次进样 ×60 通道，score≥0.90）：
    | 口径 | 指标 | 数值 |
    |------|------|------|
    | 检测 tIoU>0.95 | P / R / F1 | 0.0246 / 0.0250 / 0.0248（TP=3） |
    | 定量 tIoU>0.5 | 面积 R²（vs 人工） | **0.99998**（n=115：严格 3 + 宽松 112） |
    | 定量 | RT 起/止边界偏差中位 | 0.063 / 0.073 min |
    | 定量 | RSD 均值/中位 | 3.34% / 1.99%（n=56 通道） |
  - 报告存 `data/evaluation/quanformer/`。**解读**：模型定位与定量能力优秀（面积几乎一致、RSD<5%），但检测分极低。初判为 bbox 边界偏宽，深入分析（2026-08-16 第 7 项后）修正为：预测框相对人工边界系统性左移 ~0.05 min（框宽一致），属模型训练约定差异 → 需用第 1 项 COCO 数据集重训。
  - ⚠️ **2026-08-17 重大更正**：本表产生于 `predict_utils.py` 取类 bug 时期——score 实为背景概率、好框来自 shadow query（详见 `docs/experiment_report.md` 实验日志 001）。**表中定量指标作废**；检测指标有效但为 shadow 框口径。现行有效基线见 `data/evaluation/v1_fixed_dev/`（起止偏差 ±0.1 min 口径：v1 F1=0.008；v2 F1=0.455、面积 R²=0.99999）。

## 4. 置信度阈值统一 ☐

- **问题**：`inference/predictor.py` 默认 threshold=0.99，`tools/benchmark/runner.py` 与 README 示例用 0.90，不同入口结果不可比。
- **要干什么**：统一默认值并纳入第 2 条的配置文件；评测脚本与推理 CLI 从同一处读取，避免硬编码分叉。

## 5. build_predictor 按 args.model 路由 ☐

- **问题**：`utils/predict_utils.py` 的 `build_predictor` 硬编码 `from models.quanformer.detr import build`，将来 MRMPFormer v1 的权重无法走同一推理入口，基线对比失去公平条件（前后处理不一致）。
- **要干什么**：改用 `models/__init__.py` 已有的 `build_model(args)` 工厂，按 checkpoint 内嵌 `args.model` 自动路由到对应变体；v1 落地后零改动即可共用同一推理链路。

## 6. train.py resume 逻辑修复 ☐

- **问题**：`train.py` 恢复训练时无条件删除 `class_embed` / `query_embed` 权重再 `strict=False` 加载——从 COCO 预训练迁移是对的，但从自训 checkpoint 续训会静默丢掉分类头（随机初始化），无法严格续训。
- **要干什么**：改为按键是否存在/维度是否匹配来决定跳过，或加载后做 strict 校验并显式告警。

## 7. 生成ROI图像优化 ☑ (2026-08-16)

- **问题**：当前脚本生成的 ROI 图像采用解析谱图最高强度对应保留时间作为峰的保留时间，与真实峰的保留时间有差异，导致对比时的框边界不匹配。
- **要干什么**：
  1. 生成时按给出的标注文件内的保留时间生成ROI图像 
- **完成情况**：
  - `xic_extraction.extract_xic_with_pyopenms` 新增 `rt_center_overrides` 参数（native_id→RT 分钟覆盖表，命中则以标注 RT 为窗口中心，TIC 等未命中通道仍用最高强度点）；`coco_annotation.py` 自动从标注 xlsx `rt` 列构建覆盖表传入；
  - 重生成数据集实测：60/60 通道窗口中心与标注 RT 严格相等（max 差 0.000000），120/120 bbox 映射质量复验通过；
  - **但检测指标假设被证伪**：TP 3→4，tIoU 中位 0.741→0.736，基本不变。原因：apex RT 与标注 RT 本只差 ~0.003 min，窗口居中后预测框中位仅位移 0.004 min；
  - **真正的根因（修正第 3 项归因）**：预测框相对人工边界**系统性左移 ~0.05 min**（起 −0.051 / 止 −0.049 min，框宽几乎一致 0.449 vs 0.440）——是 quanformer.pth 训练时的边界约定与测试软件人工积分约定的模型级差异，需用第 1 项的 COCO 数据集（bbox=人工边界）重训解决。
---

## 8. 解析时不读取TIC图
- **问题**：当前脚本解析谱图时会读取 TIC 图，导致内存占用高，影响推理效率。
- **要干什么**：
  1. 解析时直接从谱图中提取 RT，不读取 TIC 图。

## 9. 参数配置外置2 ☑ (2026-08-17)
- **问题**：当前脚本参数配置没有注释，导致参数含义不清晰，影响调试。
- **要干什么**：
  1. 为每个参数添加注释，说明其作用。
- **完成情况**：
  - JSON 无注释语法 → 用 `_comment_<参数>` 键内联注释，三处 config 加载逻辑（`train.py` / `inference/cli.py` / `tools/evaluation/evaluate_baseline.py`）统一过滤 `_` 前缀键，注释不进入运行时，行为零影响；
  - 推理、评估参数外置补齐（第 2 项只外置了训练）：`cli.py` / `evaluate_baseline.py` 新增 `--config`（配置为默认值、CLI 可覆盖，与 train.py 机制一致），并修正 argparse `required=True` 不接受 `set_defaults` 的坑（改 `default=None` + parse 后校验）；
  - 四个配置文件全部逐参数注释：`quanformer_baseline.json`（42 参数）、`quanformer_v2_finetune.json`（27）、`inference_pipeline.json`（46）、`evaluation_baseline.json`（11），注释与参数一一对应；
  - 验证：4 配置 126 参数注释全覆盖、加载过滤后评估结果与之前完全一致；另修控制台 GBK 打印 `R²`（`²`）崩溃 → 打印文本改用 `R2`。

## 10. 参数生成在output_v2目录的问题
- **问题**：当前训练脚本会生成一个output，没作用

## 优先级建议

| 优先级 | 条目 | 理由 |
|--------|------|------|
| P0 | ~~1 标注生成~~✅、~~3 评测协议~~✅、~~7 ROI 按标注 RT 居中~~✅、**重训基线** | 三件套已就绪，数据闭环打通；检测分低根因为模型边界约定差异（系统性左移 0.05 min），需用 data/coco 重训 quanformer 并用第 3 项协议复测 |
| P1 | 2 配置外置、4 阈值统一、5 路由、8 跳过 TIC | 可复现与公平对比的前提 |
| P2 | 6 resume 修复 | 影响续训正确性，非首轮必需 |
