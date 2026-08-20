# 终端输出优化方案（任务 5）

> 日期：2026-08-19 ｜ 基线：commit `8717e4c`（模式 7→3）+ 任务 3/4 未提交改动
> 范围：`inference/cli.py`、`inference/predictor.py`、`postprocessing/snr_filter.py`、`postprocessing/peak_refinement.py`、`framework/util/logutil.py` 的全部终端输出
> 性质：**方案文档，不实施**。每项含现状证据、改动点、风险与优先级。

---

## 0. 输出机制现状（先厘清 logutil 如何工作）

[logutil.py](file:///d:/work/MRMPFormer/model/framework/util/logutil.py) 的机制是**替换 sys.stdout 的行前缀过滤器**：

- 按行前缀分类：`[ERROR]`→ERROR、`[WARN]`→WARNING、`[INFO]`→INFO、`[DEBUG]`→DEBUG；
- `_classify_line` 对**无标签行返回 DEBUG（放行）**——见 [logutil.py:121-127](file:///d:/work/MRMPFormer/model/framework/util/logutil.py#L121-L127)；
- 默认级别 WARNING（`-q` 设为 ERROR、`-v` 设为 INFO，[cli.py:707-711](file:///d:/work/MRMPFormer/model/inference/cli.py#L707-L711)）。

**推论（影响所有优化项）**：只要打印语句带 `[INFO]/[WARN]/[ERROR]` 前缀，`-q/-v` 就能正确过滤；但**无前缀标签**的行（`[样品]`、`[推理完成]`、`[BATCH]`、`[OK]`、`[DONE]`、分隔线、表格行）一律放行，`--quiet` 下照样刷屏。logutil 本身无需改动，**让过滤生效的办法是给"应该被过滤"的行补前缀**（或把"重要结论行"统一为无前缀保留）。

---

## 1. 术语去 legacy（高优先级，纯文案）

**问题**：阶段计时表键名引用已删除/更名的脚本：
- `1_ROI生成(testXIC)` → 现行实现是 `preprocessing/xic_extraction.py`
- `2_模型预测(newtest)` → 现行实现是 `inference/predictor.py`
- `4_框修正post_newtest(全部)` → 现行实现是 `postprocessing/peak_refinement.py`
- predictor.py:576 报错文案「Please run testXIC.py first」（[predictor.py:576](file:///d:/work/MRMPFormer/model/inference/predictor.py#L576)）——testXIC.py 已删除，误导用户

**改动点**（均字符串级，零逻辑风险）：

| 位置 | 现值 | 改为 |
|---|---|---|
| [cli.py:788](file:///d:/work/MRMPFormer/model/inference/cli.py#L788) | `1_ROI生成(testXIC)` | `1_ROI生成(xic_extraction)` |
| [cli.py:816](file:///d:/work/MRMPFormer/model/inference/cli.py#L816) | `2_模型预测(newtest)` | `2_模型预测(predictor)` |
| [cli.py:964](file:///d:/work/MRMPFormer/model/inference/cli.py#L964) | `3_SNR筛选(全部样品)` | 保留（名称无 legacy） |
| [cli.py:965](file:///d:/work/MRMPFormer/model/inference/cli.py#L965) | `4_框修正post_newtest(全部)` | `4_框修正(peak_refinement)` |
| [predictor.py:576](file:///d:/work/MRMPFormer/model/inference/predictor.py#L576) | `Please run testXIC.py first` | `请先运行 xic_extraction 生成 XIC 矩阵` |

**注意**：`stage_seconds` 键名同时是 `pipeline_timing_runs.jsonl` 的结构化字段名——改名后 JSONL 历史记录字段名会变化。建议 JSONL 中加 `"stage_schema": "v2"` 或在文档标注；如追求兼容可只改终端显示（`_build_pipeline_timing_report` 内做键名映射），JSONL 保留旧键。**倾向方案：终端与 JSONL 一起改，新增 schema 标记。**

---

## 2. 设备横幅重复打印（高优先级，2 处）

**问题**：pipeline 模式下设备信息打印两次：
1. [cli.py:722-724](file:///d:/work/MRMPFormer/model/inference/cli.py#L722-L724)：进入 pipeline 分支先 `resolve_torch_device(verbose=True)` + 双分隔线；
2. [predictor.py:719-728](file:///d:/work/MRMPFormer/model/inference/predictor.py#L719-L728)：`newtest_main` 内部又打印一次 `"=" * 60` + `resolve_torch_device(verbose=True)`。

虽然 `resolve_torch_device` 有缓存、第二次走 `cached=True` 分支，但**仍会打印一行「沿用本次进程」**，且两套 60 字符分隔线内容冗余。

**改动点**：
- 方案 A（推荐）：cli.py pipeline 分支删除 `resolve_torch_device(verbose=True)` + 分隔线（保留 predictor 内部那一次，因其同时服务独立运行的 `python -m inference.predictor` 入口）；
- 方案 B：`resolve_torch_device` 的 cached 分支不再打印（仅返回），彻底静默二次调用。**风险：独立入口首调仍正常，但批内多子目录循环时失去"沿用"提示——可接受**。

**风险**：极低。仅删两行打印。

---

## 3. 日志分级语义修正（高优先级）

**问题**：`--quiet` 声称「仅显示 ERROR 级日志」，但以下**无标签行在 quiet 下仍然打印**：

| 位置 | 内容 |
|---|---|
| [cli.py:959](file:///d:/work/MRMPFormer/model/inference/cli.py#L959) | `[样品] <stem>: ROI ... 检出 ...` |
| [cli.py:976](file:///d:/work/MRMPFormer/model/inference/cli.py#L976) | `[推理完成] ...` |
| [predictor.py:750](file:///d:/work/MRMPFormer/model/inference/predictor.py#L750) | `[BATCH n/m] <name>` |
| [predictor.py:767](file:///d:/work/MRMPFormer/model/inference/predictor.py#L767) | `[✅ BATCH DONE] ...` |
| [predictor.py:784](file:///d:/work/MRMPFormer/model/inference/predictor.py#L784) | `[✅ DONE] ...` |
| [peak_refinement.py:2942](file:///d:/work/MRMPFormer/model/postprocessing/peak_refinement.py#L2942) | `[OK] Saved refined predictions: ...` |

**方案（推荐，两段式）**：
1. **给"过程性"输出补 `[INFO]` 前缀**：`[样品]`→`[INFO] [样品]`、`[BATCH n/m]`→`[INFO] [BATCH n/m]`、`[OK]`→`[INFO] [OK]`——使 `-q` 能静默过程行；
2. **给"结论性"输出保留无前缀**：`[推理完成]`、`[✅ DONE]`、`[✅ BATCH DONE]` 是用户最终要看的，保持无前缀（quiet 下也显示）——符合"quiet = 只给结论"的直觉。

**风险**：低，纯前缀调整。需回归验证 `-q` / 默认 / `-v` 三档各跑一次输出符合预期。

---

## 4. 风格统一（中优先级）

| 项目 | 现状 | 统一目标 |
|---|---|---|
| 分隔线宽度 | `"=" * 60`（predictor）、`"=" * 64`（cli 启动块）、`"=" * 60/72`（timing report） | 统一 `"=" * 64`；timing 表内 `"-" * 72` 保留 |
| emoji | predictor 大量 ✅⚠️❌（[predictor.py:569](file:///d:/work/MRMPFormer/model/inference/predictor.py#L569) 等），cli 基本不用 | **统一不用 emoji**（与 cli 一致，避免 GBK 控制台渲染差异）；✅ 结论行改为 `[DONE]`/`[OK]` 文本 |
| 状态前缀 | `[INFO]`/`[WARN]`/`[ERROR]`（logutil 认这 4 种）+ 自由标签 | 统一为 logutil 已识别的 4 种前缀 + 结论行无前缀 |
| 中英混排 | `[INFO] Running MRMPFormer model...`（[predictor.py:535](file:///d:/work/MRMPFormer/model/inference/predictor.py#L535)）、`[INFO]  CPU 逻辑核心数`（[predictor.py:724](file:///d:/work/MRMPFormer/model/inference/predictor.py#L724) 双空格） | 全中文文案；顺手修双空格 |

**风险**：中（涉及 predictor.py 大量 print）。建议与 §3 合并为一次"print 规范化"提交，逐文件过。

---

## 5. 调试噪音清理（低优先级）

| 位置 | 内容 | 处置 |
|---|---|---|
| [predict_utils.py:168](file:///d:/work/MRMPFormer/model/utils/predict_utils.py#L168) | `[INFO] Checkpoint keys: [...]`（每次加载都打印一长串键名） | 降级为 `[DEBUG]` 或删除 |
| [predictor.py:611-617](file:///d:/work/MRMPFormer/model/inference/predictor.py#L611-L617) | `area=0 的 compound 行号` 可长达 20 个数字一行 | 改 `[DEBUG]`（默认不显示） |

**风险**：低。注意别把 `[DEBUG]` 误写为无前缀（无前缀会放行，反效果）。

---

## 6. `-q/-v` 互斥与优先级（低优先级）

**问题**：`-q` 与 `-v` 同时传时无告警，后者生效（[cli.py:707-711](file:///d:/work/MRMPFormer/model/inference/cli.py#L707-L711) 先判 quiet 后判 verbose，实际是 quiet 优先？——当前代码 `if quiet: ERROR elif verbose: INFO`，**quiet 优先**）。语义含糊。

**方案**：argparse 加 `group = parser.add_mutually_exclusive_group()` 放 `-v/-q`，互斥报错；help 注明默认 WARNING。**风险：极低。**

---

## 7. 建议实施顺序

| 批次 | 内容 | 风险 | 说明 |
|---|---|---|---|
| 第 1 批 | §1 术语 + §2 设备横幅 | 低 | 纯文案/删行，可与任务 3/4 改动一并提交 |
| 第 2 批 | §3 分级语义 + §4 风格统一 | 中 | print 规范化，需三档日志级别回归 |
| 第 3 批 | §5 噪音 + §6 互斥 | 低 | 收尾 |

**回归验证命令**（每批后用户 PowerShell 执行）：
```powershell
cd D:\work\MRMPFormer\model
# 默认（WARNING）：应只见 WARN/ERROR 与结论行
D:\Anaconda3\envs\gamstekpeaking\python.exe -m inference.cli --mode pipeline --config configs/inference_pipeline.json --mzml ..\data\test\20260715_shiyaoyuan_test\20260715_shiyaoyuan_test_1.mzML --output_dir ..\output\test\terminal_check
# -v：应见全部 [INFO] 行
# -q：应只见 ERROR 与 [推理完成] 结论行
```

---

## 8. 决策请求

| 项 | 建议 | 需确认 |
|---|---|---|
| §1 阶段键名 + JSONL schema 标记 | 终端与 JSONL 一起改，加 schema 字段 | [ ] |
| §2 方案 A（删 cli 侧横幅） | 推荐 A | [ ] |
| §3 过程行补 [INFO]、结论行留无前缀 | 推荐 | [ ] |
| §4 emoji 全删、分隔线 64 | 推荐（与 cli 一致） | [ ] |
| §5 Checkpoint keys → DEBUG | 推荐 | [ ] |
| §6 -q/-v 互斥 | 推荐 | [ ] |
