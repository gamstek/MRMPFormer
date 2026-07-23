# QuanFormer 已知问题汇总

> 更新于 2026-07-07

---

## 修复优先级说明

| 优先级 | 含义 | 判断标准 |
|:--:|------|------|
| P0 | 立即修复 | 致命崩溃 + 改动范围极小 + 几乎零副作用 |
| P1 | 紧急 | 高中影响 + 改动可控 + 需回归测试 |
| P2 | 中等 | 中高影响 + 改动有一定风险 |
| P3 | 较低 | 中低影响，或改动简单但收益有限 |
| P4 | 低 | 低影响 + 修复成本高或副作用不可控 |

---

## 待修复 - 性能

| # | 严重度 | 优先级 | 问题 | 文件 | 修复风险 |
|---|:--:|:--:|------|------|------|
| P1 | 致命 | P2 | batch_size=1 逐张推理，GPU 利用率约 15% | `predict_utils.py:83` | 改动推理管线，需验证输出一致性 |
| P2 | 高 | P2 | `plot_results` 用 `joblib(n_jobs=-1)` 全核并行 matplotlib，进程过多导致 I/O 争抢 | `predict_utils.py:56` | 改 n_jobs 为固定值，低风险 |
| P3 | 高 | P1 | `plot_single_result` 重复 `Image.open()` 读盘，数据已在 `predict()` 中读过 | `predict_utils.py:65` | 传内存数据替代路径，低风险 |
| P4 | 中 | P3 | `plot_results` 覆盖原 ROI 图（`save_path` = 原路径） | `predict_utils.py:57` | 改输出路径，需确保下游兼容 |
| P5 | 中 | P3 | 模型每次预测都重新 `torch.load`（GUI 内） | `predict_utils.py:101` | 加缓存需考虑显存生命周期 |

---

## 待修复 - 跨平台

| # | 严重度 | 优先级 | 问题 | 文件 | 修复风险 |
|---|:--:|:--:|------|------|------|
| C1 | 高 | P1 | `detection_helper.py` R 命令未加引号，Windows 路径含空格时失败 | `utils/detect_helper.py:67` | 加引号即可，几乎零风险 |
| C2 | 中 | P4 | `pycocotools` 在 Windows 需编译工具链（但 pip 已有 cp311 wheel） | `requirements.txt` | 仅文档/配置，无代码影响 |
| C3 | 中 | P3 | Linux headless 服务器 `matplotlib` 可能无 GUI 后端 | `plot_utils.py` | 需后端自动检测，改动范围中等 |
| C4 | 低 | P4 | macOS 首次启动 PySide6 需安全授权 | `GUI/ms-main.py` | 非代码问题，文档说明即可 |

---

## 待修复 - 代码质量

| # | 严重度 | 优先级 | 问题 | 文件 | 修复风险 |
|---|:--:|:--:|------|------|------|
| Q1 | 中 | P3 | `workbooks/` 中多个脚本包含原作者绝对路径硬编码 | `calcQuantificationResults.py:28` 等 | 改路径可能影响其他脚本引用 |
| Q2 | 低 | P4 | UI 类名 `PeakFormer` vs 代码 `QuanFormer` 不一致 | `ms.ui` | 改名涉及 UI 文件和信号绑定，回归面大 |
| Q3 | 低 | P4 | `quanformer` 和 `utils` 两个包职责重叠（各有 `plot_utils.py`） | 架构 | 合并包影响所有 import，需整体重构 |
| Q4 | 低 | P3 | `import bisect` 实际正确，但极易与 `bisect` 混淆 | `plot_utils.py:7` | 单行改名，低风险 |
---



## 环境约束

| 约束 | 说明 |
|------|------|
| Python | 3.10 ~ 3.11，不可 3.8 或 3.12+ |
| PyTorch | >= 2.11.0+cu128（RTX 5060 / sm_120 用户） |
| 模型权重 | `checkpoint0029.pth` > 300MB，需单独确认 |
| R | Untargeted 模式需要 R 4.0+ + MSnbase + xcms |
| 路径 | 避免中文和空格 |

---

## 说明

- **严重度**: 致命 / 高 / 中 / 低 —— 描述问题本身的影响范围
- **优先级**: P0 / P1 / P2 / P3 / P4 —— 综合考虑收益与修复风险后的排序
- **修复风险**: 评估改动可能引入的副作用，优先级越高的条目风险越低

---

## 已修复

| # | 问题 | 文件 | 修复方式 |
|---|------|------|----------|
| 1 | `postprocess.py` 硬编码 Unix 路径分隔符 `/` | `utils/postprocess.py:36` | 改用 `os.path` |
| 2 | `predict_utils.py` 硬编码 `cuda:0`，不支持 MPS | `utils/predict_utils.py:14` | 新增 `_get_best_device()` |
| 3 | 训练脚本 `--device` 默认值 `cuda`（无 GPU 崩溃） | `quanformer/main.py:98` | 改为 `auto` + 自动检测 |
| 4 | `misc.py` 分布式代码 5 处硬编码 `device='cuda'` | `quanformer/util/misc.py` | 新增 `_get_dist_device()` |
| 5 | `torch.load(weights_only=False)` PyTorch <2.0 不兼容 | 4 处 | 新增 `safe_torch_load()` |
| 6 | GUI `listWidget_2` 切换样本时图片重复累积 | `GUI/ms-main.py:120` | 加 `clear()` + `blockSignals` |
| 7 | README Python 版本 3.8 vs 实际 3.10/3.11 矛盾 | `README.md` | 统一为 3.10~3.11，全文汉化 |
| 8 | PyTorch 2.6.0 不支持 RTX 5060 (sm_120) | — | 升级到 2.11.0+cu128 |
| 9 | 空 MS1 谱图导致 IndexError 崩溃 | `utils/extract_eic.py:35-36` | 新增 `len(_mzs)==0` 守卫，跳过空谱图 |
| 10 | `detection_helper.py` R 命令未加引号，Windows 路径含空格时失败 | `utils/detect_helper.py:67` | 路径参数加双引号包裹 |
| 11 | `plot_single_result` 重复 `Image.open()` 读盘，数据已在 `predict()` 中读过  | `model/utils/predict_utils.py L57, L71` |plot_results 主进程预加载图片传入， plot_single_result 兼容 PIL Image|
| 12 | `plot_results` 用 `joblib(n_jobs=-1)` 全核并行 matplotlib，进程过多导致 I/O 争抢 | `model/utils/predict_utils.py L57` |默认值改为 n_jobs=2|
| 13 | `plot_results` 覆盖原 ROI 图（`save_path` = 原路径）| `model/utils/predict_utils.py L65-66` |输出加 _detected 后缀 + 阈值保护（≤500 张预加载）|
| 14 | `import bisect` 实际正确，但极易与 `bisect` 混淆 | `model/utils/plot_utils.py L7, L24-25` |改为 from bisect import bisect_left, bisect_right|
| 15 | `workbooks/` 中多个脚本包含原作者绝对路径硬编码 | `calcQuantificationResults.py L27-31、 peakdetective-application.py L32-34 、 calcTrueOrFalsemarker.py L6, L58 、 peakAlignment.py  L12, L64` |统一改为相对路径或空占位符|
