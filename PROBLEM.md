# QuanFormer 已知问题汇总

> 更新于 2026-07-07 · 基于全项目审查与测试

---

## 🔴 已修复

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

---

## 🟠 待修复 — 性能

| # | 严重度 | 问题 | 文件 | 预计影响 |
|---|:--:|------|------|:--:|
| P1 | 🔴 | **batch_size=1 逐张推理**，GPU 利用率 ~15% | `predict_utils.py:83` | 300 图 15s → 可优化到 2s |
| P2 | 🟠 | `plot_results` 用 `joblib(n_jobs=-1)` 全核并行 matplotlib，进程过多导致 I/O 争抢 | `predict_utils.py:56` | 16 核时反而更慢 |
| P3 | 🟠 | `plot_single_result` 重复 `Image.open()` 读盘，数据已在 `predict()` 中读过 | `predict_utils.py:65` | 每张图读盘 2 次 |
| P4 | 🟡 | `plot_results` 覆盖原 ROI 图（`save_path` = 原路径） | `predict_utils.py:57` | 重跑需重新 build ROI |
| P5 | 🟡 | 模型每次预测都重新 `torch.load`（GUI 内） | `predict_utils.py:101` | 每次多 ~1s |

---

## 🟠 待修复 — 跨平台

| # | 严重度 | 问题 | 文件 |
|---|:--:|------|------|
| C1 | 🟠 | `detection_helper.py` R 命令未加引号，Windows 路径含空格时失败 | `utils/detect_helper.py:67` |
| C2 | 🟡 | `pycocotools` 在 Windows 需编译工具链（但 pip 已有 cp311 wheel） | `requirements.txt` |
| C3 | 🟡 | Linux headless 服务器 `matplotlib` 可能无 GUI 后端 | `plot_utils.py` |
| C4 | 🟢 | macOS 首次启动 PySide6 需安全授权 | `GUI/ms-main.py` |

---

## 🟡 待修复 — 代码质量

| # | 严重度 | 问题 | 文件 |
|---|:--:|------|------|
| Q1 | 🟡 | `workbooks/` 中多个脚本包含原作者绝对路径硬编码 | `calcQuantificationResults.py:28` 等 |
| Q2 | 🟢 | UI 类名 `PeakFormer` vs 代码 `QuanFormer` 不一致 | `ms.ui` |
| Q3 | 🟢 | `quanformer` 和 `utils` 两个包职责重叠（各有 `plot_utils.py`） | 架构 |
| Q4 | 🟢 | `import bisect` 实际正确，但极易与 `bisect` 混淆 | `plot_utils.py:7` |

---

## 🔵 环境约束

| 约束 | 说明 |
|------|------|
| Python | **3.10 ~ 3.11**，不可 3.8 或 3.12+ |
| PyTorch | **≥ 2.11.0+cu128**（RTX 5060 / sm_120 用户） |
| 模型权重 | `checkpoint0029.pth` >300MB，需单独确认 |
| R | Untargeted 模式需要 R 4.0+ + MSnbase + xcms |
| 路径 | 避免中文和空格 |

---

## 📝 说明

- **已修复** = 代码已更新，待验证
- **待修复** = 已诊断未修改
- 严重度: 🔴致命 🟠高 🟡中 🟢低
