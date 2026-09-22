# 项目开发日志

## 项目概述

### 目标
QuanFormer 是一个基于深度学习的 LC-MS 代谢组学峰检测与定量工具。结合 CNN（ResNet-50）与 Transformer（DETR 架构），在提取离子色谱图（EIC/ROI）中识别真峰/假峰并定位峰边界以积分定量。

### 输入
- **原始数据**: `.mzML` 格式的高分辨率 LC-MS 数据（Centroided 或 Profile）
- **特征表**（Targeted 模式）: CSV 文件，包含 `Compound Name` / `mz` / `RT` 三列
- **模型权重**: `checkpoint0029.pth`（>300MB）

### 输出
- **定量结果**: `area.csv`（峰面积表）
- **后处理结果**: `post-area.csv`（去重转置后的面积表）
- **EIC 图像**: 每个化合物生成一张 ROI 区域的 JPEG 图像
- **预测图像**: 带检测框标注的 EIC 图像

### 方法介绍
1. 从 mzML 中按 m/z 提取 EIC → 生成 ROI 图像
2. 使用 QuanFormer（ResNet-50 + 1 层 Transformer 编解码器）检测峰
3. 根据预测框边界对 EIC 积分 → 得到峰面积
4. 后处理去重 → 输出定量结果表
支持 Targeted/Untargeted × Centroided/Profile 四种组合模式。

**centWave 模块**（`centwave/model.py`）：提供基于连续小波变换（CWT/Mexican Hat）的经典色谱峰检测算法，无需 GPU 和深度学习模型。支持直接读取 `.mzML` 色谱图数据，输出峰表和 ROI 出峰图。适用于快速筛查、数据质量检查及作为 QuanFormer 的 baseline 对比。

---

## 开发时间线

### 2026-07-07

- 需求分析(全局): 跨平台兼容性审阅 — 识别出 GPU 硬编码、路径分隔符、Python 版本矛盾等 15 个问题
- 代码生成(utils/): 新增 `_get_best_device()` (predict_utils) + `safe_torch_load()` (io_utils) + `_get_dist_device()` (misc)；替换 5 处 `device='cuda'` 硬编码为 auto
- 调试(utils/): 修复 Unix 路径分隔符硬编码 (postprocess)；`--device` 默认值改为 auto (main)；空 MS1 谱图 IndexError 守卫 (extract_eic)
- 重构(GUI/ms-main.py): 全界面中文化 + QSS 简约主题 + 信号阻塞修复 + 图片重复累积 bug 修复
- 重构(requirements.txt): 整合 GPU/CPU 双文件为统一配置，按 GPU 架构分段 (RTX 50/40/30/20 + CPU)
- 文档生成: README 全文汉化 + 跨平台兼容矩阵 + 目录结构；DEPLOY 部署说明同步；PROJECT_PANORAMA 全景文档；PROBLEM 已知问题汇总（已修复 8 项 + 待修复 9 项）
- 其他: RTX 5060 (sm_120) 与 PyTorch 2.6.0 不兼容分析；预测阶段 5 个性能瓶颈识别

### 2026-07-09

- 文档生成(CLAUDE.md): 重写项目级 AI 指令文件，新增项目身份表、技术约束和代码修改记录规则
- 代码生成(.github/skills/): 创建环境依赖检测与修复系统 —— check_env.py（全量检测/JSON/Markdown）、check_gui.py（tkinter 弹窗/一键修复）、fix_env.py（find-env/check/fix/verify）；配套 check-dependencies + fix-dependencies 两个 skill（精简后各 ~30 行）
- 调试(check_env.py + check_gui.py): 修复 Windows GBK/UTF-8 编码错配导致中文乱码 — run_cmd() 统一 utf-8、GUI 改用 --outfile 绕过管道编码
- 文档生成(README.md): 新增「环境检测」小节，一行命令启动 GUI 弹窗

### 2026-07-14

- 代码生成(centwave/model.py): 新增 `read_mzml_chromatograms()` / `process_mzml()` / `process_mzml_batch()` 三个函数，实现从 `.mzML` 色谱图数据到 centWave 峰检测的完整处理管线；输出峰表 `results.csv` 和 ROI 出峰图
- 需求分析(centwave/): 探查 20251120-01 数据集：140 个 `.mzML` + `.mzML.json` 文件对，SRM 模式，MS8000+Agilent 平台，每条色谱图 1429 点/0~300s
- 调试(centwave/model.py): 修复 mzML XML 编码兼容性问题（17/140 文件含 GBK/UTF-8 混合编码），采用 UTF-8 → GBK → GB2312 → GB18030 → replace 多级降级策略，100% 文件成功解析
- 调试(centwave/model.py): 修复 matplotlib 中文字体缺失警告，设置 `font.sans-serif` 为 SimHei/Microsoft YaHei，确保色谱图标题中文正常显示
- 测试(centwave/): 140 文件批量峰检测通过，共检出 12,599 个色谱峰，生成 140 张 ROI 出峰图。输出至 `20251120-01/output/`
- 文档生成(centwave/experiment_report.md): 撰写实验报告，涵盖数据概况、方法、结果分析和讨论

### 2026-07-16

- 调试(io_mzml.py): 按 PSI-MS CV 声明解码 mzML 二进制数组，支持 32/64 位数值、zlib/无压缩及分钟到秒转换，并对不支持或矛盾的声明给出上下文错误。
- 测试(io_mzml.py): 新增 5 项解码单元测试；回归读取 148 个真实 mzML 文件共 2692 条色谱图，全部解析成功。

### 2026-07-21

- 重构(detector.py): 将多尺度响应求和找峰改为简化 CWT 脊线跟踪与候选抑制，新增脊线长度过滤参数并保持峰边界接口兼容。
- 测试(test_detector.py): 增加脊线持续长度与参数校验用例，峰检测回归测试 5 项全部通过，并完成真实 mzML 抽样验证。
- 调试(detector.py): 根据实测反馈撤销自由脊线候选生成，改为多尺度响应定位、局部短脊线验证；真实样本由误检 245 峰恢复至历史基准 30 峰。
- 测试(test_detector.py): 新增噪声背景下不产生额外脊线候选的回归用例，专项测试 6 项全部通过。

### 2026-07-24

- 重构(detector.py): 新增全局 SNR 与局部 SNR 双重门槛，使用候选峰两侧带的中位数和 MAD 稳健估计局部基线与噪声，并输出局部质量指标。
- 测试(test_detector.py): 增加局部高波动过滤、干净真峰保留和参数校验用例，专项测试 9 项全部通过；真实样本默认双门槛下由 30 个候选收紧为 15 个。
- 调试(detector.py): 根据小峰召回反馈取消全局 SNR 硬过滤，改为仅按局部 SNR 判定候选；全局 SNR保留为诊断字段，`snr` 输出统一表示实际使用的局部 SNR。
- 测试(test_detector.py): 新增远端高噪声不应遮蔽局部清晰小峰的回归用例，专项测试 10 项全部通过。
- 重构(detector.py): 按用户要求恢复本次对话开始前的多尺度响应求和、全局突出度和全局 SNR峰检测流程，移除简化脊线及局部 SNR扩展。
- 测试(test_detector.py): 恢复最初3项峰边界回归测试并全部通过，真实样本恢复为历史基准30个峰。
- 代码生成(detector.py): 在原始多尺度候选之后增加连续脊线硬验证，要求相邻尺度均存在正局部极大值且相邻位置漂移不超过2个采样点。
- 测试(test_detector.py): 增加脊线断裂、连续漂移和过量漂移测试；默认要求至少一半尺度连续支持，6项专项测试全部通过，真实样本由30峰严格筛选为29峰。
- 代码生成(detector.py): 新增峰形左右趋势检查，以正确方向变化量占总变化量的比例衡量左升右降，默认两侧得分均需达到0.7。
- 测试(test_detector.py): 增加有效/无效峰形与参数校验测试，8项专项测试全部通过；真实样本在趋势门槛0.7下由29峰进一步筛选为25峰。

### 2026-07-27

- 代码生成(detector.py): 在连续脊线与趋势检查后新增联合质量过滤，综合平滑残差粗糙度、左右方向效率、有效连续点数、平滑峰SNR和峰顶位置筛除尖锐振荡噪声。
- 调试(detector.py): 根据真实样本质量指标分布校准默认门槛，并提供 `quality_filter` 开关用于无过滤对照与后续数据标注。
- 测试(test_detector.py): 增加干净峰/锯齿峰质量指标及端到端尖锐振荡过滤测试，11项专项测试全部通过；真实样本由25峰保留12峰。

### 2026-07-28

- 代码生成(detector.py): 新增标准定量峰验证，计算半高宽对称性、基线漂移比例、局部定量S/N和半高宽核心噪声重叠比例。
- 调试(detector.py): 严格四项标准在真实样本中判定12个联合过滤候选全部不合格，调整为默认输出 `standard_peak_pass` 标记、显式开启时才硬过滤。
- 测试(test_detector.py): 增加标准高斯峰、非对称漂移峰和参数校验测试，隔离重叠峰与联合噪声测试职责，14项专项测试全部通过。
- 代码生成(detector.py): 新增低强度尖锐噪声联合限制，仅当FWHM相对窄、峰外波动相对峰高较大且相对整条色谱强度较低时删除候选。
- 调试(detector.py): 将尖锐度比例阈值校准为0.4，真实样本在联合过滤12峰基础上额外删除2个低强度高波动候选并保留10峰。
- 测试(test_detector.py): 增加三条件缺一不可及强峰保留/弱噪声峰删除的端到端测试，17项专项测试全部通过。
- 重构(detector.py): 合并数值参数校验、MAD噪声估计与峰段平滑逻辑，移除重复切片和循环内不变量计算，保持检测公式、阈值及输出字段不变。
- 测试(detector.py): 17项峰检测回归测试通过；真实mzML样例仍输出10峰，逐字段结果哈希与重构前一致。
- 重构(detector.py): 删除左右方向效率与周围波动比三个重复公开指标及对应参数；以单一内部振荡效率保留锯齿噪声拦截能力，并复用定量S/N判定尖锐弱噪声。
- 调试(detector.py): 恢复左升右降趋势阈值始终生效，避免质量过滤开关绕过峰型趋势检查。
- 测试(test_detector.py): 17项峰检测测试通过，真实mzML样例保持10峰，输出中不再包含三个重复指标。
- 重构(detector.py): 按要求恢复至2026-07-27结束状态，仅保留连续脊线、左升右降与联合质量过滤，撤销当日标准峰验证、尖锐弱噪声过滤及相关重构。
- 测试(test_detector.py): 测试集恢复为11项并全部通过；真实mzML样例恢复为历史记录的12峰，结果中不含当日新增字段。
