# 代码修改记录

## 2026-07-21

| 改动人 | 编辑批次 | 作用域 | 修改说明 |
|---|---:|---|---|
| Codex | 1 | `detector.py`, `test_detector.py` | 引入简化 CWT 脊线跟踪、脊线候选抑制及对应回归测试。 |
| Codex | 2 | `dev_log.md`, `docs/Modified.md` | 同步项目开发时间线并创建本次代码修改记录。 |
| Codex | 3 | `detector.py`, `test_detector.py` | 根据实测反馈改为响应定位加局部脊线验证，并补充噪声误检回归测试。 |
| Codex | 4 | `dev_log.md`, `docs/Modified.md` | 记录方案修正、真实样本对比及最终测试结果。 |

## 2026-07-24

| 改动人 | 编辑批次 | 作用域 | 修改说明 |
|---|---:|---|---|
| Codex | 5 | `detector.py`, `test_detector.py` | 新增全局/局部 SNR 双重筛选、局部质量指标及初始回归测试。 |
| Codex | 6 | `test_detector.py` | 修正测试判据，仅按实际局部 SNR 判断候选是否应被过滤。 |
| Codex | 7 | `detector.py` | 局部噪声改用峰侧带中位数与 MAD 进行稳健估计。 |
| Codex | 8 | `detector.py` | 缩小峰心排除区，使局部噪声窗口能够覆盖候选附近的高频波动。 |
| Codex | 9 | `test_detector.py` | 根据稳健噪声估计结果校准局部门槛测试值。 |
| Codex | 10 | `dev_log.md`, `docs/Modified.md` | 同步本轮实现、测试与真实样本验证结果。 |
| Codex | 11 | `detector.py`, `test_detector.py` | 取消全局 SNR 硬过滤，改为局部 SNR 单独判定，并增加小峰召回回归测试。 |
| Codex | 12 | `dev_log.md`, `docs/Modified.md` | 记录局部 SNR 单门槛方案及测试结果。 |
| Codex | 13 | `detector.py`, `test_detector.py` | 恢复本次对话开始前的原始峰检测实现及3项原始测试。 |
| Codex | 14 | `dev_log.md`, `docs/Modified.md` | 记录算法恢复和验证结果。 |
| Codex | 15 | `detector.py`, `test_detector.py` | 新增连续尺度脊线验证、漂移限制、输出指标及3项回归测试。 |
| Codex | 16 | `detector.py` | 将默认连续脊线长度收紧为至少一半尺度且不少于4个尺度。 |
| Codex | 17 | `dev_log.md`, `docs/Modified.md` | 记录严格连续脊线筛选的实现和验证结果。 |
| Codex | 18 | `detector.py`, `test_detector.py` | 新增左升右降峰形趋势硬检查、趋势得分输出及回归测试。 |
| Codex | 19 | `dev_log.md`, `docs/Modified.md` | 记录峰形筛选实现和真实样本验证结果。 |

## 2026-07-27

| 改动人 | 编辑批次 | 作用域 | 修改说明 |
|---|---:|---|---|
| Codex | 20 | `detector.py`, `test_detector.py` | 新增联合质量指标、参数化硬过滤及初始合成回归测试。 |
| Codex | 21 | `detector.py`, `test_detector.py` | 调整粗糙度归一化与有效点定义，并按真实数据降低过严门槛。 |
| Codex | 22 | `detector.py`, `test_detector.py` | 校准联合过滤默认参数并验证锯齿噪声至少触发一项门槛。 |
| Codex | 23 | `test_detector.py` | 新增尖锐振荡候选被联合过滤拒绝的端到端测试。 |
| Codex | 24 | `dev_log.md`, `docs/Modified.md` | 记录联合过滤实现、调试与真实样本验证结果。 |

## 2026-07-28

| 改动人 | 编辑批次 | 作用域 | 修改说明 |
|---|---:|---|---|
| Codex | 25 | `detector.py`, `test_detector.py` | 新增对称性、基线漂移、定量S/N和噪声重叠四项标准峰验证及测试。 |
| Codex | 26 | `test_detector.py` | 对联合噪声过滤和重叠峰边界测试显式关闭标准峰硬筛选，隔离测试职责。 |
| Codex | 27 | `detector.py`, `test_detector.py` | 标准峰验证改为默认标记、可选硬过滤，并验证标准高斯峰结果字段。 |
| Codex | 28 | `dev_log.md`, `docs/Modified.md` | 记录标准峰验证实现、真实样本结果和最终测试状态。 |
| Codex | 29 | `detector.py`, `test_detector.py` | 新增尖锐度、周围波动性和相对强度三条件联合伪峰限制及参数测试。 |
| Codex | 30 | `detector.py`, `test_detector.py` | 校准尖锐度阈值并增加强峰保留、弱噪声峰删除的端到端测试。 |
| Codex | 31 | `dev_log.md`, `docs/Modified.md` | 记录低强度尖锐噪声限制及验证结果。 |
| Codex | 32 | `detector.py` | 初步抽取通用参数校验、MAD噪声估计、峰段平滑及过滤判定逻辑。 |
| Codex | 33 | `detector.py` | 合并同类阈值参数校验并移除无效的局部变量访问。 |
| Codex | 34 | `detector.py` | 收回仅使用一次的过滤包装函数，保留就地判定以减少间接层。 |
| Codex | 35 | `detector.py` | 将分散的数值参数校验合并为统一规格表。 |
| Codex | 36 | `detector.py` | 简化上界校验表达式并移除峰质量计算中的重复数组切片。 |
| Codex | 37 | `dev_log.md`, `docs/Modified.md` | 记录检测器精简重构及回归、真实样例一致性验证结果。 |
| Codex | 38 | `detector.py`, `test_detector.py` | 删除重复方向效率与周围波动比指标，改用定量S/N并恢复趋势硬检查；同步调整测试。 |
| Codex | 39 | `detector.py`, `test_detector.py` | 将必要的锯齿噪声判别收敛为内部振荡效率，恢复原有10峰筛选效果并精简测试断言。 |
| Codex | 40 | `dev_log.md`, `docs/Modified.md` | 记录重复指标清理、趋势逻辑恢复及回归验证结果。 |
| Codex | 41 | `detector.py` | 移除当日新增的通用辅助、标准峰验证和尖锐弱噪声模块，恢复联合质量指标实现及旧函数接口。 |
| Codex | 42 | `detector.py` | 恢复7月27日的参数校验、CWT响应噪声估计、联合质量过滤及结果字段。 |
| Codex | 43 | `test_detector.py` | 删除当日标准峰和尖锐弱噪声测试，恢复联合质量指标断言。 |
| Codex | 44 | `test_detector.py` | 清除残留的当日参数与结果字段断言，使测试集完整恢复至11项。 |
| Codex | 45 | `dev_log.md`, `docs/Modified.md` | 记录恢复至2026-07-27代码状态及验证结果。 |
