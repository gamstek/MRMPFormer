# C 接口与 Python massnova 数值一致改造方案

日期：2026-09-21

## 1. 目标与边界

目标是在单通道得到等价的 RT（分钟）与强度数组后，使 C 接口和 Python `inference/massnova.py` 返回相同的峰事件：

- 峰数量和排序相同；
- 每个峰的 `a`、`b`、`c` 数值相同；
- 模型命中峰使用模型边界和 Softmax 分数；
- 模型未命中但信号规则通过的峰使用信号边界和信号工程分；
- 模型峰和信号峰执行相同的深谷去重；
- C 公共函数、`QfConfig`、`QfCompoundInput` 和既有 JSON 主结构不变。

允许不同的部分只有输入适配：Python 从 mzML 读取通道，C 由软件直接传入通道数组。进入“单通道标准化 RT/强度”之后的算法必须一致。

## 2. “数值一致”的验收定义

不同推理后端和浮点实现无法承诺未量化浮点数逐 bit 相同，因此采用以下可执行定义：

1. 候选峰索引、最终峰数量、峰排序和模型/信号来源必须完全相同；
2. CPU ONNX Runtime 下，`a`、`b`、`c` 的绝对误差不超过 `1e-6`；
3. CUDA 下允许模型输出浮点误差，`a/b` 不超过 `1e-5 min`，`c` 不超过 `1e-5`；
4. 对外结果统一量化到 6 位小数后，Python 与 C JSON 中的 `a/b/c` 必须完全一致；
5. 阈值附近用同一 ONNX Runtime 参考结果决定，避免 PyTorch 与 ONNX 的微小误差改变是否过阈值。

为了满足第 2～4 项，Python 的一致性基准应改为使用发布的 `mrmpformerv2.onnx`，不能继续用 PTH 输出作为最终跨语言比较基准。

## 3. 当前主要不一致

| 环节 | Python massnova | 当前 C 接口 |
|---|---|---|
| 平滑 | SciPy `gaussian_filter1d(sigma=0.8, mode=reflect)` | 3σ核、边缘 clamp，默认 sigma=0 |
| 候选峰 | `find_peaks` + prominence + height + distance | 无 |
| 模型输入 | 每个候选峰 ±1 min 窗口 | 整条通道一张图 |
| 渲染 | Matplotlib 400×300、无边距、1.5线宽、JPEG | 40/20/20/30边距、1像素RGB线 |
| 阈值 | 0.6，Python条件为 `score > threshold` | 默认0.99，C条件等价于 `score >= threshold` |
| 框匹配 | 候选峰顶必须落入框，分数优先 | 没有候选约束 |
| 模型边界 | 模型框直接线性映射到候选RT窗口 | 映射后再按原始信号单调行走 |
| 信号补峰 | 每个模型未命中候选分别精修和门控 | 模型零峰时只取一个全局最高峰 |
| 去重 | RT距离+重叠+基线校正谷深 | 只有排序 |

当前 C 绘图使用横向 `[40,380]`，但 `pixel_to_rt` 使用 `x/400`，自身也存在绘图坐标与 RT 反映射不一致。

## 4. 必须实施的内部改动

### 4.1 固定生产参数

在不扩展 `QfConfig` 的前提下，C 内部使用当前生产参数：

- `threshold=0.6`；
- `smooth_sigma=0.8`；
- `min_chrom_points=10`；
- `min_max_intensity=1000`；
- `baseline_percentile=25`；
- `min_peak_ratio=0.04`；
- `prominence_ratio=0.055`；
- `min_peak_gap_points=3`；
- `min_peak_width_min=0.10 min`；
- `void_time_min=0.5 min`；
- `max_peaks_per_channel=50`；
- `window_half_min=1.0 min`；
- `min_snr=10`；
- `min_peak_span_points=5`；
- `dup_apex_tol=0.2 min`；
- `dup_min_overlap_fraction=0.25`；
- `dup_shallow_valley_min_ratio=0.70`；
- `width_fuse_ratio=1.5`。

`QfConfig.threshold`、`smooth_sigma` 等既有字段仍可覆盖相应默认值；不能新增或调整结构成员，以免破坏 ABI。

### 4.2 标准化输入

保留后端传数组方式。进入推理前：

1. 检查 RT 严格递增和数组有限；
2. 按 Python `_rt_to_minutes` 规则把秒转换成分钟；
3. 强度在候选枚举中执行 `max(y,0)`；
4. 使用与 SciPy 一致的高斯平滑结果作为后续候选、渲染、精修、面积和 SNR 的共同输入。

需要替换当前 C 平滑：SciPy 默认核半径为 `round(4*sigma)`，边界为 `reflect`；当前 C 的 `ceil(3*sigma)+clamp` 不等价。

### 4.3 候选峰枚举

新增内部 `PeakCandidate`，至少保存：

- `apex_index`；
- `rt_peak`；
- `apex_intensity`；
- `model_score`；
- `validated`；
- `boundary_source`；
- `rt_min/rt_max`。

移植 Python `enumerate_peaks()`：

1. `rt>=0.5` 构造有效区；
2. 25百分位基线和动态范围；
3. 计算 height、prominence、按RT步长换算的 distance；
4. 实现与 SciPy `find_peaks` 一致的局部最大值、distance选择和 prominence；
5. 使用 `_merge_insignificant_valleys()` 规则合并浅谷；
6. 超过50个时按当前真实实现的峰顶强度取前50个，再按RT排序。

候选索引必须由 Python 生成的黄金数据逐条验证，不能只验证候选数量。

### 4.4 候选窗口和图像渲染

每个候选使用 `[rt_peak-1, rt_peak+1]`，再裁剪到通道范围。窗口不足两个点则不送模型。

当前 C `render_roi()` 不能继续使用。需要新增候选窗口渲染并满足：

- 400×300；
- 横轴完整覆盖0～400像素，不保留40/20像素边距；
- 横轴与 `[rt_lo,rt_hi]` 线性对应；
- 与 Python相同的纵轴自动范围；
- 蓝色、1.5 pt、抗锯齿；
- 确认是否保留JPEG编码/解码影响。

长期可维护方案是定义一个无 Matplotlib 依赖的规范渲染器，并在 Python 和 C 同时使用；切换前必须用 test1/test2/test3 复评 special_v2。若要求不改变历史 Python 图像，则 C 需要复刻 Matplotlib Agg 与 JPEG 行为，实施和依赖成本更高。

### 4.5 ONNX推理和阈值

保持现有 ONNX 输入输出和 `OnnxInference`。修改点：

- 输入批次从“每通道一张整图”改为“每候选一张窗口图”；
- 保存 `(item_index,candidate_index,rt_lo,rt_hi)` 映射；
- 阈值使用0.6；
- 过滤条件与 Python统一为 `score > threshold`，即 C 中 `score <= threshold` 时丢弃；
- 一致性基准使用同一份 ONNX 和 CPU ONNX Runtime。

### 4.6 模型框与候选匹配

对候选窗口的每个保留框：

1. `x1/x2` 按 `[0,400]` 映射到该候选的 `[rt_lo,rt_hi]`；
2. 只接受 `left <= rt_peak <= right`；
3. 按模型分数从高到低选择；
4. 分数相同时选择框中心离候选峰顶更近的框；
5. 命中后直接采用模型框边界，不再执行当前 C 的“框内找最高点并单调下行”修正。

### 4.7 模型未命中候选的信号路径

移植以下 Python依赖：

- `refine_all_boundaries()`；
- `adjust_first_round_interval()` 及其边缘噪声、posterior lookahead、peer防碰撞辅助函数；
- `fuse_fallback_boundaries()`；
- `compute_local_snr()`；
- `roi_full_low_decile_mean_intensity()`；
- `gate_peaks()`。

模型命中框作为 peer 区间，防止信号候选边界侵入模型峰。信号候选只有满足 `SNR>=10`、有效连续点数>=5及面积门控后才保留。

信号峰的 `c` 使用：

`(SNR/(SNR+10))^0.8 * min(n_points/10,1)^0.2`

### 4.8 最终去重和返回

移植 `_dedup_overlapping_peaks()`。判为重复必须同时满足：

1. 峰顶距离不超过0.2 min；
2. 重叠宽度/较窄区间宽度至少25%；
3. 基线校正后的谷底/较小峰顶至少70%，即没有深谷。

重复时模型峰优先，同来源保留峰顶强度更高者。深谷分隔峰必须同时保留。

最终按 `a` 排序写入既有 `peaks[]`：

- 模型峰：`a/b=模型框RT边界`，`c=Softmax分数`；
- 信号峰：`a/b=信号精修边界`，`c=信号工程分`；
- 至少一个峰：`status=ok`；
- 无有效峰：`status=alert`。

不改变 `{a,b,c}` 结构，不增加后端必需字段。

## 5. C++文件级改动建议

| 文件 | 改动 |
|---|---|
| `src/signal_processing.h/.cpp` | SciPy等价平滑、百分位数、候选枚举、prominence、浅谷合并、边界精修、SNR、面积、信号分、去重 |
| `src/roi_renderer.h/.cpp` | 候选窗口渲染、移除边距、统一像素到RT映射 |
| `src/task_manager.cpp` | 从整通道推理改为候选批量推理；候选映射、模型/信号合并、去重、状态生成 |
| `src/onnx_inference.cpp` | 阈值比较与 Python 对齐；ONNX张量协议保持不变 |
| `src/json_protocol.*` | 原则上不改结构，只验证多峰完整序列化 |
| `include/mrmpformer.h` | 不改，保持 ABI |
| `tests/` | 增加Python黄金数据、渲染像素、候选、边界、信号门控、去重和端到端一致性测试 |

## 6. 测试与验收顺序

1. Python导出固定通道的平滑数组和候选索引，C逐元素比较；
2. 比较候选窗口边界和400×300输入像素；
3. 用同一ONNX Runtime CPU比较每个query的 `score/box`；
4. 比较候选框分配；
5. 比较模型未命中候选的信号边界、SNR、点数、面积和信号分；
6. 比较去重前后峰集合；
7. 比较最终有序 `{a,b,c}`；
8. 对 test1、test2、test3 全量回归，并重点覆盖宽峰、开叉峰、肩峰、相邻深谷峰和多于3峰通道；
9. 在 CPU 通过后再验证 CUDA，记录允许的浮点误差。

## 7. 实施顺序

建议分为三个可审查提交：

1. **候选和渲染对齐**：不改后处理，先保证送进ONNX的数据一致；
2. **模型匹配和信号路径对齐**：加入全部候选保留逻辑；
3. **去重、状态和端到端回归**：保证最终 `{a,b,c}` 与Python参考一致。

在每一步一致性测试通过前，不覆盖后端当前生产 DLL。
