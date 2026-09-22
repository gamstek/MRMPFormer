# MRMPFormer 项目全量文档

> 本文件为 MRMPFormer（GamstekPeaking）项目的**单一全量整合文档**，统一收录项目全部文档内容：
> README、项目报告、实验报告（含三个实验日志与全部附录）、四模型联合评估、FDR 架构设计、
> 已知问题汇总、C API 迁移指南、基线补齐清单（imporove）、QC 使用指南、特殊峰隔离实验报告、
> 以及 SimCLR / 色谱增强 / benchmark 拆分等各项设计文档。
>
> 初次整理日期：2026-09-07；接口交付基线更新：2026-09-22。
> 原始分散文档仍保留于各自原位，本文件为按主题重组的全量汇编，便于整体查阅。
>
> **阅读优先级**：第 0 章描述 2026-09-22 当前源码中的接口交付实现，是交付、联调和验收的
> 权威口径。后续章节保留项目历史与普通 Python pipeline 说明；如与第 0 章冲突，以第 0 章和
> 当前源码为准。

---

## 目录

0. [当前接口交付基线（2026-09-22）](#0-当前接口交付基线2026-09-22)
1. [项目总览](#1-项目总览)
2. [技术架构](#2-技术架构)
3. [数据体系](#3-数据体系)
4. [核心算法与模型架构](#4-核心算法与模型架构)
5. [推理管线](#5-推理管线)
6. [质量控制体系（QC）](#6-质量控制体系qc)
7. [评估体系](#7-评估体系)
8. [实验历程与关键发现](#8-实验历程与关键发现)
9. [四模型联合评估](#9-四模型联合评估)
10. [特殊峰隔离实验](#10-特殊峰隔离实验)
11. [桌面端 GUI 与工具链](#11-桌面端-gui-与工具链)
12. [C/C++ 共享库](#12-cc-共享库)
13. [工程治理与开发流程](#13-工程治理与开发流程)
14. [已知问题汇总](#14-已知问题汇总)
15. [QuanFormer 基线补齐清单](#15-quanformer-基线补齐清单)
16. [数据增强与 SimCLR 尝试](#16-数据增强与-simclr-尝试)
17. [四种分析模式教程摘要](#17-四种分析模式教程摘要)
18. [环境要求与快速开始](#18-环境要求与快速开始)
19. [当前状态与后续规划](#19-当前状态与后续规划)

---

## 0. 当前接口交付基线（2026-09-22）

本章完整描述当前用于软件接口交付的前处理、模型、后处理、C ABI、嵌入运行时、结果协议、
发布内容和验收边界。交付路径是 **MassNova 整谱模式的数组入口**，不是普通的标注驱动
`pipeline`，也不是早期“C++ 自己实现整套峰算法”的版本。

**阅读口径**：对外接口实现、联调和验收一律以本章为准。后续章节保留项目研发历史、训练实验和
普通 `pipeline` 说明；如其旧描述与本章冲突，不得据此覆盖当前 MassNova/C 接口交付行为。

### 0.1 交付结论与版本身份

| 项目 | 当前交付口径 |
|---|---|
| 公共接口 | `cpp/include/mrmpformer.h` 中的稳定 `qf_*` C ABI，工程版本 0.2.0 |
| 模型文件 | `model/checkpoint/mrmpformerv2.onnx` |
| 模型真实身份 | `mrmpformer_special_v2`，为兼容软件接入沿用 `mrmpformerv2` 文件名 |
| ONNX SHA-256 | `d0b612a786fc937ee44170a4e60e59a59d23fe7acdc2b85f5df302ae2dbee039` |
| PyTorch SHA-256 | `aa5944daec6d50428f758abc363d0901e339817fe181d0da2f4c16bc9cb3cd89` |
| 算法唯一实现 | `model/inference/massnova.py` + `massnova_runtime.py` |
| C++职责 | ABI、输入校验、数组复制、异步任务、超时/取消/回调、结果文件和 Python 桥接 |
| 推理运行时 | DLL 同进程内嵌 CPython，优先导入编译后的 Cython `.pyd`，缓存一个 ONNX Session |
| 默认设备 | `use_gpu=0`，优先 CUDA、初始化失败自动回退 CPU；`-1` 强制 CPU；`1` 强制 CUDA |
| 统一最终阈值 | `0.5`；模型候选和最终所有返回峰均严格要求 `c > 0.5`，包括信号兜底峰 |
| 当前 C 结果 | `items[].peaks[].a/b/c`，分别为峰左边界、右边界、峰自身分 |

重要说明：公开文件名不能用于判断模型代际，交付时必须同时核对
`model/checkpoint/mrmpformerv2_manifest.json` 的 `model_version`、大小和哈希。

### 0.2 交付架构与调用链

```text
MassNova/客户软件
  └─ 每条 MRM transition：uid/name/channel/mzq1/mzq3 + RT[] + intensity[]
      └─ qf_process_single() 或 qf_process(JSON 文件)
          └─ mrmpformer.dll
              ├─ api.cpp：ABI、参数校验、错误边界
              ├─ task_manager.cpp：任务队列、超时、取消、回调、结果落盘
              └─ python_bridge.cpp：同进程 CPython + JSON/数组桥接
                  └─ inference.massnova_bridge_native（Cython，失败时开发态回退 .py）
                      └─ MassNovaArrayRuntime（一次初始化、复用 ONNX Session）
                          ├─ 通道前处理与 QC
                          ├─ 候选峰枚举
                          ├─ 400×300 候选窗口渲染
                          ├─ mrmpformerv2.onnx 批量验证
                          ├─ 模型未命中候选的信号边界兜底
                          ├─ 门控、去重、最终指标和峰分
                          └─ Python 直接生成最终 JSON，C++ 原样保存/返回
```

生产 `mrmpformer` target 只编译 `api.cpp`、`json_protocol.cpp`、`python_bridge.cpp`、
`task_manager.cpp`。早期 `onnx_inference.cpp`、`signal_processing.cpp`、`roi_renderer.cpp`
不进入生产 DLL；它们仅保留给历史测试/回滚参考，不能被视为当前交付算法。

#### 0.2.1 单条 transition 的完整执行顺序

| 顺序 | 所属阶段 | 输入 | 当前实现 | 产物/失败行为 |
|---:|---|---|---|---|
| 1 | 接口前处理 | C 结构体或批量 JSON | 校验字段类型、有限值、数组长度和 RT 严格递增 | 非法输入整次提交失败，不进入模型 |
| 2 | 接口前处理 | 调用方字符串与数组 | C++ 深复制为任务私有内存 | 调用返回后调用方可释放原数组 |
| 3 | 数组前处理 | RT、原始 intensity | 转 `float64`，按单条/全局 sigma 做高斯平滑 | 得到后续全流程共用的平滑 XIC |
| 4 | 通道 QC | 平滑 XIC | 检查点数和最大强度 | 不合格 item 返回 `CHANNEL_LOW_INTENSITY` |
| 5 | 候选前处理 | 平滑 XIC | 仅检测副本裁负值为 0，排除 void-time 前沿，估计候选基线 | 得到候选检测信号，不修改面积信号 |
| 6 | 候选枚举 | 检测信号 | `find_peaks` 高度/prominence/距离门 + 浅谷合并 + 数量上限 | 得到每个候选 apex 索引 |
| 7 | 模型前处理 | 每个候选 apex | 切 `apex±1 min`，渲染 400×300 无轴 JPEG | 每个候选对应一个模型窗口 |
| 8 | 模型推理 | RGB `[B,3,H,W]`、`img_size=[W,H]` | 缓存 ONNX Session 分批推理 | 输出每个 query 的 score 与框 |
| 9 | 模型后处理 | score、像素框 | 严格 `score > threshold`，x 坐标线性映射 RT | 得到候选可用模型框 |
| 10 | 候选匹配 | 模型框、候选 apex | apex 必须落框；按高分、近中心贪心一对一匹配 | 命中候选成为模型峰 |
| 11 | 信号兜底 | 未命中候选 | 两遍边界外推、peer 防撞、半高宽保险丝 | 得到信号候选边界 |
| 12 | 信号门控 | 信号候选边界 | 本地 SNR、连续有效点数、可选 raw 面积门 | 不合格信号候选删除 |
| 13 | 合并去重 | 模型峰 + 信号峰 | apex/重叠/谷深联合去重 | 模型峰优先，同来源取更高 apex |
| 14 | 指标重算 | 最终候选边界 | 重算 SNR、连续点数、raw area | 模型峰和信号峰指标同口径 |
| 15 | 峰分 | 来源 + 指标 | 模型峰取 softmax；信号峰算 SNR/点数工程分 | 得到最终 `peak_score` |
| 16 | 最终过滤 | `peak_score` | 所有来源统一严格 `peak_score > threshold` | 等于阈值也删除 |
| 17 | 完全同框去重 | 过滤后的峰 | 左右边界均在 `1e-6 min` 内视为同框 | 模型优先，其次高分 |
| 18 | 接口输出 | 最终峰 | 只序列化 `a=rt_min,b=rt_max,c=peak_score` | 有峰 `ok`；无峰 `NO_PEAK_FOUND` |

其中第 1～7 步属于交付版前处理，第 8 步是 ONNX 模型，第 9～18 步属于模型外后处理。
模型权重不直接读取 RT/强度数组；它只读取由候选窗口渲染得到的图像。

### 0.3 客户侧前处理与输入契约

#### 0.3.1 客户软件必须完成的前处理

C 接口接收的是已经拆分好的单条 transition 时序数组，不负责读取 mzML。客户软件必须：

1. 将每个 MRM transition 独立成一个 item，提供稳定的 `uid`、`channel`、`mzq1`、`mzq3`；
2. 排除 TIC/非 transition 通道；同一真实通道不要重复提交；
3. 将 RT 转成**分钟**，保证严格递增；
4. 保证 RT 与强度等长、至少 2 点、全部为有限数；
5. 在任务完成前可释放调用方数组，因为 `qf_process_single` 返回前 DLL 已完成深复制；
6. C 与 Python 对照时使用同一个 ONNX、阈值、平滑配置和 provider。

Python mzML 入口额外执行 native ID 解析、Q1 提取、TIC 排除、真重复判定和秒/分钟启发式转换；
这些属于文件前端适配，不会在 DLL 数组入口中重复执行。因此客户若传入秒，所有峰宽、±1 min
窗口、SNR 邻域和边界都会失真，无法与 Python 结果一致。

#### 0.3.2 C 单通道结构

```c
QfCompoundInput input = {
    .uid = "caffeine-quant",
    .name = "Caffeine",
    .channel = "195.0877>138.0550",
    .mzq1 = 195.0877,
    .mzq3 = 138.0550,
    .smooth_sigma = 0.0f,
    .rt = rt_minutes,
    .intensity = intensity,
    .n_points = n_points,
};
```

`QfCompoundInput.smooth_sigma > 0` 时覆盖全局平滑值；小于或等于 0 时沿用
`QfConfig.smooth_sigma`。全局默认 `0.8`，全局设为 `0` 才表示关闭平滑。

#### 0.3.3 批量 JSON

```json
{
  "items": [
    {
      "uid": "caffeine-quant",
      "name": "Caffeine",
      "channel": "195.0877>138.0550",
      "mzq1": 195.0877,
      "mzq3": 138.0550,
      "smooth_sigma": 0.0,
      "x": [0.0, 0.1, 0.2, 0.3],
      "y": [100.0, 1200.0, 6000.0, 300.0]
    }
  ]
}
```

#### 0.3.4 字段级校验规则

| 字段 | C/JSON 规则 | 是否参与当前峰算法 |
|---|---|---|
| `items` | JSON 根节点必须是对象，`items` 必须是数组；任一 item 非法会使整个 JSON 提交失败 | 批处理容器 |
| `uid` | 必需字符串；建议业务侧保证稳定且非空，结果按原值回传 | 只用于结果关联，不参与搜峰 |
| `name` | 可选字符串，缺省为空 | 当前数组运行时不参与数值计算 |
| `channel` | 必需且非空字符串 | 接口身份字段；每个 item 独立处理，不按 channel 自动合并 |
| `mzq1/mzq3` | 必需有限数 | 接口身份字段；当前数组搜峰不使用 m/z 数值 |
| `mz` | 旧字段被明确拒绝，必须改为 `mzq1+mzq3` | — |
| `x`/`rt` | `double/float64` 一维有限数，至少 2 点，严格递增，单位必须为分钟 | 全部边界与积分的时间轴 |
| `y`/`intensity` | 与 RT 等长的一维有限数 | 平滑、检测、模型渲染、SNR 和面积信号 |
| `smooth_sigma` | 可选有限数；`>0` 覆盖全局，`<=0` 使用全局值 | 控制该 item 高斯平滑 |

批量 JSON 在 C++ `json_protocol.cpp` 先校验；单条 C 结构在 `api.cpp::copy_single_input`
校验。两条入口之后都转换成同一种 `CompoundData`，所以进入 Python 前的数组契约一致。字段错误返回
接口错误并使任务不运行；只有“结构合法但信号质量不够”才是 item 级 `alert`。

客户应提交**未平滑的原始强度**，由 DLL 按配置统一平滑。若客户已经自行平滑，应把全局
`QfConfig.smooth_sigma` 设为 0，否则会发生重复平滑。`uid/name/channel/mzq1/mzq3` 不会改变
峰边界或分数；同一 transition 被重复提交会得到两份独立结果，DLL 不负责业务去重。

### 0.4 DLL 内部前处理

数组进入 `MassNovaArrayRuntime._prepare()` 后按以下顺序处理：

1. 转为 `float64` 一维数组；检查等长、有限、至少 2 点、RT 严格递增；
2. 使用 SciPy `gaussian_filter1d` 平滑；默认 `sigma=0.8`，边界行为沿用 SciPy；
3. 在平滑后的通道上执行通道 QC：默认点数至少 10、最大强度至少 1000；
4. QC 失败不终止整个任务，而是为该 item 返回 `status="alert"`、空 `peaks` 和
   `CHANNEL_LOW_INTENSITY`，detail 中包含点数与最大强度；
5. 通过 QC 的通道进入候选枚举。候选检测使用非负化强度 `max(y, 0)`，后续模型、边界、
   SNR 和面积使用同一条平滑信号。

这条数组交付路径不会再排序 RT，也不会对 RT 重采样/插值；不会在信号前处理阶段做强度归一化、
面积基线扣除或负值整体裁零。负值裁零只用于候选枚举及部分点数门控，模型窗口渲染、边界、SNR
和最终 raw 面积仍以平滑后的原信号为输入。因此输入数组的时间单位、排序和采样质量属于接口契约。

对应的等价伪代码如下：

```python
rt = asarray(input_rt, dtype=float64)
raw = asarray(input_intensity, dtype=float64)
sigma = item_sigma if item_sigma > 0 else config.smooth_sigma
y = gaussian_filter1d(raw, sigma=sigma) if sigma > 0 else raw.copy()

if config.min_chrom_points > 0 and len(rt) < config.min_chrom_points:
    return CHANNEL_LOW_INTENSITY
if config.min_max_intensity > 0 and max(y) < config.min_max_intensity:
    return CHANNEL_LOW_INTENSITY
```

`gaussian_filter1d` 使用 SciPy 默认边界模式；没有另做 Savitzky–Golay、移动平均、基线拟合、
强度缩放或采样点补齐。QC 顺序是先点数、后最大强度；两个阈值都可用 0 关闭，但
`enumerate_peaks()` 自身仍固定要求至少 10 点。

### 0.5 候选峰枚举

`enumerate_peaks()` 的当前生产逻辑：

1. 默认忽略 `RT < 0.5 min` 的溶剂前沿；有效区不足 5 点时回退整段；
2. 用有效区第 25 百分位估计候选基线，`dynamic=max(y)-baseline`；
3. 峰高门：`baseline + 0.04 × dynamic`；
4. prominence 门：`max(0.055 × dynamic, 绝对 prominence 下限)`；
5. 最小点距同时受 3 点和 `0.10 min / 中位 RT 步长` 约束；
6. 用 SciPy `find_peaks` 枚举；平台峰取平台中点；
7. 相邻候选只有谷位置处于中央 12%～88%，且谷低于较小峰顶的 90%，才保留为两个峰；
   否则合并并保留较高候选；
8. 单通道最多 50 个候选，超限时按峰顶强度取前 50，再按 RT 排序。

`scan_baseline_mode="local_valley"` 当前与 `global_percentile` 都执行同一百分位计算，尚未形成
独立的局部基线实现。绝对 prominence 下限默认是 0。另需注意，`enumerate_peaks()` 内部固定要求
至少 10 个点；即便调用方把通道 QC 的 `min_chrom_points` 调到 10 以下，少于 10 点仍不会产生候选。

候选阶段的关键计算可写为：

```text
y_det       = max(y_smooth, 0)
body        = rt >= 0.5 min                         # 少于 5 点则回退整段
baseline    = percentile(y_det[body], 25)
dynamic     = max(y_det[body]) - baseline
height      = baseline + 0.04 × dynamic
prominence  = max(0.055 × dynamic, 0)
median_step = median(diff(sort(rt)), diff > 0)
distance    = max(3, ceil(0.10 min / median_step))
```

`dynamic<=0`、`find_peaks` 无结果或谷合并后无候选时，不调用 ONNX，最终返回
`NO_PEAK_FOUND`。最大 50 候选的截断依据是 apex 强度，而不是模型分或面积。

### 0.6 候选窗口与 ONNX 模型逻辑

每个候选以峰顶为中心切约 `±1 min`，裁剪到通道 RT 范围，使用
`render_roi_jpeg()` 生成 400×300、蓝色 1.5 线宽、无坐标轴的 JPEG；x 轴固定为实际窗口，
y 轴由 Matplotlib 自动缩放。全部候选窗口统一交给一个缓存的
`OnnxWindowPredictor`，默认 batch size 为 128。

#### 0.6.1 实际交付模型结构

交付文件名虽然是 `mrmpformerv2.onnx`，实际 checkpoint 身份是 `mrmpformer_special_v2`：

| 模型项 | 交付值 |
|---|---|
| Backbone | ResNet-50，`dilation=false` |
| 位置编码 | sine |
| Transformer | Encoder 1 层、Decoder 3 层 |
| 隐维度/注意力头/FFN | 256 / 8 / 2048 |
| Queries | 3 |
| 分类 | 二分类 Focal 训练头；推理取 softmax 峰类别分数 |
| 边界精化 | 3 层 FDR，33 bins，`roi_width` 尺度，层权重 0.5/0.7/1.0 |
| 特殊峰改动 | log-width L1 开启，PW-CIoU 开启，QFL 未用于最终 v2 分类 |
| ONNX | opset 17，动态 batch/height/width |

训练损失和 matcher 只影响权重学习，不会在部署推理时再次执行；部署图只保留网络前向和导出包装器。

#### 0.6.2 图像生成与张量进入 ONNX

1. 每个候选单独取 `[apex-1, apex+1] min`，再裁剪到整条 RT 范围；窗口不足 2 点则跳过；
2. Matplotlib `Figure(4×3 inch, dpi=100)` 生成 400×300 图，蓝线宽 1.5，四边框和刻度全部隐藏；
3. x 轴强制固定为窗口 `[rt_lo,rt_hi]`；y 轴自动缩放，不做数值归一化后再画；
4. Pillow 以 RGB 读取 JPEG，转 `float32`，从 HWC 转成 CHW；同 batch 图像 H×W 必须一致；
5. Python 只堆叠成 `[B,3,H,W]` 的 0～255 数组，并传 `img_size=[W,H]`，不在图外重复归一化；
6. ONNX 图内部完成 `/255`、ImageNet mean/std、网络前向、softmax 和框坐标转换。

#### 0.6.3 ONNX Runtime 设备与 Session

- `use_gpu=0`（默认）：CUDA provider 可见时先 `preload_dlls()`，尝试
  `CUDAExecutionProvider + CPUExecutionProvider`；初始化异常则重新创建纯 CPU Session；
- `use_gpu=-1`：只创建 CPU Session；
- `use_gpu=1`：CUDA provider 不存在、初始化失败或实际回退 CPU 都直接报错；
- `qf_init()` 创建一个 Session，候选按 `batch_size=128` 分批，但 Session 在整个 active lifecycle
  内复用；`qf_is_gpu_enabled()` 返回最终实际 provider，而不是用户请求值。

当前 `mrmpformerv2.onnx` 内部仅包含：

1. 输入 RGB 像素 `/255`；
2. ImageNet mean/std 标准化；
3. `mrmpformer_special_v2` 网络前向；
4. 二分类 softmax 后取峰类别分数；
5. `cx,cy,w,h` 转换为像素 `x1,y1,x2,y2`。

输入/输出协议：

| 张量 | 类型与形状 | 说明 |
|---|---|---|
| `image` | float32 `[B,3,H,W]` | RGB 原始 0～255；客户/桥接层不得再次 `/255` 或 Normalize |
| `img_size` | float32 `[2]` | `(W,H)`，当前窗口通常为 `(400,300)` |
| `scores` | `[B,Q]` | 每个 query 的峰类别分数 |
| `boxes_xyxy` | `[B,Q,4]` | 像素坐标，直接用于映射 RT |
| `boxes_norm` | `[B,Q,4]` | 归一化 `(cx,cy,w,h)`，主要用于诊断 |

ONNX 不包含候选枚举、阈值过滤、候选匹配、RT 边界、信号兜底、面积/SNR、去重、峰分、
状态或 JSON 组装。这些全部由共享 Python 核心完成。

模型框使用严格 `score > threshold`，默认阈值 0.5。框的 x 边界按候选 RT 窗口线性映射；只有
候选峰顶落在框内才可匹配。匹配按分数优先、框中心到候选峰顶距离次优的顺序贪心分配。
命中候选直接采用模型框 RT 边界，不再用信号规则二次修改。

像素到 RT 只使用 `x1/x2`，`y1/y2` 不参与积分边界：

```text
rt_left  = rt_lo + clip(x1, 0, 400) / 400 × (rt_hi - rt_lo)
rt_right = rt_lo + clip(x2, 0, 400) / 400 × (rt_hi - rt_lo)
```

每个候选至多分配一个框，每个窗口内的 query 框也至多使用一次。模型峰绕过信号兜底的
SNR、连续点数和面积门控；它只受模型 `score > threshold`、候选 apex 落框、后续去重和最终统一
`peak_score > threshold` 约束。未命中、低于阈值或框未覆盖 apex 的候选才进入信号兜底。

### 0.7 模型未命中候选的信号后处理

所有未命中模型的候选独立进入信号兜底，不是“整个通道只补一个最高峰”：

1. 从峰顶附近默认 `±0.05 min` 初始段出发；
2. 第一遍以已确定模型框为 peer，向两侧搜索局部稳定尾噪声截停点；
3. 第二遍再把其他候选第一遍边界作为 peer，避免相邻峰彼此穿越；
4. 默认后验观察 5 点、均值倍数 1.25、单侧最大搜索跨度 1.0 min；
5. 用半高宽保险丝只收缩不扩张：单侧边界最多为该侧半高跨度的 1.5 倍；
6. 计算排除邻峰区间后的本地 SNR；默认要求 `SNR >= 10`；
7. 以全通道低 10% 强度均值作为点数基线，要求基线上方最大连续点数至少 5；
8. 面积门默认 `scan_min_area=0`，即关闭面积下限门。

#### 0.7.1 边界外推的精确行为

- 初始半宽不是永远固定 0.05 min，而是
  `max(0.05, 3×(rt_max-rt_min)/(n-1))`，避免采样过稀时初始框没有足够点；
- 默认 `stable_tail_mean` 在 apex 单侧最多 1.0 min 内取最外侧约 10% 点（至少 4 点），按相邻
  一阶差分的第 50 百分位筛低波动点后求均值；点数不足时回退到强度第 55 百分位；
- 从初始边界逐点向外走，首次进入单侧噪声阈值后，还要求外向 5 点均值不超过
  `threshold×1.25`，避免单个低点造成过早停止；
- 第一遍把已确认模型框当外部 peer；第二遍再把其他候选第一遍边界加入 peer，防止边界穿入邻峰；
- 半高宽保险丝使用整通道最低 10% 非负强度均值为 baseline，分别计算 apex 到左右半高点的跨度，
  最终单侧跨度不得超过对应半高跨度的 1.5 倍；保险丝只收缩，不扩张。

#### 0.7.2 本地 SNR 与信号门控

本地 SNR 优先使用“本峰边界到左右最近邻峰边界之间”的区段；没有可用邻峰时，各侧最多取
2.0 min 框外扇区。每侧只保留不高于该区段第 40 百分位的安静点；不足 3 点则退回整侧区段。

```text
apex       = max(max(y_in_peak, 0))
baseline   = median(左右安静点合并)
noise_pp_L = max(left_quiet)  - min(left_quiet)
noise_pp_R = max(right_quiet) - min(right_quiet)
noise_pp   = max(noise_pp_L, noise_pp_R)
SNR        = 2 × (apex - baseline) / max(noise_pp, 1e-10)
```

峰内不足 5 个采样点、没有任何一侧有效噪声、`apex<=baseline` 或 SNR 非有限时直接拒绝。
通过 SNR 后，还要求以整通道最低 10% 非负强度均值为基线的“最大连续基线上方点数”至少 5。
`scan_min_area>0` 时再检查 raw trapz 面积；默认 0 表示关闭。上述三类门只作用于信号兜底峰，
不反向否决已命中的模型峰。

### 0.8 合并、去重、最终指标与峰分

模型峰与通过门控的信号峰合并后，只有同时满足以下三项才判为重复：

1. 峰顶距离不超过 0.2 min；
2. 重叠宽度占较窄区间至少 25%；
3. 以全通道低 10% 均值校正后，峰间谷底/较小峰顶比例至少 70%，即没有深谷。

重复时模型框优先；同来源保留峰顶更高者。深谷分隔的相邻峰必须同时保留。最终按左边界
排序并编号，在最终边界上统一重算 `n_points`、本地 `snr`、`area` 和 `peak_score`。

当前内部面积公式是：

```text
area_raw = trapz(smoothed_intensity[rt_min:rt_max], rt_minutes) × 60
```

即当前 MassNova 面积是**平滑信号的原始梯形积分，没有先扣除积分基线**。候选基线、点数基线、
SNR 基线和去重谷深基线用于检测/门控，但不等于面积扣基线。

峰自身分：

- 模型峰：`peak_score = model_score`，`score_source="model"`；
- 信号峰：
  `signal_score=(SNR/(SNR+10))^0.8 × min(n_points/10,1)^0.2`，
  `peak_score=signal_score`，`score_source="signal_rule"`；
- 信号分是可审计工程证据分，不是 softmax 概率，也不是校准后的正确概率。

完成评分后会对模型峰和信号峰统一执行最终阈值过滤：仅保留有限且
`peak_score > QfConfig.threshold` 的峰，等于阈值也会被删除。因此默认配置下所有返回的 `c`
都严格大于 0.5。最后再按 `1e-6 min` 的左右边界容差移除完全相同区间；相同区间优先保留
模型峰，其次保留分数更高者，并重新按 RT 编号。

#### 0.8.1 当前交付版后处理参数全集

| 参数 | 默认值 | 作用阶段 |
|---|---:|---|
| `scan_baseline_percentile` | 25 | 候选基线分位数 |
| `scan_baseline_mode` | `global_percentile` | `local_valley` 当前仍走同一分位实现 |
| `scan_min_peak_ratio` | 0.04 | 候选高度相对 dynamic 门 |
| `scan_prominence_ratio` | 0.055 | prominence 相对 dynamic 门 |
| `scan_min_prominence_abs` | 0 | prominence 绝对下限 |
| `scan_min_peak_gap_points` | 3 | 候选最小点距下限 |
| `scan_min_peak_width_min` | 0.10 min | 候选 RT 距离下限 |
| `scan_void_time_min` | 0.5 min | 排除前沿 |
| `scan_max_peaks_per_channel` | 50 | 单通道候选上限 |
| `scan_init_half_width_min` | 0.05 min | 兜底初始半宽下限 |
| `scan_boundary_posterior_lookahead` | 5 点 | 边界截停后验窗 |
| `scan_boundary_posterior_mean_scale` | 1.25 | 后验均值阈值倍数 |
| `scan_edge_noise_stop_mode` | `stable_tail_mean` | 单侧边界噪声估计 |
| `scan_edge_max_span_min` | 1.0 min | 边界阈值估计最大单侧跨度 |
| `scan_width_fuse_ratio` | 1.5 | 半高宽保险丝倍数；0 关闭 |
| `scan_min_snr` | 10 | 信号兜底 SNR 门 |
| `scan_min_peak_span_points` | 5 | 最大连续有效点数门 |
| `scan_min_area` | 0 | raw 面积门；0 关闭 |
| `scan_window_half_min` | 1.0 min | 候选模型窗口半宽 |
| `scan_dup_apex_tol` | 0.2 min | 重叠峰去重 apex 距离 |
| `scan_dup_min_overlap_fraction` | 0.25 | 相对较窄框最小重叠比例 |
| `scan_dup_shallow_valley_min_ratio` | 0.70 | 无深谷判定比例 |
| `signal_score_snr_pivot` | 10 | 信号分 SNR 枢轴 |
| `signal_score_points_good` | 10 | 信号分点数满分参考 |
| `signal_score_snr_weight` | 0.8 | 信号分 SNR 权重 |
| `signal_score_points_weight` | 0.2 | 信号分点数权重 |

公开 `QfConfig` 目前只暴露 threshold、smooth、设备、batch 和通道 QC 等核心项；上表扫描参数在
嵌入数组入口中使用 Python 默认值，客户不能通过现有 v0.2.0 ABI 单独修改。若要开放这些参数，
应通过版本化配置接口实现，不能直接破坏现有结构体布局。

### 0.9 C 接口结果协议与当前面积缺口

当前 DLL 数组运行时把最终峰序列化为：

```json
{
  "items": [
    {
      "uid": "caffeine-quant",
      "status": "ok",
      "peaks": [
        {"a": 7.621234, "b": 7.913456, "c": 0.932145}
      ],
      "alerts": []
    }
  ]
}
```

| 字段 | 当前含义 |
|---|---|
| `a` | 最终峰左边界，分钟 |
| `b` | 最终峰右边界，分钟 |
| `c` | 模型分或信号规则分，取决于峰来源 |
| `status="ok"` | 至少存在一个通过最终 `c > threshold` 的模型峰或信号峰 |
| `CHANNEL_LOW_INTENSITY` | 通道点数或最大强度 QC 失败 |
| `NO_PEAK_FOUND` | 通道 QC 通过，但最终无峰 |

**当前缺口必须明确**：

- Python MassNova 内部虽然计算 `area/snr/n_points/source`，但
  `MassNovaArrayRuntime._peak_result()` 只序列化 `a/b/c`，C API 没有把面积返回给客户；
- 普通标注驱动 `pipeline` 的最终 `prediction_refined_with_area.csv` 使用 SNR 线性基线扣除积分，
  该结果不会流入 MassNova 数组接口；
- `configs/massnova.json` 中的 `integration_method="linear"` 当前明确为 MassNova 忽略项；
- 因此当前接口既没有返回 raw area，也没有返回扣基线 area，更没有
  `integration_method_used` 字段。

若后续补齐接口，建议保持 `a/b/c` 向后兼容，同时增加：

```json
{
  "area_raw": 123.4,
  "area_baseline_corrected": 101.2,
  "area": 101.2,
  "integration_method_used": "snr"
}
```

并将 `raw/linear/snr/external_baseline` 作为版本化配置贯通到 `QfConfig` 或新的 v2 初始化接口。
直接扩展现有 `QfConfig` 会改变结构布局，必须评估 ABI 兼容；更安全的是新增带 `struct_size`/
`api_version` 的配置结构或新初始化函数。

需要修改的代码位置：

| 层 | 文件与位置 | 必需改动 |
|---|---|---|
| 公共 ABI | `cpp/include/mrmpformer.h` 的 `QfConfig` | 新增版本化积分模式，或增加 v2 配置 API |
| 默认值/校验 | `cpp/src/api.cpp` 的 `valid_config`、`qf_default_config_impl` | 默认方法与合法值校验 |
| C→Python 配置 | `cpp/src/python_bridge.cpp::config_json` | 把积分模式写入初始化 JSON |
| 嵌入默认值 | `model/inference/massnova_runtime.py::DEFAULT_RUNTIME_CONFIG` | 接收并保存积分模式 |
| 共享后处理 | `model/inference/massnova.py::_finalize_peak_metrics` | 同时计算 raw 与扣基线面积 |
| 调用贯通 | `finalize_channel_peaks`、`MassNovaArrayRuntime.process_items` | 传递积分模式 |
| C JSON 输出 | `MassNovaArrayRuntime._peak_result` | 返回面积、SNR、点数和方法 |
| 历史内部协议 | `cpp/src/json_protocol.h/.cpp` | 若继续保留内部 `PeakResult`，同步扩展测试；当前生产返回实际由 Python 生成 |
| 测试 | `model/tests/test_massnova_runtime*.py`、`cpp/tests/test_api.cpp` | 验证 raw/校正面积、默认值、JSON 字段和端到端一致性 |
| 文档/示例 | `cpp/README.md`、`docs/CPP_API_DIFFERENCES.md`、C examples | 更新配置和结果契约 |

### 0.10 普通 pipeline 与接口交付路径的区别

| 对比项 | 普通 `pipeline` | 接口交付 `MassNovaArrayRuntime` |
|---|---|---|
| 输入 | mzML + 人工标注/目标列表驱动 ROI | 客户传每条 transition 的 RT/强度数组 |
| 搜峰范围 | 目标 RT 附近 ROI | 整条 transition 全峰枚举 |
| 中间模型面积 | `model_prediction_*.csv` 当前 raw 积分 | 内部 raw 积分 |
| 最终面积 | `prediction_refined_with_area.csv`/`all.csv` 用 SNR 基线扣除 | 当前未向 C JSON 返回；内部仍为 raw |
| 最终接口峰 | CSV/报告字段较多 | 仅 `a/b/c` + status/alerts |

不能把普通 pipeline 的最终面积说明直接套到 C/MassNova 交付接口上。

### 0.11 C API 生命周期、线程与所有权

1. `qf_default_config(&config)` 初始化全部字段；设置 `model_path` 后调用 `qf_init()`；
2. 同一进程同时只允许一个 active runtime；`qf_shutdown()` 成功后可以重新初始化；
3. `qf_init()` 初始化内嵌 Python、导入桥接模块并创建一个缓存 ONNX Session；
4. `qf_process()` 读取批量 JSON；`qf_process_single()` 深复制单通道字符串和数组后入队；
5. 使用 `qf_query/qf_wait/qf_stop` 管理异步任务；默认超时 300 s，0 表示关闭；
6. `qf_set_callback` 注册进程级回调；回调应短小且不得在回调内调用 `qf_shutdown()`；
7. `qf_get_result_json` 返回的内存必须由 `qf_free` 释放；
8. `qf_get_result_path` 返回已落盘结果路径，结果文件不会自动由调用方之外删除；
9. `qf_get_error` 是线程局部错误文本，应在失败调用后立即读取；
10. `qf_is_gpu_enabled` 返回实际 Session 是否启用了 CUDA provider。

`QfConfig` 当前默认值：

| 字段 | 默认值 | 作用 |
|---|---:|---|
| `work_dir` | `./mrmpformer_results/` | 结果 JSON 目录 |
| `max_workers` | `0` | 自动选择 1～4 个 worker；1 为串行 |
| `threshold` | `0.5` | 模型框和最终所有返回峰的严格分数门；要求 `c > threshold` |
| `smooth_sigma` | `0.8` | 全局高斯平滑；0 关闭 |
| `task_timeout_sec` | `300` | 单任务超时；0 关闭 |
| `use_gpu` | `0` | 自动优先 CUDA、失败回退 CPU；-1 强制 CPU；1 强制 CUDA |
| `batch_size` | `128` | 候选窗口 ONNX batch |
| `min_chrom_points` | `10` | 通道最少点数 |
| `min_max_intensity` | `1000` | 平滑后通道最大强度门 |

其余扫描、边界、门控、去重和信号评分参数目前由 Python 默认配置持有，没有全部暴露到公开 ABI。

### 0.12 构建与发布包

开发构建顺序（仓库根目录，Python 3.11）：

```powershell
uv venv --python 3.11 --seed .venv
uv pip install -r cpp/requirements-build.txt
.\.venv\Scripts\Activate.ps1
python model/tools/build_massnova_bridge.py
cmake -S cpp -B cpp/build -DBUILD_TESTS=ON -DPython3_EXECUTABLE="$((Get-Command python).Source)"
cmake --build cpp/build --config Release
ctest --test-dir cpp/build -C Release --output-on-failure
```

正式 Windows 集成包在 Git Bash 中执行 `bash build.sh`，输出到 `build/windows/`；
`bash build.sh --install-deps` 可创建环境并安装依赖，`--skip-build` 用已有构建产物重新打包和验收。
发布脚本会构建 Cython 与 DLL、运行 CTest、收集私有运行时，并在移除开发路径后对搬移副本执行真实
C 接口验证；任一发布检查失败都不会产出可交付包。

Windows 交付包至少包含：

```text
  build/windows/
  mrmpformer.dll
  mrmpformer.h
  mrmpformer.lib
  mrmpformerv2.onnx
  python311.dll
  python311.zip
  python311._pth
  README_INTEGRATION.md
  python/
    DLLs/                       # Python 标准库原生扩展
    inference/                  # 编译后的 MassNova .pyd + 必需辅助模块
    utils/
    preprocessing/
    Lib/site-packages/          # numpy/scipy/Pillow/matplotlib/onnxruntime 等
```

客户机器无需另装 Python，但交付包不再是“DLL + ONNX Runtime + ONNX”三个文件。DLL 会把自身目录、
`python/` 和 `python/Lib/site-packages/` 加入 `sys.path`；`MRMPFORMER_PYTHON_PATH` 只用于开发覆盖。
数组入口不需要 `pyopenms`，因为 mzML 解析由客户软件完成。

### 0.13 当前验证状态与发布闸门

已完成：

- Python MassNova CLI 与数组入口共享 `enumerate_peaks`、`validate_with_model`、
  `finalize_channel_peaks`；
- Windows 私有 CPython、四个 Cython `.pyd`、C DLL、ONNX Runtime CPU/CUDA provider 和模型已纳入
  `build/windows/` 自动打包；集成包搬移到含中文和空格的隔离路径后，真实 C 接口可启动；
- 根目录 `tests/inputs/` 固化 256 条真实 MRM 色谱，来自 `test3_1/test3_3/test3_7/test3_56`
  四个分层验收集，每组 64 条，保留原始未平滑、未重采样的 x/y；
- 256 条输入逐例比较源码 Python、编译 Cython 和 C DLL，覆盖 batch/single，共 1536 次两两比较；
  校验 UID、状态、告警、峰数、顺序和每个 `a/b/c`，并检查所有返回 `c > threshold`；
- 数值容差为 `atol=1e-12`、`rtol=1e-10`；至少 50 条必须实际产峰，避免空结果伪通过；
- 设备策略已有 CPU、自动回退和强制 GPU 行为检查；无 NVIDIA GPU 的验收机已实测 CPU 与自动回退；
- 模型峰、多个信号峰、无峰、低强度 QC、ONNX 输入、重复窗口和严格最终阈值语义均有回归覆盖；
- ONNX 合同：opset 17，`scores/boxes_xyxy/boxes_norm`，400×300 及变尺寸 batch 已核验。

尚未完成、不能误写为已验收：

- 当前无 NVIDIA GPU 的机器不能替代目标 CUDA 12/cuDNN 9 环境上的真实 GPU 推理验收；
- 256 条回归证明的是实现一致性，不等价于有标注 LC-MS 数据上的检测准确率或 GPU/CPU 数值等价；
- C 接口扣基线面积与面积字段尚未实现。

正式发布以 `bash build.sh` 全部闸门通过为准，包括 CTest、256 条三入口一致性、设备检查、
隔离搬移冒烟、依赖/清单检查和模型哈希；GPU 版仍需在实际目标硬件上补充验证。

---

## 1. 项目总览

### 1.1 项目定位

**GamstekPeaking**（MRMPFormer）是**引力波智谱**科学智能部自主研发的 **LC-MS 代谢组学色谱峰检测与定量工具包**。其核心是 **MRMPFormer**——一种基于 **DETR（ResNet-50 + Transformer）系列** 的色谱峰检测模型。

核心思路：在提取离子色谱图（EIC）生成的 ROI 图像上训练目标检测网络，识别真实色谱峰并定位峰边界，进而对峰区间积分得到峰面积，实现定量分析。

### 1.2 输入与输出

| 项目 | 说明 |
|---|---|
| **输入** | `.mzML` 原始质谱数据（Targeted 模式另需含 `Compound Name / mz / RT` 的特征表） |
| **输出** | `prediction_refined.csv`（峰面积 + 置信度）、预测框标注图、SNR 报告、QC 系列报告表 |
| **模型** | ResNet-50 骨干 + Transformer 编解码器（hidden_dim=256, nheads=8, num_queries=3） |
| **开发版本** | v2.8.13 |
| **当前开发范围** | 仅 **Targeted × Centroided（MRM 靶向定量）** 模式；其余三组合保留现状、暂不开发 |

### 1.3 三大板块

1. **格式转换前处理**（`converters/` + `desktop/`）：厂商原始格式（`.msdata` / `.wiff`）→ 标准 `.mzML`，并提供桌面图形界面；
2. **MRMPFormer 模型**（`model/`）：训练、推理、评估三段式闭环；
3. **后处理**（`model/postprocessing/`）：SNR 筛选、峰区间精修、积分定量。

### 1.4 核心工作流：四环节闭环

| 板块 | 做什么 | 主要入口 | 核心产物 |
|---|---|---|---|
| ① 前处理 | 原始数据格式转换 → mzML；标注 QC（防线1）；EIC/ROI 提取（标注驱动，防线2） | `converters/`、`model/preprocessing/` | `data/mzml/`、`xic_roi/<样品>/`、QC 表 |
| ② 训练 | 标注 xlsx + mzML → COCO bbox 数据集（bbox 直接映射人工 `peak_start/peak_end`）→ 训练/微调 DETR | `model/preprocessing/coco_annotation.py`、`python -m train --config ...` | `data/coco/`、`checkpoint/*.pth` |
| ③ 推理（含后处理） | ROI → 模型预测 → SNR 筛选 → 峰区间精修 → 面积积分 → 推理报告 | `python -m inference.cli --mode pipeline ...` | `predictions_model/`、`prediction_refined/`、`all.csv`、`inference_report_*.md` |
| ④ 评估 | 预测值 + 人工标注计算检测与定量指标 | `python -m tools.evaluation.evaluate_baseline` | `evaluation_report.json`、`match_details.csv`、`area_pairs.csv` |

**要点**：
- 标注数据集双重身份（训练 bbox 来源 + 评估 GT），两环节共用一份标注是预期设计；
- 评估默认读**原始 `model_prediction_<样品>.csv`**（衡量模型原始检测能力），不含后处理；
- 训练/评估分样品隔离（train=`test_1`、val=`test_2`），防数据泄漏。

---

## 2. 技术架构

### 2.1 目录结构

```
MRMPFormer/
├── requirements.txt              # pip 依赖（model + desktop 合并，GPU 分段）
├── environment.yml               # Conda 环境（name: gamstekpeaking）
├── CLAUDE.md                     # AI 辅助指令（项目级全局约束）
├── README.md / User_Tutorials.md / imporove.md / blank_label.md
├── dev_log.md                    # 项目开发日志（强制维护）
├── model/                        # ⭐ 核心代码（八子包扁平结构）
│   ├── train.py                  #   训练入口（python -m train ...）
│   ├── configs/                  #   参数配置文件（--config 外置默认参数）
│   ├── inference/                #   推理：cli.py 统一入口 + predictor + massnova + two_round_detection
│   ├── models/                   #   模型定义（quanformer / mrmpformer_v1(FDR) / simclr / shared）
│   ├── preprocessing/            #   前处理：label_qc + xic_extraction + coco_annotation + ion_zenith
│   ├── postprocessing/           #   后处理：snr_filter + peak_refinement + valley_split + 面积积分
│   ├── framework/                #   DETR 训练框架（datasets / util / engine，fork 自 Facebook DETR）
│   ├── utils/                    #   推理辅助（io / quantify / mzml_load / roi_rt_mapping 等）
│   ├── tools/                    #   工具集：evaluation / batch / diagnostics / benchmark / experiments 等
│   ├── checkpoint/               #   模型权重（.pth + .onnx）
│   └── tests/                    #   单元测试
├── desktop/                      # PySide6 桌面 GUI（前处理已上线，其余占位）
├── converters/                   # 格式转换（msdata/wiff → mzML；含运行时 *_bin/）
├── cpp/                          # C++20 共享库（稳定 C ABI + 同进程 CPython/Cython 桥接）
├── data/                         # 数据（label / msdata / wiff / mzml / coco；整体 .gitignore）
└── docs/                         # 项目文档（报告 / 设计 / 本全量文档）
```

### 2.2 分层架构与单向依赖

- `model/models/`：纯 DETR 模型定义（不依赖业务逻辑）；
- `model/framework/`：DETR 训练框架（datasets / util / engine）；
- `model/inference/`：模型加载 + 预测 + 可视化 + 统一 CLI 入口；
- `model/preprocessing/`：mzML 加载 → EIC → ROI 图像；MS1 谱图聚合；
- `model/postprocessing/`：积分定量 + 峰质量 + box↔RT 映射 + SNR 筛选 + 峰区间精修；
- `model/utils/`：公共工具；`model/tools/`：批处理 / 诊断 / 可视化 / 基准测试；
- `desktop/`：PySide6 图形界面（独立于 CLI；workers 为算法模块的 Qt 薄包装）；
- `converters/`：格式转换工具。

各层单向依赖：`inference → preprocessing/postprocessing → models/framework → utils`。

### 2.3 硬性工程约束（CLAUDE.md）

| 类别 | 约束 |
|---|---|
| **开发范围** | 仅 MRM 模式；禁止新增功能/重构非 MRM 代码；公共代码改动须以 MRM 模式回归验证 |
| **环境** | Python 3.11；Conda 环境名固定 `gamstekpeaking`；PyTorch 按 GPU 选版本（RTX 50 系 cu128 ≥2.7.0 / RTX 40·30·20 系 cu124 2.6.0 / CPU·MPS 2.6.0） |
| **设备** | 禁止硬编码 `device='cuda'`，一律 `resolve_torch_device()`（CUDA > MPS > CPU） |
| **模型加载** | 禁止直接 `torch.load()`，一律 `safe_torch_load()` 兼容跨 PyTorch 版本 |
| **路径** | 禁止硬编码分隔符，一律 `os.path.join()` / `pathlib.Path` |
| **数据目录** | 「实验隔离 + 数据类型五分法」（coco/label/mzml/msdata/wiff）；中间产物一律写 `../output/` |
| **命令执行** | 训练/推理等真实运行命令整理后交用户执行（沙箱拦截 matplotlib 渲染 DLL 与 site-packages 写入）；沙箱内仅跑纯数据验证/逻辑单测/编译检查 |

### 2.4 平台与兼容性

- **Windows**：避免路径空格与中文；`pycocotools` 已有 cp311 wheel；
- **Linux**：无桌面环境运行 GUI 需 X11 转发或 xvfb；
- **macOS**：Apple Silicon 自动使用 MPS 加速。

---

## 3. 数据体系

### 3.1 数据目录规范（五分法）

```
data/
├── coco/     # COCO 训练数据集（train/ + train_coco.json、val/ + val_coco.json；_xic/ 为 XIC 中间产物）
├── label/    # 各实验人工标注（统一 data/label/<trial>.xlsx）
├── mzml/     # 转换得到的 mzML
├── msdata/   # 原始 msdata
└── wiff/     # 原始 wiff
```

数据流转：`wiff/ + msdata/ →(converters)→ mzml/ →(coco_annotation + label/)→ coco/`。

### 3.2 数据集清单

| 数据集 | 内容 | 用途 |
|---|---|---|
| **20260715_shiyaoyuan**（`data/test/`） | 试药园两次进样，60 化合物 ×2 离子标注（120 行） | 最初训练/微调与历次评测主数据 |
| **test1 标准试卷** | 11 mzML（8 农残混标 + 3 空白），96 个 GT 峰 | 四模型横评统一外域基准 |
| **test2** | 3 浓度梯度样，60 通道/样，≈187 GT 峰（含 44 特殊峰，多峰 14 行） | 低浓度/困难形态测评 |
| **traindata3** | 109 mzML（90 数字样品 + 9 空白 + 10 QC），16335 训练图 / 12677 框 | mrmpformerv1/v3 从零训练 |
| **traindatav1** | 合并 traindata1-3（label 修订后） | mrmpformerv3 / quanformerv3 全量重训 |
| **merged** | shiyaoyuan + traindata2（B 范式含负样本，7:3） | quanformerv3 微调 |
| **multisrc** | traindata3×98 + shiyaoyuan_1 + traindata2×5 + test1×7 = 17796 图 / 13929 框 | mrmpformerv2 多源联合训练 |

### 3.3 COCO 数据集构建（B 范式：标注驱动 ROI）

`preprocessing/coco_annotation.py` 从 mzML + 标注 xlsx 联合生成：

- **ROI 由标注驱动**：每行标注 `(compound, channel)` 经 `label_key` → `native_id`「化合物名-1/-2」匹配 mzML 色谱，**窗口中心 = 标注 `rt` 字段**（非谱图最高强度点）；未标注通道不生成 ROI；
- **正负样本显式区分**：`peak_label=0` 负样本（有图无 bbox）、`peak_label=1` 正样本（`peak_start1-3/peak_end1-3` 每个区间一个 bbox，最多 3 个）、其余值不入数据集；
- **bbox 映射**：`peak_start/peak_end`（分钟）经 `roi_windows.csv` 线性映射为像素（y 全高 [0,300]）；
- 图像规格与推理完全一致：400×300 JPEG，标注 RT ±1min 窗口；构建默认启用防线 1 标注 RT 一致性 QC。

### 3.4 标注格式

`data/label/<trial>.xlsx` 支持多峰格式：`peak_label` / `peak_count` / `peak_start1-3` / `peak_end1-3` / `area1-3` 等；单数 `peak_start/peak_end` 保留兼容旧文件。标注列：`compound` / `channel`（定量/定性离子）/ `rt` / `peak_label`（0=负 / 1=正）/ 多峰起止 / 面积。

---

## 4. 核心算法与模型架构

### 4.1 DETR 基础

模型基于 Facebook DETR（`model/framework/` fork 自官方），ResNet-50 骨干 + Transformer 编解码器，object queries 通过匈牙利匹配与 GT 配对，一次前向输出「框 + 类别」集合。

### 4.2 QuanFormer（基线架构）

- ResNet-50 骨干 + 1 层 Encoder + 1 层 Decoder（hidden_dim=256, nheads=8, dim_feedforward=2048, dropout=0.1）；
- num_queries=3（单图最多检出 3 个峰）；
- 损失：交叉熵分类（eos_coef=0.1）+ L1 回归（权重 5）+ GIoU/CIoU（权重 2）；
- 约 30M 参数。

**出身考据（重要）**：`checkpoint/quanformer.pth` 实为**外部下载权重**——DETR COCO 预训练 → 外部 autopeakV3 项目 peak-all 数据集（≈113 样品）微调 30+ epochs，非本项目从零训练（内嵌 `args` 字段为铁证）。其在 test1 上 F1@0.1=0.010 的差表现根因是外来框约定与评估协议系统性错位（窄框 ~0.23 min vs 人工宽积分边界 ~0.47 min，差 ~0.11 min），且训练集大概率不含空白负样本。

### 4.3 MRMPFormer v1：三层 Decoder + FDR 边界逐层精化（论文架构）

v1 是论文架构的原始实现（`model/models/mrmpformer/`），核心是 **FDR（Fine-grained Distribution Refinement，细粒度分布精化）**。

**网络结构**：

```
ResNet-50 → 1-layer Encoder → memory
Object Queries
Decoder Layer 1 → 完整二维初始框 b0=(cx,cy,w,h) + FDR Head1 → z1 → P1 → (l1,r1)
Decoder Layer 2 → FDR Head2 → Δz2; z2=z1+Δz2 → P2 → (l2,r2)
Decoder Layer 3 → FDR Head3 → Δz3; z3=z2+Δz3 → P3 → (l3,r3)

最终框：b*=(l3, t0, r3, b0_bottom)   ← 左右边界取 L3，上下边界取 L1 初始框
最终分类：仅采用 Decoder Layer 3 的峰概率
```

关键机制：
- **Logits 残差累加**：第 2、3 层预测上一层累计 logits 的残差（`z2=z1+Δz2`、`z3=z2+Δz3`），残差加在 Softmax 前 Logits 上；
- **逐层边界位置反馈**：每层精化出的左右边界 `(xL,xR)` 经 BoundaryPositionMLP 编码后加入下一层 Query 位置特征（默认可导，`detach_boundary_feedback=false` 可消融）；
- **非均匀 Bin**：33 bins，`W(n)=sign(u)|u|^p`（p=2.0），严格单调、对称覆盖 0；概率期望解码得到归一化偏移，尺度因子默认取初始框宽（`roi_width` 为另一选项）；
- 上下边界始终由第一层二维框头负责；禁止坐标残差双重累计；FDR Head 末层 Linear **零初始化**（训练初期恒等映射）。

**损失函数**（全部配置化）：
- 分类：**Softmax Focal Loss**（α=0.25, γ=2.0），主损失只取 L3，中间层辅助由配置控制；
- 定位：**动态加权 L1**（λ_c=1/(w_gt+ε)，窄峰中心定位加强）+ **PW-CIoU**（中心距离按 `bar_w/(w_gt+ε)` 加权，`bar_w` 来自训练集统计）；
- **FDR 分布监督**：三层累计分布均接受左右边界软标签 CE（层权重 0.5/0.7/1.0），软标签由真实边界偏移在非均匀 Bin 上两点线性插值生成（工程回退方案，非未经确认的论文 FGL 原式）；
- 参数量 30.45M。

**v1 已知缺陷（评估时点）**：matcher `iou_type` 未透传（恒用 GIoU，与 PW-CIoU 口径错位）；FDR 损失有效权重偏高（≈8.8）。

**FDR 架构测试**：`model/tests/test_mrmpformer_v1.py` 提示词 §15 全部 8 类测试 21 项全过（Shape / 残差零初始化恒等与逐元素累加 / 分布解码 / 最终框组装 / 分类唯一来源 L3 / 边界反馈梯度 / 损失 / legacy 迁移 / tiny-set 过拟合）。

### 4.4 mrmpformerv2：架构修复 + 多源联合训练（此前综合最优）

v2 = v1 热启动 + 三项关键优化：

| 优化项 | 内容 | 效果 |
|---|---|---|
| **P0-1 matcher 修复** | `build_matcher` 透传 `iou_type=ciou`，匈牙利匹配与 PW-CIoU 口径一致 | 边界精度提升主因 |
| **P0-2 损失再平衡** | `fdr_loss_coef` 2.0→1.0（有效权重 8.8→4.4） | 分类学习恢复 |
| **P2-8 多源联合训练** | multisrc 数据集（17796 图/13929 框），v1 热启动，AMP 30 epochs（61 分钟） | 假阳性抑制 + 跨域泛化 |

训练加速三项：`--amp` / `--tf32` / `--cudnn_benchmark`。`pw_ciou_mean_width` 按 multisrc 实测 0.1809。

### 4.5 mrmpformerv3（当前主模型）

在 traindatav1（traindata1-3 合并 + label 修订）上全量训练 30 ep 的 FDR 三层 Decoder 架构，权重名 `mrmpformer.pth`（含 `mrmpformer.pth`=v3 最终）。

### 4.6 训练入口与配置体系

- `train.py` 支持 `--config`（JSON 作为默认参数，CLI 可覆盖；无 pyyaml 故用标准库 json）；
- 配置含内联注释（`_comment_*` 键），加载统一过滤 `_` 前缀；
- 恢复训练自动跳过 `class_embed`/`query_embed`（维度不匹配），`--reset_optimizer` 支持纯权重微调；
- `build_predictor` 按 checkpoint `args.model` 路由变体；旧 QuanFormer 单层权重热启动 v1 自动走 `load_legacy_quanformer_state` 迁移（L1 参数复制初始化到 L2/L3）。

### 4.7 现成配置清单（`model/configs/`）

| 配置 | 用途 |
|---|---|
| `quanformer_baseline.json` | quanformer 基线从零训练 |
| `quanformer_v2_finetune.json` | quanformer 微调（产出 quanformerv2） |
| `quanformer_baseline_test1ft.json` | 对照：test1 微调基线 |
| `quanformer_v3_finetune.json` | quanformer 第 3 轮微调（产出 quanformerv3） |
| `mrmpformer_v1_fdr.json` | MRMPFormer v1（FDR 全开，产出 mrmpformerv1） |
| `mrmpformer_v1_multisrc.json` | v1 多源联合训练（产出 mrmpformerv2） |
| `mrmpformer_special_v1.json` / `mrmpformer_special_v2.json` | 特殊峰隔离版（QFL 止损 / focal 最终） |
| `inference_pipeline.json` / `massnova.json` | 推理管线 / 整谱扫描参数模板 |
| `evaluation_baseline.json` / `coco_annotation*.json` | 评估 / 数据集构建参数 |

### 4.8 训练期 COCO 评估指标

每 epoch 末输出 12 行标准 COCO 指标（AP 6 行 + AR 6 行）：
- AP@0.50:0.95 为横向可比主指标；AP50 宽松、AP75 对边界贴合极度敏感；
- 本项目 bbox 即人工积分边界，**AP50 与 AP75 之差 ≈ 边界贴合度 ≈ RT 起止偏差来源**；
- AR@1 贴近部署口径（每通道最终只保留面积最大框）；`num_queries=3` 使 AR@10=AR@100；
- small 档恒为 -1（val 1410 框中无 small 档，-1 表示未定义非 0）。

---

## 5. 推理管线

### 5.1 统一入口与模式

统一入口 `model/inference/cli.py`（`python -m inference.cli`），`--mode` 切换：

| 模式 | 说明 | 适用场景 |
|---|---|---|
| `pipeline` | 完整管线：ROI 提取 → 预测 → SNR 筛选 → 精修（单文件或目录递归） | ⭐ 生产环境（推荐） |
| `roi` | 仅 EIC/ROI 提取（无需 `--model`，无预测 CSV） | 检查 XIC/ROI 质量 |
| `roi2inference` | 对已有 XIC/ROI 中间结果目录批量预测+积分 | 续跑 / 断点恢复 |
| `massnova` | 整谱 XIC 全峰识别：全部 transition、不依赖标注；信号级找峰 + 可选模型验证 | MassNova 集成 / 仅时序数据 |

`roi`/`pipeline` 均支持 `--mzml` 单文件或 `--batch_dir` 目录递归扫描（含子目录）。

### 5.2 完整管线输出结构

```
../output/pipeline_batch/
├── xic_roi/<样品>/                 # ROI jpeg + feature.csv + roi_windows.csv + xic_matrix.npy + pipeline_qc_excluded.csv
├── predictions_model/<样品>/       # model_prediction_<样品>.csv ⭐ + qc3_threshold_<样品>.csv
│   └── predictions_model_all.csv / predictions_model_report.md
├── prediction_refined/<样品>/       # prediction_snr.csv + prediction_refined.csv ⭐ + with_area + qc4/qc5 + refined_plots/
│   └── prediction_refined_all.csv / prediction_refined_report.md
├── all.csv                         # ⭐ 最终跨样品合并明细
├── inference_report_<实验名>.md    # ⭐ 推理报告（自动生成）
└── pipeline_timing.log / pipeline_timing_runs.jsonl

../output/QC/<run_name>/            # qc1~qc5 统一汇总 + qc_summary.md / qc_alert.md
```

### 5.3 关键参数速查

**通用**：`--model`（必填，roi 除外）、`--labels`（roi/pipeline 必填）、`--threshold`（CLI 默认 0.5，交付版 `mrmpformerv2` 也使用 0.5）、`--integration_method`（linear/raw/external_baseline；普通 pipeline 生效，MassNova 当前忽略）、`--smooth_sigma`（统一 CLI 与 MassNova/C 接口默认 0.8）、`--plot` / `--plot_style`（xic/roi）。

**Pipeline QC**：`--pipeline_min_max_intensity`（1000）、`--pipeline_min_chrom_points`（10）、`--qc_label_rt_tol`（1.0）。

**SNR**：`--snr_min`（10.0）、`--snr_gaussian_sigma`（0.8）、`--snr_min_noise_points`（5）。

**Post 精修**：20+ 参数（次峰比例、噪声窗口、边界外推、谷值回退、置信度/SNR 门控等），完整清单见 [18 节参数明细](#184-推理完整参数模板)。

**输出控制**：`--no_timing`、`--no_report`、`--save_snr_jpeg`、`--verbose`/`--quiet`、`--config`。

### 5.4 推理 + 后处理创新点

1. **信号搜索 + 模型验证的混合整谱扫描（massnova）**——先以纯信号算法（`find_peaks` + 双门槛 prominence + 边界外推）枚举候选峰，再用模型切 ±1min 窗验证精修边界；
2. **峰边界精修三件套**（`two_round_detection.py`）——边界外推 + peer 防撞 + 宽度上限约束，解决峰谷粘连边界互侵；
3. **次峰找回 + 谷值回退双通道补峰**（`peak_refinement.py`）——模型漏掉的次要峰由信号规则找回，或谷值切分 + 比例折算置信度；
4. **五道 QC 防线 + 全防线 `qc_alert.md`**——逐层留痕、逐层剔除、统一人工复核；
5. **两遍精修防相邻截停**（整谱场景）；
6. **像素 → RT 线性映射**——预测框 x 坐标经 `roi_windows.csv` 线性映射回 RT 分钟。

---

## 6. 质量控制体系（QC）

### 6.1 五道防线总览

| # | 防线 | 位置 | 检查内容 | 参数（默认） |
|---|---|---|---|---|
| 1 | **标注 RT 一致性** | `label_qc` → 数据构建 & 推理管线 | ①跨样品同化合物同通道 RT 极差；②双离子定量/定性 RT 极差。极差 >1 min 判疑似实验有误 | `--qc_label_rt_tol`(1.0) |
| 2 | ROI 通道级 | `xic_extraction.py` | 平滑后 XIC 最大强度过低、RT 点数过少 → 不生成 ROI | `--pipeline_min_max_intensity`(1000)、`--pipeline_min_chrom_points`(10) |
| 3 | 预测框级 | `predictor.py` | score <= 阈值不输出；每 (mz,q3) 只留面积最大框 | `--threshold`(0.5) |
| 4 | SNR 框级 | `snr_filter.py` | 框外 SNR、框外噪声点数联合判定 | `--snr_min`(10.0)、`--snr_min_noise_points`(5) |
| 5 | 精修框级 | `peak_refinement.py` | 精修置信度、SNR、次峰比例、框宽上限等门控 | `--post_min_confidence`(0.99)、`--post_min_snr`(10.0) |

### 6.2 防线 1 剔除粒度（宁缺毋滥）

| 场景 | 策略 |
|---|---|
| 跨样品、组内样品数 ≥3 | 仅剔偏离组中位数 >1 min 的样品（多数派可信） |
| 跨样品、组内样品数 =2 | 两行都剔 + 警示人工复核 |
| 双离子 | 两通道都剔 |

### 6.3 统一 QC 输出

`output/QC/<run_name>/`：`qc_label_rt.csv`（标注 RT 一致性）、`qc_roi_channels.csv`（ROI 通道级）、`qc_prediction_threshold.csv`（预测阈值）、`qc_snr_boxes.csv`（SNR 框级）、`qc_post_refinement.csv`（精修框级）、`qc_summary.md`（汇总）、`qc_alert.md`（人工预警）。各表字段定义与分析建议详见 [QC 使用指南表字段](#63-统一-qc-输出) 与原始 `output/QC/READMEQC.md`。

### 6.4 真实数据发现（QC 价值实证）

- **2026-08-20**：防线 1 在 20260715 实验标注（120 行）首跑发现 **8 组双离子 RT 异常、16 行「疑似实验有误」**——乙酰甲胺磷（极差 25.8 min）、灭螨醌（15.9）、甲羧除草醚（12.8）、羟基-灭螨醌（15.9）等定量/定性离子不共流出，极可能是定性离子通道标注到干扰峰；
- **2026-08-24**：test1 评估链路启用 QC 后剔除 **22 项**（223-对硫磷定量/定性 RT 极差 ~4.2 min，跨 11 样品）。

---

## 7. 评估体系

### 7.1 评估协议（双口径）

`tools/evaluation/evaluate_baseline.py` 定义：

- **检测口径**：预测起止 vs 人工起止偏差均 ≤ 容差判 TP，算 P/R/F1；容差取 0.1 / 0.3 / 0.5 min（早期 tIoU>0.95 口径因模型框比人工边界宽 0.06~0.09 min 而过严，已废弃）；
- **定量口径**：与检测同容差配对后计算面积 R²、RT 起止偏差中位、RSD；
- 评估默认读阶段②原始 `prediction.csv`（不含后处理）；两套口径不可混用；
- 支持 `--run_inference 1` 自动跑推理 / `0` 复用已有 prediction.csv。

### 7.2 指标定义

| 指标 | 定义 | 读法 |
|---|---|---|
| TP / FP / FN | 命中 / 误检（含空白样臆造峰）/ 漏检 | ±0.1 min 最考验边界贴合 |
| P / R / F1 | 精度 / 召回 / 调和均值 | 综合检测能力 |
| RT 起/止偏差 | 配对成功预测与人工起止偏差中位数 | 边界贴合，直接影响定量 |
| 面积 R² | 预测面积 vs 人工面积线性拟合 | 定量准确性 |
| RSD | 同化合物跨进样预测面积相对标准差（中位） | 定量重复性 |
| FP_blank | 空白样（无峰通道）假阳性数 | 假阳性抑制 |

### 7.3 评估链路修复记录

2026-08-22 test1 导入时修复三处（不修则结果全错）：跳过 `peak_label=0` 伪 GT 峰；GT 匹配兼容 compound 列带通道后缀；`visualize_compare` 多峰格式解析。

---

## 8. 实验历程与关键发现

### 8.1 实验日志 001：v1「先成功后失败」之谜 —— 取类 bug 与 shadow query 假象（2026-08-17）

**现象**：同一 v1 权重、同一数据，两次评测天壤之别（08-16 面积 R²=0.99998 定量优秀；08-17 口径修正后 F1=0.008「全灭」）。

**根因**：`model/utils/predict_utils.py` 取类 bug——修复前 `probas = pred_logits.softmax(-1)[0, :, :-1]` 取到**背景概率列**（留下 shadow query 的框），修复后 `[1:]` 取峰概率列。

**逐 query 铁证**：14 张有 GT 图中「背景概率最高 query 的框更贴 GT」占 13/14。v1 三条 query 系统性分工：q0/q2 为旧训练「紧框包 apex」约定（P(峰)≈0.99），q1 为唯一符合人工积分约定的宽框（P(背)≈0.00）——**旧 bug 阴差阳错捡到唯一好框**。

**统一口径复测（±0.1 min，score≥0.90）**：v1 F1=0.008 vs v2 F1=0.455、面积 R²=0.99999、RT 起止偏差中位 0.059/0.080 min、RSD 中位 1.91%。

**方法论教训**：DETR 多 query 评测必须核验「选框依据」与「置信度语义」是否一致；单一指标好不等于流程正确；修复取类 bug 是评测口径修正（v1「显形」而非「变差」）。

**附录 A 事件时间线**：

| 日期 | 事件 |
|---|---|
| 2026-08-16 | v1 首次基线评测：定量优秀、检测 F1=0.025（shadow 框口径） |
| 2026-08-16 | v2 微调两次失败 → 定位取类 bug（`[:-1]`→`[1:]`） |
| 2026-08-17 | tIoU>0.95 口径：v1 F1=0、v2 F1=0.017（阈值过严） |
| 2026-08-17 | 评测协议改「起止偏差容差」口径，删除 tIoU 判据 |
| 2026-08-17 | v1 F1=0.008 vs v2 F1=0.455；根因分析完成 |
| 2026-08-22 | test1 导入 + 四模型横评 + 架构审查 + mrmpformerv2 诞生 |
| 2026-08-22 | 考据 quanformer.pth 为外部下载权重 |

### 8.2 实验日志 002：test1 标准试卷、四模型横评与 mrmpformerv2 的诞生（2026-08-22）

**背景**：反复出现「同一模型换数据集排名反转」，引入 test1 标准试卷（从未参与训练的 11 mzML、96 峰）作外域基准，多源联合训练作根治手段。

**四模型首轮横评（score≥0.5，±0.1 min）**：

| 模型 | TP/FP/FN | F1 | 空白样 FP |
|---|---|---|---|
| baseline | 1/111/95 | 0.010 | 17 |
| v2（shiyaoyuan 微调） | 0/0/96（零检出） | 0.000 | 0 |
| v3（merged 微调） | 4/105/92 | 0.039 | 14 |
| mrmpformerv1 | 86/4/10 | **0.925** | 2 |

**对照实验**（test1 7 样品微调基线，留出 4 个 24 峰）：微调基线 F1@0.1=0.836（边界偏差 0.11→0.009，边界约定可被数据学到），但空白臆造 7 FP 未解决；v1 仍领先 0.06 F1——架构优势（空白抑制）真实存在。

**架构审查发现**：FDR 三层精化 IoU 增益仅 +0.005、FDR 损失权重 ≈8.8 过度主导、右边界 MAE 大 23%、matcher 未透传 `iou_type`（恒 GIoU）。执行 P0-1（matcher 修复，git a14c438）/ P0-2（fdr_loss_coef 2.0→1.0）/ P0-4（阈值 0.4~0.7 平台，推荐 0.5）/ P2-8（multisrc）→ **mrmpformerv2**。

### 8.3 实验日志 003：quanformer「基线」真实出身考据（2026-08-22，仅分析）

读取 `checkpoint/quanformer.pth` 内嵌 `args`：

```
resume='D:\workspace\autopeakV3\detr-r50-e632da11.pth'   ← DETR COCO 预训练
coco_path='D:\workspace\train-dataset\peak-all'          ← 外部峰数据集
output_dir='D:\workspace\autopeakV3\output\peakdetr\...' ← 外部项目 autopeakV3
```

**真实训练链**：DETR COCO 预训练 → 外部 peak-all 微调 → 下载为本项目基线。baseline 差的真实根因：框约定错位（主因，~0.11 min）+ 空白盲区 + 成熟检测器正常泛化。

**v2 零检出根因**：外部权重 + shiyaoyuan 单样品 61 图微调（lr 1e-5 × 10 ep）= 灾难性遗忘 + 窄域概念重定义。v3 对照（同基线同 epochs，仅数据多样性提升）正常触发 105 框，证明「窄数据微调宽模型 = 遗忘快于学习」。

---

## 9. 四模型联合评估

### 9.1 评估设置

- 推理阈值 score≥0.5（统一）；四模型共享同一套 ROI 图（400×300，标注驱动 B 范式）；
- 检测口径：起止偏差均 ≤ 容差判 TP（0.1 / 0.3 / 0.5 min）。

**数据集与数据泄漏矩阵**：

| 评估集 | baseline | quanformerv3 | mrmpformerv1 | mrmpformerv2 |
|---|---|---|---|---|
| test1 全量（96 峰） | ✗ | ✗ | ✗ | ✓（7/11 在训练集） |
| test1 holdout4（24 峰） | ✗ | ✗ | ✗ | ✗（均为 val） |
| shiyaoyuan（100 峰） | ✓ 训练域 | ✓ 训练域 | ✗ 外域 | ✓ 半外域 |

### 9.2 公平留出集（holdout4，24 峰）

| 模型 | F1@0.1 | F1@0.3 | RT 起/止@0.1 | R²@0.3 | RSD | 空白 FP |
|---|---|---|---|---|---|---|
| baseline | 0.000 | 0.800 | — | 0.98004 | 1.27% | 12 |
| quanformerv3 | 0.035 | 0.842 | 0.019/0.085 | 0.98108 | 1.09% | 9 |
| mrmpformerv1 | 0.894 | 0.936 | 0.016/0.032 | 0.98968 | 1.31% | 1 |
| **mrmpformerv2** | **0.913** | **0.957** | **0.007/0.012** | **0.99052** | 1.33% | **0** |

### 9.3 test1 全量（96 峰；v2 有 7/11 训练泄漏）

| 模型 | F1@0.1 | F1@0.3 | TP/FP/FN@0.1 | RT 起/止@0.1 | R²@0.3 | 空白 FP |
|---|---|---|---|---|---|---|
| baseline | 0.010 | 0.914 | 1/111/95 | 0.0996/0.0858 | 0.98201 | 17 |
| quanformerv3 | 0.039 | 0.927 | 4/105/92 | 0.0199/0.0901 | 0.98332 | 14 |
| mrmpformerv1 | 0.925 | 0.946 | 86/4/10 | 0.0125/0.0334 | 0.99167 | 2 |
| **mrmpformerv2** | **0.931** | **0.963** | **87/4/9** | **0.0051/0.0040** | **0.99221** | **1** |

### 9.4 shiyaoyuan（100 峰）

| 模型 | F1@0.1 | F1@0.3 | F1@0.5 | RT 起/止@0.1 | R²@0.5 |
|---|---|---|---|---|---|
| baseline（域内） | 0.590 | 0.950 | 0.980 | 0.0542/0.0494 | 0.99997 |
| quanformerv3（域内） | 0.570 | 0.980 | **1.000** | 0.0191/0.0472 | 0.99997 |
| mrmpformerv1（外域） | 0.380 | 0.870 | 1.000 | 0.0295/0.0443 | 0.99998 |
| mrmpformerv2（半外域） | 0.442 | 0.874 | 0.995 | **0.0160/0.0346** | 0.99998 |

### 9.5 阈值敏感性（v2，test1 全量，tol=0.1）

```
thr    TP   FP   FN      P       R      F1
0.30   90   16    6   0.849  0.938  0.891
0.40   90    9    6   0.909  0.938  0.923
0.50   87    4    9   0.956  0.906  0.930  ← 最优
0.60   86    3   10   0.966  0.896  0.930
0.70   84    2   12   0.977  0.875  0.923
0.80   76    2   20   0.974  0.792  0.874
0.90   46    1   50   0.979  0.479  0.643  ← 高阈值召回崩塌
```

0.4~0.7 平台 F1≥0.92（最大回撤 <1%），建议生产 threshold=0.5。

### 9.6 核心结论

1. **边界贴合度是四模型真实分水岭**：baseline/v3 并非「检不到峰」（±0.3 下召回 0.99+），而是框边界与 test1 人工积分约定存在 ~0.1 min 系统性错位——训练数据约定与考卷约定差异，非架构能力问题；
2. **v1→v2 增量归因**：matcher 修复 + 损失再平衡 → 边界精度（止偏差 -63%、起 -56%）；多源空白负样本 → 假阳性抑制（空白 FP 1→0）；召回未变（剩余 3 个 FN 为同一批难例）；
3. **±0.1 严格 F1 ≈ 边界约定匹配度**：跨数据集排名反转是约定差异所致，根治靠多源联合训练；
4. **空白负样本数量与多样性是假阳性抑制第一要素**；
5. **定量能力全员优秀**（R² 0.980~1.000、RSD ≤3.6%），瓶颈在检测配对不在积分算法；
6. **v2 跨域边界一致性最佳**：多源训练在不损失主域精度前提下改善跨域泛化。

### 9.7 四模型双测试集网格（test1 + test2，2026-09-03）

模型：quanformer（baseline）/ quanformerv2（1000 样本微调）/ quanformerv3（traindatav1 全量重训）/ mrmpformer（MRMPFormer v3）。口径：score {0.1,0.5,0.8,0.9,0.99} × 容差 {0.01,0.05,0.1,0.2,0.5} min。

**test1（8 农残混标 + 3 空白，82 GT，高浓度单峰形）** 核心档（0.9/0.1）：

| 模型 | P | R | F1 | TP/FP/FN | 面积R² | RT 起/止中位(min) |
|---|---|---|---|---|---|---|
| quanformer | 0.0000 | 0.0000 | 0.0000 | 0/86/82 | N/A | N/A |
| quanformerv2 | 0.9524 | 0.9756 | 0.9639 | 80/4/2 | 0.9988 | 0.0223/0.0120 |
| quanformerv3 | 0.9877 | 0.9756 | 0.9816 | 80/1/2 | 0.9989 | 0.0047/0.0142 |
| mrmpformer | 1.0000 | 0.9756 | 0.9877 | 80/0/2 | 0.9988 | 0.0089/0.0109 |

**test2（3 浓度梯度样，≈187 GT，低浓度多峰）** 核心档（0.9/0.1）：

| 模型 | P | R | F1 | TP/FP/FN | 面积R² | RT 起/止中位(min) |
|---|---|---|---|---|---|---|
| quanformer | 0.0178 | 0.0160 | 0.0169 | 3/166/184 | 1.0000 | 0.0902/0.0867 |
| quanformerv2 | 0.8253 | 0.7326 | 0.7762 | 137/29/50 | 0.9999 | 0.0247/0.0202 |
| quanformerv3 | 0.7514 | 0.7273 | 0.7391 | 136/45/51 | 0.9466 | 0.0091/0.0157 |
| mrmpformer | 0.9773 | 0.6898 | 0.8088 | 129/3/58 | 0.9998 | 0.0131/0.0168 |

**特殊峰命中率跨集对照（0.9/0.1）**：quanformer 0/49（test1）& 1/44（test2）；quanformerv2 47/49 & 5/44；quanformerv3 47/49 & 11/44；mrmpformer 47/49 & 5/44。

**推理时间（RTX 4090）**：test1 四模型模型推理 983~1561 ms/样品；test2 1608~2401 ms/样品。

**综合结论**：
- **mrmpformer = 高精度默认选择**（两集 FP 合计 3、P ≥0.98、边界偏差 ~0.01 min 级），适合假阳性敏感的下游定量筛查；
- **quanformerv3 = 高召回选项**（test2 召回与特殊峰找回领先，FP 更多），适合『宁可复查不可漏』或作 mrmpformer 召回兜底；
- **数据是硬前提**：老 baseline 两集失效（边界约定不匹配新 ROI 范式），任何架构先决条件是 traindatav1 系数据训练；
- **0.99 超严阈值仅适用 CE 校准模型**：mrmpformer（Focal）置信度软封顶 ~0.97，0.99 档无框。

---

## 10. 特殊峰隔离实验

> 目标：test2 特殊峰在 (score≥0.8, 起止容差≤0.05 min) 下检出率 >80%。隔离原则：全部改动为新增文件/独立配置/独立输出，原版 mrmpformer 一行为未变。

### 10.1 结论先行

**「特殊峰检出率 ≥80%」在现行数据与分峰 GT 定义下几何上不可达**，受三条硬约束封顶：

| 约束 | 量化证据 |
|---|---|
| 几何覆盖上限 | 忽略分数、仅要求边界落在 GT±0.05 内：v1 标签 16/46(35%)、v2 标签 15/44(34%) |
| 置信度上限 | 44 特殊峰中最高框分 ≥0.8 的仅约 7 个；低 SNR 峰按校准无法给出 0.8 级置信 |
| 精度上限 | 全部模型在 0.8/0.05 均 ≤7/44；历史最优 quanformerv3 亦仅 5/44 |

### 10.2 改进实际增量（隔离口径复测）

| 口径 | mrmpformer 原版(v3) | 隔离版 special_v2 |
|---|---|---|
| val AP50（traindatav1/val） | 0.957-0.966 | **0.968**（22ep 热启动） |
| test2 特殊峰 0.8/0.05（v1 标签） | 4/44 = 9.1% | **7/44 = 15.9%** |
| test2 特殊峰 0.8/0.05（v2 修订标签） | 3/42 = 7.1% | 6/42 = 14.3% |
| 总召回 R（0.8/0.05, v1） | 0.626 | 0.615 |

### 10.3 特殊峰差的深层原因（三轮自剖）

1. **损失系统性偏窄峰**：动态 L1 中心权重 λ_c=1/(w_gt+ε)（窄峰 0.05 宽→20，宽峰 0.44→2.3）+ PW-CIoU 中心项，宽峰宽度回归信号被淹没；
2. **FDR 表示上限**：`fdr_scale_mode='initial_box_width'` → 边界修正量 = s0·ΣP·W，W∈[-1,1]，最终框宽 ≤ 初始框×3；初始框继承窄峰先验，宽峰逼近表示上限；
3. **分布 OOD**：训练集 17654 框 >0.8 min 的仅 2 个；test2 特殊峰 10+ 个 0.86-1.08 min；
4. **GT 自身有错**：啶虫脒 GT 起点落在缓慢基线漂移上、涕灭威 0.04 min「缝」框、甲草胺二分不一致等。

### 10.4 隔离版实现与训练

新增文件（原版零改动）：
- `model/models/mrmpformer/v1/losses_special.py`：QFL（Softmax 双类适配）+ log 宽度 L1（尺度等变宽度回归）；
- `model/models/mrmpformer/v1/detr_special.py`：SpecialSetCriterion（仅覆盖分类/定位两分支）；
- `model/configs/mrmpformer_special_v1.json`（QFL，跑 9ep 止损）、`mrmpformer_special_v2.json`（focal 最终）；
- `model/tools/evaluation/evaluate_special_isolation.py`：隔离特殊峰评估器。

训练记录：v1（QFL+log宽度+roi_width，从零）9ep 止损（QFL 把 score 锚定 IoU，窄峰 0.8 阈值被压制）；v2（focal+log宽度+roi_width，自 v1 热启动 22ep）最终 val AP50 0.968。

### 10.5 后续建议（若要逼近 80%）

1. **信号足点延拓后处理**（不动模型）：baseline-crossing 两侧延拓 + 谷底切分，可使几何可达段从 35% 提向 60-90%（v2 在 tol=0.12 已 70%）；
2. **训练分布增强**：合成/重采样宽峰与多峰样本；
3. **评估协议微调（诚实）**：0.05min 容差对窄分叉框等价 1-3 px 精度；建议 (score≥0.5, tol≤0.1) 或峰事件级命中口径（该口径下达 57-91%）；
4. **修复数据 bug**：联苯菊酯/联苯三唑醇通道名错配修复后重算。

---

## 11. 桌面端 GUI 与工具链

### 11.1 桌面端 GUI（GAMSTEKPEAKing，`desktop/`）

基于 **PySide6**，左侧边栏 + 卡片式内容区 + 深色科技风主题（QSS）：

- **已上线**「前处理」板块：功能卡片 1「格式转换」（拖拽/多选 `.msdata`，后台 QThread 调 `msdata2mzml.exe`，进度 + 文件级状态）；功能卡片 2「离子天顶」（遍历 mzML MS1 谱图提取 m/z 信号顶点，输出 `(m/z, RT, intensity, n_observations)` CSV）；
- **灰色占位**：寻峰 / 定量 / 模型 / 设置；
- 工程细节：未捕获异常写 `desktop/error.log` 并弹窗；`workers/ion_zenith.py` 为 `preprocessing/ion_zenith.py` 的 Qt 薄包装（算法已抽离、去 Qt 依赖）。

启动：`cd desktop && python main.py`（`--debug` 额外日志）。

### 11.2 格式转换工具链（`converters/`）

| 脚本 | 输入 | 输出 | 工具链 |
|---|---|---|---|
| `converters/msdata.py` | `.msdata` | `.mzML` | `msdata2mzml.exe`（OpenMS，`msdata_bin/`） |
| `converters/wiff.py` | `.wiff`/`.wiff2` | `.mzML` | `msconvert.exe`（ProteoWizard，`wiff_bin/`） |
| `converters/rename_cn.py` | — | — | 中文文件名 → 英文 |

工程教训：项目路径不得含中文（OpenMS C++ 层限制）；WIFF 需同名 `.wiff.scan`；`msdata2mzml.exe` 成功也可能返回退码 858，以「是否生成 .mzML」判定。

### 11.3 辅助工具集（`model/tools/`）

| 子模块 | 用途 |
|---|---|
| `batch/` | 批量重处理（`--stage snr/post/snr-post`） |
| `mzml/` | 色谱图查看/导出/检查 |
| `benchmark/` | 性能基准测试（sampler/aggregate/report/runner 四模块） |
| `evaluation/` | 一键评测、对比可视化、逐 query 诊断 |
| `diagnostics/` | box↔RT 映射核查、chrom-SNR 对齐等 |
| `experiments/` | 江南/欧陆实验专项脚本 |
| `visualization/` | GT vs 预测绘图、精修 XIC 绘图 |
| `maintenance/` | 强制空白负样本、结果整理、XIC 重生成 |

---

## 12. C/C++ 共享库

### 12.1 概览

`cpp/` 提供 C++20 共享库和稳定 C ABI，但当前生产实现**依赖随包交付的私有 CPython/Cython
运行时**。DLL 不再在 C++ 中复制 MassNova 峰算法，而是把已复制的 RT/强度数组送入
`MassNovaArrayRuntime`，由共享 Python 核心完成候选、ONNX、信号兜底、边界、门控、去重和评分，
再把 Python 生成的 JSON 原样落盘和返回。完整权威流程见第 0 章。

- C++ 生产层：`api.cpp`、`json_protocol.cpp`、`python_bridge.cpp`、`task_manager.cpp`；
- Python/Cython 层：`massnova_bridge_native.pyx` → `massnova_bridge.py` → `massnova_runtime.py`；
- 共享算法层：`massnova.py::finalize_channel_peaks`；
- 模型层：缓存的 `OnnxWindowPredictor` + `mrmpformerv2.onnx`；
- 默认通道 QC：`min_chrom_points=10`、`min_max_intensity=1000`（平滑后）。

### 12.2 输入 JSON 契约（新）

```json
{"items":[
  {"uid":"caffeine-quant","name":"Caffeine","channel":"195.0877>138.0550",
   "mzq1":195.0877,"mzq3":138.0550,"smooth_sigma":0.0,
   "x":[0.0,0.1,0.2,0.3],"y":[100.0,1200.0,6000.0,300.0]}
]}
```

### 12.3 结果状态与告警

| 字段 | 含义 |
|---|---|
| `status="ok"` | 至少有一个最终 `c > threshold` 的模型峰或信号规则峰；读 `peaks[].a/b/c` |
| `status="alert"` / `CHANNEL_LOW_INTENSITY` | QC 拒绝（点数少或平滑最大强度低） |
| `status="alert"` / `NO_PEAK_FOUND` | QC 通过但模型与回退均无峰 |

`c` 对模型峰是 softmax 分，对信号峰是 SNR/点数工程分。旧的
`status="review" + SIGNAL_FALLBACK` 已退出当前生产路径。当前接口未序列化面积；内部 MassNova
面积也仍是 raw trapz，不是普通 pipeline 的扣基线面积。模型分和信号分都会再经过统一的
严格最终阈值，所有返回 `c` 必须满足 `c > QfConfig.threshold`。

### 12.4 C 单条提交迁移

```c
QfCompoundInput input = {
    .uid="caffeine-quant", .name="Caffeine", .channel="195.0877>138.0550",
    .mzq1=195.0877, .mzq3=138.0550, .smooth_sigma=0.0f,
    .rt=rt, .intensity=intensity, .n_points=n_points,
};
QfError error = qf_process_single(&input, &task_id);
```

错误码：`QF_ERR_INVALID_PARAM` / `QF_ERR_NOT_INITIALIZED` / `QF_ERR_ALREADY_INITIALIZED` / `QF_ERR_FILE_NOT_FOUND|READ|WRITE` / `QF_ERR_JSON_PARSE` / `QF_ERR_MODEL_LOAD` / `QF_ERR_TASK_NOT_FOUND|CANCELLED` / `QF_ERR_TIMEOUT` / `QF_ERR_NO_WORKER` / `QF_ERR_INTERNAL`。

### 12.5 QuanFormer→MRMPFormer C API 关键迁移点

| 区域 | 旧（QuanFormer） | 新（MRMPFormer） |
|---|---|---|
| 化合物身份 | 单个 `mz` 值，无通道 | 必需 `channel` + `mzq1` + `mzq3` |
| 单条提交 | `qf_process_single(rt, intensity, n_points, mz, uid, &task_id)` | `qf_process_single(const QfCompoundInput*, int64_t*)` |
| ONNX 输入 | 单个归一化图像 | `image` + `img_size` 双张量，RGB `[0,255]` |
| ONNX 输出 | `pred_logits` + `pred_boxes` | `scores` + `boxes_xyxy` + `boxes_norm` |
| 结果 | `uid` + `peaks` | `uid` + `status` + `peaks` + `alerts` |

构建见 `cpp/README.md`；完整迁移见 `docs/CPP_API_DIFFERENCES.md`。接口交付算法、默认参数、
发布依赖、当前面积缺口和验收状态以本文第 0 章为准。

---

## 13. 工程治理与开发流程

### 13.1 项目级 AI 指令（CLAUDE.md）

强制：开发范围收敛于 MRM、环境与路径硬约束、数据/输出目录规范、外部命令执行策略（沙箱限制）、架构边界、已知陷阱清单。

### 13.2 开发日志机制（dev_log.md）

每次模型完成生成/修改/测试后须同步更新 `dev_log.md`：项目概述（目标/输入/输出/方法介绍）+ 按日期分组的时间线（`- <类型>(<作用域>): <描述>`，类型含需求分析/数据建模/代码生成/调试/文档生成/重构/测试/其他），配套 `.github/skills/dev-log-writer/` 技能。

### 13.3 测试体系

- `model/tests/test_mrmpformer_v1.py`：FDR 架构 8 类测试 21 项全过（CPU + DummyBackbone 免下载）；
- 22 项单测回归贯穿历次重构（训练终端输出改造、label_map 崩溃修复等）；
- `label_qc`、`coco_annotation` 等数据侧模块均有针对性单测。

### 13.4 代码质量改进记录

- prediction.csv「一名三义」修复（prediction/prediction_snr/prediction_refined 三分）；
- 中文目录名 ASCII 化（`筛选保留/筛选剔除/` → `snr_kept/snr_dropped/`）；
- QC 参数双阶段重复下发修复；roi2inference integration_method 透传修复；
- 模式重构（7→3）：消除文件收集重复、每图重载模型、双代码路径分叉等 6 项 bug；
- 训练终端输出优化（中英对照 → 紧凑英文短码 + 中文汇总）。

### 13.5 环境依赖检测

`.github/skills/check-dependencies/`：`check_env.py`（纯终端报告）、`check_gui.py`（GUI 弹窗 + 一键修复）、`fix_env.py`。

---

## 14. 已知问题汇总

> 更新于 2026-07-07（详见 `docs/Bugs.md`）。

### 待修复 - 性能

| # | 严重度 | 优先级 | 问题 | 文件 |
|---|:--:|:--:|------|------|
| P1 | 致命 | P2 | batch_size=1 逐张推理，GPU 利用率约 15% | `predict_utils.py:83` |
| P5 | 中 | P3 | 模型每次预测都重新 `torch.load`（GUI 内） | `predict_utils.py:101` |

### 待修复 - 跨平台

| # | 严重度 | 优先级 | 问题 |
|---|:--:|:--:|------|
| C2 | 中 | P4 | `pycocotools` Windows 需编译工具链（但已有 cp311 wheel） |
| C3 | 中 | P3 | Linux headless `matplotlib` 无 GUI 后端 |
| C4 | 低 | P4 | macOS 首次启动 PySide6 需安全授权 |

### 待修复 - 代码质量

| # | 严重度 | 优先级 | 问题 |
|---|:--:|:--:|------|
| Q2 | 低 | P4 | UI 类名 `PeakFormer` vs 代码 `QuanFormer` 不一致 |
| Q3 | 低 | P4 | `quanformer` 与 `utils` 两个包职责重叠（各有 `plot_utils.py`） |

### 已修复（15 项）

涵盖：硬编码路径分隔符、`device='cuda'` 硬编码（5 处）、`torch.load` 兼容（safe_torch_load）、GUI 图片累积、README Python 版本矛盾、PyTorch 2.6 不支持 RTX 5060、空 MS1 谱图崩溃、R 命令引号、plot_results 过度并行（n_jobs=2）、覆盖原 ROI 图（加 _detected 后缀 + 阈值保护）、`bisect` 混淆、workbooks 绝对路径硬编码等。

---

## 15. QuanFormer 基线补齐清单

> 目标：把 QuanFormer 补成「可重训、可评测、可对比」的严格实验基线（详见 `imporove.md`）。

| # | 条目 | 状态 |
|---|---|---|
| 1 | COCO 标注生成脚本 | ☑ 2026-08-16 |
| 2 | 参数配置外置 1（train --config） | ☑ 2026-08-16 |
| 3 | 基线评测协议（一键精度评测） | ☑ 2026-08-16 |
| 4 | 置信度阈值统一 | ☐ |
| 5 | build_predictor 按 args.model 路由 | ☐ |
| 6 | train.py resume 逻辑修复 | ☐ |
| 7 | 生成 ROI 图像优化（按标注 RT 居中） | ☑ 2026-08-16 |
| 8 | 解析时不读取 TIC 图 | ☐ |
| 9 | 参数配置外置 2（参数注释） | ☑ 2026-08-17 |
| 10 | 参数生成在 output_v2 目录的问题 | ☐ |

**关键结论**：检测分低根因从「bbox 边界偏宽」修正为「预测框相对人工边界系统性左移 ~0.05 min」，是 quanformer.pth 训练边界约定与人工积分约定的模型级差异，需用 data/coco 重训解决。

---

## 16. 数据增强与 SimCLR 尝试

### 16.1 SimCLR 对比学习特征提取器（2026-07-15 设计）

- 输入任意二维图像（无标签）→ 输出 2048-d 语义特征（ResNet50 backbone）；
- 训练：自监督对比学习（NT-Xent loss，温度 τ=0.5）；
- 模型：ResNet50（ImageNet 预训练，去 FC）→ Projection Head（2048→512→128，L2 归一化）；推理时去投影头输出 2048-d；
- 增强（SimCLR 标准）：RandomResizedCrop(224, scale=0.08~1.0) + RandomHorizontalFlip + ColorJitter(0.4) + RandomGrayscale + GaussianBlur(kernel=23)；
- 训练超参：AdamW lr=3e-4，CosineAnnealing，batch=256，epochs=300。

### 16.2 色谱图专用增强（方案 B，2026-07-20 设计）

针对 XIC ROI 图像（400×300，400:300 伪彩色，色谱峰有固定 RT 语义）设计专用增强，与标准 SimCLR（方案 A）对比：

| 步骤 | 方案 A (SimCLR) | 方案 B (Chromatogram) |
|---|---|---|
| 裁剪/缩放 | RandomResizedCrop 224, scale=(0.08,1.0) | Resize(168,224) + Pad(28,28)（零损失） |
| RT 漂移 | ❌ | RandomRTShift(±8%)（模拟 LC 柱 RT 漂移） |
| 翻转 | HorizontalFlip p=0.5 | HorizontalFlip p=0.5 |
| 颜色抖动 | ColorJitter(0.4)×4 | ❌（伪彩色无物理意义） |
| 灰度化 | RandomGrayscale p=0.2 | ❌ |
| 模糊 | GaussianBlur kernel=23 | GaussianBlur kernel=5 |
| 变异源 | 5 | 3 |

刻意不做：ColorJitter / RandomGrayscale / RandomVerticalFlip / RandomRotation / IntensityJitter / 极端裁剪（reason：伪彩色无意义、强度轴有方向、RT 轴水平、避免假信号与切碎峰）。

对比实验用完全相同数据（699 张）、架构、超参、损失；评估指标含训练过程（loss 收敛、防坍塌）与表征质量（正样本相似度 >0.6、负样本分离、t-SNE、维度利用率、方差分布）；判定规则 ≥3 项指标胜出方胜。

### 16.3 Benchmark 模块拆分（2026-07-29 设计）

`run_pipeline_timing_benchmark.py`（~750 行）拆为 4 模块：`sampler.py`（GPU 显存后台采样 GpuVramSampler）、`aggregate.py`（纯函数统计，去副作用）、`report.py`（格式化与文件输出）、`runner.py`（CLI 入口与编排）。依赖单向无环 `runner → {sampler, aggregate, report}`、`report → aggregate`。不变量：CLI 参数/输出格式/日志文件完全不变。

---

## 17. 四种分析模式教程摘要

> 完整教程见 `User_Tutorials.md`。

MRMPFormer 分析流程由两个正交维度决定：**分析策略**（Targeted 靶向 / Untargeted 非靶向）+ **数据形态**（Centroided 质心 / Profile 轮廓）。

| # | 组合 | 典型场景 | 开发状态 |
|---|---|---|---|
| 1 | Targeted × Centroided | MRM 已知化合物定量（最常用） | ✅ 当前唯一开发模式 |
| 2 | Targeted × Profile | 高分辨原始轮廓靶向定量 | ⚠️ 暂不开发 |
| 3 | Untargeted × Centroided | 质心化全扫描非靶向 | ⚠️ 暂不开发（R/xcms 代码已删） |
| 4 | Untargeted × Profile | CentWave 经典非靶向 | ⚠️ 暂不开发 |

`smooth_sigma` 建议：Centroided `0.0~0.5`（点稀疏已去噪）；Profile `0.8~1.5`（点密集含噪）。

`feature.csv` 格式：`Compound Name`（必填）/ `mz`（必填）/ `RT`（必填，分钟）/ `q3`（可选）/ `native_id`（可选）。

---

## 18. 环境要求与快速开始

### 18.1 环境要求

| 项目 | 要求 |
|---|---|
| Python | **3.11**（Conda 环境名固定 `gamstekpeaking`） |
| 包管理器 | Conda |
| R（可选） | 4.0+，仅 Untargeted 模式需要（暂不开发可不装） |

**PyTorch 版本**：

| GPU 系列 | CUDA | torch | torchvision |
|---|---|---|---|
| RTX 50（5060–5090） | 12.8 (cu128) | ≥2.7.0 | ≥0.22.0 |
| RTX 40 / 30 / 20 | 12.4 (cu124) | 2.6.0 | 0.21.0 |
| CPU / Apple Silicon (MPS) | — | 2.6.0 | 0.21.0 |

### 18.2 安装

```powershell
# 第一步：Python 环境
conda create -n gamstekpeaking python=3.11
conda activate gamstekpeaking
# 第二步：依赖
pip install -r requirements.txt
# 或一步到位
conda env create -f environment.yml
```

环境检测：`python .github/skills/check-dependencies/check_env.py`（终端）或 `check_gui.py`（GUI）。

### 18.3 前处理

```powershell
cd converters
python msdata.py --dry-run   # 预览
python msdata.py             # .msdata → .mzML
python wiff.py               # .wiff → .mzML（默认带峰检测）
python wiff.py --no-peak-picking   # 保留 profile 轮廓
```

### 18.4 训练

```powershell
cd model
# 生成 COCO 数据集
python -m preprocessing.coco_annotation --config configs/coco_annotation.json
# 训练
python -m train --config configs/mrmpformer_v1_multisrc.json
# 或命令行
python -m train --model mrmpformer_v1 --coco_path ../data/coco/<数据集名> `
  --output_dir ../output/train/<实验名> --device auto --epochs 30 --batch_size 4 `
  --lr 1e-4 --lr_backbone 1e-5
```

### 18.5 推理（完整管线）

```powershell
cd model
python -m inference.cli --mode pipeline `
  --model checkpoint/quanformer.pth `
  --labels ../data/label/<实验>.xlsx `
  --batch_dir ../data/mzml/<实验> `
  --output_dir ../output/pipeline_batch `
  --threshold 0.99 --plot --snr_min 10.0 `
  --pipeline_min_max_intensity 1000 --pipeline_min_chrom_points 10
```

### 18.6 评估

```powershell
cd model
python -m tools.evaluation.evaluate_baseline `
  --labels ../data/label/<实验>.xlsx --run_inference 0 `
  --prediction_csvs <样品1>=../output/<run>/predictions_model/<样品1>/model_prediction_<样品1>.csv ... `
  --output_dir ../output/eval/<实验名>
```

### 18.7 常见问题

- **CUDA 不可用**：`nvidia-smi` + `python -c "import torch; print(torch.cuda.is_available())"`，False 则重装对应 GPU 版 PyTorch；
- **无 GPU**：编辑 `requirements.txt` 启用 CPU 段；
- **Untargeted 报错 FileNotFoundError**：R 或 Bioconductor 未装；
- **Windows 路径错误**：避免空格与中文，推荐 `D:\data\mrmpformer\` 简洁路径。

---

## 19. 当前状态与后续规划

### 19.1 当前模型家族（`model/checkpoint/`）

| 权重 | 定位 | 生产建议 |
|---|---|---|
| `mrmpformerv2.pth` | 多源联合训练 + 架构修复 | ✅ 推荐部署（threshold=0.5） |
| `mrmpformer.pth`（=v3） | traindatav1 全量训练，高精度路线 | ✅ 假阳性敏感场景首选 |
| `mrmpformer_special_v2.pth` | 特殊峰隔离版（focal+log宽度+roi_width） | 实验性，val AP50 0.968 |
| `mrmpformerv1.pth` | 论文架构原始实现（traindata3） | 备选 |
| `quanformerv3.pth` | 基线架构 + traindatav1 全量重训，高召回 | 全扫/召回兜底 |
| `quanformer.pth` | 外部下载权重 | ❌ 不建议作为基线 |
| `quanformerv2.pth` | 单样品微调（灾难性遗忘） | ❌ 已证伪 |

### 19.2 输出产物实况（`output/`）

- `output/QC/`：五道防线统一 QC 表；
- `output/evaluation/`：四模型 × 多数据集 × 多容差完整评估矩阵（grid4 / grid4_test2）；
- `output/special_peak_isolation/`：特殊峰隔离实验（标签修订、叠加复核图、隔离评估产物）；
- `output/train/`：各训练 run；`output/test/`：历次试跑；`output/inference/`：推理产物。

### 19.3 后续规划

1. **P1 级架构改动**突破剩余 FN 难例：中间层 box 监督、FDR bin 值域放宽；
2. **数据面**：补充空白负样本标注；宽峰/多峰样本增强；新数据走 multisrc 联合训练增量微调；
3. **工程面**：置信度阈值统一、`build_predictor` 路由收尾、`train.py` resume 修复、batch 推理性能（batch_size=1 → GPU 利用率 ~15%）、GUI 模型缓存；
4. **实验面**：标注 QC 复核后数据重建（20260715 16 行 + test1 22 行涉事通道）、评估容差口径补充。

### 19.4 项目总结

MRMPFormer 已走完「推理可用 → 可重训可评测 → 架构升级 → 多源联合训练 → 特殊峰攻坚」的完整迭代路径：从依赖外部下载权重的黑盒基线，到建立 COCO 数据闭环、五道 QC 防线、双口径评估协议，再到实现三层 Decoder + FDR 边界精化的论文架构（v1），经 matcher 修复、损失再平衡与多源联合训练收敛到综合最优（v2 holdout F1@0.1=0.913、空白零臆造），最终在 traindatav1 全量重训后形成「高精度 mrmpformer v3 / 高召回 quanformerv3」的双路线格局。沉淀三条核心方法论：**DETR 多 query 评测必须核验置信度语义**、**严格容差 F1 度量的是边界约定匹配度**、**空白负样本多样性与数量决定假阳性抑制**。当前具备面向生产的完整能力：格式转换 → 数据构建 → 训练 → 推理 → 后处理 → 评估 → QC 全链路闭环，并有桌面端 GUI、C/C++ 共享库、环境检测、开发日志等工程化配套。

---

## 附录：原始文档索引

本全量文档整合自以下分散文档（仍保留原位，可交叉查阅）：

| 文档 | 位置 | 主要内容 |
|---|---|---|
| README.md | 根目录 | 项目概览、工作流、数据规范、QC、快速开始、各板块命令、参数、FAQ |
| dev_log.md | 根目录 | 开发时间线（按日期） |
| User_Tutorials.md | 根目录 | 四种分析模式教程 |
| imporove.md | 根目录 | QuanFormer 基线补齐清单 |
| blank_label.md | 根目录 | 所有生成 CSV/表格的用途与字段含义字典 |
| CLAUDE.md | 根目录 | 项目级全局指令 |
| docs/project_report.md | docs/ | 项目综合报告（24-08-24） |
| docs/experiment_report.md | docs/ | 实验日志 001/002/003 + 附录 A-I |
| docs/model_eval_grid_test1test2_4models.md | docs/ | 四模型 test1/test2 网格评估 |
| docs/MRMPFormer_FDR_Architecture_Agent_Prompt.md | docs/ | v1 FDR 架构设计提示词 |
| docs/Bugs.md | docs/ | 已知问题汇总 |
| docs/CPP_API_DIFFERENCES.md | docs/ | C API 迁移指南 |
| docs/superpowers/specs/* | docs/superpowers/ | SimCLR / 色谱增强 / benchmark 拆分 / 前处理 / 整谱扫描 / C++ 库设计 |
| docs/superpowers/plans/* | docs/superpowers/ | 各设计对应实施方案 |
| output/special_peak_isolation/special_peak_report.md | output/ | 特殊峰隔离实验报告（24-09-04） |
| output/QC/READMEQC.md | output/ | QC 阶段表字段使用指南 |
