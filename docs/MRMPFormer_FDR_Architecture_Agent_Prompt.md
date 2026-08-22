# MRMFormerv1: 三层 Decoder + FDR 边界逐层精化改造提示词


## 1. 你的角色与总任务

你是一名熟悉 PyTorch、DETR、Transformer Decoder、目标检测匹配与损失函数设计的资深算法工程师。请在**当前项目中构建以下代码**，把现有峰检测模型升级为“三层 Decoder + 左右边界 FDR（Fine-grained Distribution Refinement，细粒度分布精化）逐层反馈”的结构，作为MRMPFormer的v1版本，项目目录在model\models\mrmpformer中。

本次不是简单把 Transformer Decoder 从 1 层堆到 3 层。必须同时实现：

1. Decoder Layer 1 预测完整二维初始框；
2. 三个 Decoder 层均输出左右边界离散概率分布；
3. Decoder Layer 2、3 预测上一层累计 Logits 的残差；
4. 每层解码得到的左右边界，经位置编码后反馈给下一层 Decoder；
5. 最终二维框采用第三层精化后的左右边界，以及第一层初始框的上下边界；
6. **最终分类结果只采用 Decoder Layer 3 的峰概率**；
7. 分类损失改为 Focal Loss，并加入可配置的动态加权 L1、PW-CIoU、FDR 分布监督及召回率优化选项；
8. 保持现有训练、验证、推理、数据格式和旧配置尽量兼容。

请先检查项目结构、配置系统、模型构建入口、Decoder 实现、Matcher、Criterion、推理后处理和现有测试，再给出简短修改计划并直接实施。不要脱离现有代码另写一套无法接入的示例网络。

---

## 2. 不可偏离的核心语义

### 2.1 这不是普通的三层 Decoder

- **仅堆叠三层 Decoder**：只能提升 Query 特征质量；
- **三层 Decoder + 中间层输出 + Logits 残差累加 + 边界位置反馈**：才是真正的 FDR 式逐层峰框精化。

原始 QuanFormer 使用类似原始 DETR 的普通 Transformer Decoder。默认情况下，只有上一层 Query 特征会传到下一层，上一层预测的左右边界不会自动传入下一层。本次必须显式增加边界位置反馈通路。

### 2.2 最终输出的唯一口径

设第三层分类 Logits 为 `class_logits_3`，类别定义仍与现有项目一致（例如 `[background, peak]`）。推理阶段最终峰概率必须为：

$$
p_{\text{peak}}=\operatorname{Softmax}(\text{class\_logits}_3)_{\text{peak class}}.
$$

禁止平均三层分类结果，禁止使用第一层或第二层分类概率替代最终结果。中间层分类输出可以用于辅助监督，但必须由配置控制，并且不能改变最终推理口径。

### 2.3 必须修正架构图中的层号笔误

正确残差关系为：

$$
z^{(2)}=z^{(1)}+\Delta z^{(2)},
$$

$$
z^{(3)}=z^{(2)}+\Delta z^{(3)}.
$$

第二层输出必须命名为 `delta_z2` 或等价清晰名称，不能照搬原图中把第二层写成 `ΔZ3` 的笔误。

---

## 3. 目标网络结构与张量形状

以下以架构图中的默认值说明：

- Backbone：ResNet-50；
- Transformer Encoder：1 层；
- Encoder Memory：`[B, 128, 256]`（以现有实现的实际 token 数为准，不要硬编码 128）；
- Object Queries：图中为 3 个，形状 `[Q, 256]`，其中 `Q=num_queries` 必须配置化；
- Transformer Decoder：3 层；
- 隐藏维度：`D=256`；
- FDR Bin 数：默认 `N=33`，必须配置化；
- 每层 Query 特征：`h_k: [B, Q, D]`；
- 每层分类输出：`class_logits_k: [B, Q, num_classes]`；
- 每层 FDR 输出：`[B, Q, 2, N]`，维度 2 依次代表左边界和右边界。

如果现有项目采用 `[Q, B, D]` 等布局，可以在 Decoder 内部保持原布局，但模型对外输出必须有明确且统一的形状约定，不能依赖含糊的 `view()`。

整体数据流如下：

```text
ResNet-50
    ↓
1-layer Transformer Encoder → memory
    ↓
Object Queries
    ↓
Decoder Layer 1 → h1
    ├─ shared class head → class_logits_1
    ├─ initial 2D box head → b0=(cx0,cy0,w0,h0)
    └─ FDR Head 1 → z1 → P1 → (l1,r1)
                         ↓ Boundary Position MLP
                    boundary_pos_embed_1
                         ↓ 加入 Layer 2 的 Query position
Decoder Layer 2 → h2
    ├─ shared class head → class_logits_2
    └─ FDR Head 2 → delta_z2; z2=z1+delta_z2 → P2 → (l2,r2)
                                                       ↓ Boundary Position MLP
                                                  boundary_pos_embed_2
                                                       ↓ 加入 Layer 3 的 Query position
Decoder Layer 3 → h3
    ├─ shared class head → class_logits_3 → 最终峰概率
    └─ FDR Head 3 → delta_z3; z3=z2+delta_z3 → P3 → (l3,r3)

最终二维框：b*=(l3,t0,r3,b0_bottom)
```

注意：公式中的初始框记为 `b^(0)`，其下边界不要与变量名 `b0` 混淆。代码中建议使用：

- `initial_box_cxcywh`；
- `initial_edges_ltrb`；
- `initial_top`；
- `initial_bottom`；
- `final_boxes_xyxy`。

---

## 4. Decoder Layer 1：初始二维框与第一次边界分布

### 4.1 保留原二维框头

第一个 Decoder 输出 Query 特征：

$$
h^{(1)}\in\mathbb{R}^{B\times Q\times D}.
$$

原有二维框头保持不变，例如：

```text
MLP(D → D → D → 4) + Sigmoid
```

得到归一化初始框：

$$
b^{(0)}=(c_x^{(0)},c_y^{(0)},w^{(0)},h^{(0)}).
$$

转换为四条边：

$$
x_L^{(0)}=c_x^{(0)}-\frac{w^{(0)}}{2},\qquad
x_R^{(0)}=c_x^{(0)}+\frac{w^{(0)}}{2},
$$

$$
y_T^{(0)}=c_y^{(0)}-\frac{h^{(0)}}{2},\qquad
y_B^{(0)}=c_y^{(0)}+\frac{h^{(0)}}{2}.
$$

本方案**不是删除二维框头**。上下边界始终由第一层初始二维框头负责；FDR 只精化左右边界。

### 4.2 新增 FDR Head 1

在 `h1` 后增加小型 FFN：

```text
Linear(D,D) → ReLU → Linear(D,D) → ReLU → Linear(D,2N)
```

得到：

$$
z^{(1)}=\operatorname{FFN}_{\text{FDR},1}(h^{(1)}),
$$

reshape 为：

$$
z^{(1)}\in\mathbb{R}^{B\times Q\times 2\times N}.
$$

沿最后一个 Bin 维度分别做 Softmax：

$$
P_L^{(1)}(n)=\operatorname{Softmax}(z_L^{(1)})(n),
$$

$$
P_R^{(1)}(n)=\operatorname{Softmax}(z_R^{(1)})(n).
$$

FDR Head 1 预测的不是另一套独立二维框，而是初始框左右边界的候选偏移分布。

---

## 5. Decoder Layer 2、3：Logits 残差精化

第二、三层不从头预测完整分布，而是预测上一层**累计 Logits** 的残差：

$$
\Delta z^{(2)}=\operatorname{FFN}_{\text{FDR},2}(h^{(2)}),
$$

$$
z^{(2)}=z^{(1)}+\Delta z^{(2)},
$$

$$
\Delta z^{(3)}=\operatorname{FFN}_{\text{FDR},3}(h^{(3)}),
$$

$$
z^{(3)}=z^{(2)}+\Delta z^{(3)}.
$$

每次残差累加后均重新计算：

$$
P^{(k)}=\operatorname{Softmax}(z^{(k)}),\qquad k\in\{1,2,3\}.
$$

实现要求：

1. 三个 FDR Head 应是各层独立参数，除非现有项目明确要求共享；
2. `z1`、`z2`、`z3` 和三层边界必须保留在模型输出中，供训练监督、调试和可视化使用；
3. 不允许只保留最后一层坐标；
4. 不允许对 Softmax 后的概率直接做无说明的残差相加；残差必须加在 Softmax 前的 Logits 上；
5. 保证 `delta_z2=0` 时 `z2=z1`，`delta_z3=0` 时 `z3=z2`，并为此编写单元测试。

---

## 6. 概率分布到左右边界的解码

### 6.1 Bin 候选偏移

设每个 Bin 对应一个候选偏移量：

$$
W(0),W(1),\ldots,W(N-1).
$$

`W` 必须注册为模型 buffer，以便随设备移动但不参与训练，形状为 `[N]`。它必须满足：

- 严格单调递增；
- 包含或对称覆盖 0；
- 左右两侧尽量对称；
- 长度严格等于 `N`。

优先复用项目或论文中已有的非均匀 Bin 定义。若项目中没有定义，则实现以下可配置默认方案，并在代码注释中标明它是工程默认值而非冒充论文原式：

$$
u_n=\frac{2n}{N-1}-1,
$$

$$
W(n)=\operatorname{sign}(u_n)|u_n|^p,
$$

其中默认 `p=2.0`，并允许配置文件直接传入显式 `bin_values` 覆盖该生成方式。

### 6.2 期望偏移量

第 `k` 层左右边界的归一化修正量为：

$$
\Delta x_L^{(k)}=s^{(0)}\sum_{n=0}^{N-1}P_L^{(k)}(n)W(n),
$$

$$
\Delta x_R^{(k)}=s^{(0)}\sum_{n=0}^{N-1}P_R^{(k)}(n)W(n).
$$

其中 `s^(0)` 是尺度因子：

- 默认采用第一层初始框宽度 `w0`；
- 允许配置为 ROI 宽度；
- 必须与当前坐标系一致，不能混用像素坐标与 `[0,1]` 归一化坐标。

符号约定统一为：

- `Δx > 0`：向右移动；
- `Δx < 0`：向左移动。

每层精化边界为：

$$
x_L^{(k)}=x_L^{(0)}+\Delta x_L^{(k)},
$$

$$
x_R^{(k)}=x_R^{(0)}+\Delta x_R^{(k)}.
$$

### 6.3 禁止双重累计坐标残差

由于 `z^(k)` 已经包含前面各层的 Logits 残差，`x_L^(k)`、`x_R^(k)` 应由**累计分布相对初始边界解码**。不要再写成：

```text
x_k = x_(k-1) + expectation(z_k)
```

否则会把前面层的偏移重复累计。只有在重新定义 FDR Head 为“单层增量分布”并给出严格推导时才可采用逐坐标相加，但本任务不采用该口径。

### 6.4 最终二维框

第三层得到 `x_L^(3)`、`x_R^(3)` 后，最终二维框为：

$$
b^*=(x_L^{(3)},y_T^{(0)},x_R^{(3)},y_B^{(0)}).
$$

若当前项目统一使用 `cxcywh`，则在组装完 `xyxy` 后通过公共工具函数转换，禁止在各模块重复手写不一致的转换公式。

训练和推理均需处理：

- 边界越界；
- `x_L >= x_R`；
- 极小宽度；
- FP16 下的数值稳定性。

但不要通过 `.detach()`、离散 `argmax` 或不可导排序切断 FDR 到边界损失的梯度。建议把训练使用的连续边界和推理阶段安全裁剪后的边界区分命名，并记录非法框比例。

---

## 7. 真正的逐层边界位置反馈

### 7.1 反馈路径

每层精化得到的边界区间必须编码为下一层的 Query 位置特征：

```text
(x_L^(k), x_R^(k))
        ↓
Boundary Position MLP
        ↓
boundary_pos_embed_k: [B,Q,D]
        ↓
加入下一层 Query position encoding
```

可采用：

```text
BoundaryPositionMLP: Linear(2,D) → ReLU → Linear(D,D)
```

如果现有 Decoder 已有 `query_pos`、reference point 或 positional embedding 机制，优先接入该机制，不要粗暴改变 attention 输入语义。建议：

$$
q_{pos}^{(k+1)}=q_{pos}^{base}+g_k\cdot\operatorname{MLP}_{pos}([x_L^{(k)},x_R^{(k)}]),
$$

其中 `g_k` 可为固定 1，或通过小型门控层产生；默认先使用固定 1，门控作为配置化扩展，避免一次引入过多变量。

### 7.2 梯度要求

默认必须允许后一层损失通过位置反馈回传到前一层边界预测。增加配置：

```yaml
detach_boundary_feedback: false
```

仅用于消融实验，默认不得 detach。

编写梯度测试：只对第三层输出构造损失并反向传播时，第一层 FDR Head 与 Boundary Position MLP 应获得非零有限梯度。

---

## 8. 分类头与 Focal Loss

### 8.1 分类头

保持共享峰分类头，例如：

```text
Linear(D, num_classes)
```

三层均可输出分类 Logits，但：

- 训练主分类损失以第三层输出为准；
- 第一、二层辅助分类损失由 `aux_class_loss` 配置控制；
- 推理只使用第三层峰概率。

### 8.2 用 Focal Loss 替换分类交叉熵

峰检测中背景样本远多于真实峰，采用 Focal Loss：

$$
L_{FL}=-\alpha_t(1-p_t)^\gamma\log(p_t).
$$

默认参数：

```yaml
focal_alpha: 0.25
focal_gamma: 2.0
```

补充材料中“β=2”应按 Focal Loss 标准符号修正为 `γ=2`，代码、配置和日志中统一使用 `gamma`。

必须先确认现有模型采用：

- 两类 Softmax（背景/峰）；还是
- Sigmoid 二分类/多标签形式。

然后选择与现有输出语义一致的 Focal Loss 实现，禁止把 Sigmoid Focal Loss 直接套在含显式背景类的 Softmax 输出上而不调整目标编码。

必须正确处理 DETR 的 `no-object/background` 权重、有效 Query mask、Hungarian Matching 后的正负样本，以及空目标 ROI。禁止先算 Softmax 再把概率传入要求原始 Logits 的损失实现。

预期目标：降低大量易分类背景样本对训练的主导，提高弱峰、小峰和困难峰的检出能力。

---

## 9. FDR 分布监督损失

### 9.1 总体要求

三层累计分布均接受左右边界监督：

$$
L_{FDR}=\sum_{k=1}^{3}\alpha_k L_{FGL}^{(k)},
$$

初始层权重建议：

```yaml
fdr_layer_weights: [0.5, 0.7, 1.0]
```

中后层权重更高，以鼓励逐层精化。

### 9.2 不得凭名称猜测论文公式

补充说明只给出了 `L_FGL` 的用途，没有给出其完整数学定义。实施时按以下优先级处理：

1. 搜索当前仓库、项目文档及随项目提供的论文，若存在明确 `FGL` 定义，严格按原定义实现，并在代码注释中标注来源位置；
2. 若没有精确定义，不得自行编造后声称是论文原式；
3. 在这种情况下，实现一个独立、可替换的 `DistributionBoundaryLoss` 作为工程回退，并通过配置名明确区分。

工程回退方案如下：

1. 对 Hungarian 匹配后的正样本，计算真实左右边界相对初始边界的目标偏移：

$$
d_L^{gt}=\frac{x_L^{gt}-x_L^{(0)}}{s^{(0)}},\qquad
d_R^{gt}=\frac{x_R^{gt}-x_R^{(0)}}{s^{(0)}}.
$$

2. 在有序非均匀 Bin `W` 上寻找包围目标值的两个相邻 Bin；
3. 按距离做线性插值，生成和为 1 的两点软标签；超出范围时裁到端点，并统计越界比例；
4. 使用软标签交叉熵监督每层的累计 Logits `z1/z2/z3`；
5. 仅匹配到真实峰的 Query 参与边界分布损失，背景 Query 不生成伪边界目标；
6. 对 `s0` 使用 `eps` 防止除零；
7. 记录每层左右边界分布损失、目标越界率和期望偏移误差。

该回退实现应命名清楚，不能在论文或报告中冒充未经确认的 `FGL` 原始公式。

---

## 10. 定位损失优化

### 10.1 动态加权 L1 Loss

原始 L1：

$$
L_1=|c_x-\hat c_x|+|c_y-\hat c_y|+|w-\hat w|+|h-\hat h|.
$$

改为：

$$
L_{dyn\text{-}L1}
=\lambda_c\left(|c_x-\hat c_x|+|c_y-\hat c_y|\right)
+\lambda_w|w-\hat w|
+\lambda_h|h-\hat h|,
$$

其中：

$$
\lambda_c=\frac{1}{w_{gt}+\varepsilon},
$$

默认：

$$
\lambda_w=1,\qquad\lambda_h=1.
$$

实现要求：

- 明确预测和 GT 的坐标顺序，不能把公式中的 `x,y` 与左上角坐标混淆；这里指中心坐标 `cx,cy`；
- `w_gt` 必须与当前归一化坐标一致；
- 提供 `eps` 配置；
- 原始公式可能对极窄峰产生很大梯度，因此提供可选 `center_weight_clip` 和 `normalize_dynamic_weights`，默认先忠实复现公式，同时在训练日志记录权重的 P50/P90/P99/Max；
- `dynamic_l1_enabled` 必须可开关，方便和原始 L1 做消融。

这里的动态 L1 主要突出峰中心定位。峰宽与二维重叠质量同时交由 FDR 和 PW-CIoU 约束。

### 10.2 PW-CIoU

原始 CIoU：

$$
CIoU=IoU-\frac{\rho^2(b,b^{gt})}{c^2}-\alpha v,
$$

其中：

$$
\rho^2(b,b^{gt})=(x-x_{gt})^2+(y-y_{gt})^2.
$$

引入峰宽权重后：

$$
PW\text{-}CIoU
=IoU-\frac{\bar w}{w_{gt}+\varepsilon}\cdot
\frac{\rho^2(b,b^{gt})}{c^2+\varepsilon}-\alpha v.
$$

对应最小化损失：

$$
L_{PW\text{-}CIoU}=1-PW\text{-}CIoU.
$$

其中：

- `bar_w` 是训练集真实峰宽的平均值；
- `w_gt` 是当前真实峰宽；
- `(x,y)`、`(x_gt,y_gt)` 是预测框与真实框中心；
- `v` 是宽高比一致性项；
- `alpha` 沿用标准 CIoU 的定义。

性质：

- `w_gt = bar_w` 时，中心距离项权重为 1，与原 CIoU 一致；
- `w_gt < bar_w`（窄峰）时权重大于 1，增强定位约束；
- `w_gt > bar_w`（宽峰）时权重小于 1，降低过度惩罚。

实现要求：

1. `bar_w` 必须来自训练集统计并写入配置、数据集元数据或 checkpoint；不要用当前 mini-batch 均值，否则损失会随批次组成漂移；
2. 提供两种权重模式：

```yaml
pw_ciou_weight_mode: ratio          # bar_w / (w_gt + eps)，默认
# 或
pw_ciou_weight_mode: one_plus_ratio # 1 + bar_w / (w_gt + eps)，探索实验
```

3. `one_plus_ratio` 在 `w_gt=bar_w` 时权重为 2，并不等同于原 CIoU；日志和文档中不得写成“此时与原 CIoU 一致”；
4. 提供可选权重裁剪，防止极窄峰造成梯度爆炸；
5. 复用项目中成熟的 box area、IoU、enclosing box 工具；
6. 对无匹配目标和退化框安全返回有限值；
7. `pw_ciou_enabled` 可开关，保留原 CIoU 作为基线。

---

## 11. 总损失函数

在不破坏现有损失框架的前提下，将总损失组织为：

$$
L_{total}
=\lambda_{cls}L_{FL}^{(3)}
+\lambda_{box}L_{dyn\text{-}L1}
+\lambda_{iou}L_{PW\text{-}CIoU}
+\lambda_{fdr}\sum_{k=1}^{3}\alpha_kL_{FGL}^{(k)}
+L_{aux}.
$$

其中：

- `L_FL^(3)`：第三层主分类损失；
- `L_dyn-L1`：定位回归损失；
- `L_PW-CIoU`：峰宽感知的 CIoU 损失；
- `L_FGL^(k)`：第 `k` 层左右边界概率分布监督；
- `L_aux`：可选中间层分类辅助损失或现有项目必须保留的其他损失。

所有权重必须配置化，训练日志必须分别输出各项未经加权和加权后的数值，禁止只记录一个总损失而无法定位问题。

建议初始配置：

```yaml
num_decoder_layers: 3
num_fdr_bins: 33
fdr_bin_power: 2.0
fdr_scale_mode: initial_box_width
fdr_layer_weights: [0.5, 0.7, 1.0]
detach_boundary_feedback: false

classification_loss: focal
focal_alpha: 0.25
focal_gamma: 2.0
aux_class_loss: true

dynamic_l1_enabled: true
dynamic_l1_eps: 1.0e-6
dynamic_l1_lambda_w: 1.0
dynamic_l1_lambda_h: 1.0
center_weight_clip: null
normalize_dynamic_weights: false

pw_ciou_enabled: true
pw_ciou_weight_mode: ratio
pw_ciou_eps: 1.0e-6
pw_ciou_weight_clip: null

recall_loss_enabled: false
```

不要假定这些初始值一定最优；先保证实现正确，再通过消融实验调参。

---

## 12. Hungarian Matching 与提高召回率

本项目宁愿增加一定误判，也应优先减少漏峰。实现以下可配置策略：

### 12.1 增加 `num_queries`

- 取消任何把 Query 数量硬编码为 3 的逻辑；
- 配置、模型、Matcher、后处理和可视化统一读取 `num_queries`；
- 不直接拍脑袋改成某个固定值；提供至少 `3 / 5 / 10 / 20` 的实验入口，并结合单个 ROI 最大真实峰数选择范围；
- 记录“GT 数量超过 Query 数量”的 ROI 比例，该比例不应大于 0。

### 12.2 降低 Matcher 中的 `cost_class`

- `cost_class` 必须配置化；
- 设计基线值和降低后的实验值；
- 定位准确但分类置信度暂时较低的 Query 应更容易完成匹配；
- 修改后检查 `cost_class/cost_bbox/cost_giou` 的量纲，避免分类代价降得过头后匹配完全由框决定；
- 若训练使用 Focal Loss，Matcher 的分类代价必须明确采用何种概率定义，保持数学语义一致并加注释。

### 12.3 降低推理置信度阈值

- `confidence_threshold` 配置化，不写死；
- 验证时输出 Precision、Recall、F1、PR 曲线或不同阈值表；
- 以 Recall 优先选择阈值，但不能只报告 Recall 而隐藏误检变化；
- 最终阈值必须基于验证集确定，不能使用测试集调参。

### 12.4 Recall Loss 探索实验

- 仅作为实验开关，默认关闭；
- 先复现并验证 Focal Loss 基线，再尝试联合 Recall 导向损失；
- 若补充材料没有给出 Recall Loss 的精确定义，不得随意造公式后当作既定方案；
- 新增实现必须说明来源、输入、正负样本定义、空目标处理与梯度稳定性。

---

## 13. 模型输出接口

在尽量兼容现有接口的前提下，建议模型输出至少包含：

```python
{
    # 最终推理使用的第三层分类 Logits
    "pred_logits": class_logits_3,              # [B,Q,C]

    # 最终二维框；格式与项目现有约定一致
    "pred_boxes": final_boxes,                  # [B,Q,4]

    # 第一层原始二维框
    "initial_boxes": initial_box_cxcywh,        # [B,Q,4]

    # 三层 FDR 累计 Logits
    "fdr_logits": [z1, z2, z3],                 # each [B,Q,2,N]

    # 第二、三层实际预测的 Logits 残差
    "fdr_deltas": [delta_z2, delta_z3],         # each [B,Q,2,N]

    # 每层解码得到的左右边界
    "refined_lr": [lr1, lr2, lr3],              # each [B,Q,2]

    # 可选中间层分类结果，仅供辅助监督与调试
    "aux_outputs": [...],
}
```

如果现有 Criterion 依赖 DETR 标准 `aux_outputs`，请按现有约定扩展，不要破坏训练脚本。`pred_boxes` 必须明确是 `xyxy` 还是 `cxcywh`，并在模型、Matcher、Loss 和 PostProcess 之间保持一致。

---

## 14. 兼容性与初始化

### 14.1 旧 checkpoint

增加 checkpoint 迁移逻辑或清晰的加载报告：

- 旧模型只有一层 Decoder 时，可把第一层 Decoder 参数复制初始化到新增的第二、三层；
- 新增 FDR Heads、Boundary Position MLP 按项目现有初始化策略初始化；
- 不能使用完全静默的 `strict=False`；必须打印并分类说明 expected missing keys、unexpected keys；
- 旧分类头和初始二维框头尽量加载复用；
- 保存新配置和模型结构版本号，防止新旧 checkpoint 混淆。

### 14.2 数值与设备兼容

至少兼容：

- CPU 单元测试；
- CUDA 训练；
- FP32；
- 项目当前如启用 AMP，则兼容 FP16/BF16；
- 空目标 batch；
- 一个 ROI 多峰；
- `B=1`，不能因无参数 `squeeze()` 丢掉 batch 或 query 维度。

---

## 15. 必须完成的测试

不要只保证代码能 import。至少增加并运行以下测试：

### 15.1 Shape Test

使用小型随机输入验证：

- 三层 Query 特征存在；
- `z1/z2/z3` 均为 `[B,Q,2,N]`；
- `pred_logits` 为第三层分类输出；
- `pred_boxes` 为 `[B,Q,4]`；
- 修改 `B/Q/N` 后形状仍正确。

### 15.2 Residual Refinement Test

人工把第二、三层残差置零，验证：

$$
z^{(2)}=z^{(1)},\qquad z^{(3)}=z^{(2)}.
$$

再给定已知残差，验证逐元素累加完全正确。

### 15.3 Distribution Decode Test

构造 one-hot 或高度集中的 Bin 分布，验证期望偏移等于对应 `W(n)`；验证正偏移向右、负偏移向左；验证尺度使用初始框宽度时结果正确。

### 15.4 Final Box Assembly Test

验证最终框：

- 左右边界来自第三层；
- 上下边界严格来自第一层初始二维框；
- 不会错误使用第二、三层不存在的上下边界。

### 15.5 Final Classification Source Test

人为设置三层分类 Logits 不同，验证 `pred_logits` 和最终 `p_peak` 只来自 Decoder Layer 3。

### 15.6 Boundary Feedback Gradient Test

仅基于第三层输出反向传播，验证：

- Boundary Position MLP 有非零有限梯度；
- 第一层 FDR Head 有非零有限梯度；
- `detach_boundary_feedback=true` 时行为符合消融设计。

### 15.7 Loss Test

分别测试：

- 全背景/空目标；
- 单峰；
- 多峰；
- 极窄峰；
- 退化预测框；
- Focal Loss 不产生 NaN/Inf；
- 动态 L1 权重符合公式；
- `w_gt=bar_w` 时 PW-CIoU 中心项权重为 1；
- FDR 软标签和为 1；
- 所有损失均可 backward。

### 15.8 Tiny-set Overfit Test

选少量训练样本做短时过拟合测试，确认：

- 总损失明显下降；
- 三层 FDR 边界误差总体呈精化趋势；
- 模型能够记住小样本；
- 若做不到，优先检查匹配、坐标格式、目标分布编码和边界反馈，而不是盲目延长训练。

---

## 16. 训练日志与评估

训练日志至少增加：

- `loss_cls_focal_main`；
- `loss_cls_aux_1/2`（若启用）；
- `loss_dynamic_l1`；
- `loss_pw_ciou`；
- `loss_fdr_layer_1/2/3_left/right`；
- 每层左、右边界 MAE；
- 每层框 IoU；
- FDR 目标超出 Bin 范围的比例；
- 动态中心权重和 PW-CIoU 权重分位数；
- 无效框比例；
- 正样本匹配数、空目标 ROI 数。

验证结果至少报告：

- Precision；
- Recall；
- F1；
- AP 或项目原有主指标；
- 不同 confidence threshold 下的结果；
- 弱峰、小峰、窄峰等困难子集结果（若数据标签支持）；
- 左右边界误差与峰宽误差；
- 三层逐层边界误差变化，证明精化机制确实有效。

若第三层并未比第一层更好，不要只汇报最终结果；检查边界反馈、残差尺度、Bin 覆盖范围、层间监督权重和学习率。

---

## 17. 建议消融实验顺序

不要一次打开所有改动后无法判断收益来源。建议保持同一数据划分和随机种子，按以下顺序实验：

1. 原始 QuanFormer 基线；
2. 仅把 Decoder 增加到 3 层，不做 FDR；
3. 三层 Decoder + 三层 FDR Logits 残差监督，但不做边界反馈；
4. 加入 Boundary Position MLP 和逐层边界反馈；
5. 分类 CE 改为 Focal Loss；
6. 加入动态加权 L1；
7. 原 CIoU 改为 PW-CIoU；
8. 调整 `num_queries`；
9. 降低 Matcher `cost_class`；
10. 在验证集调低 confidence threshold；
11. 最后才探索 Recall Loss。

每一步至少比较 Recall、Precision、F1、边界 MAE、IoU、漏峰数和误检数。核心目标是提高召回率和弱峰/小峰检测能力，同时确保左右边界精度确实提升。

---

## 18. 实施原则

1. **先读代码再改**：找出真实模型入口和坐标约定，不凭文件名猜测；
2. **最小侵入**：优先扩展现有模块，不复制整套训练管线；
3. **配置化**：所有新结构、损失和召回策略均可独立开关；
4. **不破坏兼容性**：保持旧数据与推理接口，必要时提供迁移层；
5. **不掩盖不确定项**：`L_FGL`、Recall Loss 若缺少论文原式，明确报告并使用可替换回退模块；
6. **不硬编码**：禁止硬编码 Batch、Query、Bin、token 数、类别索引和设备；
7. **不切断梯度**：边界期望解码和位置反馈默认全程可导；
8. **不只改模型文件**：同步检查 Matcher、Criterion、PostProcess、配置、checkpoint、日志、可视化和测试；
9. **详细注释**：对 Logits 残差、边界解码、反馈通路、坐标格式、Focal Loss 形式和 PW-CIoU 权重写清楚中英文注释；
10. **拒绝伪完成**：只有三层 Decoder、没有逐层边界反馈，不算完成本任务。

---

## 19. 最终交付内容

完成后请提供：

1. 修改文件清单；
2. 每个文件的修改目的；
3. 最终前向传播路径说明；
4. 新增配置项及默认值；
5. 新旧 checkpoint 的兼容方式；
6. 运行过的测试命令与测试结果；
7. 一个最小训练命令和一个推理命令；
8. 模型输出字典及各张量形状；
9. 尚未确认的设计项，尤其是 `L_FGL` 和 Recall Loss 的论文精确定义；
10. 推荐的第一轮消融实验表。

最终自检时必须逐项回答：

- 最终峰概率是否只来自 Decoder Layer 3？
- Layer 1 是否仍预测完整二维初始框？
- 三层是否都预测左右边界分布？
- Layer 2、3 是否分别预测 `Δz2`、`Δz3`？
- Logits 是否按 `z2=z1+Δz2`、`z3=z2+Δz3` 累加？
- 每层边界是否反馈到下一层 Query 位置编码？
- 最终框上下边界是否来自初始二维框？
- 是否避免了坐标残差的双重累计？
- Focal Loss 是否使用 `alpha=0.25, gamma=2.0` 的正确参数名？
- PW-CIoU 的 `bar_w` 是否来自训练集而非 mini-batch？
- 所有关键路径是否已有 Shape、数值、梯度和小样本过拟合测试？

任意一项为“否”，都不能宣称改造已经完成。
