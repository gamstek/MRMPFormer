# 项目开发日志

## 项目概述

### 目标
MRMPFormer 是一个基于深度学习的 LC-MS 代谢组学峰检测与定量工具。结合 CNN（ResNet-50）与 Transformer（DETR 架构），在提取离子色谱图（EIC/ROI）中识别真峰/假峰并定位峰边界以积分定量。

### 输入
- **原始数据**: `.mzML` 格式的高分辨率 LC-MS 数据（Centroided 或 Profile）
- **特征表**（Targeted 模式）: CSV 文件，包含 `Compound Name` / `mz` / `RT` 三列
- **模型权重**: `quanformer.pth`（>300MB）

### 输出
- **定量结果**: `area.csv`（峰面积表）
- **后处理结果**: `post-area.csv`（去重转置后的面积表）
- **EIC 图像**: 每个化合物生成一张 ROI 区域的 JPEG 图像
- **预测图像**: 带检测框标注的 EIC 图像

### 方法介绍
1. 从 mzML 中按 m/z 提取 EIC → 生成 ROI 图像
2. 使用 MRMPFormer（ResNet-50 + 1 层 Transformer 编解码器）检测峰
3. 根据预测框边界对 EIC 积分 → 得到峰面积
4. 后处理去重 → 输出定量结果表
支持 Targeted/Untargeted × Centroided/Profile 四种组合模式。

---

## 开发时间线

### 2026-08-21

- 需求分析(修复后首轮 v1 训练日志诊断): eos 修复生效确认 —— train class_error 100→29.7→25.1→23.1，Focal 分类头已正常学习；"分类误差跳变"定性为小分母统计噪声（每 batch 仅 ~13 个正匹配 query，错 1 个即跳 7.7%，epoch 级 7.3↔9.7 仅约 30 样本差）；数量误差 0.99→1.30 缓涨系背景降权+召回优先策略的已知副作用（负样本图多报 1 峰即贡献 1.0，推理阈值可滤），暂不干预；健康证据：val_loss 9.82→8.19、L3 IoU 0.75→0.80、epoch2 AP 0.269→0.377 / AP50 0.766 / AR100 0.658 大幅跃升；另发现 log.txt 首行混有崩溃前旧 run 遗留记录（读日志需按 run 边界区分）
- 重构(model/framework/engine.py+util/misc.py): 终端输出第三版（自我攻击后重设计，两轮批判：机械逐词双语=词典式噪音、括号均值重复、行长失控、信息层级错位、补丁式叠加概念）—— 设计原则改为"逐步行紧凑英文短码可 grep + 图例只打一次 + 汇总纯中文短标签一组一行"；逐步行精简为 loss/cls + eta/步时/显存GB 单行≤90 字符（去掉 data 取数耗时逐步打印）；汇总块 64 字符宽框 ◆ 头行 + 分类/损失/精化/匹配/权重 五组，FDR 六项损失与逐层 MAE/IoU 用箭头串成趋势线（左MAE 0.024→0.022→0.021），权重分位数 P50/P90/max 三值压缩单项，p99/期望偏差等纯诊断只进 log.txt；图例（_print_legend_once）训练开始打一次；quanformer 基线键兼容（CE/L1/CIoU 短标签 + 未知 loss_ 键兜底）；22 项单测回归通过
- 调试(model/framework/engine.py+util/misc.py): 修复 label_map AttributeError 训练崩溃 —— 中英对照改造时同文件并行编辑互相覆盖（misc.__init__ 丢 label_map=None、engine.train_one_epoch 丢 label_map 赋值，evaluate 侧完好），str(ml) 触发 __getattr__ 抛错；补回两处赋值并将 __str__ 改为 getattr 防御式读取（display_keys/label_map 缺失回退全量旧行为，日志层永不炸训练），三场景模拟（正常/半缺/全缺）与 22 项单测通过；另释疑 creating index 打印两次系 train/val 数据集各建一次 COCO 索引，属正常
- 重构(model/framework/engine.py+util/misc.py): 训练终端输出改为中英对照 —— engine 新增 _METRIC_LABELS 精确表 + _LABEL_PATTERNS 正则表（FDR 逐层/辅助分类/权重分位数等模式化 key），_metric_label 输出 "en/中文" 显示名；逐步行（含 lr，经 MetricLogger.label_map）与分组汇总全部应用；misc.log_every 模板字段 eta/time/data/max mem → eta/剩余、time/步时、data/取数、mem/显存；组名升级 分类/Cls、损失/Loss、FDR、匹配/Match、权重/Weight、其他/Other；每组 3 项/行；log.txt 与 train_stats 仍用原始英文 key 不变（分析脚本零影响）；22 项单测回归通过
- 重构(model/framework/engine.py+util/misc.py): 训练终端输出结构优化 —— MetricLogger 新增 display_keys 子集（None=旧行为全量），训练/验证逐步行只打印 loss/class_error/cardinality_error + eta/time/mem（原先单行 20~40 项无法扫读；训练打印间隔 10→20，验证 10→25）；_print_avg_stats 重写为分组多行汇总（分类/损失分量/FDR 逐层/匹配诊断/权重分位数，未知键归"其他"组，兼容基线与新变体），log.txt 全量 JSON 与 train_stats 返回值不受影响；'Test:' 头改为 'Val'
- 调试(models/mrmpformer/v1/fdr.py+detr.py): 修复 Focal 分类头梯度平衡卡死 —— 首轮 traindata3 训练 1000 步 class_error 恒 100、focal 损失纹丝不动（0.0566→0.0548）；根因：softmax_focal_loss 未接入 DETR no-object 权重（eos_coef），QuanFormer CE 的 empty_weight=0.1 背景降权在换 Focal 后丢失，负样本质量（~34/batch×0.75）压倒正样本（~14/batch×0.25），模型卡死在 p_peak≈0.3 平衡点；修复为背景项乘 eos_coef（前景恒 1），新增 test_focal_eos_weighting 回归测试（eos 缩放精确性 + p=0.5 下 14 正 vs 34 负梯度质量对比 + 无 eos 反证），22 项单测全过
- 需求分析(首轮 v1 验证集诊断): FDR 精化机制 epoch 0 即被证实有效 —— 逐层左/右边界 MAE L1(0.0239/0.0379)→L2(0.0219/0.0241)→L3(0.0210/0.0240) 单调下降，逐层 IoU 0.700→0.763→0.771；overflow=0/invalid_box=0/gt_exceeds=0 全部干净；唯分类头因上述 bug 卡死
- 调试(model/framework/engine.py): 恢复被覆盖丢失的 train_one_epoch 诊断日志钩子 —— 此前对 engine.py 的两处并行编辑互相覆盖，导致 weight_dict 之外的诊断指标（fdr_target_overflow_ratio/fdr_lr_mae_layer_*/dyn_l1_center_w_*/invalid_box_ratio/匹配统计）未进 metric_logger，首轮 v1 训练日志中缺失即由此发现；已补回并与 evaluate 侧对齐
- 需求分析(traindata3 首轮 v1 训练观察): epoch 0 输出健康（loss 17.7→11.6，动态 L1/PW-CIoU 均在降）；class_error=100 属 Focal 早期正常（正样本仅 ~26% query，alpha=0.25 收敛慢，判据 epoch 2-3 应开始下降）；FDR 损失逐层升高（1.56→2.06→2.93）为残差头梯度权重不对称（head1 收三层损失等效 2.2，head3 仅 1.0）+ 深层学小残差慢，非精化失效；发现 pw_ciou_mean_width=0.2297 系 merged 数据集统计值，当前训练集 traindata3 实测 0.1808（12677 峰），用户决定本轮跑完后用 0.1808 重训对比
- 调试(models/mrmpformer/v1/fdr.py): 修复 CUDA 训练崩溃 —— dynamic_l1_weights 与 _ciou_with_center_weight 中 torch.quantile 的 q 张量建在 CPU，GPU 上设备不一致报 "q tensor must be on the same device"；两处改为 device=输入张量.device；CUDA 冒烟（前向+全损失+backward）与 CPU 21 项单测回归均通过（CPU 单测跑不出此问题，根因是诊断统计代码跨设备建张量）
- 代码生成(models/mrmpformer/v1/): 新建 MRMPFormer v1 完整实现（按 docs/MRMPFormer_FDR_Architecture_Agent_Prompt.md）—— fdr.py（FDRHead 零初始化末层/非均匀 Bin W=sign(u)|u|^p 生成与 buffer/期望偏移可导解码/BoundaryPositionMLP/DistributionBoundaryLoss 两点插值软标签工程回退（非论文 FGL 原式）/Softmax Focal（alpha=0.25,gamma=2.0，含显式背景类）/动态加权 L1 λc=1/(w_gt+eps)/PW-CIoU 双权重模式+分位数统计）、transformer.py（FDRTransformer：encoder 复用 QuanFormer 层类保证旧权重可加载，decoder 逐层循环+每层后 boundary_pos_fn 回调实现 q_pos^(k+1)=q_pos^base+MLP([xL,xR]) 边界位置反馈）、detr.py（MRMPFormer 模型：L1 初始二维框+三层 FDR Logits 残差累计 z2=z1+Δz2/z3=z2+Δz3+每层边界相对初始边解码（禁双重累计）+最终框左右取 L3 上下取 L1+宽度下限保护；MRMPSetCriterion：Focal 主分类（仅 L3）+中间层辅助分类+动态 L1+PW-CIoU+FDR 三层分布监督+召回诊断指标；load_legacy_quanformer_state 旧单层 checkpoint 迁移 L1→L2/L3 复制初始化+分类报告）
- 代码生成(model/train.py): 新增 MRMPFormer v1 全量配置参数（FDR 结构 8 项/分类损失 7 项/动态 L1 6 项/PW-CIoU 5 项/recall_loss_enabled 默认关闭且启用即报错防编造公式）；resume 路径接入旧 QuanFormer 权重迁移（mrmpformer_v1 时自动识别 legacy checkpoint）
- 代码生成(model/configs/mrmpformer_v1_fdr.json): v1 训练配置 —— dec_layers=3/num_queries=3/N=33/α=[0.5,0.7,1.0]/Focal+动态 L1+PW-CIoU 全开；pw_ciou_mean_width=0.2297 取自 data/coco/merged/train 实测训练集 GT 平均峰宽（70 峰，禁止 mini-batch 均值）
- 代码生成(model/framework/engine.py): 训练/评估通用日志钩子 —— weight_dict 之外的诊断指标（FDR 逐层 MAE/IoU、越界率、无效框比例、匹配统计、权重分位数）自动进 metric_logger 与 log.txt，基线 quanformer 零影响
- 代码生成(model/utils/predict_utils.py): build_predictor 按 checkpoint args.model 路由模型变体（原硬编码 quanformer）；权重加载非静默化（打印 missing/unexpected 分类报告）+ v1 自动触发 legacy 迁移
- 测试(model/tests/test_mrmpformer_v1.py): 提示词 §15 全部 8 类测试 21 项全过（CPU + DummyBackbone 免下载）—— Shape（B/Q/N 可变）/残差零初始化恒等+已知残差逐元素累加/分布解码（one-hot 期望=W(n)、正右负左、w0 尺度）/最终框组装（左右=L3 上下=L1、退化宽度保护）/分类唯一来源 L3（hook 验证）/边界反馈梯度（非 detach 时 BoundaryMLP 与 FDR Head1 非零有限梯度、detach 时 MLP 梯度恰为零）/损失（空目标/单峰/多峰/极窄峰/退化框/Focal 无 NaN/动态 L1 公式/PW-CIoU w_gt=bar_w 时权重=1/软标签和=1 与越界裁剪/全部可 backward）/legacy 迁移（L2/L3=L1 复制、FDR 头不被污染）/4 样本 tiny-set 过拟合（总损失显著下降、L3 边界 MAE ≤ L1）
- 其他(model/models/mrmpformer/__init__.py): v1 子包文档说明

### 2026-08-20

- 代码生成(model/preprocessing/label_qc.py): 新建标注 RT 一致性 QC 模块（docs/plan_qc.md 防线1）—— 跨样品极差 groupby(compound,channel)（n≥3 仅剔偏离中位数者、n=2 双剔）+ 双离子极差 groupby(sample_id,compound)（超阈值双通道剔）；极差>tol(默认1.0min) 判疑似实验有误，WARN 警示人工复核；返回全量 qc_rows 与 exclude_keys，配套 mark_excluded_labels（打标记不删行保持行序）/write_qc_table
- 代码生成(model/preprocessing/coco_annotation.py): 挂接标注 QC —— 新增 --qc_label_rt_tol(默认1.0，0=关闭)；parse_labels_xlsx 后做一致性检查并标记命中行，by_key 构建与行序回退匹配跳过被标记者（ROI 降级负样本，rt_overrides 同步隔离）；QC 表输出 output/QC/coco_<实验名>_<时间戳>/qc_label_rt.csv
- 调试(数据验证): label_qc 单测通过（构造 B/C/D 三类异常：n=2 双剔、n=3 仅剔离群、双离子双剔；exclude_keys/WARN/CSV 断言）；真实标注回归发现 20260715 实验 8 组双离子 RT 极差 12.8~25.9 min（乙酰甲胺磷/灭螨醌/烯啶虫胺?/苄嘧磺隆? 等定量定性离子不共流出），16 行判疑似实验有误待人工复核 —— QC 首跑即捕获真实数据问题

### 2026-08-16

- 文档生成(imporove.md): 新建 QuanFormer 基线补齐清单 —— 基于模型缺口分析整理 6 项待办（COCO 标注生成脚本（testcase_data.xlsx→COCO）、参数配置外置、一键精度评测协议、置信度阈值统一、build_predictor 按模型变体路由、train.py resume 逻辑修复），每项含问题说明与大致做法，附 P0/P1/P2 优先级表
- 代码生成(model/preprocessing/coco_annotation.py): 新建 COCO 训练数据集生成脚本 —— mzML（复用 extract_xic_with_pyopenms，图像与推理管线完全一致 400x300/apex±1min）+ 标注 xlsx（标准库 zipfile+ElementTree 解析、列字母定位免疫稀疏单元格错位，无 openpyxl 依赖）→ peak_start/peak_end 分钟值经 roi_windows.csv 窗口线性映射为像素 bbox（y 全高 [0,300]）→ 按 mzML 分组划分 train/val；native_id「化合物名-1/-2」对齐定量/定性离子（实测两文件 60/60 精确匹配），TIC 等无标注 ROI 纳入为负样本
- 数据集生成(data/coco/): 产出首个可重训 COCO 数据集 —— train 61 图 60 框+1 TIC 负样本（20260715_shiyaoyuan_test_1 ↔ sample_id 方法1）、val 61 图 60 框+1 负样本（_2 ↔ 方法1-2）；标注源 data/test/testcase_data.xlsx 共 120 行=30 化合物×2 通道×2 进样
- 调试(数据验证): 三层验证全部通过 —— json 结构等价断言（id 唯一/引用完整/bbox 合法/无缺图）+ pycocotools 官方 API 加载 + 全量 120 框映射质量（峰顶落框内 120/120、框边界强度<50%峰高 120/120）；抽查图（PIL 画红框）存 data/coco/_inspect/
- 文档生成(README.md): 训练节路径约定修齐 —— 删除虚构的 train2017/ + annotations/instances_*.json 结构，改为 coco.py 实际读取的 train/ + train_coco.json / val/ + val_coco.json；训练/微调/评估命令 --coco_path 统一改为 ../data/coco；新增「生成 COCO 数据集」小节（coco_annotation.py 用法与标注 xlsx 布局说明）
- 其他(环境): gamstekpeaking 环境补装 pycocotools（requirements.txt 已声明但环境缺失，训练必需）；排查确认 TRAE 沙箱会拦截 matplotlib 渲染 DLL 延迟加载（0xc06d007f 崩溃），XIC 提取/数据集生成需在系统终端执行
- 代码生成(model/tools/evaluation/evaluate_baseline.py): 新建基线一键精度评测脚本（improve.md 第 3 项）—— 双口径协议：检测口径（tIoU>0.95 判 TP → P/R/F1）+ 定量口径（宽松配对 tIoU>0.5 → RT 边界偏差/面积R²/RSD，不影响 P/R/F1 计数）；复用 linear_fit_r2 与 coco_annotation 标注对齐；--run_inference 0 可复用已有 prediction.csv（5s/两样品）；输出 evaluation_report.json + match_details.csv + area_pairs.csv
- 重构(model/postprocessing/evaluation/standard_curves.py): matplotlib 导入移入 main() 绘图分支 —— 顶层 import 会连累纯数据使用者（evaluate_baseline 仅需 linear_fit_r2）在沙箱/无头环境崩溃
- 重构(model/preprocessing/coco_annotation.py): _LABEL_COLS 补 area/snr 列映射 —— 此前漏解析 xlsx 面积列导致评测脚本定量指标无人工面积可比
- 需求分析(评测协议): 确定匹配规则——用户指定 RT 区间重叠（tIoU）>0.95 判命中；实测发现 0.95 过严（模型框比人工积分边界宽 0.06~0.09min，tIoU 中位 0.738），增设定量宽松口径分离检测/定量指标
- 数据集生成(data/evaluation/quanformer/): 产出 quanformer.pth 首份基线参考分数 —— 检测 P/R/F1=0.0246/0.0250/0.0248（TP=3）；定量：面积R²=0.99998（n=115）、RT 起止偏差中位 0.063/0.073min、RSD 均值 3.34%/中位 1.99%（n=56 通道）；结论：模型定量能力优秀、边界偏宽是检测分低的主因，根因指向 improve.md 第 7 项（ROI 以最高强度点居中而非标注 RT）
- 文档生成(imporove.md): 第 3 项打勾（含基线分数表与解读）；优先级表更新——第 7 项升 P0（评测数据证实其为主因），第 1/3 项标记完成
- 代码生成(model/preprocessing/xic_extraction.py): extract_xic_with_pyopenms 新增 rt_center_overrides 参数（improve.md 第 7 项）—— {native_id: RT分钟} 覆盖表命中时以标注 RT 为 ROI 窗口中心，替代默认最高强度点；TIC 等未命中通道不受影响
- 代码生成(model/preprocessing/coco_annotation.py): build_coco_for_mzml 自动从标注 xlsx rt 列构建覆盖表传入提取函数（修复初始版本覆盖表构建晚于提取调用的顺序 bug）
- 数据集生成(data/coco/ 重生成): 标注 RT 居中版 —— 60/60 通道窗口中心与标注 RT 严格相等（max 差 0）、120/120 bbox 映射质量复验通过；--force 必带否则复用旧 _xic
- 需求分析(根因修正): 第 7 项对检测指标无改善（TP 3→4，tIoU 中位 0.741→0.736）——同通道新旧预测中位位移仅 0.004 min，证伪"窗口居中是主因"；有符号偏差分析揭示真相：预测框相对人工边界系统性左移 ~0.05 min（起 −0.051/止 −0.049，框宽一致 0.449 vs 0.440），属 quanformer.pth 训练约定与测试软件人工积分约定的模型级差异；重训（data/coco，bbox=人工边界）是解决路径
- 文档生成(imporove.md): 第 7 项打勾（含假设证伪与根因修正记录）；第 3 项解读同步修正；优先级表更新为"重训基线"为 P0
- 文档生成(imporove.md 第 8 项): 用户新增「解析时不读取 TIC 图」待办
- 代码生成(model/train.py): 修复 --device auto 设备解析 bug —— 原代码先 torch.device('auto') 再判断分支，'auto' 非法直接抛 RuntimeError，训练入口从未实测跑通；改为先按分支解析再创建 device
- 代码生成(model/models/quanformer/detr.py): 修复 build() 同类 --device auto bug —— build 入口直接 torch.device(args.device) 消费 'auto' 抛错；改为 auto 时复用 utils.torch_device.resolve_torch_device，显式 device 时按原逻辑
- 调试(model/framework/datasets/coco.py): 修复 Windows 下标注 json 读取 UnicodeDecodeError —— torchvision CocoDetection 内部用系统默认编码(GBK) open() 读 UTF-8 标注（含中文化合物名）报错；改为保留继承关系但绕过 __init__，显式 UTF-8 读入 + COCO().createIndex() 重建索引（get_coco_api_from_dataset 的 isinstance 检查不受影响）；数据集加载冒烟测试通过（61 图 60 框、train transforms、coco api 链路）
- 调试(model/framework/datasets/coco.py): 修复 build() 标注路径不匹配 —— coco.py 原读 <coco_path>/<split>_coco.json（根目录），而 coco_annotation.py 生成的是 <coco_path>/<split>/<split>_coco.json（split 目录内），导致 FileNotFoundError；改为兼容两种布局（优先 split 目录内，回退根目录），train/val 构建验证通过（各 61 样本）
- 调试(v2 微调失败): quanformerv2.pth 首次微调检测全崩 —— 推理检出 0/122 峰（v1 61/61），score 全 0。根因：make_coco_transforms 为 DETR 自然照片增强（RandomHorizontalFlip 峰形镜像 + RandomResize 放大 480~800px + RandomSizeCrop(384,600) 裁高 600>图高 300），而推理端预处理仅 ToTensor+Normalize 且不缩放 400×300 原图，训练/推理分布严重错位 → 模型学会翻转+放大分布，正常图全判背景；v1 为旧大数据集成熟模型故多尺度泛化未崩
- 重构(model/framework/datasets/coco.py): make_coco_transforms 重写为色谱图专用 —— train/val 统一仅 ToTensor+Normalize（与 utils/predict_utils.py 推理预处理完全对齐）；训练/验证对齐验证通过（400×300、batch stack 正常）
- 代码生成(model/train.py): 新增 --config（JSON）参数配置外置机制（improve.md 第 2 项）—— 配置文件作默认参数，CLI 仍可覆盖；环境无 pyyaml 故用标准库 json
- 代码生成(model/configs/): 新增两个配置文件 —— quanformer_baseline.json（从 quanformer.pth 的 args 提取全部超参：batch_size=16/num_queries=3/enc_dec_layers=1，修正 lr_drop 35→20、coco_path 指向 ../data/coco、resume 置空）；quanformer_v2_finetune.json（微调专用：lr=1e-5/lr_backbone=1e-6/epochs=10/batch_size=4/reset_optimizer=true）
- 代码生成(model/train.py): 新增 --reset_optimizer 开关 —— 微调语义：只加载模型权重，跳过 optimizer/lr_scheduler 恢复，start_epoch 归零以当前 lr 从头训练
- 调试(配置验证): 两配置加载 + CLI 覆盖验证通过（num_queries=3 生效、epochs/batch_size 覆盖正常、resume 空值=从零训练）
- 文档生成(imporove.md): 第 2 项（参数配置外置）打勾

### 2026-08-14

- 重构(preprocessing/ion_zenith.py): 新建纯算法模块 `extract_ions_from_ms1()` + CLI 入口 —— 从 `desktop/workers/ion_zenith.py` 抽离核心算法（去 Qt 依赖），与 `xic_extraction.py` 同层组织；可用 `python -m preprocessing.ion_zenith --input_mzml ... --output_csv ...` 直接调用
- 重构(desktop/workers/ion_zenith.py): 改为薄包装 —— 保留 `IonZenithWorker(QThread)` 接口不变，`run()` 方法改为调用 `preprocessing.ion_zenith.extract_ions_from_ms1()`，通过 Signal 转发进度/统计/结果；前端 `IonZenithCard` 零改动
- 代码生成(inference/cli.py): 新增 `--no_timing` 开关 —— 启用时跳过 `pipeline_timing.log` / `pipeline_timing_runs.jsonl` 写入（终端仍打印计时汇总）；同步修正顶部 docstring 中 `python main.py` → `python -m inference.cli` 与 `from main import` → `from inference.cli import`
- 代码生成(inference/cli.py): 新增 `--save_snr_jpeg` 开关 —— pipeline 模式默认不生成 `筛选保留/筛选剔除/` 标注图（省磁盘）；启用时生成
- 代码生成(postprocessing/snr_filter.py): `run()` 新增 `save_jpeg: bool = True` 参数 —— `save_jpeg=False` 时跳过 `save_roi_jpeg_with_box` 调用且不创建 `筛选保留/筛选剔除/` 目录；独立 CLI 同步加 `--no_save_jpeg` 反向开关
- 文档生成(README.md): 项目结构树重写 —— 删除已不存在的 `model/main.py` / `getFeature.py` / `mrmpformer/` 子包 / `gamstekpeaking/` 目录，改为反映实际扁平结构（inference / models / preprocessing / postprocessing / framework / utils / tools）；推理命令全部改为 `python -m inference.cli --mode ...`；训练命令改为 `python -m train`；Untargeted 节精简为「代码已删除」说明；删除 GamSTekPeaking Web 工作台节
- 文档生成(User_Tutorials.md): 全部 `python main.py` 改为 `python -m inference.cli`；模式三/四标注「暂不开发 · 代码已删除」并注释历史命令；Q2 改为说明 Untargeted 已不可用
- 文档生成(CLAUDE.md): 项目身份表与架构边界段同步实际结构 —— `model/mrmpformer/` → `model/` 扁平结构；新增推理入口与训练入口行；开发范围段说明 `getFeature.py`/`testXIC.py` 已删除；设备路径段更新 `resolve_torch_device` / `safe_torch_load` 实际位置
- 文档生成(desktop/workers/README_workers.md): `ion_zenith.py` 描述改为「Qt 线程包装（纯算法在 `model/preprocessing/ion_zenith.py`）」；IonZenithWorker 接口段加引用块说明算法位置 + CLI 调用方式
- 文档生成(README.md): 推理参数速查新增「输出控制」段 —— `--no_timing` / `--save_snr_jpeg` 两个开关；Pipeline 参数速查表新增对应两行

### 2026-08-13

- 重构(converters/): 目录精简 —— 删除零代码引用的 desktop_bin/ 工具链副本（115.9MB，MD5 比对与 desktop/bin 逐字节一致）；合并 readme.txt 要点至 readme.md、归档 exp_log.md 至 dev_log.md 后删除；删除历史产物 conversion_report.txt（UTF-16 终端输出副本）
- 重构(requirements.txt): 合并 model/requirements.txt 与 desktop/requirements.txt 至根目录统一依赖文件并删除旧文件 —— 默认启用 RTX 40 系/4090D 段（cu124, torch 2.6.0 + torchvision 0.21.0），合并 tqdm 等 desktop 独有依赖；同步更新 CLAUDE.md / README / desktop/README / environment.yml（name→gamstekpeaking）/ check-dependencies skill 的路径引用
- 文档生成(CLAUDE.md/README): 写死 Conda 环境名为 `gamstekpeaking` —— CLAUDE.md 技术约束新增环境名硬约束；README 环境要求表、安装步骤、FAQ 同步改用根目录 requirements.txt 与环境名
- 文档生成(README.md): 修正推理模式表描述 —— mzml/batch_mzml 由“仅预测”改为“仅 EIC/ROI 提取（--plot 附加预测画图，无预测 CSV）”（经代码核对：两者只调 extract_xic_with_pyopenms / run_batch_mzml，不输出 prediction.csv）；轻量模式小节补充行为提醒
- 文档生成(README.md): 推理章节前新增「前处理（原始数据 → mzML）」模块 —— 介绍 converters/ 三个脚本（msdata/wiff/rename_cn）与工具链（OpenMS/ProteoWizard）、使用步骤（重命名→dry-run→批量转换）、注意事项（中文路径、.wiff.scan、退码 858 判定）
- 文档生成(User_Tutorials.md): 新建四模式用户教程 —— 基于代码事实（extract_xic_with_pyopenms 通道提取与 QC、getFeature CentWave 参数、extract_xic_from_arrays 外部数组、smooth_sigma 建议）撰写 Targeted/Untargeted × Centroided/Profile 四组合的原理、命令、参数建议与 FAQ；README 推理节添加链接
- 需求分析(CLAUDE.md/README): 声明开发范围 —— 当前仅开发 Targeted × Centroided（MRM）模式，其余三组合保留现状暂不开发；CLAUDE.md 新增「开发范围」硬性约束（禁止动 getFeature/R、extract_xic_from_arrays 等非 MRM 代码，公共代码改动以 MRM 模式回归验证）；README 简介、环境要求（R 可选标注）、推理节链接提示、Untargeted 节同步声明

### 2026-08-10

- 文档生成(README.md): 基于当前模型架构与 conda 环境校正 README —— 修正编解码器层数（3层Decoder→1层Decoder，与 quanformer.pth 一致）、补充 num_queries=3 / hidden_dim=256 / nheads=8 等架构细节、conda 环境名切换为 `gamstekpeaking`（Python 3.11.15 + PyTorch 2.6.0+cu124，含全部依赖）、标注本机 GPU 为 8× RTX 4090 D、训练参数表注明 checkpoint 实际 num_queries 与默认值差异、新增 checkpoint 完整训练参数说明
- 代码生成(mrmpformer/util/logutil.py): 新建运行时日志过滤模块 —— 通过替换 sys.stdout 为带缓冲的 `_FilteredStdout` 实现按行前缀级别（[INFO]/[WARN]/[ERROR]）拦截输出；默认级别 WARNING（抑制 [INFO]），支持环境变量 `MRMPFORMER_LOG_LEVEL` 与 `configure_log_level()` API 控制
- 代码生成(main.py): CLI 新增 `--verbose` / `--quiet` 参数；解析后调用 `install_filter()` 安装全局日志过滤器
- 调试(mrmpformer/util/__init__.py): 修复预存导入错误 —— `safe_torch_load` 实际位于 `misc.py` 而非 `io.py`，拆分导入语句避免 ImportError

### 2026-07-29

- 重构(tools/batch/): 合并 batch_post_newtest_under_snr_filtered.py 与 rerun_snr_under_snr_filtered.py → 新建 reprocess.py，统一 `--stage snr/post/snr-post` 三种运行模式，消除 post_newtest 参数分叉
- 代码生成(tools/batch/reprocess.py): 合并脚本支持分阶段批量处理 —— SNR 阶段调用 mzml_box_outside_snr_pipeline.run()，post 阶段通过 subprocess 调用 run_unified_peak_workflow.py；统一暴露所有 post_newtest 参数至 CLI
- 文档生成(tools/batch/): 旧脚本加弃用标记，指向 reprocess.py
- 重构(tools/mzml/): 合并 mzml_export_one_chrom.py 与 read_mzml_one_group.py → 新建 chromatogram.py，统一 `list/show/export` 三个子命令
- 代码生成(tools/mzml/chromatogram.py): 共享 inspect.py 依赖，修复旧脚本 import 路径 bug（mzml_inspect_to_csv→inspect）；统一色谱定位、编码处理、数据提取逻辑
- 文档生成(tools/mzml/): 旧脚本加弃用标记，指向 chromatogram.py
- 文档生成(README.md): 全面更新 —— 修正核心目录名 main_model→model、重写项目结构树（新增 gamstekpeaking/converters/tools/管线脚本）、版本号 v0.3.0→v2.0.0、新增「辅助工具」章节

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
- 其他(converters/): 归档实验日志 exp_log.md（2026-08-13 合并入本日志后删除）—— 完成 191/191 个 .msdata 全量转换（100.0%）；关键教训：msdata2mzml.exe 仅接受位置参数（不支持 -in/-out）、需设置 TMP/TEMP 指向纯英文 tmp/ 目录绕过中文用户名、以 mzML 文件生成判定成败（退码 858 不可信）。注：该工具链脚本现已更名为 msdata.py
- 调试(check_env.py + check_gui.py): 修复 Windows GBK/UTF-8 编码错配导致中文乱码 — run_cmd() 统一 utf-8、GUI 改用 --outfile 绕过管道编码
- 文档生成(README.md): 新增「环境检测」小节，一行命令启动 GUI 弹窗

### 2026-07-15

- 需求分析(MRMPFormer/): 确定采用 SimCLR 对比学习方案（方案 B），ResNet50 骨干 + 128-d 投影头，自监督训练无需标注
- 代码生成(MRMPFormer/): 创建完整 SimCLR 训练框架 —— simclr.py（ResNet50 + ProjectionHead）、losses.py（NT-Xent）、augmentations.py（5 种 SimCLR 增强）、dataset.py（无标签图像数据集）、train.py（CosineAnnealing + 梯度累积）、extract_features.py（推理输出 2048-d 特征）
- 文档生成(docs/superpowers/specs/): 创建 2026-07-15-mrmpformer-simclr-design.md 设计文档

### 2026-08-03

- 文档生成(README.md): 修复完整管线部分 —— 数据路径改为 `../data/...`（`cd model` 后根目录数据）、输出结构图修正（`prediction_refined.csv` 实际位于 `snr_filtered/<样品>/SNR_box_<阈值>/` 下，补充 `predicted_plots/`、`refined_plots/`、计时日志）、轻量模式路径统一
- 文档生成(README.md): 安装部分拆分为「第一步 安装 Python 环境」+「第二步 安装项目依赖」两节，明确 Python 3.10~3.11 与 pip/conda 两种路径
- 文档生成(README.md): 推理参数速查新增「完整参数模板」—— 覆盖 `model/main.py` argparse 全部 42 个参数（基础/QC/SNR/Post 四组），每个参数配注释填写说明，可直接复制运行（bash 语法已验证）
- 其他(model/environment.yml): 补充缺失的 `pyopenms==3.3.0`，与 `requirements.txt` 对齐（testXIC.py 顶层硬依赖，缺失会导致管线无法启动）
