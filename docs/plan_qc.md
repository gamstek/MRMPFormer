# 标注数据质量 QC 行动方案（plan_qc）

> 日期：2026-08-20 ｜ 状态：**方案已定稿，未实施**（本文档即实施蓝图）
> 需求来源：用户提出「同一标注文件中同化合物 RT 差别过大则不生成 ROI，作为 QC 一部分」；扩展为「训练/评估数据加质量防线 + 推理时跨样品/双离子 RT 极差检查 + 各环节 QC 表统一输出 `../output/QC/`」。

---

## 1. 目标与判定规则

### 1.1 两类 RT 一致性检查（均基于标注 xlsx）

| 检查 | 分组键 | 计算量 | 物理依据 |
|---|---|---|---|
| **A. 跨样品极差** | `(compound, channel)` × 各 `sample_id` | `max(rt) − min(rt)` | 同一色谱方法下同化合物 RT 稳定（连续进样漂移通常 <0.1 min）；>1 min 提示某样品标注画错或仪器异常 |
| **B. 样品内双离子极差** | `(sample_id, compound)` × 各 `channel`（-1 定量 / -2 定性） | `max(rt) − min(rt)` | 定量/定性离子必须共流出；>1 min 提示通道张冠李戴或干扰峰被误标 |

**判定**：极差 **> 1.0 min**（`--qc_label_rt_tol`，默认 1.0，可调）→ 判「疑似实验有误」：
1. 终端 `[WARN]` 警示**人工复核**（列出化合物、通道、各样品 RT）；
2. 涉事行**剔除、不生成 ROI / 不进训练 bbox**；
3. 写入 QC 结果表。

### 1.2 剔除粒度策略（关键设计决策）

| 场景 | 策略 | 理由 |
|---|---|---|
| A 检查、组内样品数 ≥3 | 只剔**偏离组中位数 >1 min 的样品**的行 | 多数派可信，仅离群样品可疑，避免误伤好样品 |
| A 检查、组内样品数 =2 | **两行都剔** | 无法仲裁谁错（双方各偏 >0.5 min），宁缺毋滥 + 人工复核 |
| B 检查（双离子） | **两通道都剔** | 同上，无法判断 -1/-2 谁错 |

### 1.3 高效计算方法（向量化，单次解析复用）

标注表很小（当前 120 行），但按可扩展写法一次 `groupby` 完成，无 Python 双重循环：

```python
df = parse_labels_xlsx(labels_path)            # 复用现有解析（含 parse_rt_field 容错）

# A. 跨样品极差：一次 groupby-agg
cross = (df.groupby(["compound", "channel"])["rt"]
           .agg(rt_min="min", rt_max="max", rt_median="median", n="count"))
cross["rt_range"] = cross["rt_max"] - cross["rt_min"]

# B. 双离子极差：另一次 groupby-agg
inner = (df.groupby(["sample_id", "compound"])["rt"]
           .agg(rt_min="min", rt_max="max", n="count"))
inner["rt_range"] = inner["rt_max"] - inner["rt_min"]

# 剔除行定位：A 用 |rt − 组中位| > tol（n≥3）或整组（n=2）；B 取整组行
```

复杂度 O(n)，两遍 groupby；labels 仅解析一次，A/B/（训练侧）三方共用。

---

## 2. 新增模块与改动点

### 2.1 新模块 `model/preprocessing/label_qc.py`

```python
def check_label_rt_consistency(labels_df, tol=1.0):
    """返回 (qc_rows: DataFrame, exclude_keys: set[(sample_id, compound, channel)])"""
```

- `qc_rows` 列：`check_type`(cross_sample/ion_pair)、`compound`、`channel`、`sample_id`、`rt`、`group_median`、`rt_range`、`n_group`、`action`(excluded/kept)、`suggest_review`(bool)
- 同时返回终端 WARN 文本所需信息

### 2.2 挂点一：训练/评估数据防线（coco_annotation.py）

- 位置：`parse_labels_xlsx` 之后、逐 mzML 生成 bbox 之前
- 行为：exclude_keys 命中的标注行**不参与 bbox 映射**（对应 ROI 自动降级为负样本，与现有「无标注 ROI → 负样本」路径一致）
- 新参数：`--qc_label_rt_tol`（默认 1.0；0=关闭）
- QC 表输出：`../output/QC/coco_<实验名>_<时间戳>/qc_label_rt.csv`
- 效果链：错标注 → 不进 COCO bbox → 不污染训练/评估

### 2.3 挂点二：推理管线（cli.py pipeline + xic_extraction.py）

- cli 新参数：`--labels`（可选，指向 `data/<实验>/label/*.xlsx`）、`--qc_label_rt_tol`（默认 1.0；不传 `--labels` 则此检查整体跳过，**不破坏"推理无需标注"契约**）
- 流程：pipeline 分支 ROI 循环前调 `check_label_rt_consistency` → 由 exclude_keys 经 `label_key()` 转成 `exclude_native_ids: set[str]` → 传入 `extract_xic_with_pyopenms` 新参数
- `xic_extraction.extract_xic_with_pyopenms` 新增参数 `exclude_native_ids=None`：命中通道 `continue`，`qc_excluded` 追加行，reason=`label_rt_cross_sample` / `label_rt_ion_pair`（复用现有 `pipeline_qc_excluded.csv` 结构与 reason 列，零下游改动）
- 双离子检查不依赖多样品，单样品 pipeline 同样生效

### 2.4 各环节 QC 表统一输出 `../output/QC/<run_name>/`

`run_name` = pipeline `--output_dir` 的目录名（如 `full_pipeline` / `test/xxx`），训练侧为 `coco_<实验名>_<时间戳>`：

| 文件 | 来源环节 | 内容 | 现状 |
|---|---|---|---|
| `qc_label_rt.csv` | 新·标注 RT 一致性 | §1.1 A+B 全量结果（含 kept 行，便于复核） | **新建** |
| `qc_roi_channels.csv` | ① ROI 通道级 | 各样品 `pipeline_qc_excluded.csv` 合并（含新 label_rt reason） | 已有散件 → **汇总复制** |
| `qc_prediction_threshold.csv` | ② 预测阈值 | 每 ROI 图低于 threshold 被丢弃的框数统计 | **新建**（predictor 记录 dropped 计数） |
| `qc_snr_boxes.csv` | ④ SNR 框级 | `box_outside_snr_report.csv` 合并（逐框 SNR + passed_* 列） | 已有散件 → **汇总复制** |
| `qc_post_refinement.csv` | ⑤ 精修框级 | 精修各门控剔除的框（confidence/SNR/次峰比例/框宽上限） | **新建**（peak_refinement 需补剔除记录，工作量最大） |
| `qc_summary.md` | 汇总 | 各环节：检查数/剔除数/剔除率/人工复核清单 | **新建** |

汇总动作在 pipeline 结束、timing 输出前统一执行（一个 `_collect_qc_tables(base_out, qc_root)` 函数）。

---

## 3. 实施分期

| 期 | 内容 | 改动文件 | 风险 |
|---|---|---|---|
| **P1** ✅ 已完成(2026-08-20) | `label_qc.py` + coco_annotation 挂点（训练/评估防线） | 新 1 + 改 1 | 低——只影响数据构建，且默认阈值温和（1.0 min） |
| **P2** | cli `--labels` + `exclude_native_ids` 贯通 + `output/QC` 目录落地（label_rt / roi_channels 两表 + summary） | 改 3 | 低——不传 labels 行为零变化 |
| **P3** | SNR/预测阈值/精修三张 QC 表接入汇总（精修剔除记录为最大增量） | 改 3 | 中——peak_refinement 需在不改变输出行维度的前提下记录被剔框 |

### P1 实施记录（2026-08-20）

- 新模块 `model/preprocessing/label_qc.py`：`check_label_rt_consistency(labels, tol)` / `mark_excluded_labels` / `write_qc_table`；RT 解析内联（解除 pyopenms import 链依赖，可独立单测）
- `coco_annotation.py`：新增 `--qc_label_rt_tol`（默认 1.0，0=关闭）；QC 在 parse_labels_xlsx 后执行；被标行不进 by_key / rt_overrides / 行序回退匹配（ROI 降级负样本，行序对齐保持）；QC 表写 `output/QC/coco_<实验名>_<时间戳>/qc_label_rt.csv`
- 单测通过（n=2 双剔 / n=3 仅剔离群 / 双离子双剔 / WARN 计数 / CSV 列）；**真实标注回归（20260715）首跑即发现 8 组双离子 RT 极差 12.8~25.9 min（16 行判疑似实验有误，待人工复核）**——此前 v1/v2 评估 GT 与 COCO bbox 均包含这些可疑通道

每期验证命令（用户 PowerShell）：
```powershell
cd D:\work\MRMPFormer\model
# P1: 重建数据集，观察 WARN 与 output/QC 表
D:\Anaconda3\envs\gamstekpeaking\python.exe -m preprocessing.coco_annotation --mzmls ..\data\test\mzml\20260715_shiyaoyuan_test_1.mzML ..\data\test\mzml\20260715_shiyaoyuan_test_2.mzML --labels ..\data\test\label\testcase_data.xlsx --output_dir ..\data\test\coco --qc_label_rt_tol 1.0
# P2: 推理带 labels
D:\Anaconda3\envs\gamstekpeaking\python.exe -m inference.cli --mode pipeline --config configs/inference_pipeline.json --mzml ..\data\test\mzml\20260715_shiyaoyuan_test_1.mzML --labels ..\data\test\label\testcase_data.xlsx --output_dir ..\output\test\qc_check
# 验收：..\output\test\qc_check\QC\ 或 ..\output\QC\ 下出现 qc_label_rt.csv 等
```

## 4. 待用户确认项

- [ ] 阈值 1.0 min 作为默认（`--qc_label_rt_tol` 可调）
- [ ] n=2 跨样品场景「两行都剔」策略（宁缺毋滥）vs「仅警示不剔」
- [ ] QC 根目录：`../output/QC/`（pipeline 运行时写入 `<run_name>/` 子目录）是否符合预期
- [ ] P1→P2→P3 分期顺序
