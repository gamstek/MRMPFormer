# 推理报告（massnova）

- 生成时间: 2026-09-14 17:22:49
- 推理模式: `massnova`（整谱 XIC 全峰识别，模型前置、精修兜底） | 输出目录: `D:\yinlibo\MRMPFormer\model\project_review\massnova_v2_smoke_test3_1`
- 模型: `checkpoint\mrmpformer_special_v2.pth` | 置信度阈值: 0.6
- Phase1 候选检测器: `scipy`（scipy=find_peaks+prominence；centwave=CentWave CWT，隔离实验）
- 样本数: 1 | 总耗时: 140.9s

## 1. 样本摘要

| 样品 | 通道 | 候选 | 峰数 | 模型框 | 信号兜底 | 模型验证率 | 平均置信度 | 平均 SNR |
|---|---|---|---|---|---|---|---|---|
| test3_1 | 669 | 5695 | 949 | 169 | 780 | 17.8% | 0.710 | 4.62e+11 |

## 2. 化合物 × 样品 检出面积矩阵（主峰, intensity·min）

> 单元格 = 该样品该化合物各离子通道最大峰面积（—=未检出）；离子通道（uid 后缀 -1/-2）已合并。

| 化合物 | 检出/样品 | test3_1 |
|---|---|---
| 3-hydroxy carbofuran | 1/1 | 33454.5 |
| Chloridazon | 1/1 | 10460.7 |
| Diclobutrazol | 1/1 | 4396.0 |
| EPN | 1/1 | 106198.1 |
| Fenoxycarb | 1/1 | 21137.1 |
| Penflufen | 1/1 | 25563.9 |
| Picolinafen | 1/1 | 1825074.1 |
| Proquinazid | 1/1 | 87664.6 |
| acetamiprid | 1/1 | 10061.9 |
| acetochlor | 1/1 | 18108.2 |
| aldicarb sulfoxide | 1/1 | 7205.9 |
| ametoctradin | 1/1 | 24050.3 |
| ametryn | 1/1 | 27338.3 |
| amidosulfuron | 1/1 | 366624.0 |
| amisulbrom | 1/1 | 18024.3 |
| anilofos | 1/1 | 12744.6 |
| atrazine | 1/1 | 35032.3 |
| azinphos-methyl | 1/1 | 12724.6 |
| azoxystrobin | 1/1 | 18469.9 |
| benalaxyl | 1/1 | 9450.1 |
| benazolin-ethyl | 1/1 | 78019.3 |
| bendiocarb | 1/1 | 16235.0 |
| bensulfuron-methyl | 1/1 | 17299.9 |
| benzoximate | 1/1 | 9232.0 |
| bifenox | 1/1 | 15809.2 |
| bifenthrin | 1/1 | 1272.4 |
| bioresmethrin | 1/1 | 18505.5 |
| bitertanol | 1/1 | 21232.3 |
| boscalid | 1/1 | 152609.9 |
| bromuconazole | 1/1 | 32082.9 |
| bupirimate | 1/1 | 11275.9 |
| buprofezin | 1/1 | 17641.5 |
| butachlor | 1/1 | 41658.2 |
| butralin | 1/1 | 151412.3 |
| cadusafos | 1/1 | 168300.2 |
| carbaryl | 1/1 | 29383.8 |
| carbendazim | 1/1 | 21590.0 |
| carbofuran | 1/1 | 83375.2 |
| carboxin | 1/1 | 26116.0 |
| carfentrazone-ethyl | 1/1 | 33111.0 |
| chlorantraniliprole | 1/1 | 193820.2 |
| chlorbenzuron | 1/1 | 1724.5 |
| chlorfluazuron | 1/1 | 15238.3 |
| chlorimuron-ethyl | 1/1 | 15282.2 |
| chlorpyrifos | 1/1 | 17118.8 |
| chlorpyrifos-methyl | 1/1 | 14641.5 |
| chlorsulfuron | 1/1 | 6829.3 |
| chromafenozide | 1/1 | 23267.6 |
| cinosulfuron | 1/1 | 7138.0 |
| clethodim | 1/1 | 30393.5 |
| clethodim sulfone | 1/1 | 95436.8 |
| clethodim sulfoxide | 1/1 | 7672.3 |
| clomazone | 1/1 | 41542.9 |
| clothianidin | 1/1 | 7417.3 |
| coumaphos | 1/1 | 40935.6 |
| coumoxystrobin | 1/1 | 42695.0 |
| cyanazine | 1/1 | 13520.9 |
| cyantraniliprole | 1/1 | 17051.8 |
| cyclosulfamuron | 1/1 | 43996.2 |
| cycloxydim | 1/1 | 125629.6 |
| cyflufenamid | 1/1 | 87169.5 |
| cyflumetofen | 1/1 | 14525.7 |
| cymoxanil | 1/1 | 69205.3 |
| cyproconazole | 1/1 | 8749.7 |
| cyprodinil | 1/1 | 6747.8 |
| demeton-S-methyl | 1/1 | 6762.2 |
| demeton-S-methyl-sulfone | 1/1 | 406068.8 |
| demeton-S-sulfone | 1/1 | 113168.3 |
| demeton-S-sulfoxide | 1/1 | 38473.4 |
| diazinon | 1/1 | 10888.7 |
| dichlorvos | 1/1 | 452310.5 |
| dicrotophos | 1/1 | 163217.3 |
| diethyl aminoethyl hexanoate | 1/1 | 28943.5 |
| difenoconazole | 1/1 | 8359.6 |
| diflubenzuron | 1/1 | 19414.0 |
| diflufenican | 1/1 | 64990.3 |
| dimepiperate | 1/1 | 55920.5 |
| dimethoate | 1/1 | 12441.0 |
| dimethomorph | 1/1 | 38712.6 |
| dimoxystrobin | 1/1 | 21846.8 |
| dinotefuran | 1/1 | 4539064.0 |
| disulfoton sulfone | 1/1 | 13897.8 |
| disulfoton sulfoxide | 1/1 | 92349.0 |
| edifenphos | 1/1 | 82111.1 |
| enestroburin | 1/1 | 16466.0 |
| epoxiconazole | 1/1 | 21847.4 |
| ethion | 1/1 | 106017.6 |
| ethiprole | 1/1 | 15076.3 |
| ethirimol | 1/1 | 6418.4 |
| ethofumesate | 1/1 | 17792.1 |
| ethoxysulfuron | 1/1 | 36181.1 |
| etofenprox | 1/1 | 248497.2 |
| etoxazole | 1/1 | 17785.3 |
| etrimfos | 1/1 | 10933.5 |
| fenamidone | 1/1 | 43459.2 |
| fenaminstrobin | 1/1 | 9284.0 |
| fenamiphos | 1/1 | 475729.6 |
| fenamiphos sulfone | 1/1 | 53983.9 |
| fenamiphos sulfoxide | 1/1 | 49866.4 |
| fenarimol | 1/1 | 25897.6 |
| fenazaquin | 1/1 | 147343.9 |
| fenbuconazole | 1/1 | 3251.2 |
| fenhexamid | 1/1 | 29210.2 |
| fenobucarb | 1/1 | 153710.4 |
| fenothiocarb | 1/1 | 36429.3 |
| fenoxanil | 1/1 | 1967.2 |
| fenoxaprop-P-ethyl | 1/1 | 33368.1 |
| fenpropathrin | 1/1 | 16642.3 |
| fenpropidin | 1/1 | 8802.3 |
| fenpropimorph | 1/1 | 183502.8 |
| fenpyroximate | 1/1 | 111417.7 |
| fensulfothion oxon | 1/1 | 31291.6 |
| fensulfothion oxon sulfone | 1/1 | 56488.6 |
| fensulfothion sulfone | 1/1 | 70708.0 |
| fenthion | 1/1 | 617232.3 |
| fenthion sulfone | 1/1 | 347577.2 |
| fenthion sulfoxide | 1/1 | 17538.2 |
| flonicamid | 1/1 | 812097.7 |
| florasulam | 1/1 | 2924.6 |
| fluazifop | 1/1 | 189840.6 |
| flubendiamide | 1/1 | 20190.1 |
| flucetosulfuron | 1/1 | 16514.9 |
| flufenacet | 1/1 | 11748.9 |
| flufenoxuron | 1/1 | 44938.7 |
| flumetralin | 1/1 | 6876.7 |
| flumetsulam | 1/1 | 2538.8 |
| flumorph | 1/1 | 38811.4 |
| fluopyram | 1/1 | 97233.2 |
| fluoroglycofen-ethyl | 1/1 | 21729.8 |
| flusilazole | 1/1 | 112944.4 |
| flutriafol | 1/1 | 16358.9 |
| fonofos | 1/1 | 32144.6 |
| forchlorfenuron | 1/1 | 170739.8 |
| fosthiazate | 1/1 | 64178.3 |
| furathiocarb | 1/1 | 162951.7 |
| halosulfuron-methyl | 1/1 | 22606.7 |
| heptenophos | 1/1 | 20863.7 |
| hexaconazole | 1/1 | 16489.5 |
| hexaflumuron | 1/1 | 5833.7 |
| hexazinone | 1/1 | 33539.9 |
| hexythiazox | 1/1 | 71229.2 |
| imazalil | 1/1 | 188998.7 |
| imibenconazole | 1/1 | 66698.0 |
| imidacloprid | 1/1 | 39583.3 |
| imidaclothiz | 1/1 | 7881.7 |
| indoxacarb | 1/1 | 3366.4 |
| iodosulfuron-methyl-sodium | 1/1 | 4000.7 |
| ipconazole | 1/1 | 5922.1 |
| iprobenfos | 1/1 | 16032.0 |
| iprodione | 1/1 | 68236.7 |
| iprovalicarb | 1/1 | 18621.5 |
| isazofos | 1/1 | 11239.0 |
| isocarbophos | 1/1 | 118881.1 |
| isofenphos-methyl | 1/1 | 56025.7 |
| isoprocarb | 1/1 | 29764.3 |
| isoproturon | 1/1 | 3513.7 |
| isopyrazam | 1/1 | 12091.3 |
| isoxaflutole | 1/1 | 35381.8 |
| kresoxim-methyl | 1/1 | 11903.7 |
| lactofen | 1/1 | 27007.0 |
| linuron | 1/1 | 11854.2 |
| lufenuron | 1/1 | 7464.4 |
| malaoxon | 1/1 | 40193.3 |
| malathion | 1/1 | 44023.1 |
| mandipropamid | 1/1 | 53577.3 |
| mefenacet | 1/1 | 27768.0 |
| mepronil | 1/1 | 95530.4 |
| mesosulfuron-methyl | 1/1 | 3418.0 |
| metaflumizone | 1/1 | 2067.5 |
| metamifop | 1/1 | 25634.3 |
| metamitron | 1/1 | 48077.3 |
| metazachlor | 1/1 | 28717.4 |
| metazosulfuron | 1/1 | 7154.3 |
| methacrifos | 1/1 | 886624.9 |
| methamidophos | 1/1 | 124928.9 |
| methidathion | 1/1 | 55230.7 |
| methiocarb | 1/1 | 79882.5 |
| methiocarb sulfone | 1/1 | 22214.1 |
| methiocarb sulfoxide | 1/1 | 33529.5 |
| methomyl | 1/1 | 7305.5 |
| methoprene | 1/1 | 1056746.1 |
| methoxyfenozide | 1/1 | 7311.5 |
| metolachlor | 1/1 | 522249.6 |
| metolcarb | 1/1 | 22465.7 |
| metrafenone | 1/1 | 25809.6 |
| metsulfuron-methyl | 1/1 | 4906.1 |
| mevinphos | 1/1 | 19449.9 |
| molinate | 1/1 | 110056.2 |
| monocrotophos | 1/1 | 392412.2 |
| myclobutanil | 1/1 | 19647.7 |
| napropamide | 1/1 | 12535.0 |
| orthosulfamuron | 1/1 | 9247.1 |
| oxadiargyl | 1/1 | 63081.9 |
| oxadiazon | 1/1 | 67668.1 |
| oxadixyl | 1/1 | 1208911.6 |
| oxamyl | 1/1 | 2850.6 |
| oxaziclomefone | 1/1 | 13112.4 |
| oxydemeton-methyl | 1/1 | 63161.1 |
| oxyfluorfen | 1/1 | 34108.7 |
| paclobutrazol | 1/1 | 21541.9 |
| penconazole | 1/1 | 23701.9 |
| pencycuron | 1/1 | 9796.7 |
| pendimethalin | 1/1 | 145914.3 |
| penoxsulam | 1/1 | 6514.2 |
| penthiopyrad | 1/1 | 69807.2 |
| phenamacril | 1/1 | 9019.6 |
| phenmedipham | 1/1 | 6884.4 |
| phenthoate | 1/1 | 2671070.8 |
| phorate | 1/1 | 6044.3 |
| phorate sulfoxide | 1/1 | 38403.7 |
| phosalone | 1/1 | 159481.5 |
| phosfolan | 1/1 | 60507.9 |
| phosfolan-methyl | 1/1 | 6777.5 |
| phosmet | 1/1 | 8273.1 |
| phosmet oxon | 1/1 | 15675.9 |
| phosphamidon | 1/1 | 10366.4 |
| phoxim | 1/1 | 9475.4 |
| picoxystrobin | 1/1 | 12509.2 |
| piperonyl butoxide | 1/1 | 49596.9 |
| pirimicarb | 1/1 | 34775.1 |
| pirimicarb-desmethyl | 1/1 | 41037.5 |
| pirimicarb-desmethyl-formamido | 1/1 | 6921.0 |
| pirimiphos-methyl | 1/1 | 13817.7 |
| pretilachlor | 1/1 | 311710.7 |
| prochloraz | 1/1 | 18329.2 |
| prochloraz metabolite BTS44595 | 1/1 | 68192.2 |
| procymidone | 1/1 | 101451.9 |
| profenofos | 1/1 | 21233.2 |
| promecarb | 1/1 | 64458.6 |
| prometryn | 1/1 | 28383.0 |
| propachlor | 1/1 | 176626.0 |
| propaquizafop | 1/1 | 157298.8 |
| propargite | 1/1 | 27938.9 |
| propiconazole | 1/1 | 10856.5 |
| propoxur | 1/1 | 9685.7 |
| propyrisulfuron | 1/1 | 12924.8 |
| propyzamide | 1/1 | 193028.6 |
| prosulfocarb | 1/1 | 1743135.6 |
| pyraclostrobin | 1/1 | 5606.1 |
| pyraflufen-ethyl | 1/1 | 120548.8 |
| pyrametostrobin | 1/1 | 30929.7 |
| pyraoxystrobin | 1/1 | 5378.1 |
| pyrazosulfuron-ethyl | 1/1 | 6012.7 |
| pyrethrins I | 1/1 | 3961.8 |
| pyrethrins II | 1/1 | 12094.2 |
| pyribenzoxim | 1/1 | 7131.2 |
| pyridaben | 1/1 | 731133.4 |
| pyridalyl | 1/1 | 10385.6 |
| pyridaphenthion | 1/1 | 80417.7 |
| pyriftalid | 1/1 | 31680.6 |
| pyrimethanil | 1/1 | 5971.7 |
| pyrimorph | 1/1 | 12896.0 |
| pyriproxyfen | 1/1 | 23828.7 |
| pyrisoxazole | 1/1 | 14194.9 |
| quinalphos | 1/1 | 33512.7 |
| quizalofop-ethyl | 1/1 | 556996.2 |
| rotenone | 1/1 | 31792.5 |
| saflufenacil | 1/1 | 18644.0 |
| sedaxane | 1/1 | 23384.0 |
| sethoxydim | 1/1 | 36704.3 |
| simazine | 1/1 | 29858.2 |
| simetryn | 1/1 | 39598.3 |
| spinetoram L | 1/1 | 908.8 |
| spiromesifen | 1/1 | 50036.1 |
| spirotetramat-enol | 1/1 | 27632.8 |
| spirotetramat-enol-glucoside | 1/1 | 59456.4 |
| spirotetramat-keto-hydroxy | 1/1 | 35928.2 |
| spirotetramat-mono-hydroxy | 1/1 | 36043.1 |
| sulfentrazone | 1/1 | 3642.6 |
| sulfotep | 1/1 | 15710.4 |
| sulfoxaflor | 1/1 | 25611.3 |
| tau-fluvalinate | 1/1 | 203816.4 |
| tebuconazole | 1/1 | 6675.8 |
| tebufenozide | 1/1 | 39076.1 |
| tebuthiuron | 1/1 | 22720.7 |
| teflubenzuron | 1/1 | 17030.8 |
| terbufos | 1/1 | 289898.9 |
| terbufos sulfone | 1/1 | 131565.1 |
| terbufos sulfoxide | 1/1 | 63720.6 |
| terbuthylazine | 1/1 | 206766.6 |
| tetraconazole | 1/1 | 7307.6 |
| thiabendazole | 1/1 | 7158.7 |
| thiacloprid | 1/1 | 30267.2 |
| thiamethoxam | 1/1 | 24884.2 |
| thidiazuron | 1/1 | 46209.2 |
| thifensulfuron-methyl | 1/1 | 15047.7 |
| thifluzamide | 1/1 | 3384.0 |
| thiophanate-methyl | 1/1 | 8769.2 |
| tralkoxydim | 1/1 | 11202489.9 |
| triadimefon | 1/1 | 54152.6 |
| triadimenol | 1/1 | 6652.9 |
| triallate | 1/1 | 11023.1 |
| triasulfuron | 1/1 | 1187.3 |
| triazophos | 1/1 | 13175.3 |
| tribenuron-methyl | 1/1 | 6939.4 |
| trichlorfon | 1/1 | 1309364.5 |
| tricyclazole | 1/1 | 5328.2 |
| trifloxystrobin | 1/1 | 8407.2 |
| triflumizole | 1/1 | 10745.7 |
| triflumizole metabolite FM-6-1 | 1/1 | 500631.4 |
| triflumuron | 1/1 | 19055.1 |
| triticonazole | 1/1 | 43621.8 |
| tritosulfuron | 1/1 | 5502.6 |
| uniconazole | 1/1 | 23179.7 |
| vamidothion | 1/1 | 73220.4 |
| zoxamide | 1/1 | 44805.9 |

## 3. 输出布局

```text
massnova_v2_smoke_test3_1/
  prediction_signal/<样品>/    2min 窗口图（仅信号兜底峰边界，无模型分）
  prediction_model/<样品>/     2min 窗口图（仅模型框峰边界，含 score）
  prediction_refined/<样品>/   最终结果：massnova_peaks.csv / scan_summary.csv /
                               scan_qc_excluded.csv + 2min 窗口图（全部峰）
  scan_plots/<样品>/           长 XIC 整谱标注图（红=模型验证，灰=信号兜底）
  model_plots/<样品>/          长 XIC 标注图（与其他模式同款，全峰）
  all.csv                      跨样本合并明细（每峰一行）
  inference_report_<实验名>.md  本报告
```

## 4. massnova_peaks.csv 关键列

| 列 | 含义 |
|---|---|
| rt_min / rt_peak / rt_max | 峰边界与峰顶（min）；模型框峰取模型框边界 |
| area / snr / n_points | 最终边界上的面积 / 本地信噪比 / baseline 以上连续点数 |
| validated | 模型验证通过（model_score ≥ 阈值） |
| model_score | 模型置信度；未命中模型为空 |
| boundary_source | model=模型框边界；signal=信号兜底精修边界（建议人工复核） |

## 5. 已知问题与解决方案（实验记录）

开发/实验过程中遇到的问题抽象化归档，含根因层与对策状态：

| 编号 | 抽象问题 | 根因层 | 典型实例 | 状态与对策 |
|---|---|---|---|---|
| P1 | 多候选架构下同一色谱簇被多个候选独立切窗验证，输出重叠重复框（跨窗口无协调机制；pipeline 有单 ROI 天然防线 + (mz,q3) 面积去重，massnova 缺等价环节） | 架构层 | 恶虫威-1 5.0–5.6 三头簇曾输出 3 个 validated 峰，面积重复计约 3 遍 | ✅ 已修：跨候选去重（区间重叠且 apex 间距 ≤ scan_dup_apex_tol 判同一峰；模型框优先保留代表框） |
| P2 | 边界走查截停阈值（stable_tail_mean）在宽峰/双驼峰缓降尾上不收敛，兜底框远大于真实峰跨度 | 信号层 | 莠去津-1 兜底框 1.59min vs 真实 0.71min；恶虫威-2 兜底框 4.58–6.18 | ✅ 已修：兜底宽度保险丝（单侧跨度 > scan_width_fuse_ratio×半高跨度即回缩，且先回缩再门控） |
| P3 | SNR 邻居区段噪声估计把本峰边界拖尾的信号衰减计入 p-p 噪声 → SNR 被抬高数十倍低估 → 真峰被 SNR 门误杀 | 后处理层 | 甲羧除草醚-2 16.97min 峰 SNR=3.6（修复后 >800） | ✅ 已修：compute_local_snr 安静点过滤统一作用于邻居区段与回退扇区 |
| P4 | 模型对非「单峰居中」形态（多头簇/双驼峰/偏心峰）框回归偏窄、偏移甚至缺失 | 模型层（训练分布：ROI 均为 2min 窗单峰居中） | 恶虫威-1 去重后保留框只罩主头；莠去津-1 漏检、-2 只框半个峰 | ⚠️ 未根治：训练端需补多峰/偏心样本重训；推理端可做「模型框+信号延拓」混合边界校正（待做） |
| P5 | GPU 推理在阈值边缘（score≈阈值）存在运行间数值抖动，validated/兜底归属可能翻转 | 工程层 | 联苯三唑醇-1 相邻两次运行模型命中翻转 | ℹ️ 已知现象：阈值附近峰建议结合 boundary_source 列人工复核 |

P1/P2 对应参数：`scan_dup_apex_tol`（默认 0.2min，0=关闭）、`scan_width_fuse_ratio`（默认 1.5，0=关闭）。

---
> 由 `model/inference/massnova.py` 生成。