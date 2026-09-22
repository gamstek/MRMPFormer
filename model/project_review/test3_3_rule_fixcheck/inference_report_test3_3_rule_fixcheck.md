# 推理报告（massnova）

- 生成时间: 2026-09-16 11:57:11
- 推理模式: `massnova`（整谱 XIC 全峰识别，模型前置、精修兜底） | 输出目录: `D:\yinlibo\MRMPFormer\model\project_review\test3_3_rule_fixcheck`
- 模型: `checkpoint/mrmpformer_special_v2.pth` | 置信度阈值: 0.6
- Phase1 候选检测器: `scipy`（scipy=find_peaks+prominence；centwave=CentWave CWT，隔离实验）
- 样本数: 1 | 总耗时: 315.9s

## 1. 样本摘要

| 样品 | 通道 | 候选 | 峰数 | 模型框 | 信号兜底 | 模型验证率 | 平均置信度 | 平均 SNR |
|---|---|---|---|---|---|---|---|---|
| test3_3 | 700 | 3139 | 823 | 665 | 158 | 80.8% | 0.895 | 4.07e+14 |

## 2. 化合物 × 样品 检出面积矩阵（主峰, intensity·min）

> 单元格 = 该样品该化合物各离子通道最大峰面积（—=未检出）；离子通道（uid 后缀 -1/-2）已合并。

| 化合物 | 检出/样品 | test3_3 |
|---|---|---
| 3-hydroxy carbofuran | 1/1 | 104966.6 |
| Chloridazon | 1/1 | 85784.4 |
| Diclobutrazol | 1/1 | 1727942.3 |
| Dimethenamid-P | 1/1 | 1222464.6 |
| EPN | 1/1 | 469738.5 |
| Fenoxycarb | 1/1 | 182392.9 |
| Penflufen | 1/1 | 2830652.7 |
| Picolinafen | 1/1 | 823055.4 |
| Proquinazid | 1/1 | 1665604.8 |
| acetamiprid | 1/1 | 230277.6 |
| acetochlor | 1/1 | 217219.3 |
| alachlor | 1/1 | 316635.0 |
| albendazole | 1/1 | 2044461.7 |
| aldicarb sulfoxide | 1/1 | 63898.1 |
| ametoctradin | 1/1 | 653541.8 |
| ametryn | 1/1 | 4202637.6 |
| amidosulfuron | 1/1 | 713564.6 |
| amisulbrom | 1/1 | 36934.7 |
| anilofos | 1/1 | 1242362.4 |
| atrazine | 1/1 | 25563833.9 |
| azoxystrobin | 1/1 | 2824482.7 |
| benalaxyl | 1/1 | 1322303.6 |
| benazolin-ethyl | 1/1 | 128691.2 |
| bendiocarb | 1/1 | 140416.3 |
| bensulfuron-methyl | 1/1 | 1000796.1 |
| benzoximate | 1/1 | 58481.3 |
| bifenox | 1/1 | 12204.6 |
| bifenthrin | 1/1 | 30007.1 |
| bioresmethrin | 1/1 | 247160.5 |
| bitertanol | 1/1 | 20492.9 |
| boscalid | 1/1 | 485903.6 |
| bromuconazole | 1/1 | 153960.8 |
| bupirimate | 1/1 | 1167627.5 |
| buprofezin | 1/1 | 3870796.1 |
| butachlor | 1/1 | 371413.3 |
| butralin | 1/1 | 617889.6 |
| cadusafos | 1/1 | 905351.2 |
| carbaryl | 1/1 | 43928.5 |
| carbendazim | 1/1 | 2364408.7 |
| carbofuran | 1/1 | 756098.2 |
| carboxin | 1/1 | 645291.3 |
| carfentrazone-ethyl | 1/1 | 338940.0 |
| chlorantraniliprole | 1/1 | 217273.6 |
| chlorbenzuron | 1/1 | 171995.8 |
| chlordimeform | 1/1 | 391948.9 |
| chlorfluazuron | 1/1 | 355611.9 |
| chlorimuron-ethyl | 1/1 | 1096195.2 |
| chlorpropham | 1/1 | 307231.8 |
| chlorpyrifos | 1/1 | 152176.2 |
| chlorpyrifos-methyl | 1/1 | 70994.0 |
| chlorsulfuron | 1/1 | 331427.9 |
| chlortoluron | 1/1 | 186627.9 |
| chromafenozide | 1/1 | 51177.1 |
| cinosulfuron | 1/1 | 1051084.1 |
| clethodim | 1/1 | 95382.1 |
| clethodim sulfone | 1/1 | 220120.3 |
| clethodim sulfoxide | 1/1 | 175451.8 |
| clomazone | 1/1 | 596205.0 |
| clothianidin | 1/1 | 5139872.3 |
| coumaphos | 1/1 | 615133.9 |
| coumoxystrobin | 1/1 | 125836.3 |
| cyanazine | 1/1 | 718909.3 |
| cyantraniliprole | 1/1 | 52803.8 |
| cyazofamid | 1/1 | 77992.7 |
| cyclosulfamuron | 1/1 | 1399399.3 |
| cycloxydim | 1/1 | 473841.3 |
| cyflufenamid | 1/1 | 739103.9 |
| cyflumetofen | 1/1 | 71723.8 |
| cyproconazole | 1/1 | 154163.7 |
| cyprodinil | 1/1 | 320285.4 |
| demeton-S-methyl | 1/1 | 5684.8 |
| demeton-S-methyl-sulfone | 1/1 | 226107.6 |
| demeton-S-sulfone | 1/1 | 362551.5 |
| demeton-S-sulfoxide | 1/1 | 508771.2 |
| diazinon | 1/1 | 3572665.5 |
| dichlorvos | 1/1 | 395925.6 |
| dicrotophos | 1/1 | 721568.9 |
| diethofencarb | 1/1 | 211181.1 |
| diethyl aminoethyl hexanoate | 1/1 | 2571486.4 |
| difenoconazole | 1/1 | 28173719.0 |
| diflubenzuron | 1/1 | 220164.6 |
| diflufenican | 1/1 | 1166908.3 |
| dimepiperate | 1/1 | 23644.6 |
| dimethoate | 1/1 | 254531.9 |
| dimethomorph | 1/1 | 8784916.5 |
| dimoxystrobin | 1/1 | 396096.5 |
| diniconazole | 1/1 | 123860.3 |
| dinotefuran | 1/1 | 1631768.0 |
| disulfoton sulfone | 1/1 | 92890.2 |
| disulfoton sulfoxide | 1/1 | 173192.3 |
| diuron | 1/1 | 61182.5 |
| edifenphos | 1/1 | 1211698.6 |
| emamectin B1a | 1/1 | 325287.9 |
| enestroburin | 1/1 | 3265789.8 |
| epoxiconazole | 1/1 | 190732.6 |
| ethion | 1/1 | 125617.0 |
| ethiprole | 1/1 | 159865.3 |
| ethirimol | 1/1 | 481953.1 |
| ethofumesate | 1/1 | 33129.3 |
| ethoprophos | 1/1 | 378790.6 |
| ethoxysulfuron | 1/1 | 1965321.4 |
| etofenprox | 1/1 | 148340.8 |
| etoxazole | 1/1 | 3788968.0 |
| etrimfos | 1/1 | 2334711.6 |
| fenamidone | 1/1 | 305842.3 |
| fenaminstrobin | 1/1 | 1895459.0 |
| fenamiphos | 1/1 | 880552.8 |
| fenamiphos sulfone | 1/1 | 499927.1 |
| fenamiphos sulfoxide | 1/1 | 415531.8 |
| fenarimol | 1/1 | 213291.6 |
| fenazaquin | 1/1 | 1839873.0 |
| fenbuconazole | 1/1 | 223302.9 |
| fenhexamid | 1/1 | 187041.7 |
| fenobucarb | 1/1 | 274304.4 |
| fenothiocarb | 1/1 | 136074.4 |
| fenoxanil | 1/1 | 178462.5 |
| fenoxaprop-P-ethyl | 1/1 | 1762140.7 |
| fenpropidin | 1/1 | 2210553.7 |
| fenpropimorph | 1/1 | 1534482.2 |
| fenpyrazamine | 1/1 | 1021092.1 |
| fenpyroximate | 1/1 | 3031842.9 |
| fensulfothion | 1/1 | 421926.6 |
| fensulfothion oxon | 1/1 | 802797.0 |
| fensulfothion oxon sulfone | 1/1 | 649337.8 |
| fensulfothion sulfone | 1/1 | 396921.7 |
| fenthion | 1/1 | 483975.5 |
| fenthion sulfone | 1/1 | 210792.0 |
| fenthion sulfoxide | 1/1 | 350757.4 |
| flonicamid | 1/1 | 184746.1 |
| florasulam | 1/1 | 273493.1 |
| fluazifop | 1/1 | 2176667.5 |
| flubendiamide | 1/1 | 7244.1 |
| flucetosulfuron | 1/1 | 781957.8 |
| flufenacet | 1/1 | 110210.5 |
| flufenoxuron | 1/1 | 420598.5 |
| flumetralin | 1/1 | 40327.6 |
| flumetsulam | 1/1 | 343668.9 |
| flumorph | 1/1 | 1419495.4 |
| fluopicolide | 1/1 | 19812109.0 |
| fluopyram | 1/1 | 65298183.5 |
| fluoroglycofen-ethyl | 1/1 | 13420.5 |
| flurtamone | 1/1 | 1114075.9 |
| flusilazole | 1/1 | 1475206.9 |
| fluthiacet-methyl | 1/1 | 170227.7 |
| flutolanil | 1/1 | 1114313.3 |
| flutriafol | 1/1 | 99328.3 |
| fluxapyroxad | 1/1 | 738989.1 |
| fonofos | 1/1 | 284580.7 |
| forchlorfenuron | 1/1 | 652969.1 |
| fosthiazate | 1/1 | 413141.6 |
| furathiocarb | 1/1 | 926252.5 |
| halosulfuron-methyl | 1/1 | 954822.5 |
| heptenophos | 1/1 | 193585.1 |
| hexaconazole | 1/1 | 184261.2 |
| hexaflumuron | 1/1 | 137828.0 |
| hexazinone | 1/1 | 2351025.4 |
| hexythiazox | 1/1 | 418281.2 |
| imazalil | 1/1 | 215518.1 |
| imibenconazole | 1/1 | 329952.5 |
| imidacloprid | 1/1 | 179767.6 |
| imidaclothiz | 1/1 | 104194.9 |
| indoxacarb | 1/1 | 229733.3 |
| iodosulfuron-methyl-sodium | 1/1 | 577473.1 |
| ipconazole | 1/1 | 197516.7 |
| iprobenfos | 1/1 | 111685.2 |
| iprodione | 1/1 | 452050.3 |
| iprovalicarb | 1/1 | 190498.6 |
| isazofos | 1/1 | 236243.8 |
| isocarbophos | 1/1 | 421792.4 |
| isofenphos-methyl | 1/1 | 912128.1 |
| isoprocarb | 1/1 | 218875.8 |
| isoprothiolane | 1/1 | 460985.3 |
| isoproturon | 1/1 | 924370.4 |
| isopyrazam | 1/1 | 770387.6 |
| isoxaflutole | 1/1 | 203826.0 |
| kresoxim-methyl | 1/1 | 15707.2 |
| lactofen | 1/1 | 31784.4 |
| linuron | 1/1 | 194362.8 |
| lufenuron | 1/1 | 336884.4 |
| malaoxon | 1/1 | 524990.4 |
| malathion | 1/1 | 43235.9 |
| mandipropamid | 1/1 | 481454.9 |
| mefenacet | 1/1 | 1382114.3 |
| mepronil | 1/1 | 1164437.5 |
| mesosulfuron-methyl | 1/1 | 581683.4 |
| metaflumizone | 1/1 | 229924.7 |
| metalaxyl | 1/1 | 4674508.3 |
| metamifop | 1/1 | 1824685.9 |
| metamitron | 1/1 | 360535.4 |
| metazachlor | 1/1 | 277695.1 |
| metazosulfuron | 1/1 | 591528.0 |
| metconazole | 1/1 | 142192.5 |
| methacrifos | 1/1 | 540415.5 |
| methamidophos | 1/1 | 1950288.0 |
| methiocarb | 1/1 | 163386.1 |
| methiocarb sulfone | 1/1 | 68572.4 |
| methiocarb sulfoxide | 1/1 | 745432.4 |
| methomyl | 1/1 | 18470.2 |
| methoprene | 1/1 | 26969.0 |
| metolachlor | 1/1 | 1699981.6 |
| metolcarb | 1/1 | 300363.6 |
| metrafenone | 1/1 | 916424.0 |
| metribuzin | 1/1 | 491938.4 |
| metsulfuron-methyl | 1/1 | 699949.9 |
| mevinphos | 1/1 | 125132.1 |
| molinate | 1/1 | 340953.6 |
| monocrotophos | 1/1 | 2569785.1 |
| myclobutanil | 1/1 | 129452.8 |
| napropamide | 1/1 | 774430.5 |
| nitenpyram | 1/1 | 913320.0 |
| novaluron | 1/1 | 284295.0 |
| omethoate | 1/1 | 280619.8 |
| orthosulfamuron | 1/1 | 176091.7 |
| oxadiargyl | 1/1 | 97685.8 |
| oxadiazon | 1/1 | 182973.3 |
| oxadixyl | 1/1 | 382746.8 |
| oxamyl | 1/1 | 4434.6 |
| oxamyl oxime | 1/1 | 75509.8 |
| oxaziclomefone | 1/1 | 989336.8 |
| oxydemeton-methyl | 1/1 | 650226.2 |
| oxyfluorfen | 1/1 | 163810.3 |
| paclobutrazol | 1/1 | 504234.2 |
| parathion | 1/1 | 333434.1 |
| penconazole | 1/1 | 390997.5 |
| pencycuron | 1/1 | 1482888.7 |
| pendimethalin | 1/1 | 207483.8 |
| penoxsulam | 1/1 | 640363.2 |
| penthiopyrad | 1/1 | 1628256.3 |
| phenamacril | 1/1 | 698612.0 |
| phenmedipham | 1/1 | 62798.1 |
| phenthoate | 1/1 | 2195515.1 |
| phorate sulfone | 1/1 | 15319.4 |
| phorate sulfoxide | 1/1 | 197633.1 |
| phosalone | 1/1 | 92622.9 |
| phosfolan | 1/1 | 961051.5 |
| phosfolan-methyl | 1/1 | 1265564.0 |
| phosmet | 1/1 | 20426.8 |
| phosmet oxon | 1/1 | 494598.3 |
| phosphamidon | 1/1 | 5978353.4 |
| phoxim | 1/1 | 231407.0 |
| picoxystrobin | 1/1 | 103866.1 |
| piperonyl butoxide | 1/1 | 256329.7 |
| pirimicarb | 1/1 | 1414054.1 |
| pirimicarb-desmethyl | 1/1 | 871306.2 |
| pirimicarb-desmethyl-formamido | 1/1 | 289947.4 |
| pirimiphos-methyl | 1/1 | 2744985.9 |
| pretilachlor | 1/1 | 2600303.3 |
| probenazole | 1/1 | 9147.6 |
| prochloraz | 1/1 | 138226.5 |
| prochloraz metabolite BTS44595 | 1/1 | 547389.2 |
| prochloraz metabolite BTS44596 | 1/1 | 194849.7 |
| procymidone | 1/1 | 116198.6 |
| profenofos | 1/1 | 409079.6 |
| promecarb | 1/1 | 314605.5 |
| prometryn | 1/1 | 3578844.9 |
| propachlor | 1/1 | 1267314.5 |
| propamocarb | 1/1 | 109946874.5 |
| propanil | 1/1 | 309328.0 |
| propaquizafop | 1/1 | 284891.3 |
| propargite | 1/1 | 34323.1 |
| propiconazole | 1/1 | 318085.7 |
| propisochlor | 1/1 | 114892.4 |
| propyrisulfuron | 1/1 | 475341.7 |
| propyzamide | 1/1 | 229241.7 |
| prosulfocarb | 1/1 | 761044.6 |
| pyraclostrobin | 1/1 | 1890735.0 |
| pyraflufen-ethyl | 1/1 | 706251.4 |
| pyrametostrobin | 1/1 | 1131658.1 |
| pyraoxystrobin | 1/1 | 688782.6 |
| pyrazosulfuron-ethyl | 1/1 | 1152154.2 |
| pyrethrins I | 1/1 | 101464.2 |
| pyrethrins II | 1/1 | 72316.2 |
| pyribenzoxim | 1/1 | 26600.4 |
| pyridaben | 1/1 | 847116.3 |
| pyridalyl | 1/1 | 139540.3 |
| pyridaphenthion | 1/1 | 898757.4 |
| pyriftalid | 1/1 | 448653.4 |
| pyrimethanil | 1/1 | 347193.7 |
| pyrimorph | 1/1 | 417685.6 |
| pyriproxyfen | 1/1 | 1609432.4 |
| pyrisoxazole | 1/1 | 418193.5 |
| quinalphos | 1/1 | 635893.5 |
| quizalofop-ethyl | 1/1 | 1610340.3 |
| rotenone | 1/1 | 243051.7 |
| saflufenacil | 1/1 | 105757.6 |
| sedaxane | 1/1 | 1103855.9 |
| sethoxydim | 1/1 | 415797.3 |
| silthiofam | 1/1 | 438007.5 |
| simazine | 1/1 | 433070.8 |
| simetryn | 1/1 | 1268988.9 |
| spinetoram J | 1/1 | 723770.9 |
| spinetoram L | 1/1 | 618649.7 |
| spinosad A | 1/1 | 439534.8 |
| spinosad D | 1/1 | 571295.5 |
| spirodiclofen | 1/1 | 183420.2 |
| spirotetramat | 1/1 | 556347.3 |
| spirotetramat-enol | 1/1 | 124914.4 |
| spirotetramat-keto-hydroxy | 1/1 | 55250.2 |
| spirotetramat-mono-hydroxy | 1/1 | 208186.4 |
| sulfentrazone | 1/1 | 51108.6 |
| sulfotep | 1/1 | 611126.9 |
| sulfoxaflor | 1/1 | 11991.8 |
| tau-fluvalinate | 1/1 | 65833.1 |
| tebuconazole | 1/1 | 214258.1 |
| tebufenozide | 1/1 | 159959.4 |
| tebuthiuron | 1/1 | 2205786.7 |
| terbufos | 1/1 | 22003.6 |
| terbufos sulfone | 1/1 | 25872.2 |
| terbuthylazine | 1/1 | 6294300.9 |
| tetraconazole | 1/1 | 461622.0 |
| thiabendazole | 1/1 | 498453.0 |
| thiacloprid | 1/1 | 192585.0 |
| thiamethoxam | 1/1 | 269926.1 |
| thidiazuron | 1/1 | 149013.9 |
| thifensulfuron-methyl | 1/1 | 692825.4 |
| thifluzamide | 1/1 | 102954.0 |
| thiophanate-methyl | 1/1 | 275899.0 |
| tolclofos-methyl | 1/1 | 166613.3 |
| tolfenpyrad | 1/1 | 695261.0 |
| tralkoxydim | 1/1 | 1544581.7 |
| triadimefon | 1/1 | 363577.9 |
| triadimenol | 1/1 | 222321.7 |
| triallate | 1/1 | 93357.1 |
| triasulfuron | 1/1 | 378019.1 |
| triazophos | 1/1 | 1166353.1 |
| tribenuron-methyl | 1/1 | 494675.1 |
| trichlorfon | 1/1 | 76221.5 |
| tricyclazole | 1/1 | 346904.3 |
| trifloxystrobin | 1/1 | 2092727.1 |
| triflumizole | 1/1 | 88489.9 |
| triflumizole metabolite FM-6-1 | 1/1 | 266047.2 |
| triflumuron | 1/1 | 112531.7 |
| triflusulfuron-methyl | 1/1 | 2223817.8 |
| triticonazole | 1/1 | 167771.5 |
| uniconazole | 1/1 | 162538.9 |
| vamidothion | 1/1 | 941225.1 |
| zoxamide | 1/1 | 428461.8 |

## 3. 输出布局

```text
test3_3_rule_fixcheck/
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