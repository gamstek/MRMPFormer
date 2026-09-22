# MRMPFormer vs CentreWave 可视化复核

绿色为人工标签，蓝色为 MRMPFormer，橙色为 CentreWave。单 ROI 判定容差为 0.1 min。

## 总览图

![核心指标](core_metrics.png)

![F1-容差曲线](f1_vs_tolerance.png)

![边界误差箱线图](boundary_error_boxplots.png)

![典型差异案例](representative_failures_montage.jpg)

## ROI 画廊

- [test1 全部 ROI](test1/index.html)
- [test2 全部 ROI](test2/index.html)

## 0.1 min 命中数量对比

### test1

- MRMPFormer 更准: 9 个 ROI
- CentreWave 更准: 0 个 ROI
- 两者均命中: 78 个 ROI
- 两者均正确无峰: 21 个 ROI
- 两者均未命中: 0 个 ROI
- 两者均有误差: 2 个 ROI
- 两者均误报: 0 个 ROI

### test2

- MRMPFormer 更准: 78 个 ROI
- CentreWave 更准: 29 个 ROI
- 两者均命中: 49 个 ROI
- 两者均正确无峰: 0 个 ROI
- 两者均未命中: 11 个 ROI
- 两者均有误差: 1 个 ROI
- 两者均误报: 0 个 ROI
