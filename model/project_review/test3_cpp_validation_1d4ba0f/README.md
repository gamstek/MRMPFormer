# test3 C++ 推理验收包

本包对应 Git 提交 `1d4ba0f966a14ef1b2058ebe4f2284447cb8fba2`，用于验证新的 `mrmpformer.dll`
是否与 Python MassNova 数组入口返回相同的逐峰 `a/b/c`。

## 固定配置

- ONNX SHA256: `D0B612A786FC937EE44170A4E60E59A59D23FE7ACDC2B85F5DF302AE2DBEE039`
- threshold: `0.5`，严格执行 `score > threshold`
- smooth_sigma: `0.8`
- use_gpu: `-1`（金标准固定 CPU）
- batch_size: `128`
- min_chrom_points: `10`
- min_max_intensity: `1000.0`

`inputs/full/` 是 58 个样本的完整 DLL JSON 输入；每个 `item` 是一条独立
MRM transition，`x` 为分钟单位原始 RT，`y` 为未平滑强度。
`smooth_sigma=0` 表示使用全局 0.8，禁止调用方提前再次平滑。

`golden/inputs/` 与 `golden/expected/` 是严格数值验收集。先逐个调用
`qf_process`，将 `qf_get_result_json` 得到的纯 JSON 保存到文件，再执行：

```powershell
python tools/compare_cpp_result.py `
  golden/expected/test3_3_stratified64.expected.json `
  actual/test3_3_stratified64.actual.json `
  --atol 1e-6
```

必须满足：item 顺序、uid、status、峰数量完全一致；每组 a/b/c 的绝对误差
不超过 1e-6。信号兜底峰同样位于 `peaks[]` 且通道状态为 `ok`。

`source_mzml/` 只用于数据溯源；DLL 本身不解析 mzML。旧的
`output/inference/massnova_test3` 使用 PTH 和阈值 0.8，不能作为本包标准答案。
传统算法 Excel、置信度预警图和候选窗口图片不属于 C++ 推理验收范围。
