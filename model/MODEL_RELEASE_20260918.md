# 当前 MRMPFormer 模型发布说明（2026-09-18）

本次同步当前 `mrmpformer_special_v2` 权重、ONNX 导出与 Python 推理更新，便于软件团队取得同一版本。AI 与传统 MassNova 结果比较的置信度和预警功能已交后端实现，不包含在本次模型发布中。

## 权重与导出

- PyTorch：`checkpoint/mrmpformerv2.pth`
- ONNX：`checkpoint/mrmpformerv2.onnx`
- 文件大小和 SHA-256：`checkpoint/mrmpformerv2_manifest.json`
- 两个大文件使用 Git LFS。克隆后执行 `git lfs pull --include="model/checkpoint/mrmpformerv2.*"`，确保取得实际文件而不是 LFS 指针。
- 按软件接入约定沿用旧名 `mrmpformerv2`，文件内容覆盖为最新 special_v2。旧内容可从 Git 历史恢复；文件名本身不能用于判断模型版本，请核对 manifest 的版本和 SHA-256。

在仓库 `model/` 目录中重新导出并验证：

```powershell
python -m tools.export_onnx --checkpoint checkpoint/mrmpformerv2.pth --out checkpoint/mrmpformerv2.onnx
```

ONNX 为 opset 17，预处理内置；输入 `image` 是 RGB float32、原始 0–255 像素，`img_size` 为 `(W,H)`。输出为 `scores`、`boxes_xyxy`、`boxes_norm`。不要在软件端重复归一化。模型本身不输出 RT、面积、复合置信度或报警状态。

## 当前 Python massnova 模式

```powershell
python -m inference.cli --config configs/massnova.json --mode massnova
```

配置默认读取 `../data/mzml/test3`。切换输入时，应只提供单文件 `mzml` 或目录 `batch_dir` 中的一项。

- 权重：special_v2；模型检测框阈值：0.6。
- SciPy `find_peaks + prominence` 生成候选，模型按峰切窗验证。
- 模型未验证的候选进入信号边界精修与门控；兜底 SNR 门槛为 10。
- 最终去重同时要求峰顶间距 ≤0.2 min、重叠占较窄区间 ≥25%、基线校正谷底/较小峰顶 ≥70%。深谷分隔的相邻峰保留。
- 最终输出补算面积、SNR、点数，并写回各通道峰集合，避免空结果。
- `model_score` 是网络分类分。信号兜底的 `signal_score` 使用 SNR 与点数的规则分，`peak_score` 按来源选择；分数来源写在 `score_source`，供后端使用。
- `rt_peak` 来自信号候选峰顶采集点，模型框用于映射起止时间。模型阈值不是最终 GREEN/YELLOW/RED 门槛。

信号峰自身分：

`signal_score = (SNR/(SNR+10))^0.8 × min(n_points/10, 1)^0.2`

这是工程分数，不是校准后的正确概率。

## 软件接入边界

本次不主动修改 `cpp/` 接口源码。远端已有的软件团队提交应保留。现有 C++ 接口与 Python massnova 的候选/兜底/边界处理并不相同：C++ 默认检测阈值 0.99，且没有有效模型峰时才启用单个最高峰兜底。更换为 special_v2 ONNX 或把阈值改为 0.6，不会自动使两端后处理完全一致。

CentreWave/CentWave 实验、临时测试输入、原始采集数据、3000 张评审图片和比较置信度后端参考模块不作为当前生产模型依赖发布。发布前的实验代码保留在本地备份中，生产候选检测继续使用 SciPy。

## 验证

- 深谷保留、浅谷重复框去重的 3 项回归测试。
- 模型峰与信号剩余峰同时进入最终 CSV，面积、点数和自身分完整的输出回归测试。
- ONNX 与 PyTorch 在 320×256、448×192 两个尺寸比较，最大分数误差 ≤1.49e-7，最大像素框误差 ≤1.22e-4。
- `test3_3` 的 napropamide-2、spinetoram L-1 真实推理验证通过；前者保留 3 个峰（含此前误删的中间峰），两个通道共 4 个峰，面积与自身分均完整输出。
- 现有 C++ `validate_contract` 要求的输入输出名称、float32 类型、张量维数通过检查；批量 1/2/3、400×300 和 448×192 下 ONNX Runtime 推理通过，像素框与归一化框映射最大误差 3.06e-5。复查工具：`tools/diagnostics/check_onnx_interface_compatibility.py`。
- 本机未找到 CMake/C++ 编译器，未运行编译后的 DLL/C API 集成测试。上述结果证明模型的张量协议及 Runtime 推理兼容，不代表已验证软件端 DLL 部署、CUDA 依赖或最终峰结果等价。
- 历史全量结果来自 2026-09-16 的 54 个样本、随机 3000 通道评审；本次发布前额外进行关键通道真实推理验证，不把历史统计当作此次完整重跑。

```powershell
python -m unittest tests.test_massnova_peak_dedup tests.test_massnova_output tests.test_mrmpformer_v1
```
