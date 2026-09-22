# MRMPFormer Windows 集成包

本目录是一份同时支持 CPU 和 NVIDIA CUDA GPU 的 Windows x64 发布包。客户机器无需安装 Python、pip、Conda、PyTorch 或配置 PYTHONPATH。GPU 加速需要目标机具有兼容的 NVIDIA 显卡、驱动及 CUDA/cuDNN 运行环境；不可用时自动使用 CPU。

## 如何集成

1. 将整个目录（包括 `python/` 子目录）复制到软件的 `third_party/mrmpformer/runtime/windows/`。
2. 编译时包含本目录的 `mrmpformer.h`，链接本目录的 `mrmpformer.lib`。
3. 发布软件时，将本目录运行文件和 `python/` 原样复制到宿主 EXE 所在目录。不要只复制 `mrmpformer.dll`，不要混用旧版 DLL、头文件或 ONNX Runtime DLL。
4. 调用 `qf_default_config`，设置 `model_path` 为实际 `mrmpformerv2.onnx` 绝对路径、`work_dir` 为可写目录，然后调用 `qf_init`。默认 `use_gpu=0`：优先 GPU，不可用时回退 CPU；`-1` 强制 CPU，`1` 强制 GPU，不可用时初始化失败。调用 `qf_is_gpu_enabled()` 查询实际设备。
5. 使用原有 `qf_process` / `qf_process_single`、等待和取结果接口。返回字符串仍由 `qf_free` 释放，退出时调用 `qf_shutdown`。

若需要将运行文件保留在独立子目录，宿主必须在加载库前正确设置 Windows DLL 搜索路径（静态链接时应由启动器设置，或改用延迟加载/动态加载）；仅给出 mrmpformer.dll 的路径不等于能找到其依赖。

## 文件用途

| 文件 | 用途 |
|---|---|
| `mrmpformer.h` / `mrmpformer.lib` | 开发时包含和链接 |
| `mrmpformer.dll` | 原有 C 接口 |
| `mrmpformerv2.onnx` | 神经网络模型 |
| `python311.dll`、`python3.dll`、`python311.zip`、`python311._pth`、VC 运行库 DLL | 私有 Python 运行时，保持相对位置 |
| `python/` | 编译算法、辅助模块及 NumPy/SciPy/ONNX Runtime 等运行依赖，整体复制 |
| `licenses/`、依赖目录中的许可证 | 第三方许可说明，随运行文件保留 |

## 验证

交付目录只保留集成文件、模型、运行依赖、许可证和本说明。自检 EXE、示例输入、HTML/JSON 测试报告不放入交付包。调用示例在仓库 `cpp/examples/`；自检由 `build.sh` 在临时目录完成。

开发方在 `tests/results/` 查看 `report.html`、`report.json`、`device_verification.json`、`verification.json` 和 `package_manifest.json`（文件哈希及依赖版本）。

打包脚本会将包复制到临时目录，移除开发环境路径，并故意设置无效的 PYTHONHOME/PYTHONPATH，通过实际 C EXE 加载 DLL、执行 ONNX 推理，并与源码 Python 结果逐值对照。该检查验证本机搬移后的独立运行；仍建议在目标软件及干净 Windows 测试机验收。

同一份包包含 CUDAExecutionProvider 和 CPUExecutionProvider，无需选择或更换包。`onnxruntime_providers_cuda.dll` 位于 `python/Lib/site-packages/onnxruntime/capi/`。这是 Python ONNX Runtime 的加载布局，不要把旧集成目录的 DLL 混入新包。GPU 模式使用目标机已有的 CUDA 12 / cuDNN 9 运行库，其 DLL 所在目录需可被进程找到（例如在启动软件前加入 PATH）。包内不附带 NVIDIA pip 包。

打包时额外生成 `device_verification.json`，记录强制 CPU、自动选择和强制 GPU 的实际测试结果。在没有 NVIDIA GPU 的机器上只能验证 CPU 与回退行为，不能据此声称 GPU 推理已实测。构建机使用 `onnxruntime-gpu==1.23.2` 获取 ONNX Runtime；该发行包也包含 CPU provider，客户机不需要 pip。不要同时安装 CPU-only `onnxruntime`。

### Python 与 C 一致性回归

每次一键打包会读取 256 份真实 MRM 输入，对比 Python 源码、`model/build/` 中的 Cython 扩展
和 C DLL 接口。批量和单条调用分别做三方两两比较，共 1536 次对照。
数据来自 `test3_cpp_validation_1d4ba0f` 的 4 组分层验收输入，保留完整原始 x/y，使用验收包配置。
逐项检查峰数量、顺序、`a/b/c`、状态和告警；数值容差为绝对 `1e-12`、相对 `1e-10`。
单条与批量各自使用相同调用方式的 Python 参考结果，避免不同 ONNX 批次形状带来的浮点舍入差异。
至少 50 个用例必须实际返回峰。任何失败都会阻止新包发布。

单独重跑（仓库根目录、已配置构建环境）：

```powershell
uv run tests/run_parity.py
```

对比测试集中在根目录 `tests/`。先打开 `tests/inputs/README.md` 查看 256 份真实输入，
每个 JSON 都包含实际使用的配置和完整数组，可直接修改。
运行命令后打开 `tests/results/README.md` 查看逐例结果，点击用例可并排查看 Python、Cython build、C DLL 三方的返回值。
测试驱动自动独立编译到 `tests/build/`，运行时不会重新生成输入。
从原验收包导入到空目录使用 `uv run tests/import_inputs.py <验收包目录> --output <空目录>`。
这些用例验证 CPU 实现一致性，不代替真实样本的检出准确率评估。

## 重新生成（模型开发方）

打包入口统一为仓库根目录的 `build.sh`，在 Windows 的 Git Bash 中运行。
它使用根目录 `.venv/Scripts/python.exe`，无需激活环境；脚本可从任意工作目录调用。
`build.sh` 只负责参数处理、环境准备和调用；依赖收集、打包与校验逻辑位于 `cpp/tools/build_windows_package.py`，与 `cpp/requirements-runtime.txt`、`cpp/requirements-build.txt` 一起维护。模型的 Cython 编译脚本仍位于 `model/tools/build_massnova_bridge.py`。根目录不再需要 `tools/`。

`cpp/requirements-runtime.txt` 是运行依赖的唯一清单，打包器读取它并根据已安装包的元数据递归收集传递依赖、检查版本约束和环境条件。`cpp/requirements-build.txt` 引用运行清单并加入 Cython 等构建工具，供 `--install-deps` 安装；构建工具不会仅因为已安装就进入交付包。

```bash
# 首次使用：需要 uv；创建缺失的 .venv 并安装依赖
bash build.sh --install-deps
# 日常使用：完整编译、测试、打包
bash build.sh
# 复用编译产物，仍执行完整集成验证
bash build.sh --skip-build
```

需预先安装 Git Bash、CMake 和 Visual Studio C++ 构建工具。成功返回 0，失败返回非零。

中间文件分别留在 `cpp/build/` 和 `model/build/`。只将最终 `build/windows/` 交给集成方。
