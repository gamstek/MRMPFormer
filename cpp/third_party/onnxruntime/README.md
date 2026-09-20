# ONNX Runtime SDK

当前版本：1.23.2（Windows x64 GPU，含 CUDA provider）。本目录用于
MRMPFormer C++ 的离线构建，不包含 Linux 运行库。

来源：Microsoft ONNX Runtime 官方发布包
`onnxruntime-win-x64-gpu-1.23.2.zip`，按 `LICENSE` 中的 MIT License 分发。

## 目录结构

```
third_party/onnxruntime/
├── include/          # C/C++ 头文件 (ORT_API_VERSION=23)
│   ├── onnxruntime_c_api.h
│   ├── onnxruntime_cxx_api.h
│   └── ...
├── lib/
│   ├── onnxruntime.dll            # Windows 运行时
│   ├── onnxruntime.lib            # Windows 导入库
│   ├── onnxruntime_providers_cuda.dll   # CUDA provider
│   └── onnxruntime_providers_shared.dll # 共享模块
└── README.md
```

## 如需重新下载

**Windows GPU**:
```
https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-gpu-1.23.2.zip
```

将 include/ 和 lib/ 内容复制到对应目录。
