# converters — 格式转换工具集

将厂商原始质谱数据批量转换为标准 `.mzML` 格式。

## 功能一览

| 脚本 | 输入格式 | 输出格式 | 工具链 |
|------|----------|----------|--------|
| `msdata.py` | `.msdata` | `.mzML` | `msdata2mzml.exe`（OpenMS） |
| `wiff.py` | `.wiff` / `.wiff2` | `.mzML` | `msconvert.exe`（ProteoWizard） |

## 目录结构

```
converters/
├── msdata.py              # msdata → mzML 批量转换
├── wiff.py                # wiff → mzML 批量转换
├── readme.md              # 本文档（含 msdata 转换详细说明）
├── msdata_bin/            # msdata2mzml.exe + OpenMS 运行时
└── wiff_bin/              # msconvert.exe + ProteoWizard 运行时
```

## 快速开始

```bash
cd converters

# === msdata → mzML ===
python msdata.py --dry-run     # 预览待转换文件
python msdata.py               # 批量转换全部

# === wiff → mzML ===
python wiff.py --dry-run       # 预览待转换文件
python wiff.py                 # 批量转换全部
python wiff.py --no-peak-picking  # 保留 profile 模式
```

## 注意事项

- 项目路径不得含中文（OpenMS C++ 层限制）
- WIFF 文件需要同名 `.wiff.scan` 配套文件
- 大文件转换耗时长、吃内存
