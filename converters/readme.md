# converters — 格式转换工具集

将厂商原始质谱数据批量转换为标准 `.mzML` 格式。

## 功能一览

| 脚本 | 输入格式 | 输出格式 | 工具链 |
|------|----------|----------|--------|
| `msdata.py` | `.msdata` | `.mzML` | `msdata2mzml.exe`（OpenMS） |
| `wiff.py` | `.wiff` / `.wiff2` | `.mzML` | `msconvert.exe`（ProteoWizard） |
| `rename_cn.py` | — | — | 中文文件名 → 英文（msdata 预处理） |

## 目录结构

```
converters/
├── msdata.py              # msdata → mzML 批量转换
├── wiff.py                # wiff → mzML 批量转换
├── rename_cn.py           # 中文文件名重命名工具
├── readme.md              # 本文档
├── readme.txt             # msdata 转换详细说明
├── conversion_report.txt  # 最近一次转换报告
├── exp_log.md             # 实验日志
├── msdata_bin/            # msdata2mzml.exe + OpenMS 运行时
├── wiff_bin/              # msconvert.exe + ProteoWizard 运行时
└── desktop_bin/           # desktop/ 自包含副本
```

## 快速开始

```bash
cd converters

# === msdata → mzML ===
python rename_cn.py            # 预览中文文件名映射
python rename_cn.py --no-dry-run  # 确认执行重命名
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
- 详见各 `readme.txt` 和 `exp_log.md`
