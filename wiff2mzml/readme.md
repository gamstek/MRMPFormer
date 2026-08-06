# wiff2mzml

使用 ProteoWizard msconvert 将 AB Sciex 的 `.wiff` / `.wiff2` 文件批量转换为 `.mzML` 格式。

## 目录结构

```
wiff2mzml/
├── wiff2mzml.py          # 批量转换脚本
├── data/                 # 放 .wiff 文件（需自行创建）
│   ├── sample.wiff
│   ├── sample.wiff.scan  # WIFF 通常需要配套的 .wiff.scan 文件
│   └── sample/           # 转换输出自动生成到此
└── bin/                  # msconvert.exe 及运行时依赖
```

## 快速开始

```powershell
cd wiff2mzml

# 1. 预览待转换文件
python wiff2mzml.py --dry-run

# 2. 批量转换
python wiff2mzml.py
```

## 命令行参数

| 参数 | 说明 |
|------|------|
| `--input <路径>` | 仅转换指定文件 |
| `--dry-run` | 仅列出文件，不转换 |
| `--no-peak-picking` | 保留原始 profile 数据（默认会做峰检测，输出更小） |

示例：

```powershell
# 转换单个文件
python wiff2mzml.py --input "data/20230222_YYF_Blood_Meta_POS.wiff"

# 保留 profile 模式
python wiff2mzml.py --no-peak-picking
```

## 输出

每个 `.wiff` 转换后在 `data/<文件名>/` 下生成若干 `.mzML` 文件（数量取决于 WIFF 中包含的 sample 数）。脚本末尾会打印汇总统计。

转换过程中如遇损坏的 run 会自动跳过并继续处理其余数据。

## 注意事项

- 路径不能包含中文，否则会报错
- WIFF 文件需要同名 `.wiff.scan` 配套文件在相同目录下
- 大文件转换耗时长、吃内存，建议转换时关闭其他程序
- 默认开启 `peakPicking`（centroid 化），输出文件更小

## 依赖

- Python 3（标准库即可，无需额外安装）
- `bin/msconvert.exe`（ProteoWizard，已附带）
