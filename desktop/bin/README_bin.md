# bin/ — msdata2mzml 运行时

基于 OpenMS 工具链编译的 `.msdata → .mzML` 格式转换工具，由 `workers/converter.py` 通过 subprocess 调用。

## 文件清单

| 文件 | 作用 |
|------|------|
| `msdata2mzml.exe` | 格式转换主程序（C++/OpenMS） |
| `OpenMS.dll` | OpenMS 核心库 |
| `Qt5Core.dll`, `Qt5Network.dll` | Qt 运行时依赖 |
| `*.dll` | 其他运行时依赖（zlib, xerces, lapack 等） |
| `share/OpenMS/` | OpenMS 数据文件（CHEMISTRY, CV 术语等） |

## 使用方式

```
msdata2mzml.exe <input.msdata>
```
输出自动生成在输入文件的同级目录下的 `<basename>/` 子目录中。

## 注意事项

- 路径不能包含中文字符（OpenMS C++ 层限制）
- 需要设置 `OPENMS_DATA_PATH` 环境变量指向 `share/OpenMS/`
- 来源：`ms2mzml/bin/`（历史工具链，复制至此以实现自包含）
