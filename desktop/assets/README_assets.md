# assets/ — 静态资源

存放 GAMSTEKPEAKing 的图标、图片等静态资源文件。

## 文件清单

| 文件 | 作用 |
|------|------|
| `logo.png` | 应用 Logo（待设计，可用 256×256 的 PNG 图标；缺失时不影响启动） |
| `spin_up_arrow.png` | SpinBox 上箭头（`theme.py::_ensure_assets()` 缺失时自动生成） |
| `combo_down_arrow.png` | ComboBox/SpinBox 下箭头（同上，自动生成） |
| `check_icon.png` | 复选框勾号（同上，自动生成） |

> 后三个图标由 `theme.py` 在启动时检查并生成，无需手工维护；删除后会自动重建。
