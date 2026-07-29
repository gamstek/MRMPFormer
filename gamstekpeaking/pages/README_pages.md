# pages/ — 功能页面模块

GAMSTEKPEAKing 的各功能板块。每个板块一个 `.py` 文件，由 `app.py` 的侧边栏通过 `add_page()` 注册并路由。页面切换由 `QStackedWidget` 管理。

## 文件清单

| 文件 | 导出类 | 作用 | 状态 |
|------|--------|------|------|
| `preprocessing.py` | `PreprocessingPage`, `ConversionCard`, `IonZenithCard` | 前处理板块：格式转换（msdata→mzML）+ 离子天顶（MS1→CSV） | ✅ 已上线 |
| `peak_finding.py` | `PeakFindingPage` | 寻峰板块：DETR 模型预测、EIC 可视化、峰面积定量 | 🔒 占位 |
| `settings.py` | `SettingsPage` | 设置板块：主题/语言/路径/GPU 偏好 | 🔒 占位 |

## 页面开发指南

每个页面是一个 `QWidget` 子类，在 `main.py` 中注册：

```python
window.add_page("🏔", "前处理", PreprocessingPage(), enabled=True)
```

### add_page 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `icon` | `str` | emoji 或 Unicode 图标字符 |
| `name` | `str` | 侧边栏显示的文字标签 |
| `widget` | `QWidget` | 页面 Widget 实例 |
| `enabled` | `bool` | `True`=可导航, `False`=灰色禁用态 |
