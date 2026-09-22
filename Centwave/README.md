# CentWave

基于连续小波变换（CWT）的色谱峰检测工具，灵感来源于 XCMS 的 centWave 算法。

## 功能

- 多尺度 Ricker（Mexican Hat）小波变换
- 脊线持久性验证：候选峰必须在多个尺度上持续出现
- 自适应边界划分：重叠峰自动分离，边界精炼到局部最小值
- 多维度质量控制：方向趋势、残差粗糙度、基线漂移、对称性、尖锐噪声过滤
- 支持 `.mzML` 原始质谱数据和 `.csv` 表格数据

## 模块结构

```
centwave-2/
├── wavelet.py          # 小波变换（ricker / cwt）
├── detector.py         # 核心峰检测算法
├── io_mzml.py          # .mzML 文件读写与批处理
├── io_csv.py           # CSV 文件读写
├── plot.py             # 色谱图可视化
├── model.py            # 统一入口 & 命令行接口
├── test_detector.py    # 检测器单元测试
└── test_io_mzml.py     # mzML I/O 测试
```

## 快速开始

### 安装依赖

```bash
pip install numpy scipy matplotlib pandas lxml
```

### 命令行使用

```bash
# 批量处理 .mzML 目录
python model.py /path/to/mzml_dir

# 指定 SNR 阈值
python model.py /path/to/mzml_dir 5
```

### Python API

```python
import numpy as np
from model import detect_peaks_centwave, plot_roi_with_peaks

rt = np.linspace(0, 200, 400)
intensity = (
    3000 * np.exp(-0.5 * ((rt - 40) / 3) ** 2)
    + 8000 * np.exp(-0.5 * ((rt - 100) / 8) ** 2)
    + np.random.normal(0, 100, len(rt))
)

peaks, baseline, noise = detect_peaks_centwave(rt, intensity, snr_thresh=3)

for p in peaks:
    print(f"RT={p['apex_rt']:.1f}, SNR={p['snr']:.1f}")

fig = plot_roi_with_peaks(rt, intensity, peaks)
fig.savefig("output.png", dpi=150)
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `snr_thresh` | 3 | 信噪比阈值 |
| `scales` | `np.arange(1, 15)` | CWT 尺度序列 |
| `min_ridge_length` | scales 的 1/4 | 脊线最小持续尺度数 |
| `max_ridge_drift` | 2 | 脊线相邻尺度最大漂移 |
| `min_trend_score` | 0.7 | 峰形状最小方向趋势分 |

### 运行测试

```bash
python -m unittest test_detector -v
python -m unittest test_io_mzml -v
```

## License

MIT
