# workers/ — 后台工作线程模块

封装耗时操作为 `QThread` 子类，通过 Qt Signal 与 UI 通信，确保界面不阻塞。

## 文件清单

| 文件 | 导出类 | 作用 | 核心 Signal |
|------|--------|------|------------|
| `converter.py` | `FormatConverter` | msdata/wiff→mzML 格式转换 Qt 线程包装（**纯转换逻辑在 `converters/msdata.py` / `converters/wiff.py`**，本文件仅线程适配 + 信号转发） | `progress(current, total)`, `file_done(index, ok, info)`, `error(msg)` |
| `ion_zenith.py` | `IonZenithWorker` | 离子天顶算法 Qt 线程包装（**纯算法在 `model/preprocessing/ion_zenith.py`**：遍历 mzML MS1 → 按 m/z 聚合 → 输出 CSV） | `progress(scanned, total)`, `stats(ms1, peaks)`, `finished(ions, elapsed, path)`, `error(msg)` |

## FormatConverter 接口

> 💡 本文件只负责 Qt 线程适配（QThread + Signal 转发进度/结果）。
> **纯转换逻辑**位于 [`converters/msdata.py`](../../converters/msdata.py) 与 [`converters/wiff.py`](../../converters/wiff.py)（bin 定位 / OPENMS_DATA_PATH / subprocess 均在 converters/ 中），
> 也可用 `python converters/msdata.py --input ...` / `python converters/wiff.py --input ...` 直接调用，无需 Qt 依赖。

```python
class FormatConverter(QThread):
    progress  = Signal(int, int)           # (当前文件索引, 文件总数)
    file_done = Signal(int, bool, str)     # (索引, 是否成功, 信息)
    error     = Signal(str)                # 全局错误消息

    def __init__(self, files: list[str], fmt: str = "msdata", output_dir: str | None = None):
        """
        Args:
            files: 待转换文件的绝对路径列表
            fmt: 源格式 ("msdata" | "wiff")
            output_dir: 自定义输出目录，None=使用默认（同输入目录）
        """
```

## IonZenithWorker 接口

> 💡 本文件只负责 Qt 线程适配（QThread + Signal 转发进度/统计/结果）。
> **纯算法实现**位于 [`model/preprocessing/ion_zenith.py`](../../model/preprocessing/ion_zenith.py)，可用 `python -m preprocessing.ion_zenith --input_mzml ... --output_csv ...` 直接调用，无需 Qt 依赖。

```python
class IonZenithWorker(QThread):
    progress = Signal(int, int)            # (已扫描谱图数, 总数/0)
    stats    = Signal(int, int)            # (MS1 谱图数, 累计扫描峰数)
    finished = Signal(int, float, str)     # (离子数, 耗时秒, 输出路径)
    error    = Signal(str)                 # 错误消息

    def __init__(self, params: dict):
        """
        Args:
            params: {
                input_mzml (str):       输入 mzML 路径
                output_csv (str):       输出 CSV 路径
                mz_min (float):         m/z 下限
                mz_max (float):         m/z 上限
                ppm_tol (float):        ppm 容差
                da_tol (float):         Da 容差
                intensity_min (float|None): 强度下限
                intensity_max (float|None): 强度上限
                max_spectra (int):      最大谱图数 (0=全部)
                build_index (bool):     是否重建 mzML 索引
            }
        """
```

## 线程开发指南

1. 继承 `QThread`，重写 `run()` 方法
2. 定义 Signal 类属性（非实例属性）用于对外通信
3. 在 `run()` 中捕获所有异常，通过 `error.emit()` 传递给 UI
4. UI 层通过 `worker.start()` 启动，结束时线程自动终止
5. 避免在 `run()` 中直接操作 UI widget（线程安全）
