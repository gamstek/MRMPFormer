# workers/ — 后台工作线程模块

封装耗时操作为 `QThread` 子类，通过 Qt Signal 与 UI 通信，确保界面不阻塞。

## 文件清单

| 文件 | 导出类 | 作用 | 核心 Signal |
|------|--------|------|------------|
| `converter.py` | `MsdataConverter` | msdata→mzML 格式转换（调用内嵌的 `bin/msdata2mzml.exe`） | `progress(current, total)`, `file_done(index, ok, info)`, `error(msg)` |
| `ion_zenith.py` | `IonZenithWorker` | 离子天顶算法（遍历 mzML MS1 → 按 m/z 聚合 → 输出 CSV） | `progress(scanned, total)`, `stats(ms1, peaks)`, `finished(ions, elapsed, path)`, `error(msg)` |

## MsdataConverter 接口

```python
class MsdataConverter(QThread):
    progress  = Signal(int, int)           # (当前文件索引, 文件总数)
    file_done = Signal(int, bool, str)     # (索引, 是否成功, 信息)
    error     = Signal(str)                # 全局错误消息

    def __init__(self, files: list[str], output_dir: str | None = None):
        """
        Args:
            files: .msdata 文件的绝对路径列表
            output_dir: 自定义输出目录，None=使用默认（同输入目录）
        """
```

## IonZenithWorker 接口

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
