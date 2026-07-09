msdata -> mzML 批量转换工具
====================================

使用 msdata2mzml.exe（OpenMS 工具链）将 data/ 目录下的 .msdata 文件批量转换为 .mzML 格式。


一、目录结构
------------------------------------

AAms_mzml/
├── ms2mzml.py              # 批量转换主脚本
├── rename_cn.py            # 中文文件名 -> 英文重命名工具
├── conversion_report.txt   # 最近一次转换的完整报告
│
├── bin/                    # msdata2mzml.exe 及其依赖
│   ├── msdata2mzml.exe
│   ├── *.dll               # OpenMS 运行时 DLL
│   └── share/OpenMS/       # OpenMS 数据文件（CV、CHEMISTRY 等）
│
├── data/                   # 输入 {name}.msdata，输出到 {name}/ 子目录
│   ├── 20251120-01.msdata
│   ├── 20251120-01/        # 转换生成的 mzML
│   │   ├── 20251120-01_1.mzML
│   │   └── ...
│   └── ...
│
└── tmp/                    # 运行时临时目录（自动创建）


二、前置条件
------------------------------------

1. bin/ 目录需完整：
   将 msdata2mzml_*/Release/ 下的 msdata2mzml.exe、所有 .dll 和 share/ 复制到 bin/。

2. 项目路径不得含中文：
   OpenMS C++ 层不支持中文路径。

3. Python 环境：标准库即可，无需安装第三方包。


三、快速开始
------------------------------------

步骤一：将中文文件名重命名为英文（仅首次需要）

    python rename_cn.py                    # 预览
    python rename_cn.py --no-dry-run       # 确认后执行

步骤二：批量转换

    python ms2mzml.py

步骤三：查看结果
    - 终端末尾会打印成功率和失败详情
    - 完整报告保存在 conversion_report.txt

转换后输出会自动放在 data/{basename}/ 下，
例如 data/20251120-01.msdata -> data/20251120-01/20251120-01_1.mzML。


四、命令行参数
------------------------------------

ms2mzml.py:

    --input <路径>    仅转换指定文件，不传则处理 data/*.msdata
    --dry-run         仅列出待处理文件，不执行转换

    示例：
    python ms2mzml.py --dry-run                        # 预览有多少文件
    python ms2mzml.py --input data/20251120-01.msdata   # 只转换一个
    python ms2mzml.py                                  # 转换全部

rename_cn.py:

    --dry-run         预览模式（默认），仅显示映射，不修改文件
    --no-dry-run      确认执行重命名


五、输出说明
------------------------------------

每个 .msdata 转换后在同目录下生成 <basename>/ 文件夹，
内含若干 *.mzML 文件（数量按内部 Experiment 个数）。

转换完成后脚本会打印汇总：

    ========================================
    成功: 191/191 (100.0%)  |  失败: 0/191 (0.0%)
    ========================================

失败样本会逐条显示原因（stderr / stdout / 进程退码）。


六、注意事项
------------------------------------

1. 中文 Windows 用户名：
   脚本已将 TMP/TEMP 环境变量指向项目 tmp/ 目录，
   绕过 C:\Users\许恒\ 导致的 OpenMS 路径错误。

2. exe 参数：
   msdata2mzml.exe 仅接受位置参数，不支持 -in/-out 标志。

3. 输出位置：
   exe 自动输出到输入文件同级目录，无法通过参数指定。

4. 退码 858：
   exe 即使成功也会返回 858，脚本不以退码判定成败，
   改为检查是否确实生成了 .mzML 文件。
