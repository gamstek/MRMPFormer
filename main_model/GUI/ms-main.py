import os
import logging
from PySide6 import QtCore
from PySide6.QtCore import Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QHeaderView
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from natsort import natsorted
from GUI.ms import Ui_QuanFormer
from main import *
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.postprocess_thread = None
        self.area = None
        self.results = None
        self.xic_list = None
        self.xic_info = None
        self.threshold = None
        self.paths = None
        self.current_time = None
        self.args = get_args_parser().parse_args([])
        self.ui = Ui_QuanFormer()
        self.ui.setupUi(self)
        self._apply_theme()
        self.bind()

        self.mzml_thread = None
        self.feature_thread = None
        self.export_thread = None
        self.model_thread = None
        self.eic_build_thread = None
        self.eic_output_thread = None
        self.predict_thread = None
        self.quantify_thread = None

    def _apply_theme(self):
        """应用简约中文主题"""
        # ── 窗口 ──
        self.setWindowTitle("QuanFormer · 峰检测与定量")
        self.resize(960, 680)
        self.setMinimumSize(900, 620)

        # ── 标题 ──
        self.ui.label_8.setText("QuanFormer")
        self.ui.label_8.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #1E293B; padding: 4px 0;"
        )

        # ── 标签文字中文化 ──
        self.ui.label.setText("导入 mzML 文件")
        self.ui.label_2.setText("导入特征表")
        self.ui.label_10.setText("加载模型")
        self.ui.label_9.setText("设置输出目录")
        self.ui.label_3.setText("构建 ROI")
        self.ui.label_4.setText("预测峰")
        self.ui.label_5.setText("峰定量")
        self.ui.label_7.setText("导出结果")
        self.ui.label_6.setText("结果后处理")

        # ── 按钮文字 ──
        self.ui.mzmlImport.setText("浏览")
        self.ui.featureImport.setText("浏览")
        self.ui.modelLoad.setText("浏览")
        self.ui.pushButton.setText("浏览")
        self.ui.eicBuild.setText("开始构建")
        self.ui.eicPredict.setText("开始预测")
        self.ui.eicQuantify.setText("开始定量")
        self.ui.resultsExport.setText("导出 CSV")
        self.ui.eicPostprogress.setText("执行后处理")

        # ── 复选框 ──
        self.ui.checkBox.setText("绘制预测框")

        # ── 全局样式表 ──
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F1F5F9;
            }
            QWidget#centralwidget {
                background-color: #F1F5F9;
            }

            /* ── 标签 ── */
            QLabel {
                color: #475569;
                font-size: 13px;
            }

            /* ── 按钮 ── */
            QPushButton {
                background-color: #3B82F6;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 7px 16px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }

            /* ── 列表 ── */
            QListWidget {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                padding: 4px;
                font-size: 12px;
                color: #334155;
                outline: none;
            }
            QListWidget::item {
                padding: 6px 10px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #DBEAFE;
                color: #1E40AF;
            }
            QListWidget::item:hover {
                background-color: #F1F5F9;
            }

            /* ── 表格 ── */
            QTableWidget {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                gridline-color: #F1F5F9;
                font-size: 11px;
                color: #334155;
            }
            QTableWidget::item {
                padding: 4px 8px;
            }
            QTableWidget::item:selected {
                background-color: #DBEAFE;
                color: #1E40AF;
            }
            QHeaderView::section {
                background-color: #F8FAFC;
                color: #64748B;
                border: none;
                border-bottom: 1px solid #E2E8F0;
                padding: 6px 8px;
                font-size: 11px;
                font-weight: 600;
            }

            /* ── 日志区 ── */
            QTextBrowser {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
                color: #475569;
            }

            /* ── 复选框 ── */
            QCheckBox {
                color: #475569;
                font-size: 13px;
                spacing: 6px;
            }

            /* ── EIC 图片区 ── */
            QLabel#showEIC {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
            }

            /* ── 状态栏 ── */
            QStatusBar {
                background-color: #F8FAFC;
                color: #94A3B8;
                font-size: 11px;
                border-top: 1px solid #E2E8F0;
            }

            /* ── 滚动条 ── */
            QScrollBar:vertical {
                background: #F1F5F9;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #CBD5E1;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #94A3B8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        # ── 图片区域占位提示 ──
        self.ui.showEIC.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.showEIC.setText("EIC 预览区")

    def bind(self):
        self.current_time = datetime.now().strftime("%H:%M:%S")
        self.ui.mzmlImport.clicked.connect(self.mzml_import)
        self.ui.featureImport.clicked.connect(self.feature_import)
        self.ui.modelLoad.clicked.connect(self.model_load)
        self.ui.pushButton.clicked.connect(self.set_output_dir)
        self.ui.eicBuild.clicked.connect(self.on_eic_build_clicked)
        self.ui.listWidget.currentItemChanged.connect(self.on_list_item_changed)
        self.ui.listWidget_2.currentItemChanged.connect(self.show_eic)
        self.ui.eicPredict.clicked.connect(self.on_eic_predict_clicked)
        self.ui.eicQuantify.clicked.connect(self.on_eic_quantify_clicked)
        self.ui.resultsExport.clicked.connect(self.on_results_export_clicked)
        self.ui.eicPostprogress.clicked.connect(self.on_eic_postprogress_clicked)

    def mzml_import(self):
        mzml_dir = QFileDialog.getExistingDirectory(self, "选择 mzML 文件夹", "")
        if mzml_dir:
            self.args.source = mzml_dir
            self.ui.textBrowser.append(f"{self.current_time}  ✓ 已选择 mzML 目录：{self.args.source}")
            self.mzml_thread = MzmlImportThread(mzml_dir)
            self.mzml_thread.import_finished.connect(
                lambda msg: self.ui.textBrowser.append(f"{self.current_time}  {msg}"))
            self.mzml_thread.start()
        else:
            return None

    def feature_import(self):
        table_path, _ = QFileDialog.getOpenFileName(self, "选择特征文件", "", "CSV 文件 (*.csv)")
        if table_path:
            self.args.feature = table_path
            self.ui.textBrowser.append(f"{self.current_time}  ✓ 已导入特征表：{self.args.feature}")
            self.feature_thread = FeatureImportThread(table_path, self.ui.tableWidget)
            self.feature_thread.import_finished.connect(
                lambda msg: self.ui.textBrowser.append(f"{self.current_time}  {msg}"))
            self.feature_thread.start()
        else:
            return None

    def model_load(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.pth)")
        if model_path:
            self.args.model = model_path
            self.ui.textBrowser.append(f"{self.current_time}  ✓ 已加载模型：{self.args.model}")
            self.model_thread = ModelImportThread(model_path)
            self.model_thread.import_finished.connect(
                lambda msg: self.ui.textBrowser.append(f"{self.current_time}  {msg}"))
            self.model_thread.start()
        else:
            return None

    def set_output_dir(self):
        eic_output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录", "")
        if eic_output_dir:
            self.args.images_path = eic_output_dir
            self.ui.textBrowser.append(f"{self.current_time}  ✓ 已设置输出目录：{self.args.images_path}")
            self.eic_output_thread = EicOutputThread(eic_output_dir)
            self.eic_output_thread.import_finished.connect(
                lambda msg: self.ui.textBrowser.append(f"{self.current_time}  {msg}"))
            self.eic_output_thread.start()
        else:
            return None

    def on_eic_build_clicked(self):
        if self.args.source and self.args.feature:
            self.eic_build_thread = EicBuildThread(self.args.source, self.args.feature, self.args)
            self.eic_build_thread.build_finished.connect(self.on_eic_build_finished)
            self.ui.textBrowser.append(f"{self.current_time}  ⏳ 正在构建 ROI，请稍候…")
            self.eic_build_thread.start()
        else:
            self.ui.textBrowser.append(f"{self.current_time}  ⚠ 请先导入 mzML 文件和特征表。")

    def on_list_item_changed(self):
        if self.args.images_path:
            new_path = os.path.join(self.args.images_path, self.ui.listWidget.currentItem().text())
            self.ui.listWidget_2.blockSignals(True)
            self.ui.listWidget_2.clear()
            self.ui.listWidget_2.blockSignals(False)
            self.ui.listWidget_2.addItems(natsorted(d for d in os.listdir(new_path)))

    def show_eic(self):
        if self.ui.listWidget_2.currentItem():
            new_path = os.path.join(self.args.images_path,
                                    self.ui.listWidget.currentItem().text(),
                                    self.ui.listWidget_2.currentItem().text())
            fixed_size = QtCore.QSize(328, 251)
            self.ui.showEIC.setPixmap(QPixmap(new_path).scaled(fixed_size))

    def on_eic_predict_clicked(self):
        self.current_time = datetime.now().strftime("%H:%M:%S")
        self.ui.textBrowser.append(f"{self.current_time}  ⏳ 正在预测峰，请稍候…")
        if self.xic_list is not None:  # 确保xic_list已初始化且不为空
            self.predict_thread = EicPredictThread(self.args.model, self.args.images_path, self.args.threshold,
                                                   self.ui.checkBox.isChecked())
            self.predict_thread.predict_finished.connect(self.on_eic_predict_finished)
            self.predict_thread.start()
        else:
            self.ui.textBrowser.append(f"{self.current_time}  ⚠ 没有可用的 ROI 数据进行预测。")

    def on_eic_quantify_clicked(self):
        self.current_time = datetime.now().strftime("%H:%M:%S")
        if self.results is not None:
            self.quantify_thread = EicQuantifyThread(self.xic_list, self.results, self.xic_info)
            self.quantify_thread.quantify_finished.connect(self.on_eic_quantify_finished)
            self.quantify_thread.start()
        else:
            self.ui.textBrowser.append(f"{self.current_time}  ⚠ 没有可用的预测结果进行定量。")

    def on_results_export_clicked(self):
        self.current_time = datetime.now().strftime("%H:%M:%S")
        if self.area is not None:
            output_path, _ = QFileDialog.getSaveFileName(self, "导出结果", "", "CSV 文件 (*.csv)")
            if output_path:
                self.args.output = output_path
                self.export_thread = ResultsExportThread(self.area, self.args.output, self.ui.tableWidget)
                self.export_thread.export_finished.connect(
                    lambda msg: self.ui.textBrowser.append(f"{self.current_time}  {msg}"))
                self.export_thread.start()
            else:
                return None
        else:
            self.ui.textBrowser.append(f"{self.current_time}  ⚠ 没有可用的定量结果进行导出。")

    def on_eic_postprogress_clicked(self):
        self.current_time = datetime.now().strftime("%H:%M:%S")
        self.ui.textBrowser.append(f"{self.current_time}  ⏳ 正在执行后处理…")
        if self.args.output:
            self.postprocess_thread = EicPostProcessThread(self.args.output, self.xic_info, self.ui.tableWidget)
            self.postprocess_thread.postprocess_finished.connect(
                lambda msg: self.ui.textBrowser.append(f"{self.current_time}  {msg}"))
            self.postprocess_thread.start()

    def on_eic_build_finished(self, msg, xic_info, xic_list, image_filenames):
        self.ui.textBrowser.append(f"{self.current_time}  ✓ ROI 构建完成")
        self.xic_info = xic_info
        self.xic_list = xic_list
        self.ui.listWidget.addItems(image_filenames)

    def on_eic_predict_finished(self, msg, results):
        self.ui.textBrowser.append(f"{self.current_time}  ✓ 峰预测完成")
        self.results = results

    def on_eic_quantify_finished(self, msg, area):
        self.ui.textBrowser.append(f"{self.current_time}  ✓ 定量完成")
        self.area = area



class MzmlImportThread(QtCore.QThread):
    import_finished = Signal(str)

    def __init__(self, mzml_dir):
        super().__init__()
        self.mzml_dir = mzml_dir

    def run(self):
        self.import_finished.emit("✓ mzML 导入完成")


class FeatureImportThread(QtCore.QThread):
    import_finished = Signal(str)

    def __init__(self, table_path, table_widget):
        super().__init__()
        self.table_path = table_path
        self.ui = table_widget

    def run(self):
        try:
            import pandas as pd
            df = pd.read_csv(self.table_path)
            self.ui.setRowCount(len(df))
            self.ui.setColumnCount(len(df.columns))
            self.ui.setHorizontalHeaderLabels(df.columns)
            self.ui.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            for i in range(len(df)):
                for j in range(len(df.columns)):
                    self.ui.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))
            self.import_finished.emit("✓ 特征表导入完成")
        except Exception as e:
            logging.error(f"导入特征表出错: {e}")
            self.import_finished.emit(f"✗ 错误: {str(e)}")


class ModelImportThread(QtCore.QThread):
    import_finished = Signal(str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        self.import_finished.emit("✓ 模型加载完成")


class EicOutputThread(QtCore.QThread):
    import_finished = Signal(str)

    def __init__(self, eic_output_dir):
        super().__init__()
        self.eic_output_dir = eic_output_dir

    def run(self):
        self.import_finished.emit("✓ 输出目录设置完成")


class EicBuildThread(QtCore.QThread):
    build_finished = Signal(str, object, object, list)

    def __init__(self, mzml_path, feature, args):
        super().__init__()

        self.source = mzml_path
        self.feature = feature
        self.args = args
        self.plot = True

    def run(self):
        paths = get_files(self.source, "mzML")
        xic_info = read_targeted_features(self.feature)
        xic_list = build_roi(paths, xic_info, self.plot, self.args)
        self.build_finished.emit(
            f"ROI 构建完成", xic_info, xic_list, [d for d in os.listdir(self.args.images_path)])


class EicPredictThread(QtCore.QThread):
    predict_finished = Signal(str, list)  # 发射一个字符串和结果字典

    def __init__(self, model_path, images_path, threshold, with_plot):
        super().__init__()
        self.model_path = model_path
        self.images_path = images_path
        self.threshold = threshold
        self.with_plot = with_plot

    def run(self):
        if self.with_plot:
            results = build_predictor(self.model_path, self.images_path, self.threshold, plot=True)
            self.predict_finished.emit("✓ 预测完成（含绘图）", results)
        else:
            results = build_predictor(self.model_path, self.images_path, self.threshold, plot=False)
            self.predict_finished.emit("✓ 预测完成", results)
            

class EicQuantifyThread(QtCore.QThread):
    quantify_finished = Signal(str, list)

    def __init__(self, xic_list, results, xic_info):
        super().__init__()
        self.xic_list = xic_list
        self.results = results
        self.xic_info = xic_info

    def run(self):
        area = quantify(self.xic_list, self.results, self.xic_info)
        self.quantify_finished.emit("✓ 定量完成", area)


class ResultsExportThread(QtCore.QThread):
    export_finished = Signal(str)

    def __init__(self, area, output, table_widget):
        super().__init__()
        self.area = area
        self.output = output
        self.table_widget = table_widget

    def run(self):
        export_results(self.area, self.output)
        self.export_finished.emit(f"✓ 结果已导出至：{self.output}")
        df = pd.read_csv(self.output)
        self.populate_table(df)

    def populate_table(self, df):
        self.table_widget.setRowCount(len(df))
        self.table_widget.setColumnCount(len(df.columns))
        self.table_widget.setHorizontalHeaderLabels(df.columns)
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for i in range(len(df)):
            for j in range(len(df.columns)):
                self.table_widget.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))


class EicPostProcessThread(QtCore.QThread):
    postprocess_finished = Signal(str)

    def __init__(self, output, xic_info, table_widget):
        super().__init__()
        self.output = output
        self.xic_info = xic_info
        self.table_widget = table_widget

    def run(self):
        post_process(self.output, self.xic_info)
        self.postprocess_finished.emit("✓ 后处理完成")
        df = pd.read_csv(self.output)
        self.populate_table(df)

    def populate_table(self, df):
        self.table_widget.setRowCount(len(df))
        self.table_widget.setColumnCount(len(df.columns))
        self.table_widget.setHorizontalHeaderLabels(df.columns)
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for i in range(len(df)):
            for j in range(len(df.columns)):
                self.table_widget.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))


if __name__ == "__main__":
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec()
