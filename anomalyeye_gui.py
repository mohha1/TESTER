from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from inference_engine import (
    InferenceResult,
    InferenceSettings,
    discover_model_package,
    export_ng_results,
    run_batch,
)


ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "LOGO2.png"


class Worker(QObject):
    progress = Signal(int, int, object)
    finished = Signal(list, str)
    failed = Signal(str)

    def __init__(self, model_path: str, images_dir: str, out_dir: str, settings: InferenceSettings):
        super().__init__()
        self.model_path = model_path
        self.images_dir = images_dir
        self.out_dir = out_dir
        self.settings = settings

    def run(self):
        try:
            results, report_path = run_batch(
                self.model_path,
                self.images_dir,
                self.out_dir,
                self.settings,
                progress=lambda done, total, result: self.progress.emit(done, total, result),
            )
            self.finished.emit(results, str(report_path))
        except Exception as exc:
            self.failed.emit(str(exc))


class PathRow(QWidget):
    changed = Signal(str)

    def __init__(self, title: str, placeholder: str):
        super().__init__()
        self.value = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel(title)
        label.setObjectName("FieldLabel")
        row = QHBoxLayout()
        row.setSpacing(8)
        self.path_label = QLabel(placeholder)
        self.path_label.setObjectName("PathValue")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setMinimumHeight(34)
        self.path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button = QPushButton("Browse")
        button.clicked.connect(self.browse)
        row.addWidget(self.path_label, 1)
        row.addWidget(button)
        layout.addWidget(label)
        layout.addLayout(row)

    def browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder", self.value or str(ROOT))
        if path:
            self.set_value(path)

    def set_value(self, path: str):
        self.value = path
        self.path_label.setText(path)
        self.path_label.setToolTip(path)
        self.changed.emit(path)


class ImagePanel(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("ImagePanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(12, 8, 12, 8)
        self.title = QLabel(title)
        self.title.setObjectName("PanelTitle")
        self.dim = QLabel("")
        self.dim.setObjectName("DimBadge")
        head.addWidget(self.title)
        head.addStretch()
        head.addWidget(self.dim)
        self.image = QLabel("No image")
        self.image.setObjectName("ImageCanvas")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumHeight(280)
        self.image.setScaledContents(False)
        self.footer = QLabel("")
        self.footer.setObjectName("ImageFooter")
        self.footer.setMinimumHeight(28)
        layout.addLayout(head)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.footer)
        self._pixmap = QPixmap()

    def set_image(self, path: str, footer: str = "", dim: str = ""):
        self.footer.setText(footer)
        self.dim.setText(dim)
        self._pixmap = QPixmap(path) if path else QPixmap()
        self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._pixmap.isNull():
            self.image.setText("No image")
            self.image.setPixmap(QPixmap())
            return
        self.image.setText("")
        target = self.image.size()
        self.image.setPixmap(self._pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DENSO AnomalyEye")
        self.resize(1180, 760)
        self.results: list[InferenceResult] = []
        self.report_path = ""
        self.thread: QThread | None = None
        self.worker: Worker | None = None

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._topbar())
        root_layout.addWidget(self._body(), 1)
        root_layout.addWidget(self._footer())
        self.setCentralWidget(central)

        self._wire()
        self.setStyleSheet(STYLE)

    def _topbar(self):
        bar = QFrame()
        bar.setObjectName("TopBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 8, 18, 8)
        layout.setSpacing(14)
        logo = QLabel()
        logo.setObjectName("LogoBox")
        if LOGO_PATH.is_file():
            logo.setPixmap(QPixmap(str(LOGO_PATH)).scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo.setText("D")
        title = QLabel("AnomalyEye")
        title.setObjectName("TopTitle")
        self.runtime = QLabel("Runtime ready")
        self.runtime.setObjectName("ReadyBadge")
        layout.addWidget(logo)
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.runtime)
        return bar

    def _body(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._setup_panel())
        splitter.addWidget(self._results_panel())
        splitter.setSizes([340, 840])
        return splitter

    def _setup_panel(self):
        panel = QFrame()
        panel.setObjectName("SetupPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        heading = QLabel("Inspection setup")
        heading.setObjectName("SectionTitle")
        sub = QLabel("Select the DASH package, images, and output folder.")
        sub.setObjectName("Muted")
        sub.setWordWrap(True)

        self.model_row = PathRow("Model package", "Folder containing model.xml")
        self.images_row = PathRow("Images", "Folder containing product images")
        self.output_row = PathRow("Output", "Folder for artifacts and Excel")

        options = QFrame()
        options.setObjectName("OptionBox")
        grid = QGridLayout(options)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        mode_label = QLabel("Threshold mode")
        mode_label.setObjectName("FieldLabel")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["auto", "manual"])
        device_label = QLabel("OpenVINO device")
        device_label.setObjectName("FieldLabel")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "AUTO", "GPU"])
        self.auto_start_review = QCheckBox("Show first result after inference")
        self.auto_start_review.setChecked(True)
        grid.addWidget(mode_label, 0, 0)
        grid.addWidget(self.mode_combo, 0, 1)
        grid.addWidget(device_label, 1, 0)
        grid.addWidget(self.device_combo, 1, 1)
        grid.addWidget(self.auto_start_review, 2, 0, 1, 2)

        self.detected = QLabel("Model package not checked yet.")
        self.detected.setObjectName("Hint")
        self.detected.setWordWrap(True)
        self.start_btn = QPushButton("Start inference")
        self.start_btn.setObjectName("PrimaryButton")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)

        layout.addWidget(heading)
        layout.addWidget(sub)
        layout.addSpacing(4)
        layout.addWidget(self.model_row)
        layout.addWidget(self.images_row)
        layout.addWidget(self.output_row)
        layout.addWidget(options)
        layout.addWidget(self.detected)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.progress)
        layout.addStretch()
        return panel

    def _results_panel(self):
        panel = QFrame()
        panel.setObjectName("ResultsPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._stats_row())

        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)
        side = QFrame()
        side.setObjectName("ListPanel")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        list_title = QLabel("Processed images")
        list_title.setObjectName("SectionTitleSmall")
        filters = QHBoxLayout()
        self.all_btn = QPushButton("All")
        self.ok_btn = QPushButton("OK")
        self.ng_btn = QPushButton("NG")
        for btn in [self.all_btn, self.ok_btn, self.ng_btn]:
            btn.setObjectName("FilterButton")
            btn.setCheckable(True)
            filters.addWidget(btn)
        self.all_btn.setChecked(True)
        self.list_widget = QListWidget()
        self.list_widget.setObjectName("ResultList")
        self.export_ng_btn = QPushButton("Export NG Excel")
        self.export_ng_btn.setEnabled(False)
        self.open_output_btn = QPushButton("Open output folder")
        self.open_output_btn.setEnabled(False)
        side_layout.addWidget(list_title)
        side_layout.addLayout(filters)
        side_layout.addWidget(self.list_widget, 1)
        side_layout.addWidget(self.export_ng_btn)
        side_layout.addWidget(self.open_output_btn)

        viewer = QFrame()
        viewer.setObjectName("Viewer")
        viewer_layout = QVBoxLayout(viewer)
        viewer_layout.setContentsMargins(16, 16, 16, 16)
        viewer_layout.setSpacing(12)
        self.stack = QStackedWidget()
        empty = QLabel("Run inference to review results.")
        empty.setObjectName("EmptyState")
        empty.setAlignment(Qt.AlignCenter)
        result_page = QWidget()
        result_layout = QVBoxLayout(result_page)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(12)
        panels = QHBoxLayout()
        panels.setSpacing(12)
        self.original_panel = ImagePanel("Original")
        self.masked_panel = ImagePanel("Masked anomaly map")
        panels.addWidget(self.original_panel)
        panels.addWidget(self.masked_panel)
        self.detail = QLabel("")
        self.detail.setObjectName("DetailLine")
        self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        result_layout.addLayout(panels, 1)
        result_layout.addWidget(self.detail)
        self.stack.addWidget(empty)
        self.stack.addWidget(result_page)
        viewer_layout.addWidget(self.stack, 1)

        body.addWidget(side)
        body.addWidget(viewer)
        body.setSizes([260, 700])
        layout.addWidget(body, 1)
        return panel

    def _stats_row(self):
        row = QFrame()
        row.setObjectName("StatsRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.ok_stat = self._stat("OK", "0", "OkStat")
        self.ng_stat = self._stat("NG", "0", "NgStat")
        self.total_stat = self._stat("Total", "0", "TotalStat")
        layout.addWidget(self.ok_stat)
        layout.addWidget(self.ng_stat)
        layout.addWidget(self.total_stat)
        return row

    def _stat(self, label: str, value: str, name: str):
        box = QFrame()
        box.setObjectName("StatBox")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(18, 12, 18, 12)
        title = QLabel(label)
        title.setObjectName(name + "Label")
        number = QLabel(value)
        number.setObjectName(name)
        layout.addWidget(title)
        layout.addWidget(number)
        box.number = number
        return box

    def _footer(self):
        footer = QFrame()
        footer.setObjectName("Footer")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(18, 6, 18, 6)
        self.status = QLabel("Idle")
        self.status.setObjectName("FooterText")
        self.report = QLabel("")
        self.report.setObjectName("FooterText")
        self.report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        layout.addStretch()
        layout.addWidget(self.report)
        return footer

    def _vline(self):
        line = QFrame()
        line.setObjectName("VLine")
        line.setFixedWidth(1)
        return line

    def _wire(self):
        self.model_row.changed.connect(self.check_model)
        self.start_btn.clicked.connect(self.start_inference)
        self.list_widget.currentRowChanged.connect(self.show_result)
        self.all_btn.clicked.connect(lambda: self.apply_filter("ALL"))
        self.ok_btn.clicked.connect(lambda: self.apply_filter("OK"))
        self.ng_btn.clicked.connect(lambda: self.apply_filter("NG"))
        self.export_ng_btn.clicked.connect(self.export_ng_excel)
        self.open_output_btn.clicked.connect(self.open_output)

    def check_model(self, path: str):
        try:
            package = discover_model_package(path)
            meta = package.meta_json.name if package.meta_json else "not found"
            self.detected.setText(
                f"Detected model.xml: {package.model_xml.name}\n"
                f"Detected info: {package.info_yml.name}\n"
                f"Detected meta_data.json: {meta}"
            )
            self.detected.setProperty("state", "ok")
        except Exception as exc:
            self.detected.setText(str(exc))
            self.detected.setProperty("state", "bad")
        self.detected.style().unpolish(self.detected)
        self.detected.style().polish(self.detected)

    def start_inference(self):
        missing = []
        if not self.model_row.value:
            missing.append("model package")
        if not self.images_row.value:
            missing.append("image folder")
        if not self.output_row.value:
            missing.append("output folder")
        if missing:
            QMessageBox.warning(self, "Missing input", "Select: " + ", ".join(missing))
            return

        settings = InferenceSettings(
            mode=self.mode_combo.currentText(),
            device=self.device_combo.currentText(),
            save_artifacts=True,
        )
        self.results = []
        self.list_widget.clear()
        self.progress.setValue(0)
        self.report.setText("")
        self.export_ng_btn.setEnabled(False)
        self.open_output_btn.setEnabled(False)
        self.set_running(True)

        self.thread = QThread()
        self.worker = Worker(self.model_row.value, self.images_row.value, self.output_row.value, settings)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.cleanup_thread)
        self.thread.start()

    def set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.runtime.setText("Inference running" if running else "Runtime ready")
        self.status.setText("Running inference..." if running else "Idle")

    def on_progress(self, done: int, total: int, result):
        if total:
            self.progress.setValue(int(done * 100 / total))
        if result is not None:
            self.results.append(result)
            self.add_result_item(result)
            self.update_stats()
            self.status.setText(f"Processed {done} of {total}: {Path(result.file).name}")

    def on_finished(self, results: list, report_path: str):
        self.results = results
        self.report_path = report_path
        self.set_running(False)
        self.progress.setValue(100)
        self.report.setText(f"Report: {report_path}")
        self.open_output_btn.setEnabled(True)
        self.export_ng_btn.setEnabled(any(result.product == "NG" for result in results))
        self.update_stats()
        self.status.setText(f"Done. {len(results)} images processed.")
        if results and self.auto_start_review.isChecked():
            self.list_widget.setCurrentRow(0)

    def on_failed(self, message: str):
        self.set_running(False)
        self.status.setText("Inference failed.")
        QMessageBox.critical(self, "Inference failed", message)

    def cleanup_thread(self):
        self.thread = None
        self.worker = None

    def add_result_item(self, result: InferenceResult):
        name = Path(result.file).name
        item = QListWidgetItem(f"{result.product}   {name}\nscore {result.anomaly_score:g}  ·  blobs {result.blob_count}")
        item.setData(Qt.UserRole, result)
        item.setData(Qt.UserRole + 1, result.product)
        self.list_widget.addItem(item)

    def update_stats(self):
        ok = sum(1 for r in self.results if r.product == "OK")
        ng = sum(1 for r in self.results if r.product == "NG")
        self.ok_stat.number.setText(str(ok))
        self.ng_stat.number.setText(str(ng))
        self.total_stat.number.setText(str(len(self.results)))

    def show_result(self, row: int):
        item = self.list_widget.item(row)
        if item is None:
            return
        result = item.data(Qt.UserRole)
        if result is None:
            return
        image_name = Path(result.file).name
        dim = f"{result.width} x {result.height}"
        self.original_panel.set_image(result.file, image_name, dim)
        masked = result.masked_path or result.overlay_path or result.candidate_path
        self.masked_panel.set_image(masked, Path(masked).name if masked else "", dim)
        self.detail.setText(
            f"{image_name}  |  {result.product}  |  score {result.anomaly_score:g}  |  "
            f"blobs {result.blob_count}  |  NG pixels {result.ng_pixels_total}"
        )
        self.stack.setCurrentIndex(1)

    def apply_filter(self, status: str):
        self.all_btn.setChecked(status == "ALL")
        self.ok_btn.setChecked(status == "OK")
        self.ng_btn.setChecked(status == "NG")
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(status != "ALL" and item.data(Qt.UserRole + 1) != status)

    def open_output(self):
        path = self.output_row.value
        if path:
            QDesktopServices.openUrl(Path(path).resolve().as_uri())

    def export_ng_excel(self):
        if not self.results:
            QMessageBox.information(self, "No results", "Run inference before exporting NG results.")
            return
        default = Path(self.output_row.value or ROOT) / "ng_results.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export NG results",
            str(default),
            "Excel workbook (*.xlsx)",
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            exported = export_ng_results(self.results, path)
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self.status.setText(f"NG Excel exported: {exported}")
        QMessageBox.information(self, "Export complete", f"NG results exported to:\n{exported}")


STYLE = """
* {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 12px;
    color: #c9ced8;
}
QMainWindow, QWidget {
    background: #12151b;
}
QLabel, QCheckBox {
    background: transparent;
}
#TopBar, #Footer {
    background: #0c0e12;
    border-bottom: 1px solid #202432;
}
#Footer {
    border-top: 1px solid #202432;
    border-bottom: none;
}
#LogoBox {
    min-width: 36px;
    min-height: 36px;
}
#TopTitle {
    color: #d7dce7;
    font-size: 14px;
    font-weight: 650;
}
#ReadyBadge {
    color: #49d17d;
    background: #0b2115;
    border: 1px solid #1a4a2d;
    border-radius: 12px;
    padding: 4px 10px;
}
#VLine {
    background: #242938;
}
#SetupPanel, #ListPanel {
    background: #0f1218;
    border-right: 1px solid #202432;
}
#ResultsPanel, #Viewer {
    background: #141820;
}
#SectionTitle {
    color: #edf0f5;
    font-size: 18px;
    font-weight: 650;
}
#SectionTitleSmall {
    color: #d8dde8;
    font-size: 12px;
    font-weight: 650;
    letter-spacing: 0.5px;
}
#Muted, #FooterText, #ImageFooter {
    color: #697185;
}
#FieldLabel {
    color: #9ca4b6;
    font-weight: 600;
}
#PathValue {
    color: #747d91;
    background: #0a0c10;
    border: 1px solid #252a38;
    border-radius: 6px;
    padding: 8px 10px;
}
#OptionBox {
    background: #0a0c10;
    border: 1px solid #252a38;
    border-radius: 8px;
}
#Hint {
    color: #7f8798;
    background: #0a0c10;
    border: 1px solid #252a38;
    border-radius: 6px;
    padding: 10px;
}
#Hint[state="ok"] {
    color: #49d17d;
    border-color: #1a4a2d;
}
#Hint[state="bad"] {
    color: #ef6868;
    border-color: #573030;
}
QPushButton {
    color: #cbd1df;
    background: #1a1f2a;
    border: 1px solid #303649;
    border-radius: 6px;
    padding: 7px 12px;
}
QPushButton:hover {
    background: #222938;
    border-color: #424a62;
}
QPushButton:disabled {
    color: #535a6c;
    background: #151922;
    border-color: #222735;
}
#PrimaryButton {
    background: #233750;
    border-color: #33577d;
    color: #8ecaff;
    font-weight: 700;
    padding: 11px;
}
#FilterButton:checked {
    color: #ffffff;
    background: #252c3b;
    border-color: #4d5872;
}
QComboBox {
    color: #d5dae5;
    background: #121722;
    border: 1px solid #2b3245;
    border-radius: 6px;
    padding: 6px 8px;
}
QProgressBar {
    background: #0a0c10;
    border: 1px solid #252a38;
    border-radius: 5px;
    height: 8px;
}
QProgressBar::chunk {
    background: #d94f4f;
    border-radius: 5px;
}
#StatsRow {
    background: #131720;
    border-bottom: 1px solid #202432;
}
#StatBox {
    border-right: 1px solid #202432;
}
#OkStatLabel, #OkStat {
    color: #49d17d;
}
#NgStatLabel, #NgStat {
    color: #ef5757;
}
#TotalStatLabel {
    color: #7f8798;
}
#TotalStat {
    color: #dce1eb;
}
#OkStat, #NgStat, #TotalStat {
    font-size: 28px;
    font-weight: 650;
}
#ResultList {
    background: #0a0c10;
    border: 1px solid #202432;
    border-radius: 8px;
    padding: 4px;
}
#ResultList::item {
    border-radius: 6px;
    padding: 8px;
    margin: 2px;
}
#ResultList::item:selected {
    background: #202636;
}
#ImagePanel {
    background: #0a0c10;
    border: 1px solid #202432;
    border-radius: 8px;
}
#PanelTitle {
    color: #9fa8bb;
    font-weight: 650;
}
#DimBadge {
    color: #596174;
    background: #111620;
    border: 1px solid #202432;
    border-radius: 4px;
    padding: 2px 7px;
}
#ImageCanvas {
    background: #07090d;
    border-top: 1px solid #202432;
    border-bottom: 1px solid #202432;
    color: #444b5d;
}
#ImageFooter {
    padding: 6px 10px;
}
#DetailLine {
    color: #9ca4b6;
    background: #0a0c10;
    border: 1px solid #202432;
    border-radius: 6px;
    padding: 10px 12px;
}
#EmptyState {
    color: #656d80;
    background: #0a0c10;
    border: 1px solid #202432;
    border-radius: 8px;
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", help="Save a setup-screen screenshot and exit.")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if args.screenshot:
        def grab():
            Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(args.screenshot)
            app.quit()
        QTimer.singleShot(500, grab)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
