from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT / "cropper_settings.json"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_EXPORT_FORMAT = "PNG"


def find_logo() -> Path | None:
    candidates = [
        ROOT / "LOGO2.png",
        ROOT.parent / "LOGO2.png",
        Path.cwd() / "LOGO2.png",
        Path(r"C:\Users\TESTER\Desktop\PROJECTE OPENVINO\NEW\LOGO2.png"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "ROI"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for index in range(2, 10000):
        candidate = parent / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique output name for {path.name}")


def compact_path_label(path_text: str) -> str:
    path = Path(path_text)
    parts = path.parts
    if len(parts) <= 4:
        return path_text
    prefix = parts[0]
    tail = "\\".join(parts[-3:])
    return f"{prefix}\\...\\{tail}"


@dataclass
class ROI:
    name: str = "ROI_1"
    x1: int = 0
    y1: int = 0
    x2: int = 100
    y2: int = 100

    def normalized(self) -> "ROI":
        return ROI(self.name, min(self.x1, self.x2), min(self.y1, self.y2), max(self.x1, self.x2), max(self.y1, self.y2))

    def clamped(self, width: int, height: int) -> "ROI":
        roi = self.normalized()
        return ROI(
            roi.name,
            max(0, min(roi.x1, width)),
            max(0, min(roi.y1, height)),
            max(0, min(roi.x2, width)),
            max(0, min(roi.y2, height)),
        )

    @property
    def width(self) -> int:
        roi = self.normalized()
        return max(0, roi.x2 - roi.x1)

    @property
    def height(self) -> int:
        roi = self.normalized()
        return max(0, roi.y2 - roi.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {"name": self.name, "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_dict(cls, data: dict) -> "ROI":
        return cls(
            str(data.get("name", "ROI")),
            int(data.get("x1", 0)),
            int(data.get("y1", 0)),
            int(data.get("x2", 100)),
            int(data.get("y2", 100)),
        )


class CropWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(int, list, str)
    failed = Signal(str)

    def __init__(
        self,
        image_paths: list[Path],
        rois: list[ROI],
        output_dir: Path,
        export_format: str,
        per_roi_folder: bool,
        manifest_enabled: bool,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.rois = [ROI.from_dict(roi.to_dict()) for roi in rois]
        self.output_dir = output_dir
        self.export_format = export_format.upper()
        self.per_roi_folder = per_roi_folder
        self.manifest_enabled = manifest_enabled
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            total = len(self.image_paths) * len(self.rois)
            done = 0
            saved = 0
            errors: list[str] = []
            manifest_rows: list[dict] = []
            suffix = ".jpg" if self.export_format == "JPEG" else f".{self.export_format.lower()}"

            for image_path in self.image_paths:
                if self.cancelled:
                    break
                try:
                    with Image.open(image_path) as img:
                        img.load()
                        base_name = image_path.stem
                        for roi in self.rois:
                            if self.cancelled:
                                break
                            bounded = roi.clamped(img.width, img.height)
                            if bounded.width < 1 or bounded.height < 1:
                                done += 1
                                errors.append(f"{image_path.name}: skipped empty ROI {roi.name}")
                                self.progress.emit(done, total, image_path.name)
                                continue

                            crop = img.crop((bounded.x1, bounded.y1, bounded.x2, bounded.y2))
                            target_dir = self.output_dir / slugify(roi.name) if self.per_roi_folder else self.output_dir
                            target_dir.mkdir(parents=True, exist_ok=True)
                            out_path = unique_path(target_dir / f"{base_name}_{slugify(roi.name)}{suffix}")

                            save_img = crop
                            save_kwargs = {}
                            if self.export_format == "JPEG":
                                save_img = crop.convert("RGB")
                                save_kwargs["quality"] = 95
                            elif self.export_format == "WEBP":
                                save_kwargs["quality"] = 95

                            save_img.save(out_path, self.export_format, **save_kwargs)
                            saved += 1
                            manifest_rows.append(
                                {
                                    "source": str(image_path),
                                    "roi": roi.name,
                                    "x1": bounded.x1,
                                    "y1": bounded.y1,
                                    "x2": bounded.x2,
                                    "y2": bounded.y2,
                                    "width": bounded.width,
                                    "height": bounded.height,
                                    "output": str(out_path),
                                }
                            )
                            done += 1
                            self.progress.emit(done, total, image_path.name)
                except Exception as exc:
                    errors.append(f"{image_path.name}: {exc}")
                    remaining = len(self.rois)
                    done = min(total, done + remaining)
                    self.progress.emit(done, total, image_path.name)

            manifest_path = ""
            if self.manifest_enabled and manifest_rows:
                manifest = unique_path(self.output_dir / "crop_manifest.csv")
                with manifest.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(manifest_rows)
                manifest_path = str(manifest)

            if self.cancelled:
                errors.insert(0, "Batch cancelled by user.")
            self.finished.emit(saved, errors, manifest_path)
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
        self.path_label.setText(compact_path_label(path))
        self.path_label.setToolTip(path)
        self.changed.emit(path)


class ImageCanvas(QWidget):
    roi_created = Signal(object)
    roi_selected = Signal(int)
    cursor_changed = Signal(str)
    zoom_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("ImageCanvas")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.pixmap = QPixmap()
        self.image_path: Path | None = None
        self.rois: list[ROI] = []
        self.selected_index: int | None = None
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self._panning = False
        self._pan_start = QPointF()
        self._creating = False
        self._create_start: tuple[int, int] | None = None
        self._create_current: tuple[int, int] | None = None
        self._fit_on_resize = True

    def set_image(self, path: Path | None):
        self.image_path = path
        self.pixmap = QPixmap(str(path)) if path else QPixmap()
        self._fit_on_resize = True
        self.fit_to_window()
        self.update()

    def set_rois(self, rois: list[ROI], selected_index: int | None):
        self.rois = rois
        self.selected_index = selected_index
        self.update()

    def fit_to_window(self):
        if self.pixmap.isNull():
            self.scale = 1.0
            self.offset = QPointF(0, 0)
        else:
            margin = 36
            available_w = max(1, self.width() - margin)
            available_h = max(1, self.height() - margin)
            self.scale = min(available_w / self.pixmap.width(), available_h / self.pixmap.height(), 1.0)
            self.offset = QPointF(0, 0)
        self._fit_on_resize = True
        self.zoom_changed.emit(self.zoom_label())
        self.update()

    def actual_size(self):
        if not self.pixmap.isNull():
            self.scale = 1.0
            self.offset = QPointF(0, 0)
            self._fit_on_resize = False
            self.zoom_changed.emit(self.zoom_label())
            self.update()

    def zoom_by(self, factor: float, anchor: QPointF | None = None):
        if self.pixmap.isNull():
            return
        old_scale = self.scale
        self.scale = max(0.05, min(8.0, self.scale * factor))
        if anchor is not None:
            before = self.screen_to_image(anchor, clamp=False)
            self.offset += QPointF((anchor.x() - self.image_rect().x()) / old_scale - before[0], (anchor.y() - self.image_rect().y()) / old_scale - before[1])
        self._fit_on_resize = False
        self.zoom_changed.emit(self.zoom_label())
        self.update()

    def zoom_label(self) -> str:
        return f"{int(round(self.scale * 100))}%"

    def image_rect(self) -> QRectF:
        if self.pixmap.isNull():
            return QRectF()
        width = self.pixmap.width() * self.scale
        height = self.pixmap.height() * self.scale
        x = (self.width() - width) / 2 + self.offset.x()
        y = (self.height() - height) / 2 + self.offset.y()
        return QRectF(x, y, width, height)

    def screen_to_image(self, point: QPointF, clamp: bool = True) -> tuple[int, int]:
        rect = self.image_rect()
        x = int((point.x() - rect.x()) / self.scale)
        y = int((point.y() - rect.y()) / self.scale)
        if clamp and not self.pixmap.isNull():
            x = max(0, min(x, self.pixmap.width()))
            y = max(0, min(y, self.pixmap.height()))
        return x, y

    def roi_at(self, point: QPointF) -> int | None:
        if self.pixmap.isNull():
            return None
        x, y = self.screen_to_image(point)
        for index in range(len(self.rois) - 1, -1, -1):
            roi = self.rois[index].normalized()
            if roi.x1 <= x <= roi.x2 and roi.y1 <= y <= roi.y2:
                return index
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#07090d"))

        if self.pixmap.isNull():
            self._paint_empty(painter)
            return

        rect = self.image_rect()
        painter.fillRect(rect.adjusted(-1, -1, 1, 1), QColor("#0a0c10"))
        painter.drawPixmap(rect, self.pixmap, QRectF(self.pixmap.rect()))
        self._paint_grid(painter, rect)
        self._paint_rois(painter, rect)

        if self._creating and self._create_start and self._create_current:
            self._paint_temp_roi(painter, rect)

    def _paint_empty(self, painter: QPainter):
        painter.setPen(QPen(QColor("#202432"), 1))
        step = 42
        for x in range(0, self.width(), step):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            painter.drawLine(0, y, self.width(), y)
        painter.setPen(QColor("#697185"))
        painter.setFont(QFont("Segoe UI", 18, QFont.Weight.DemiBold))
        painter.drawText(self.rect().adjusted(0, -28, 0, 0), Qt.AlignCenter, "No image loaded")
        painter.setPen(QColor("#4f586b"))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(self.rect().adjusted(0, 22, 0, 0), Qt.AlignCenter, "Select an input folder to begin defining crop regions.")

    def _paint_grid(self, painter: QPainter, rect: QRectF):
        spacing = max(24, int(100 * self.scale))
        if spacing > 140:
            return
        painter.save()
        painter.setClipRect(rect)
        painter.setPen(QPen(QColor(32, 36, 50, 130), 1))
        x = rect.x()
        while x < rect.right():
            painter.drawLine(int(x), int(rect.y()), int(x), int(rect.bottom()))
            x += spacing
        y = rect.y()
        while y < rect.bottom():
            painter.drawLine(int(rect.x()), int(y), int(rect.right()), int(y))
            y += spacing
        painter.restore()

    def _paint_rois(self, painter: QPainter, rect: QRectF):
        for index, roi in enumerate(self.rois):
            bounded = roi.clamped(self.pixmap.width(), self.pixmap.height())
            roi_rect = QRectF(
                rect.x() + bounded.x1 * self.scale,
                rect.y() + bounded.y1 * self.scale,
                bounded.width * self.scale,
                bounded.height * self.scale,
            )
            selected = index == self.selected_index
            color = QColor("#d94f4f") if selected else QColor("#e3a13d")
            fill = QColor(color)
            fill.setAlpha(36 if selected else 16)
            painter.fillRect(roi_rect, fill)
            painter.setPen(QPen(color, 2.4 if selected else 1.6))
            painter.drawRect(roi_rect)

            label = f" {roi.name} "
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            label_rect = painter.fontMetrics().boundingRect(label).adjusted(-6, -3, 6, 4)
            label_rect.moveTopLeft(roi_rect.topLeft().toPoint() + QPointF(0, -label_rect.height() - 4).toPoint())
            if label_rect.top() < rect.top():
                label_rect.moveTopLeft(roi_rect.topLeft().toPoint() + QPointF(0, 5).toPoint())
            painter.fillRect(label_rect, color)
            painter.setPen(QColor("#f3f5f9"))
            painter.drawText(label_rect, Qt.AlignCenter, label)

            painter.setFont(QFont("Consolas", 8))
            painter.setPen(color)
            painter.drawText(roi_rect.bottomLeft() + QPointF(0, 15), f"{bounded.width} x {bounded.height}")

            if selected:
                painter.setBrush(QColor("#edf0f5"))
                painter.setPen(QPen(color, 1.5))
                size = 5
                for point in [roi_rect.topLeft(), roi_rect.topRight(), roi_rect.bottomLeft(), roi_rect.bottomRight()]:
                    painter.drawRect(QRectF(point.x() - size, point.y() - size, size * 2, size * 2))

    def _paint_temp_roi(self, painter: QPainter, rect: QRectF):
        x1, y1 = self._create_start
        x2, y2 = self._create_current
        temp = ROI("New ROI", x1, y1, x2, y2).clamped(self.pixmap.width(), self.pixmap.height())
        roi_rect = QRectF(
            rect.x() + temp.x1 * self.scale,
            rect.y() + temp.y1 * self.scale,
            temp.width * self.scale,
            temp.height * self.scale,
        )
        painter.setPen(QPen(QColor("#49d17d"), 2, Qt.DashLine))
        painter.drawRect(roi_rect)
        painter.setPen(QColor("#49d17d"))
        painter.setFont(QFont("Consolas", 9))
        painter.drawText(roi_rect.center() + QPointF(8, -8), f"{temp.width} x {temp.height}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_on_resize:
            QTimer.singleShot(0, self.fit_to_window)

    def mousePressEvent(self, event):
        if self.pixmap.isNull():
            return
        self.setFocus()
        point = QPointF(event.position())
        if event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = point
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.LeftButton:
            index = self.roi_at(point)
            if index is not None:
                self.roi_selected.emit(index)
                return
            self._creating = True
            self._create_start = self.screen_to_image(point)
            self._create_current = self._create_start
            self.setCursor(Qt.CrossCursor)
            self.update()

    def mouseMoveEvent(self, event):
        point = QPointF(event.position())
        if self._panning:
            delta = point - self._pan_start
            self.offset += delta
            self._pan_start = point
            self._fit_on_resize = False
            self.update()
        elif self._creating and self._create_start:
            self._create_current = self.screen_to_image(point)
            self.update()

        if not self.pixmap.isNull():
            x, y = self.screen_to_image(point)
            inside = 0 <= x < self.pixmap.width() and 0 <= y < self.pixmap.height()
            self.cursor_changed.emit(f"{x}, {y}" if inside else "")

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        if event.button() == Qt.LeftButton and self._creating and self._create_start and self._create_current:
            roi = ROI(f"ROI_{len(self.rois) + 1}", *self._create_start, *self._create_current).normalized()
            self._creating = False
            self._create_start = None
            self._create_current = None
            self.setCursor(Qt.ArrowCursor)
            if roi.width >= 8 and roi.height >= 8:
                self.roi_created.emit(roi)
            self.update()

    def wheelEvent(self, event):
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        self.zoom_by(factor, QPointF(event.position()))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DENSO Cropper")
        self.resize(1420, 860)
        self.setMinimumSize(1120, 680)

        self.input_folder: Path | None = None
        self.output_folder: Path | None = None
        self.image_files: list[Path] = []
        self.current_index = -1
        self.rois: list[ROI] = []
        self.selected_roi: int | None = None
        self.thread: QThread | None = None
        self.worker: CropWorker | None = None
        self._updating_roi_table = False
        self._suspend_settings = False

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._topbar())
        root.addWidget(self._body(), 1)
        root.addWidget(self._footer())
        self.setCentralWidget(central)

        self._wire()
        self._install_shortcuts()
        self.setStyleSheet(STYLE)
        self._load_settings()
        self._refresh_state()

    def _topbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("TopBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 8, 18, 8)
        layout.setSpacing(14)

        logo = QLabel()
        logo.setObjectName("LogoBox")
        logo_path = find_logo()
        if logo_path:
            logo.setPixmap(QPixmap(str(logo_path)).scaled(34, 34, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo.setText("D")

        title_group = QVBoxLayout()
        title_group.setSpacing(0)
        title = QLabel("DENSO Cropper")
        title.setObjectName("TopTitle")
        subtitle = QLabel("ROI crop preparation for inspection datasets")
        subtitle.setObjectName("TopSubtitle")
        title_group.addWidget(title)
        title_group.addWidget(subtitle)

        self.ready_badge = QLabel("Ready")
        self.ready_badge.setObjectName("ReadyBadge")

        layout.addWidget(logo)
        layout.addLayout(title_group)
        layout.addStretch()
        layout.addWidget(self.ready_badge)
        return bar

    def _body(self) -> QWidget:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._setup_panel())
        splitter.addWidget(self._viewer_panel())
        splitter.addWidget(self._roi_panel())
        splitter.setSizes([350, 630, 460])
        return splitter

    def _setup_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SetupPanel")
        panel.setMinimumWidth(330)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        heading = QLabel("Crop setup")
        heading.setObjectName("SectionTitle")
        layout.addWidget(heading)

        self.input_row = PathRow("Input images", "Select source folder")
        self.output_row = PathRow("Output crops", "Select output folder")
        layout.addWidget(self.input_row)
        layout.addWidget(self.output_row)

        options = QFrame()
        options.setObjectName("OptionBox")
        options.setFixedHeight(92)
        option_layout = QVBoxLayout(options)
        option_layout.setContentsMargins(12, 12, 12, 12)
        option_layout.setSpacing(7)

        format_label = QLabel("Export format")
        format_label.setObjectName("FieldLabel")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPEG", "WEBP"])
        self.format_combo.setCurrentText(DEFAULT_EXPORT_FORMAT)
        self.format_combo.setFixedHeight(34)

        self.recursive_check = QCheckBox("Include subfolders")
        self.folder_check = QCheckBox("Group by ROI name")
        self.manifest_check = QCheckBox("Write crop manifest")
        self.manifest_check.setChecked(True)

        option_layout.addWidget(format_label)
        option_layout.addWidget(self.format_combo)
        layout.addWidget(options)

        export_options = QFrame()
        export_options.setObjectName("OptionBox")
        export_options.setFixedHeight(90)
        export_layout = QVBoxLayout(export_options)
        export_layout.setContentsMargins(12, 10, 12, 10)
        export_layout.setSpacing(7)
        export_layout.addWidget(self.recursive_check)
        export_layout.addWidget(self.folder_check)
        export_layout.addWidget(self.manifest_check)
        layout.addWidget(export_options)

        self.stats_line = QLabel("0 images | 0 ROIs | 0 crops ready")
        self.stats_line.setObjectName("StatsLine")
        self.stats_line.setMinimumHeight(34)
        layout.addWidget(self.stats_line)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        self.crop_all_btn = QPushButton("Crop all images")
        self.crop_all_btn.setObjectName("PrimaryButton")
        self.crop_current_btn = QPushButton("Crop current")
        self.cancel_btn = QPushButton("Cancel batch")
        self.cancel_btn.setEnabled(False)
        self.open_output_btn = QPushButton("Open output folder")

        layout.addWidget(self.crop_all_btn)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addWidget(self.crop_current_btn)
        actions.addWidget(self.cancel_btn)
        layout.addLayout(actions)
        layout.addWidget(self.open_output_btn)
        layout.addStretch()
        return panel

    def _viewer_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Viewer")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        toolbar = QFrame()
        toolbar.setObjectName("ViewerToolbar")
        tools = QHBoxLayout(toolbar)
        tools.setContentsMargins(10, 8, 10, 8)
        tools.setSpacing(8)
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.fit_btn = QPushButton("Fit")
        self.actual_btn = QPushButton("100%")
        self.zoom_out_btn = QPushButton("-")
        self.zoom_in_btn = QPushButton("+")
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("DimBadge")
        self.image_title = QLabel("No image selected")
        self.image_title.setObjectName("PanelTitle")

        tools.addWidget(self.prev_btn)
        tools.addWidget(self.next_btn)
        tools.addWidget(self._vline())
        tools.addWidget(self.fit_btn)
        tools.addWidget(self.actual_btn)
        tools.addWidget(self.zoom_out_btn)
        tools.addWidget(self.zoom_in_btn)
        tools.addWidget(self.zoom_label)
        tools.addStretch()
        tools.addWidget(self.image_title)

        self.canvas = ImageCanvas()
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)
        return panel

    def _roi_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setMinimumWidth(420)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        files_header = QHBoxLayout()
        files_title = QLabel("Images")
        files_title.setObjectName("SectionTitleSmall")
        self.file_count = QLabel("0")
        self.file_count.setObjectName("DimBadge")
        files_header.addWidget(files_title)
        files_header.addStretch()
        files_header.addWidget(self.file_count)
        layout.addLayout(files_header)

        self.image_list = QListWidget()
        self.image_list.setObjectName("ResultList")
        self.image_list.setMinimumHeight(150)
        self.image_list.setMaximumHeight(210)
        layout.addWidget(self.image_list)

        roi_header = QHBoxLayout()
        roi_title = QLabel("ROI regions")
        roi_title.setObjectName("SectionTitleSmall")
        roi_header.addWidget(roi_title)
        roi_header.addStretch()
        layout.addLayout(roi_header)

        roi_actions = QHBoxLayout()
        roi_actions.setSpacing(8)
        self.add_roi_btn = QPushButton("Add")
        self.duplicate_roi_btn = QPushButton("Duplicate")
        self.delete_roi_btn = QPushButton("Delete")
        roi_actions.addWidget(self.add_roi_btn)
        roi_actions.addWidget(self.duplicate_roi_btn)
        roi_actions.addWidget(self.delete_roi_btn)
        layout.addLayout(roi_actions)

        self.roi_table = QTableWidget(0, 6)
        self.roi_table.setObjectName("RoiTable")
        self.roi_table.setHorizontalHeaderLabels(["Name", "X1", "Y1", "X2", "Y2", "Size"])
        self.roi_table.verticalHeader().setVisible(False)
        self.roi_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.roi_table.setSelectionMode(QTableWidget.SingleSelection)
        self.roi_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            self.roi_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Fixed)
            self.roi_table.setColumnWidth(column, 48)
        self.roi_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Fixed)
        self.roi_table.setColumnWidth(5, 78)
        self.roi_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.roi_table, 1)

        config_actions = QHBoxLayout()
        config_actions.setSpacing(8)
        self.save_config_btn = QPushButton("Save config")
        self.load_config_btn = QPushButton("Load config")
        config_actions.addWidget(self.save_config_btn)
        config_actions.addWidget(self.load_config_btn)
        layout.addLayout(config_actions)
        return panel

    def _footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("Footer")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(18, 7, 18, 7)
        self.status = QLabel("Idle")
        self.status.setObjectName("FooterText")
        self.cursor = QLabel("")
        self.cursor.setObjectName("FooterText")
        self.shortcuts = QLabel("Drag: create ROI | Right-drag: pan | Wheel: zoom | Delete: remove ROI")
        self.shortcuts.setObjectName("FooterText")
        layout.addWidget(self.status)
        layout.addStretch()
        layout.addWidget(self.shortcuts)
        layout.addWidget(self._vline())
        layout.addWidget(self.cursor)
        return footer

    def _metric(self, label: str, value: str) -> QWidget:
        box = QFrame()
        box.setObjectName("Metric")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(1)
        title = QLabel(label)
        title.setObjectName("MetricLabel")
        number = QLabel(value)
        number.setObjectName("MetricValue")
        layout.addWidget(title)
        layout.addWidget(number)
        return box

    def _vline(self) -> QFrame:
        line = QFrame()
        line.setObjectName("VLine")
        line.setFixedWidth(1)
        return line

    def _wire(self):
        self.input_row.changed.connect(self.set_input_folder)
        self.output_row.changed.connect(self.set_output_folder)
        self.recursive_check.stateChanged.connect(lambda _: self.reload_images())
        self.format_combo.currentTextChanged.connect(lambda _: self._save_settings())
        self.folder_check.stateChanged.connect(lambda _: self._save_settings())
        self.manifest_check.stateChanged.connect(lambda _: self._save_settings())

        self.image_list.currentRowChanged.connect(self.select_image)
        self.prev_btn.clicked.connect(self.previous_image)
        self.next_btn.clicked.connect(self.next_image)
        self.fit_btn.clicked.connect(self.canvas.fit_to_window)
        self.actual_btn.clicked.connect(self.canvas.actual_size)
        self.zoom_out_btn.clicked.connect(lambda: self.canvas.zoom_by(1 / 1.18))
        self.zoom_in_btn.clicked.connect(lambda: self.canvas.zoom_by(1.18))
        self.canvas.roi_created.connect(self.add_roi_from_canvas)
        self.canvas.roi_selected.connect(self.select_roi)
        self.canvas.cursor_changed.connect(self.cursor.setText)
        self.canvas.zoom_changed.connect(self.zoom_label.setText)

        self.add_roi_btn.clicked.connect(self.add_roi)
        self.duplicate_roi_btn.clicked.connect(self.duplicate_roi)
        self.delete_roi_btn.clicked.connect(self.delete_selected_roi)
        self.roi_table.currentCellChanged.connect(lambda row, _col, _old_row, _old_col: self.select_roi(row if row >= 0 else None))
        self.roi_table.itemChanged.connect(self.update_roi_from_item)
        self.save_config_btn.clicked.connect(self.save_config)
        self.load_config_btn.clicked.connect(self.load_config)

        self.crop_all_btn.clicked.connect(lambda: self.start_crop(self.image_files))
        self.crop_current_btn.clicked.connect(self.crop_current)
        self.cancel_btn.clicked.connect(self.cancel_batch)
        self.open_output_btn.clicked.connect(self.open_output)

    def _install_shortcuts(self):
        shortcuts = [
            ("Left", self.previous_image),
            ("Right", self.next_image),
            ("Delete", self.delete_selected_roi),
            ("Ctrl+S", self.save_config),
            ("Ctrl+O", self.load_config),
            ("F", self.canvas.fit_to_window),
            ("Ctrl+0", self.canvas.fit_to_window),
            ("Ctrl+1", self.canvas.actual_size),
            ("+", lambda: self.canvas.zoom_by(1.18)),
            ("-", lambda: self.canvas.zoom_by(1 / 1.18)),
            ("Ctrl+D", self.duplicate_roi),
        ]
        for key, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)

    def set_input_folder(self, path: str):
        self.input_folder = Path(path)
        self.reload_images()
        self._save_settings()

    def set_output_folder(self, path: str):
        self.output_folder = Path(path)
        self._save_settings()
        self._refresh_state()

    def reload_images(self):
        if not self.input_folder or not self.input_folder.is_dir():
            self.image_files = []
            self.current_index = -1
            self.image_list.clear()
            self.canvas.set_image(None)
            self._refresh_state()
            return

        iterator = self.input_folder.rglob("*") if self.recursive_check.isChecked() else self.input_folder.iterdir()
        self.image_files = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMG_EXTS)
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for path in self.image_files:
            label = path.name if path.parent == self.input_folder else str(path.relative_to(self.input_folder))
            item = QListWidgetItem(label)
            item.setToolTip(str(path))
            self.image_list.addItem(item)
        self.image_list.blockSignals(False)

        if self.image_files:
            self.image_list.setCurrentRow(0)
            self.select_image(0)
            self.status.setText(f"Loaded {len(self.image_files)} images")
        else:
            self.current_index = -1
            self.canvas.set_image(None)
            self.status.setText("No supported images found")
        self._refresh_state()

    def select_image(self, row: int):
        if row < 0 or row >= len(self.image_files):
            return
        self.current_index = row
        path = self.image_files[row]
        self.image_title.setText(f"{row + 1} / {len(self.image_files)}   {path.name}")
        self.canvas.set_image(path)
        self._refresh_state()

    def previous_image(self):
        if self.current_index > 0:
            self.image_list.setCurrentRow(self.current_index - 1)

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.image_list.setCurrentRow(self.current_index + 1)

    def add_roi(self):
        if self.canvas.pixmap.isNull():
            QMessageBox.warning(self, "No image", "Load an image before adding an ROI.")
            return
        width = self.canvas.pixmap.width()
        height = self.canvas.pixmap.height()
        roi_w = max(40, width // 2)
        roi_h = max(40, height // 2)
        x1 = max(0, (width - roi_w) // 2)
        y1 = max(0, (height - roi_h) // 2)
        self.add_roi_from_canvas(ROI(f"ROI_{len(self.rois) + 1}", x1, y1, x1 + roi_w, y1 + roi_h))

    def add_roi_from_canvas(self, roi: ROI):
        roi.name = self._unique_roi_name(roi.name)
        self.rois.append(roi.normalized())
        self.select_roi(len(self.rois) - 1)
        self.refresh_roi_table()
        self._refresh_state()

    def duplicate_roi(self):
        if self.selected_roi is None or not (0 <= self.selected_roi < len(self.rois)):
            return
        source = self.rois[self.selected_roi]
        copy = ROI(self._unique_roi_name(f"{source.name}_copy"), source.x1 + 12, source.y1 + 12, source.x2 + 12, source.y2 + 12)
        if not self.canvas.pixmap.isNull():
            copy = copy.clamped(self.canvas.pixmap.width(), self.canvas.pixmap.height())
        self.rois.append(copy)
        self.select_roi(len(self.rois) - 1)
        self.refresh_roi_table()
        self._refresh_state()

    def delete_selected_roi(self):
        if self.selected_roi is None or not (0 <= self.selected_roi < len(self.rois)):
            return
        self.rois.pop(self.selected_roi)
        self.selected_roi = min(self.selected_roi, len(self.rois) - 1) if self.rois else None
        self.refresh_roi_table()
        self._refresh_state()

    def select_roi(self, index: int | None):
        if index is None or index < 0 or index >= len(self.rois):
            self.selected_roi = None
        else:
            self.selected_roi = index
        self.canvas.set_rois(self.rois, self.selected_roi)
        if self.selected_roi is not None and self.roi_table.currentRow() != self.selected_roi:
            self.roi_table.blockSignals(True)
            self.roi_table.selectRow(self.selected_roi)
            self.roi_table.blockSignals(False)
        self._refresh_state()

    def refresh_roi_table(self):
        self._updating_roi_table = True
        self.roi_table.setRowCount(len(self.rois))
        for row, roi in enumerate(self.rois):
            data = [roi.name, roi.x1, roi.y1, roi.x2, roi.y2, f"{roi.width} x {roi.height}"]
            for column, value in enumerate(data):
                item = QTableWidgetItem(str(value))
                if column == 5:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if column in {1, 2, 3, 4, 5}:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.roi_table.setItem(row, column, item)
        self._updating_roi_table = False
        if self.selected_roi is not None and 0 <= self.selected_roi < len(self.rois):
            self.roi_table.selectRow(self.selected_roi)
        self.canvas.set_rois(self.rois, self.selected_roi)

    def update_roi_from_item(self, item: QTableWidgetItem):
        if self._updating_roi_table:
            return
        row = item.row()
        column = item.column()
        if row < 0 or row >= len(self.rois):
            return
        roi = self.rois[row]
        try:
            if column == 0:
                roi.name = self._unique_roi_name(item.text(), ignore_index=row)
            elif column == 1:
                roi.x1 = int(item.text())
            elif column == 2:
                roi.y1 = int(item.text())
            elif column == 3:
                roi.x2 = int(item.text())
            elif column == 4:
                roi.y2 = int(item.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid coordinate", "ROI coordinates must be whole numbers.")
        if not self.canvas.pixmap.isNull():
            self.rois[row] = roi.clamped(self.canvas.pixmap.width(), self.canvas.pixmap.height())
        self.refresh_roi_table()
        self._refresh_state()

    def _unique_roi_name(self, name: str, ignore_index: int | None = None) -> str:
        base = slugify(name)
        existing = {roi.name for index, roi in enumerate(self.rois) if index != ignore_index}
        if base not in existing:
            return base
        for index in range(2, 1000):
            candidate = f"{base}_{index}"
            if candidate not in existing:
                return candidate
        return f"{base}_{len(existing) + 1}"

    def crop_current(self):
        if 0 <= self.current_index < len(self.image_files):
            self.start_crop([self.image_files[self.current_index]])

    def start_crop(self, image_paths: list[Path]):
        if self.thread is not None:
            return
        if not image_paths:
            QMessageBox.warning(self, "No images", "Load at least one image before cropping.")
            return
        if not self.rois:
            QMessageBox.warning(self, "No ROIs", "Define at least one ROI before cropping.")
            return
        if not self.output_folder:
            QMessageBox.warning(self, "No output", "Select an output folder before cropping.")
            return

        self.worker = CropWorker(
            image_paths,
            self.rois,
            self.output_folder,
            self.format_combo.currentText(),
            self.folder_check.isChecked(),
            self.manifest_check.isChecked(),
        )
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_crop_progress)
        self.worker.finished.connect(self.on_crop_finished)
        self.worker.failed.connect(self.on_crop_failed)
        self.worker.finished.connect(self.cleanup_thread)
        self.worker.failed.connect(self.cleanup_thread)
        self.thread.start()
        self.progress.setValue(0)
        self.status.setText("Cropping started")
        self.ready_badge.setText("Running")
        self.ready_badge.setProperty("state", "running")
        self._set_running(True)

    def on_crop_progress(self, done: int, total: int, name: str):
        percent = int((done / total) * 100) if total else 0
        self.progress.setValue(percent)
        self.status.setText(f"Cropping {done} / {total}: {name}")

    def on_crop_finished(self, saved: int, errors: list, manifest_path: str):
        self.progress.setValue(100 if saved else 0)
        suffix = f" Manifest: {Path(manifest_path).name}" if manifest_path else ""
        self.status.setText(f"Saved {saved} crops.{suffix}")
        if errors:
            QMessageBox.warning(self, "Crop completed with notes", f"Saved {saved} crops.\n\n" + "\n".join(errors[:8]))
        else:
            QMessageBox.information(self, "Crop complete", f"Saved {saved} crops to:\n{self.output_folder}")

    def on_crop_failed(self, message: str):
        self.progress.setValue(0)
        self.status.setText("Crop failed")
        QMessageBox.critical(self, "Crop failed", message)

    def cleanup_thread(self):
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.thread = None
        self.worker = None
        self.ready_badge.setText("Ready")
        self.ready_badge.setProperty("state", "")
        self.ready_badge.style().unpolish(self.ready_badge)
        self.ready_badge.style().polish(self.ready_badge)
        self._set_running(False)

    def cancel_batch(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("Cancelling batch...")

    def _set_running(self, running: bool):
        for widget in [
            self.crop_all_btn,
            self.crop_current_btn,
            self.input_row,
            self.output_row,
            self.format_combo,
            self.folder_check,
            self.manifest_check,
        ]:
            widget.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

    def save_config(self):
        if not self.rois:
            QMessageBox.warning(self, "No ROIs", "There are no ROI regions to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save ROI configuration", str(ROOT / "roi_config.json"), "JSON files (*.json)")
        if not path:
            return
        data = {
            "version": 2,
            "source": "DENSO Cropper",
            "rois": [roi.to_dict() for roi in self.rois],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.status.setText(f"Saved ROI config: {Path(path).name}")

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load ROI configuration", str(ROOT), "JSON files (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.rois = [ROI.from_dict(item) for item in data.get("rois", [])]
            self.selected_roi = 0 if self.rois else None
            self.refresh_roi_table()
            self._refresh_state()
            self.status.setText(f"Loaded ROI config: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Config error", f"Could not load ROI config:\n{exc}")

    def open_output(self):
        if self.output_folder and self.output_folder.exists():
            QDesktopServices.openUrl(self.output_folder.as_uri())

    def _refresh_state(self):
        crop_count = len(self.image_files) * len(self.rois)
        self.stats_line.setText(f"{len(self.image_files)} images | {len(self.rois)} ROIs | {crop_count} crops ready")
        self.file_count.setText(str(len(self.image_files)))
        self.canvas.set_rois(self.rois, self.selected_roi)
        has_image = bool(self.image_files)
        has_roi = bool(self.rois)
        has_output = bool(self.output_folder)
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(0 <= self.current_index < len(self.image_files) - 1)
        self.crop_current_btn.setEnabled(has_image and has_roi and has_output and self.thread is None)
        self.crop_all_btn.setEnabled(has_image and has_roi and has_output and self.thread is None)
        self.duplicate_roi_btn.setEnabled(self.selected_roi is not None)
        self.delete_roi_btn.setEnabled(self.selected_roi is not None)
        self.open_output_btn.setEnabled(has_output and self.output_folder.exists() if self.output_folder else False)

    def _load_settings(self):
        if not SETTINGS_PATH.is_file():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if data.get("input_folder") and Path(data["input_folder"]).is_dir():
                self.input_row.set_value(data["input_folder"])
            if data.get("output_folder"):
                self.output_row.set_value(data["output_folder"])
            if data.get("format") in {"PNG", "JPEG", "WEBP"}:
                self.format_combo.setCurrentText(data["format"])
            self.recursive_check.setChecked(bool(data.get("recursive", False)))
            self.folder_check.setChecked(bool(data.get("per_roi_folder", False)))
            self.manifest_check.setChecked(bool(data.get("manifest", True)))
        except Exception:
            pass

    def _save_settings(self):
        if self._suspend_settings:
            return
        data = {
            "input_folder": str(self.input_folder) if self.input_folder else "",
            "output_folder": str(self.output_folder) if self.output_folder else "",
            "format": self.format_combo.currentText(),
            "recursive": self.recursive_check.isChecked(),
            "per_roi_folder": self.folder_check.isChecked(),
            "manifest": self.manifest_check.isChecked(),
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load_demo(self):
        self._suspend_settings = True
        demo_dir = Path(tempfile.gettempdir()) / "denso_cropper_demo"
        demo_dir.mkdir(exist_ok=True)
        image_path = demo_dir / "sample_part.png"
        if not image_path.exists():
            img = Image.new("RGB", (1280, 760), "#111720")
            draw = ImageDraw.Draw(img)
            draw.rectangle((120, 120, 1160, 640), fill="#202938", outline="#3b455d", width=3)
            draw.rectangle((220, 205, 510, 450), fill="#2f394c", outline="#7c879d", width=2)
            draw.ellipse((725, 175, 1015, 465), fill="#283243", outline="#8993a8", width=2)
            for x in range(160, 1120, 90):
                draw.line((x, 120, x, 640), fill="#2b3345")
            for y in range(160, 620, 70):
                draw.line((120, y, 1160, y), fill="#2b3345")
            draw.text((140, 675), "Demo inspection image", fill="#7f8798")
            img.save(image_path)
        self.input_row.set_value(str(demo_dir))
        self.output_row.set_value(str(demo_dir / "crops"))
        self.rois = [ROI("Connector_A", 210, 190, 525, 470), ROI("Housing_Ring", 705, 155, 1035, 490)]
        self.selected_roi = 0
        self.refresh_roi_table()
        self._refresh_state()
        self._suspend_settings = False


STYLE = """
* {
    font-family: "Segoe UI", system-ui, sans-serif;
    font-size: 10pt;
}
QMainWindow {
    background: #141820;
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
    min-width: 38px;
    min-height: 38px;
}
#TopTitle {
    color: #d7dce7;
    font-size: 15px;
    font-weight: 650;
}
#TopSubtitle {
    color: #697185;
    font-size: 9px;
}
#ReadyBadge {
    color: #49d17d;
    background: #0b2115;
    border: 1px solid #1a4a2d;
    border-radius: 12px;
    padding: 4px 10px;
}
#ReadyBadge[state="running"] {
    color: #8ecaff;
    background: #102032;
    border-color: #33577d;
}
#SetupPanel, #SidePanel {
    background: #0f1218;
    border-right: 1px solid #202432;
}
#Viewer {
    background: #141820;
}
#ViewerToolbar {
    background: #0f1218;
    border: 1px solid #202432;
    border-radius: 8px;
}
#ImageCanvas {
    background: #07090d;
    border: 1px solid #202432;
    border-radius: 8px;
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
}
#Muted, #FooterText {
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
#OptionBox, #StatsBox, #StatsLine {
    background: #0a0c10;
    border: 1px solid #252a38;
    border-radius: 8px;
}
#StatsLine {
    color: #8f98aa;
    padding: 8px 12px;
}
#Metric {
    background: #101620;
    border: 1px solid #202432;
    border-radius: 6px;
}
#MetricLabel {
    color: #697185;
    font-size: 9px;
}
#MetricValue {
    color: #dce1eb;
    font-size: 16px;
    font-weight: 650;
}
#PanelTitle {
    color: #9fa8bb;
    font-weight: 650;
}
#DimBadge {
    color: #7f8798;
    background: #111620;
    border: 1px solid #202432;
    border-radius: 4px;
    padding: 2px 7px;
}
#VLine {
    background: #242938;
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
QPushButton:pressed {
    background: #111722;
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
QComboBox, QCheckBox {
    color: #d5dae5;
}
QComboBox {
    background: #121722;
    border: 1px solid #2b3245;
    border-radius: 6px;
    padding: 6px 8px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
}
QCheckBox::indicator:unchecked {
    background: #121722;
    border: 1px solid #2b3245;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background: #d94f4f;
    border: 1px solid #e36a6a;
    border-radius: 3px;
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
#ResultList, #RoiTable {
    background: #0a0c10;
    border: 1px solid #202432;
    border-radius: 8px;
    color: #cbd1df;
    gridline-color: #202432;
}
#ResultList::item {
    border-radius: 6px;
    padding: 8px;
    margin: 2px;
}
#ResultList::item:selected {
    background: #202636;
}
QHeaderView::section {
    background: #121722;
    color: #7f8798;
    border: none;
    border-bottom: 1px solid #202432;
    padding: 6px;
    font-weight: 650;
}
QTableWidget::item {
    padding: 5px;
}
QTableWidget::item:selected {
    background: #202636;
    color: #edf0f5;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #0a0c10;
    border: none;
    width: 10px;
    height: 10px;
}
QScrollBar::handle {
    background: #293143;
    border-radius: 5px;
}
QScrollBar::handle:hover {
    background: #3a4358;
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", help="Save a screenshot and exit.")
    parser.add_argument("--demo", action="store_true", help="Load generated demo data for screenshots.")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow()
    if args.demo:
        window.load_demo()
    window.show()

    if args.screenshot:
        def grab():
            target = Path(args.screenshot)
            target.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(target))
            app.quit()

        QTimer.singleShot(900, grab)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
