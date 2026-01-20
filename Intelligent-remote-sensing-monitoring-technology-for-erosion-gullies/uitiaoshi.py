import sys
from PyQt5.QtCore import Qt, QPoint, QSize, QRect, QRectF
import numpy as np
from PyQt5.QtGui import QImage, QPixmap, QTransform, QPainter, QCursor
from PyQt5.QtWidgets import *
from PyQt5 import uic
import cv2
from skimage import morphology
import subprocess
import tkinter

global img
global img_path
import os
from tkinter import messagebox
import shutil
import csv
import metricsv5
import tifffile
import tempfile
from PyQt5.QtWidgets import QShortcut, QGroupBox
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeySequence, QPainter

import warnings
from PyQt5.QtCore import pyqtRemoveInputHook

warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=".*sipPyTypeDict().*")
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=".*sipPyTypeDict.*")
try:
    import sip
    sip.setapi('QVariant', 2)
    sip.setapi('QString', 2)
except ImportError:
    pass

pyqtRemoveInputHook()
global pathsavebatch
global pathsaveone

pathsavebatch = ""
pathsaveone = ""

def area1(image):
    frame = cv2.imread(image)

    bright_frame = cv2.add(frame, 50)
    # Conver image after appying CLAHE to HSV
    hsv_frame = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2HSV)
    sensitivity = 70  # Higher value allows wider color range to be considered white color
    low_white = np.array([0, 0, 255 - sensitivity])
    high_white = np.array([255, sensitivity, 255])
    white_mask = cv2.inRange(hsv_frame, low_white, high_white)
    white = cv2.bitwise_and(frame, frame, mask=white_mask)  # white为处理后的图片

    sum = 0
    h, w = white.shape[0], white.shape[1]
    for i in range(h):
        for j in range(w):
            if white[i][j].all() > 0:
                sum += 1
    return sum

def area2(img):
    sum = 0
    h, w = img.shape[0], img.shape[1]
    for i in range(h):
        for j in range(w):
            if img[i][j].all() > 0:
                sum += 1
    return sum

def tongji(image):
    frame = cv2.imread(image)
    bright_frame = cv2.add(frame, 50)
    # Conver image after appying CLAHE to HSV
    hsv_frame = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2HSV)
    sensitivity = 70
    low_white = np.array([0, 0, 255 - sensitivity])
    high_white = np.array([255, sensitivity, 255])
    white_mask = cv2.inRange(hsv_frame, low_white, high_white)
    white = cv2.bitwise_and(frame, frame, mask=white_mask)
    img_gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
    num_objects, labels = cv2.connectedComponents(img_gray)
    return num_objects

class ZoomableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale_factor = 1.0
        self.panning = False
        self.drag_start_pos = QPoint()
        self.offset = QPoint()
        self.original_pixmap = None
        self.setStyleSheet("background-color: #002d50;")
        self.initial_scale_set = False
        self.base_scale_factor = 1.0

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.initial_scale_set = False
        self.resetView()

    def resetView(self):
        if not self.original_pixmap:
            return

        label_size = self.size()
        pixmap_size = self.original_pixmap.size()
        width_ratio = label_size.width() / pixmap_size.width()
        height_ratio = label_size.height() / pixmap_size.height()

        self.base_scale_factor = max(width_ratio, height_ratio)

        if self.base_scale_factor > 2.0:
            self.base_scale_factor = 2.0

        self.scale_factor = self.base_scale_factor
        self.offset = QPoint()
        self.initial_scale_set = True
        self.updatePixmap()

    def updatePixmap(self):
        if self.original_pixmap is None:
            return

        scaled_size = self.original_pixmap.size() * self.scale_factor
        scaled_pixmap = self.original_pixmap.scaled(
            scaled_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        x = (self.width() - scaled_pixmap.width()) // 2 + self.offset.x()
        y = (self.height() - scaled_pixmap.height()) // 2 + self.offset.y()

        painter.drawPixmap(x, y, scaled_pixmap)
        painter.end()
        super().setPixmap(pixmap)

    def wheelEvent(self, event):

        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        old_scale = self.scale_factor

        if event.angleDelta().y() > 0:
            self.scale_factor *= zoom_in_factor
        else:
            self.scale_factor *= zoom_out_factor

        self.scale_factor = max(0.1, min(self.scale_factor, 10.0))
        self.updatePixmap()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.panning = True
            self.drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning:
            delta = event.pos() - self.drag_start_pos
            self.drag_start_pos = event.pos()
            self.offset += delta
            self.updatePixmap()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.panning = False
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        self.updatePixmap()
        super().resizeEvent(event)

class HistoryRecordWidget(QWidget):
    """自定义历史记录项部件"""

    def __init__(self, image_path, result_data, timestamp, record_type="distribution", parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.result_data = result_data
        self.timestamp = timestamp
        self.record_type = record_type  # "distribution" 或 "measurement"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(60, 60)
        self.thumbnail_label.setStyleSheet("border: 1px solid #64ffda;")
        self.load_thumbnail()
        layout.addWidget(self.thumbnail_label)

        time_label = QLabel(timestamp)
        time_label.setStyleSheet("color: #e6f1ff; font-size: 10pt;")
        layout.addWidget(time_label, 1)

        detail_btn = QPushButton("详情")
        detail_btn.setFixedSize(60, 30)
        detail_btn.setStyleSheet("""
            QPushButton {
                background-color: #0c7b93;
                color: white;
                border-radius: 4px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #00a8cc;
            }
        """)
        detail_btn.clicked.connect(self.show_details)
        layout.addWidget(detail_btn)

        self.setStyleSheet("""
            QWidget {
                background-color: #002d50;
                border-bottom: 1px solid #1e2a3a;
            }
            QWidget:hover {
                background-color: #1f3b5d;
            }
        """)
        self.setFixedHeight(70)

    def sizeHint(self):
        """返回建议的大小"""
        return QSize(280, 70)

    def load_thumbnail(self):
        """加载并显示缩略图"""
        try:
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

                w, h = scaled_pixmap.width(), scaled_pixmap.height()
                if w > h:
                    x = (w - h) // 2
                    cropped_pixmap = scaled_pixmap.copy(x, 0, h, h)
                else:
                    y = (h - w) // 2
                    cropped_pixmap = scaled_pixmap.copy(0, y, w, w)

                self.thumbnail_label.setPixmap(cropped_pixmap.scaled(
                    60, 60, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                ))
        except Exception as e:
            print(f"加载缩略图失败: {str(e)}")

    def show_details(self):
        """显示评估详情，根据记录类型显示不同内容"""
        dialog = QDialog(self)

        if self.record_type == "distribution":
            dialog.setWindowTitle("评估详情")
        else:  # "measurement"
            dialog.setWindowTitle("测量详情")
        if self.record_type == "measurement":
            dialog.setFixedSize(800, 600)
        else:
            dialog.setFixedSize(600, 500)

        layout = QVBoxLayout(dialog)
        preview_label = QLabel()
        preview_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(self.image_path)

        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(600, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            preview_label.setPixmap(scaled_pixmap)
        layout.addWidget(preview_label)

        if self.record_type == "distribution":
            time_text = f"评估时间: {self.timestamp}"
        else:  # "measurement"
            time_text = f"测量时间: {self.timestamp}"

        time_label = QLabel(time_text)
        time_label.setStyleSheet("color: #64ffda; font-size: 10pt;")
        time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(time_label)

        if self.record_type == "distribution":
            details_label = QLabel(self.result_data)
            details_label.setStyleSheet("""
                QLabel {
                    color: #e6f1ff;
                    font-size: 14pt;
                    padding: 10px;
                    background-color: #002d50;
                    border-radius: 5px;
                }
            """)
            details_label.setWordWrap(True)
            layout.addWidget(details_label)
        else:
            # 形态测量类型 - 显示参数表格
            table_widget = QTableWidget()
            table_widget.setColumnCount(2)
            table_widget.setHorizontalHeaderLabels(["参数名称", "值"])
            table_widget.verticalHeader().setVisible(False)
            table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table_widget.setStyleSheet("""
                QTableWidget {
                    background-color: #002d50;
                    color: #e6f1ff;
                    font-size: 11pt;
                    gridline-color: #1e2a3a;
                    border: none;
                }
                QHeaderView::section {
                    background-color: #0c7b93;
                    color: white;
                    padding: 4px;
                    border: 1px solid #1e2a3a;
                }
            """)

            params = self.result_data
            table_widget.setRowCount(len(params))
            for i, (key, value) in enumerate(params.items()):
                table_widget.setItem(i, 0, QTableWidgetItem(key))
                table_widget.setItem(i, 1, QTableWidgetItem(f"{value:.2f}" if isinstance(value, float) else str(value)))

            table_widget.resizeColumnsToContents()
            table_widget.horizontalHeader().setStretchLastSection(True)
            layout.addWidget(table_widget)

        close_btn = QPushButton("关闭")
        close_btn.setFixedSize(100, 30)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #0c7b93;
                color: white;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #00a8cc;
            }
        """)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, 0, Qt.AlignCenter)

        dialog.exec_()

class MaskEditorDialog(QDialog):
    def __init__(self, rgb_path, mask, parent=None):
        super().__init__(parent)
        self.setWindowTitle("掩模编辑器")
        self.setGeometry(500, 250, 1500, 800)

        # 更新样式表，突出显示选中按钮
        self.setStyleSheet("""
            QPushButton {
                padding: 5px 10px;
                min-width: 80px;
                background-color: #2c3e50;
                color: white;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:checked {
                background-color: #e74c3c;
                font-weight: bold;
                border: 2px solid #f1c40f;
            }
            QSlider::handle:horizontal {
                width: 10px;
            }
            QGroupBox {
                border: 1px solid #7f8c8d;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QGraphicsView {
                background-color: #f0f0f0;
                border: 1px solid #34495e;
            }
        """)

        self.rgb_path = rgb_path
        self.original_mask = mask
        self.current_mask = mask.copy()
        self.result_mask = None
        self.drawing = False
        self.last_point = None
        self.tool = "pen"
        self.brush_size = 5
        self.draw_color = 255
        self.history = [self.current_mask.copy()]
        self.history_index = 0
        self.history_limit = 20
        self.zoom_factor = 1.0
        self.first_display = True
        self.panning = False
        self.pan_start = QPoint()
        self.last_update_rect = None
        self.partial_update_enabled = True
        self.brush_optimization = False
        self.temp_mask = None

        self.init_ui()
        self.setup_context_menu()
        self.update_display()

    def showEvent(self, event):
        """窗口显示事件 - 首次显示时适应视图"""
        super().showEvent(event)
        if self.first_display:
            self.fit_view()
            self.first_display = False

    def fit_view(self):
        """调整视图以完整显示图像"""
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.zoom_factor = 1.0
        self.view.setToolTip("视图已适应")

    def setup_context_menu(self):
        """设置右键上下文菜单"""
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.show_context_menu)

    def show_context_menu(self, pos):
        """显示右键菜单"""
        menu = QMenu(self)

        # 添加"撤销"菜单项
        undo_action = menu.addAction("撤销 (Ctrl+Z)")
        undo_action.setEnabled(self.history_index > 0)
        undo_action.triggered.connect(self.undo)

        # 添加"重置掩膜"菜单项
        reset_mask_action = menu.addAction("重置掩膜 (Ctrl+R)")
        reset_mask_action.triggered.connect(self.reset_mask)

        # 添加分隔线
        menu.addSeparator()

        # 添加"放大"菜单项
        zoom_in_action = menu.addAction("放大 (Ctrl++)")
        zoom_in_action.triggered.connect(self.zoom_in)

        # 添加"缩小"菜单项
        zoom_out_action = menu.addAction("缩小 (Ctrl+-)")
        zoom_out_action.triggered.connect(self.zoom_out)

        # 添加"重置视图"菜单项
        reset_view_action = menu.addAction("重置视图 (Ctrl+0)")
        reset_view_action.triggered.connect(self.reset_view)

        # 添加分隔线
        menu.addSeparator()

        # 添加"画笔"、"橡皮擦"、"填充"工具选项
        tools_group = QActionGroup(self)
        tools_group.setExclusive(True)

        pen_action = menu.addAction("画笔 (P)")
        pen_action.setCheckable(True)
        pen_action.setChecked(self.tool == "pen")
        pen_action.triggered.connect(lambda: self.set_tool("pen"))
        tools_group.addAction(pen_action)

        eraser_action = menu.addAction("橡皮擦 (E)")
        eraser_action.setCheckable(True)
        eraser_action.setChecked(self.tool == "eraser")
        eraser_action.triggered.connect(lambda: self.set_tool("eraser"))
        tools_group.addAction(eraser_action)

        fill_action = menu.addAction("填充 (F)")
        fill_action.setCheckable(True)
        fill_action.setChecked(self.tool == "fill")
        fill_action.triggered.connect(lambda: self.set_tool("fill"))
        tools_group.addAction(fill_action)

        # 添加分隔线
        menu.addSeparator()

        # 添加颜色选择
        colors_group = QActionGroup(self)
        colors_group.setExclusive(True)

        white_action = menu.addAction("侵蚀沟区域 (白色)")
        white_action.setCheckable(True)
        white_action.setChecked(self.draw_color == 255)
        white_action.triggered.connect(lambda: self.set_color(255))
        colors_group.addAction(white_action)

        black_action = menu.addAction("非侵蚀沟区域 (黑色)")
        black_action.setCheckable(True)
        black_action.setChecked(self.draw_color == 0)
        black_action.triggered.connect(lambda: self.set_color(0))
        colors_group.addAction(black_action)

        # 执行菜单
        action = menu.exec_(self.view.mapToGlobal(pos))

    def reset_view(self):
        """重置视图（缩放和平移）"""
        self.view.resetTransform()
        self.zoom_factor = 1.0
        self.fit_view()
        self.view.setToolTip("视图已重置")

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        # 图像显示区域
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setMouseTracking(True)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        layout.addWidget(self.view, 4)

        # 工具栏
        tool_layout = QHBoxLayout()
        tool_layout.setSpacing(10)

        # 工具按钮
        tool_group = QGroupBox("")
        tool_group_layout = QHBoxLayout()
        tool_group_layout.setSpacing(5)

        self.pen_btn = QPushButton("画笔")
        self.pen_btn.setCheckable(True)
        self.pen_btn.setChecked(True)
        self.pen_btn.setToolTip("绘制侵蚀沟区域 (快捷键: P)")
        self.pen_btn.clicked.connect(lambda: self.set_tool("pen"))
        tool_group_layout.addWidget(self.pen_btn)

        self.eraser_btn = QPushButton("橡皮擦")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.setToolTip("擦除侵蚀沟区域 (快捷键: E)")
        self.eraser_btn.clicked.connect(lambda: self.set_tool("eraser"))
        tool_group_layout.addWidget(self.eraser_btn)

        self.fill_btn = QPushButton("填充")
        self.fill_btn.setCheckable(True)
        self.fill_btn.setToolTip("区域填充 (快捷键: F)")
        self.fill_btn.clicked.connect(lambda: self.set_tool("fill"))
        tool_group_layout.addWidget(self.fill_btn)
        tool_group.setLayout(tool_group_layout)
        tool_layout.addWidget(tool_group)

        # 颜色选择
        color_group = QGroupBox("")
        color_layout = QHBoxLayout()
        color_layout.setSpacing(5)

        self.white_btn = QPushButton("侵蚀沟")
        self.white_btn.setCheckable(True)
        self.white_btn.setChecked(True)
        self.white_btn.setToolTip("标记为侵蚀沟区域 (白色)")
        self.white_btn.clicked.connect(lambda: self.set_color(255))
        color_layout.addWidget(self.white_btn)

        self.black_btn = QPushButton("非侵蚀沟")
        self.black_btn.setCheckable(True)
        self.black_btn.setToolTip("标记为背景区域 (黑色)")
        self.black_btn.clicked.connect(lambda: self.set_color(0))
        color_layout.addWidget(self.black_btn)
        color_group.setLayout(color_layout)
        tool_layout.addWidget(color_group)

        # 画笔大小
        size_group = QGroupBox("")
        size_layout = QHBoxLayout()

        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 20)
        self.size_slider.setValue(5)
        self.size_slider.setToolTip("调整画笔大小")
        self.size_slider.valueChanged.connect(self.update_brush_size)
        size_layout.addWidget(self.size_slider)

        self.size_label = QLabel(f"{self.size_slider.value()}px")
        size_layout.addWidget(self.size_label)
        size_group.setLayout(size_layout)
        tool_layout.addWidget(size_group)

        # 撤销按钮
        self.undo_btn = QPushButton("撤销")
        self.undo_btn.setToolTip("撤销上一步操作")
        self.undo_btn.clicked.connect(self.undo)
        tool_layout.addWidget(self.undo_btn)

        # 保存/取消按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.reset_btn = QPushButton("重置")
        self.reset_btn.setToolTip("恢复到原始掩模")
        self.reset_btn.clicked.connect(self.reset_mask)
        btn_layout.addWidget(self.reset_btn)

        self.save_btn = QPushButton("保存")
        self.save_btn.setToolTip("保存编辑结果")
        self.save_btn.clicked.connect(self.save_mask)
        btn_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setToolTip("取消编辑")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        tool_layout.addLayout(btn_layout)

        layout.addLayout(tool_layout)
        self.setLayout(layout)

        # 安装事件过滤器
        self.view.viewport().installEventFilter(self)

        # 添加快捷键
        QShortcut(QKeySequence("P"), self).activated.connect(lambda: self.set_tool("pen"))
        QShortcut(QKeySequence("E"), self).activated.connect(lambda: self.set_tool("eraser"))
        QShortcut(QKeySequence("F"), self).activated.connect(lambda: self.set_tool("fill"))
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.undo)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.reset_mask)
        QShortcut(QKeySequence("Ctrl+0"), self).activated.connect(self.reset_view)
        QShortcut(QKeySequence("Ctrl+="), self).activated.connect(self.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self).activated.connect(self.zoom_out)

    def set_tool(self, tool):
        self.tool = tool
        self.pen_btn.setChecked(tool == "pen")
        self.eraser_btn.setChecked(tool == "eraser")
        self.fill_btn.setChecked(tool == "fill")
        tool_names = {"pen": "画笔", "eraser": "橡皮擦", "fill": "填充"}
        self.view.setToolTip(f"当前工具: {tool_names[tool]} | 使用鼠标左键进行绘制")

    def set_color(self, color):
        self.draw_color = color
        self.white_btn.setChecked(color == 255)
        self.black_btn.setChecked(color == 0)
        color_name = "侵蚀沟区域 (白色)" if color == 255 else "背景区域 (黑色)"
        self.view.setToolTip(f"当前绘制颜色: {color_name}")

    def update_brush_size(self, size):
        self.brush_size = size
        self.size_label.setText(f"{size}px")
        self.view.setToolTip(f"画笔大小: {size}px")

    def reset_mask(self):
        """重置为原始掩模"""
        self.current_mask = self.original_mask.copy()
        self.history = [self.current_mask.copy()]
        self.history_index = 0
        self.update_button_states()
        self.update_display()
        self.view.setToolTip("已重置为原始掩模")

    def toggle_partial_update(self, state):
        """切换局部更新模式"""
        self.partial_update_enabled = (state == Qt.Checked)
        if not self.partial_update_enabled:
            self.update_display()

    def update_display(self, update_rect=None):
        """优化后的图像更新方法，支持局部更新"""
        try:
            if not self.partial_update_enabled or update_rect is None or self.first_display:
                self.scene.clear()

                rgb_img = cv2.imread(self.rgb_path)
                if rgb_img is None:
                    raise ValueError(f"无法读取图像: {self.rgb_path}")

                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                h, w, c = rgb_img.shape
                rgb_img = np.ascontiguousarray(rgb_img)

                qimg = QImage(rgb_img.data, w, h, 3 * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.bg_item = self.scene.addPixmap(pixmap)
                self.bg_item.setZValue(0)

                if self.current_mask.shape[:2] != (h, w):
                    self.current_mask = cv2.resize(self.current_mask, (w, h))

                self.mask_overlay = np.zeros((h, w, 4), dtype=np.uint8)
                self.update_mask_overlay()

                mask_img = np.ascontiguousarray(self.mask_overlay)
                qmask = QImage(mask_img.data, w, h, 4 * w, QImage.Format_RGBA8888)
                self.mask_pixmap = QPixmap.fromImage(qmask)
                self.mask_item = self.scene.addPixmap(self.mask_pixmap)
                self.mask_item.setZValue(1)
                self.scene.setSceneRect(0, 0, w, h)
            else:
                if self.mask_item is not None:
                    x, y, width, height = update_rect
                    self.update_mask_overlay(update_rect)

                    x1, y1, x2, y2 = x, y, x + width, y + height
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(self.mask_overlay.shape[1], x2)
                    y2 = min(self.mask_overlay.shape[0], y2)

                    if x1 >= x2 or y1 >= y2:
                        return

                    roi = self.mask_overlay[y1:y2, x1:x2]
                    if roi.size == 0:
                        return

                    roi_bytes = roi.tobytes()

                    qimg = QImage(roi_bytes,
                                  roi.shape[1],
                                  roi.shape[0],
                                  4 * roi.shape[1],  # bytesPerLine
                                  QImage.Format_RGBA8888)

                    # 更新Pixmap的指定区域
                    painter = QPainter(self.mask_pixmap)
                    painter.drawImage(QRect(x1, y1, width, height), qimg)
                    painter.end()

                    # 更新场景
                    self.mask_item.setPixmap(self.mask_pixmap)
                    self.scene.update(QRectF(x1, y1, width, height))

        except Exception as e:
            import traceback
            error_msg = f"Error in update_display: {e}\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self, "渲染错误", f"无法更新显示:\n{str(e)}")

    def update_mask_overlay(self, rect=None):
        """更新掩模覆盖层，支持局部更新"""
        h, w = self.current_mask.shape

        if rect is None:
            self.mask_overlay[:, :, 0] = 255
            self.mask_overlay[:, :, 3] = np.where(self.current_mask > 0, 100, 0)
        else:
            x, y, width, height = rect
            x1, y1, x2, y2 = x, y, x + width, y + height

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x1 >= x2 or y1 >= y2:
                return

            roi = self.current_mask[y1:y2, x1:x2]
            mask_roi = self.mask_overlay[y1:y2, x1:x2]
            mask_roi[:, :, 0] = 255
            mask_roi[:, :, 3] = np.where(roi > 0, 100, 0)

    def zoom_in(self):
        """放大图像"""
        self.zoom_factor *= 1.2
        self.view.scale(1.2, 1.2)
        self.view.setToolTip(f"缩放比例: {self.zoom_factor:.1f}x")

    def zoom_out(self):
        """缩小图像"""
        self.zoom_factor /= 1.2
        self.view.scale(1 / 1.2, 1 / 1.2)
        self.view.setToolTip(f"缩放比例: {self.zoom_factor:.1f}x")

    def viewportMousePressEvent(self, event):
        """处理视口鼠标按下事件"""
        if event.modifiers() & Qt.ControlModifier and event.button() == Qt.LeftButton:
            self.panning = True
            self.pan_start = event.pos()
            self.view.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            scene_pos = self.view.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())

            self.drawing = True
            self.last_point = QPoint(x, y)

            if self.tool == "pen":
                self.draw_point(x, y, self.draw_color)
            elif self.tool == "eraser":
                self.draw_point(x, y, 0)
            elif self.tool == "fill":
                self.fill_area(x, y, self.draw_color)

            if self.tool in ["pen", "eraser"]:
                self.save_state()

    def viewportMouseMoveEvent(self, event):
        """处理视口鼠标移动事件"""
        if self.panning:
            delta = event.pos() - self.pan_start
            self.pan_start = event.pos()
            self.view.horizontalScrollBar().setValue(
                self.view.horizontalScrollBar().value() - delta.x()
            )
            self.view.verticalScrollBar().setValue(
                self.view.verticalScrollBar().value() - delta.y()
            )
            return

        if self.drawing and event.buttons() & Qt.LeftButton:
            scene_pos = self.view.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())

            if self.last_point:
                if self.tool == "pen":
                    self.draw_line(self.last_point.x(), self.last_point.y(), x, y, self.draw_color)
                elif self.tool == "eraser":
                    self.draw_line(self.last_point.x(), self.last_point.y(), x, y, 0)

                self.last_point = QPoint(x, y)

    def viewportMouseReleaseEvent(self, event):
        """处理视口鼠标释放事件"""
        if self.panning and event.button() == Qt.LeftButton:
            self.panning = False
            self.view.setCursor(Qt.ArrowCursor)
            return

        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.last_point = None

            if self.tool in ["pen", "eraser"]:
                self.save_state()

            if self.last_update_rect:
                self.update_display()
                self.last_update_rect = None

    def eventFilter(self, source, event):
        """事件过滤器，实现缩放和平移功能"""
        if source is self.view.viewport():

            if event.type() == QEvent.Wheel:

                if event.modifiers() & Qt.ControlModifier:

                    if event.angleDelta().y() > 0:
                        self.zoom_in()
                    else:
                        self.zoom_out()
                    return True

            elif event.type() == QEvent.MouseButtonPress:

                if event.modifiers() & Qt.ControlModifier and event.button() == Qt.LeftButton:
                    self.panning = True
                    self.pan_start = event.pos()
                    self.view.setCursor(Qt.ClosedHandCursor)
                    return True
                else:

                    self.viewportMousePressEvent(event)
                    return True

            elif event.type() == QEvent.MouseMove:
                if self.panning:
                    self.viewportMouseMoveEvent(event)
                    return True
                else:
                    self.viewportMouseMoveEvent(event)
                    return True

            elif event.type() == QEvent.MouseButtonRelease:
                if self.panning and event.button() == Qt.LeftButton:
                    self.viewportMouseReleaseEvent(event)
                    return True
                else:
                    self.viewportMouseReleaseEvent(event)
                    return True

        return super().eventFilter(source, event)

    def draw_point(self, x, y, value):
        h, w = self.current_mask.shape
        if 0 <= x < w and 0 <= y < h:
            # 计算更新区域
            radius = self.brush_size
            update_rect = (
                max(0, x - radius),
                max(0, y - radius),
                min(w, x + radius) - max(0, x - radius),
                min(h, y + radius) - max(0, y - radius)
            )
            cv2.circle(self.current_mask, (x, y), self.brush_size, value, -1)

            # 更新显示
            self.last_update_rect = update_rect
            self.update_display(update_rect)

    def draw_line(self, x1, y1, x2, y2, value):
        h, w = self.current_mask.shape
        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
            # 计算更新区域
            radius = self.brush_size
            min_x = max(0, min(x1, x2) - radius)
            min_y = max(0, min(y1, y2) - radius)
            max_x = min(w, max(x1, x2) + radius)
            max_y = min(h, max(y1, y2) + radius)

            update_rect = (
                min_x,
                min_y,
                max_x - min_x,
                max_y - min_y
            )
            cv2.line(self.current_mask, (x1, y1), (x2, y2), value, self.brush_size)
            self.last_update_rect = update_rect
            self.update_display(update_rect)

    def fill_area(self, x, y, value):
        h, w = self.current_mask.shape
        if 0 <= x < w and 0 <= y < h:
            temp_mask = self.current_mask.copy()
            if temp_mask.dtype != np.uint8:
                temp_mask = temp_mask.astype(np.uint8)

            flags = 4 | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
            cv2.floodFill(temp_mask, None, (x, y), value, flags=flags)

            self.current_mask = temp_mask
            self.save_state()
            self.update_display()

    def save_state(self):
        # 限制历史记录数量
        if len(self.history) > self.history_limit * 2:
            self.history = self.history[len(self.history)//2:]
            self.history_index = len(self.history) - 1
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append(self.current_mask.copy())
        self.history_index = len(self.history) - 1
        if len(self.history) > self.history_limit:
            self.history.pop(0)
            self.history_index -= 1
        self.update_button_states()

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.current_mask = self.history[self.history_index].copy()
            self.update_display()
            self.update_button_states()
            self.view.setToolTip(f"已撤销操作 (剩余可撤销: {self.history_index})")

    def update_button_states(self):
        self.undo_btn.setEnabled(self.history_index > 0)

    def save_mask(self):
        self.result_mask = self.current_mask
        self.accept()

class MyWindow(QMainWindow):
# 基于 PyQt5 的主窗口类，用于创建图像处理应用程序的主要界面
    def __init__(self):
        super().__init__()           # 调用父类 QMainWindow 的构造函数
        self.lbl10 = None            # 初始化所有UI组件引用为None
        self.lbl19 = None
        self.lbl20 = None
        self.lbl21 = None
        self.lbl5 = None
        self.B18 = None              # 添加第一部分的掩膜编辑按钮引用
        self.ui = None               # 用于存储UI文件加载的界面对象
        self.B1 = None               # 按钮引用初始化（B1-B12）
        self.B2 = None
        self.B3 = None
        self.B4 = None
        self.B5 = None
        self.B6 = None
        self.B7 = None
        self.B8 = None
        self.B9 = None
        self.B10 = None
        self.B11 = None
        self.B12 = None
        self.B19 = None
        self.B20 = None
        self.lbl = None
        self.lbl2 = None
        self.stackedWidget = None                    # 堆叠窗口部件，用于多页面切换
        self.actionFunc1 = None                      # 菜单动作引用
        self.actionFunc2 = None
        self.actionFunc3 = None
        self.rgb_path = None                            # 存储32位RGB图像路径
        self.mask_image = None                          # 存储掩模图像数据
        self.edited_mask = None                         # 存储编辑后的掩模
        self.mask_edit_tool = "pen"                     # 当前编辑工具
        self.mask_brush_size = 5                        # 画笔大小
        self.history_records = []                       # 第一页面历史记录
        self.history_records_2 = []                     # 第二页面历史记录
        self.max_history = 20                           # 最大历史记录数
        self.history_placeholder_2 = None
        self.big_image_path = None
        self.split_rows = 0
        self.split_cols = 0
        self.temp_dir = "./test/temp_split_images"
        self.result_dir = "./test/temp_split_results"

        self.init_ui()

    def init_ui(self):
        self.ui = uic.loadUi("./shiyan1.ui", self)

        # 替换所有需要缩放功能的标签并更新引用
        self.lbl = self.replace_label_with_zoomable(self.ui.label)
        self.lbl2 = self.replace_label_with_zoomable(self.ui.label_2)
        self.lbl10 = self.replace_label_with_zoomable(self.ui.label_10)
        self.lbl11 = self.replace_label_with_zoomable(self.ui.label_11)
        self.lbl19 = self.replace_label_with_zoomable(self.ui.label_19)
        self.lbl20 = self.replace_label_with_zoomable(self.ui.label_20)
        self.lbl21 = self.replace_label_with_zoomable(self.ui.label_21)

        # 提取要操作的控件
        self.B1 = self.ui.pushButton_1
        self.B2 = self.ui.pushButton_2
        self.B3 = self.ui.pushButton_3
        self.B4 = self.ui.pushButton_4
        self.B5 = self.ui.pushButton_5
        self.B6 = self.ui.pushButton_6
        self.B7 = self.ui.pushButton_7
        self.B8 = self.ui.pushButton_8
        self.B9 = self.ui.pushButton_9
        self.B10 = self.ui.pushButton_10
        self.B11 = self.ui.pushButton_11
        self.B12 = self.ui.pushButton_12
        self.B13 = self.ui.pushButton_13
        self.B14 = self.ui.pushButton_14
        self.B19 = self.ui.pushButton_19
        self.B20 = self.ui.pushButton_20
        self.B18 = self.ui.pushButton_18
        self.lbl5 = self.ui.label_5

        # 获取stackedWidget引用
        self.stackedWidget = self.ui.stackedWidget

        self.actionFunc1 = self.ui.actionFunc1
        self.actionFunc2 = self.ui.actionFunc2
        self.actionFunc3 = self.ui.actionFunc3

        self.actionFunc1.triggered.connect(lambda: self.switch_page(0))
        self.actionFunc2.triggered.connect(lambda: self.switch_page(1))
        self.actionFunc3.triggered.connect(lambda: self.switch_page(2))

        self.B1.clicked.connect(self.process_big_image)
        self.B2.clicked.connect(self.xuanqu)
        self.B3.clicked.connect(self.fenge)
        self.B4.clicked.connect(self.pinggu)
        self.B5.clicked.connect(self.select_and_show_big_image)
        self.B6.clicked.connect(self.saveonepath)
        self.B7.clicked.connect(self.saveone)
        self.B8.clicked.connect(self.dsm_image_selected)
        self.B9.clicked.connect(self.generate_mask)
        self.B10.clicked.connect(self.calculate_parameters)
        self.B11.clicked.connect(self.set_save_path_metrics)
        self.B12.clicked.connect(self.save_metrics)
        self.B13.clicked.connect(self.rgb_image_selected)
        self.B14.clicked.connect(self.edit_mask_page2)
        self.B19.clicked.connect(self.set_save_path_for_big_image)
        self.B20.clicked.connect(self.save_big_image_results)
        self.B18.clicked.connect(self.edit_mask_page1)

        self.dsm_path = None
        self.mask_path = None
        self.metrics_result = None
        self.metrics_save_path = ""
        self.display_dsm_path = None

        # 初始化功能1历史记录
        self.historyListWidget = self.ui.findChild(QListWidget, "historyListWidget")
        self.historyListWidget.setSpacing(5)
        self.historyListWidget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.historyListWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 初始化历史记录列表
        self.history_records = []
        self.max_history = 20
        self.history_placeholder = QLabel("暂无历史记录", self.historyListWidget)
        self.history_placeholder.setAlignment(Qt.AlignCenter)
        self.history_placeholder.setStyleSheet("""
                    color: #8892b0;
                    font-size: 11pt;
                    font-style: italic;
                    padding: 20px;
                    text-align: center;
                """)
        self.history_placeholder.setGeometry(0, 0,
                                             self.historyListWidget.width(),
                                             self.historyListWidget.height())
        self.history_placeholder.show()

        # 初始化第二页面的历史记录列表
        self.historyListWidget_2 = self.ui.findChild(QListWidget, "historyListWidget_2")
        self.historyListWidget_2.setSpacing(5)
        self.historyListWidget_2.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.historyListWidget_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history_placeholder_2 = QLabel("暂无历史记录", self.historyListWidget_2)
        self.history_placeholder_2.setAlignment(Qt.AlignCenter)
        self.history_placeholder_2.setStyleSheet("""
                color: #8892b0;
                font-size: 11pt;
                font-style: italic;
                padding: 20px;
                text-align: center;
            """)
        self.history_placeholder_2.setGeometry(0, 0,
                                               self.historyListWidget_2.width(),
                                               self.historyListWidget_2.height())
        self.history_placeholder_2.show()

    def replace_label_with_zoomable(self, label):
        """将普通QLabel替换为ZoomableLabel并返回新标签"""
        parent = label.parent()
        geometry = label.geometry()

        zoom_label = ZoomableLabel(parent)
        zoom_label.setObjectName(label.objectName())
        zoom_label.setGeometry(geometry)
        zoom_label.setAlignment(label.alignment())
        # 设置白色背景
        zoom_label.setStyleSheet("background-color: #002d50;")
        # 添加右键菜单用于重置视图
        zoom_label.setContextMenuPolicy(Qt.CustomContextMenu)
        zoom_label.customContextMenuRequested.connect(
            lambda: self.show_reset_context_menu(zoom_label))
        # 隐藏原标签并返回新标签
        label.hide()
        return zoom_label

    def show_reset_context_menu(self, label):
        """显示重置视图的右键菜单"""
        menu = QMenu(self)
        reset_action = menu.addAction("重置视图")
        action = menu.exec_(QCursor.pos())
        if action == reset_action:
            label.resetView()

    def convert_32bit_to_8bit(self, input_path, output_path, coord_sys="projected"):
        """将TIFF图像（8位、16位、32位）转换为适合显示的8位图像。支持投影坐标系下的对比度增强（如 gamma 校正）"""
        try:
            img = tifffile.imread(input_path)
            data_type = img.dtype
            if data_type == np.uint8:
                img_normalized = img.astype(np.float32) / 255.0
            elif data_type == np.uint16:
                valid_mask = img > 0
                if np.any(valid_mask):
                    valid_values = img[valid_mask]
                    low1, high1 = np.percentile(valid_values, [0.1, 99.9])
                    mid_range = valid_values[(valid_values >= low1) & (valid_values <= high1)]
                    vmin = np.percentile(mid_range, 1)
                    vmax = np.percentile(mid_range, 99)
                else:
                    vmin, vmax = img.min(), img.max()

                img_normalized = (img.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6)
                img_normalized = np.clip(img_normalized, 0, 1)
            elif data_type in [np.float32, np.float64]:
                valid_mask = img > 0
                if np.any(valid_mask):
                    valid_values = img[valid_mask]
                    low1, high1 = np.percentile(valid_values, [0.1, 99.9])
                    mid_range = valid_values[(valid_values >= low1) & (valid_values <= high1)]
                    vmin = np.percentile(mid_range, 1)
                    vmax = np.percentile(mid_range, 99)
                else:
                    vmin, vmax = img.min(), img.max()

                img_normalized = (img.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6)
                img_normalized = np.clip(img_normalized, 0, 1)
            else:
                raise ValueError(f"不支持的数据类型: {data_type}")

            if coord_sys.lower() == "projected":
                gamma = 0.8
                img_normalized = np.power(img_normalized, gamma)

            img_8bit = (img_normalized * 255).astype(np.uint8)
            tifffile.imwrite(output_path, img_8bit)
            return True
        except Exception as e:
            print(f"转换过程中出错: {str(e)}")
            return False

    # 添加页面切换方法
    def switch_page(self, index):
        """切换到指定索引的页面"""
        print(f"切换到页面 {index+1}")
        self.stackedWidget.setCurrentIndex(index)

# 下面为功能1函数：
    def select_and_show_big_image(self):
        """选择并显示大图，切换到 page_2 界面"""
        # 切换到 page_2
        self.ui.stackedWidget_2.setCurrentIndex(1)
        path = QFileDialog.getOpenFileName(self.ui, "选择大图", "./", "Image Files (*.png *.jpg *.tif)")
        self.big_image_path = path[0]

        if not self.big_image_path:
            return
        pix = QPixmap(self.big_image_path)
        self.lbl11.setPixmap(pix)
        self.lbl11.resetView()
        self.B5.setText("重新选择大图")

    def process_big_image(self):
        """处理大图：切分、批量检测、重组结果"""
        if not self.big_image_path:
            QMessageBox.warning(self.ui, "警告", "请先选择大图")
            return

        self.split_rows = self.ui.splitRowsSpinBox.value()
        self.split_cols = self.ui.splitColsSpinBox.value()

        if self.split_rows <= 0 or self.split_cols <= 0:
            QMessageBox.warning(self.ui, "警告", "请设置有效的行数和列数")
            return

        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        for folder in [self.temp_dir, self.result_dir]:
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))

        if not self.split_image():
            return

        global pathsavebatch
        pathsavebatch = self.result_dir

        self.ceshi_batch()
        self.recombine_results()
        self.display_final_result()

        if hasattr(self, 'B20'):  # B20是保存按钮
            self.B20.setEnabled(True)
            self.B20.setText("保存结果")

    def split_image(self):
        try:
            big_image = cv2.imread(self.big_image_path)
            if big_image is None:
                QMessageBox.critical(self.ui, "错误", "无法读取大图")
                return False
            base_name = os.path.splitext(os.path.basename(self.big_image_path))[0]
            h, w = big_image.shape[:2]
            tile_height = h // self.split_rows
            tile_width = w // self.split_cols
            # 切分并保存
            for row in range(self.split_rows):
                for col in range(self.split_cols):
                    y1 = row * tile_height
                    y2 = (row + 1) * tile_height if row < self.split_rows - 1 else h
                    x1 = col * tile_width
                    x2 = (col + 1) * tile_width if col < self.split_cols - 1 else w
                    tile = big_image[y1:y2, x1:x2]
                    tile_name = f"{base_name}_{row}_{col}.png"
                    cv2.imwrite(os.path.join(self.temp_dir, tile_name), tile)

            return True
        except Exception as e:
            QMessageBox.critical(self.ui, "错误", f"切分图像失败: {str(e)}")
            return False

    def ceshi_batch(self):
        try:
            path1 = ".\\test\\data\\VOCdevkit\\VOC2012\\JPEGImages"
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            del_file(path1)
            for file in os.listdir(self.temp_dir):
                if file.endswith(".png"):
                    shutil.copy(
                        os.path.join(self.temp_dir, file),
                        os.path.join(path1, file)
                    )

            file_list = []
            for file in os.listdir(path1):
                if file.endswith(".png"):
                    file_name = file.split(".")[0]
                    file_list.append(file_name)

            with open(r".\test\data\VOCdevkit\VOC2012\ImageSets\Segmentation\val.txt", "w") as write_file:
                for file_name in file_list:
                    write_file.write(file_name + "\n")

            new_path = r'.\\test\\data\\VOCdevkit\\VOC2012\\SegmentationClassAug\\'
            for file in os.listdir(path1):
                if file.endswith(".png"):
                    img = cv2.imread(os.path.join(path1, file))
                    if img is not None:
                        gray = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                        cv2.imwrite(os.path.join(new_path, file), gray)

            # 执行检测
            a = subprocess.Popen(".\\test\\test.exe")
            a.wait()

            # 结果
            result_path = r'.\\test\\result\\'
            for file in os.listdir(result_path):
                if file.endswith(".png"):
                    shutil.copy(
                        os.path.join(result_path, file),
                        os.path.join(self.result_dir, file)
                    )

            try:                                # 删除目录
                for root, dirs, files in os.walk(new_path, topdown=False):
                    for name in files:
                        file_path = os.path.join(root, name)
                        os.remove(file_path)
                    for name in dirs:
                        dir_path = os.path.join(root, name)
                        os.rmdir(dir_path)

            except Exception as e:
                QMessageBox.critical(self.ui, "错误", f"清理文件夹内容")

            return True
        except Exception as e:
            QMessageBox.critical(self.ui, "错误", f"批量检测失败: {str(e)}")
            return False

    def recombine_results(self):  # 检测后重组
        try:
            # 读取原始大图
            big_image = cv2.imread(self.big_image_path)
            base_name = os.path.splitext(os.path.basename(self.big_image_path))[0]  #
            h, w = big_image.shape[:2]

            combined_mask = np.zeros((h, w), dtype=np.uint8)
            tile_height = h // self.split_rows
            tile_width = w // self.split_cols

            for row in range(self.split_rows):
                for col in range(self.split_cols):

                    result_name = f"{base_name}_{row}_{col}.png"  #
                    result_path = os.path.join(self.result_dir, result_name)

                    if not os.path.exists(result_path):  # 跳过缺失文件
                        continue

                    mask = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue

                    y1 = row * tile_height
                    y2 = (row + 1) * tile_height if row < self.split_rows - 1 else h
                    x1 = col * tile_width
                    x2 = (col + 1) * tile_width if col < self.split_cols - 1 else w
                    mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))
                    combined_mask[y1:y2, x1:x2] = mask_resized

            self.combined_mask_path = os.path.join(self.result_dir, "combined_mask.png")
            cv2.imwrite(self.combined_mask_path, combined_mask)
            return True
        except Exception as e:
            QMessageBox.critical(self.ui, "错误", f"重组结果失败: {str(e)}")
            return False

    def display_final_result(self):  # 最终结果展现
        """显示最终结果：原始大图 + 检测边界 + 分割线"""
        try:
            big_image = cv2.imread(self.big_image_path)
            if big_image is None:
                return

            combined_mask = cv2.imread(self.combined_mask_path, cv2.IMREAD_GRAYSCALE)
            if combined_mask is None:
                return

            edges = cv2.Canny(combined_mask, 30, 100)  # 降低阈值让更多边缘显现
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            boundary_layer = np.zeros_like(big_image)
            boundary_layer[edges > 0] = [0, 0, 255]  # 纯红色边界
            h, w = big_image.shape[:2]
            tile_height = h // self.split_rows
            tile_width = w // self.split_cols
            light_lavender = (200, 160, 255)

            for i in range(1, self.split_rows):
                y = i * tile_height
                for x in range(0, w, 10):
                    if x + 3 < w:
                        cv2.line(big_image, (x, y), (x + 3, y), light_lavender, 1)

            for j in range(1, self.split_cols):
                x = j * tile_width
                # 绘制更虚化的虚线: 每3像素实线，7像素空白
                for y in range(0, h, 10):  # 每10像素一个周期
                    if y + 3 < h:  # 确保不超过图像边界
                        cv2.line(big_image, (x, y), (x, y + 3), light_lavender, 1)

            enhanced_red = big_image.copy()
            enhanced_red[edges > 0] = [0, 0, 255]

            result = cv2.addWeighted(enhanced_red, 0.9, boundary_layer, 0.8, 0)
            result_path = os.path.join(self.result_dir, "final_result.png")
            cv2.imwrite(result_path, result)

            pix = QPixmap(result_path)
            self.lbl11.setPixmap(pix)
            self.lbl11.resetView()

        except Exception as e:
            QMessageBox.critical(self.ui, "错误", f"显示结果失败: {str(e)}")

    def set_save_path_for_big_image(self):  # 设置保存路径
        """设置大图处理结果的保存路径"""
        try:
            print("设置保存路径按钮被点击")

            save_dir = QFileDialog.getExistingDirectory(
                self.ui,
                "选择保存目录",
                "./",
                QFileDialog.ShowDirsOnly
            )

            if save_dir:
                self.custom_save_dir = save_dir
                QMessageBox.information(
                    self.ui,
                    "提示",
                    f"保存路径已设置为: {save_dir}"
                )
                if hasattr(self, 'B20'):
                    self.B20.setEnabled(True)
                    print("保存按钮已启用")
            else:
                QMessageBox.warning(self.ui, "警告", "未选择保存路径")
        except Exception as e:
            print(f"设置保存路径错误: {e}")
            QMessageBox.critical(self.ui, "错误", f"设置保存路径失败: {str(e)}")


    def save_big_image_results(self):  # 保存
        """保存大图处理结果到用户指定的目录"""
        try:
            print("保存结果按钮被点击")

            if not hasattr(self, 'custom_save_dir') or not self.custom_save_dir:
                QMessageBox.warning(self.ui, "警告", "请先设置保存路径")
                return

            if not hasattr(self, 'big_image_path') or not self.big_image_path:
                QMessageBox.warning(self.ui, "警告", "没有找到大图文件")
                return

            base_name = os.path.splitext(os.path.basename(self.big_image_path))[0]
            files_to_save = []

            # 1. 分块后的掩码图片（从temp_split_results目录）
            if hasattr(self, 'result_dir') and os.path.exists(self.result_dir):
                for file in os.listdir(self.result_dir):
                    if file.startswith(base_name) and file.endswith('.png'):
                        files_to_save.append(os.path.join(self.result_dir, file))

            # 2. combined_mask.png（组合掩码）
            mask_file = os.path.join(self.result_dir, "combined_mask.png")
            if os.path.exists(mask_file):
                files_to_save.append(mask_file)

            # 3. final_result.png（最终可视化结果）
            final_file = os.path.join(self.result_dir, "final_result.png")
            if os.path.exists(final_file):
                files_to_save.append(final_file)
            if not files_to_save:
                QMessageBox.warning(self.ui, "警告", "没有找到可保存的结果文件")
                return

            save_subdir = os.path.join(self.custom_save_dir, f"{base_name}_results")
            os.makedirs(save_subdir, exist_ok=True)

            # 复制文件
            saved_files = []
            for src_file in files_to_save:
                if os.path.exists(src_file):
                    filename = os.path.basename(src_file)
                    dst_file = os.path.join(save_subdir, filename)
                    shutil.copy2(src_file, dst_file)
                    saved_files.append(filename)
            self.clear_test_result_directory()

            if saved_files:
                report = f"已保存 {len(saved_files)} 个文件到:\n{save_subdir}\n\n保存的文件:\n"
                for i, filename in enumerate(saved_files, 1):
                    report += f"{i}. {filename}\n"

                QMessageBox.information(
                    self.ui,
                    "保存成功",
                    report
                )
            else:
                QMessageBox.warning(self.ui, "警告", "没有成功保存任何文件")

        except Exception as e:
            print(f"保存结果错误: {e}")
            QMessageBox.critical(self.ui, "错误", f"保存失败: {str(e)}")

    def clear_test_result_directory(self):
        """清空test\\result文件夹中的内容"""
        try:
            test_result_path = r'.\\test\\result\\'

            if os.path.exists(test_result_path):
                print(f"开始清空test\\result目录: {test_result_path}")
                for file in os.listdir(test_result_path):
                    file_path = os.path.join(test_result_path, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"已删除文件: {file}")
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            print(f"已删除目录: {file}")
                    except Exception as e:
                        print(f"删除文件 {file} 时出错: {e}")
                print(f"test\\result目录已清空")

        except Exception as e:
            print(f"清空test\\result目录错误: {e}")


    def xuanqu(self):                       # 功能图片选取
        global img_path
        path = QFileDialog.getOpenFileName(self.ui, "Open File", "./")
        img_path = path[0]
        # 切换到 page_1
        self.ui.stackedWidget_2.setCurrentIndex(0)
        if img_path == "":
            root = tkinter.Tk()
            root.withdraw()
            messagebox.showinfo(title='侵蚀沟检测', message='请正确选择数据')
        else:
            path1 = ".\\test\\data\\VOCdevkit\\VOC2012\\JPEGImages"

            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            del_file(path1)
            image = cv2.imread(img_path)  # 原图
            cv2.imwrite('.\\test\\data\\VOCdevkit\\VOC2012\\JPEGImages\\' + os.path.basename(img_path), image)

            print(img_path)
            if path[0]:
                pix = QPixmap(img_path)
                self.lbl.setPixmap(pix)
                self.lbl.setScaledContents(True)
                self.B2.setText("继续选取")

    def fenge(self):            # 智能检测
        global result_path
        image = r".\\test\\data\\VOCdevkit\\VOC2012\\JPEGImages\\" + os.path.basename(img_path)
        save_path = r".\\test\\data\\VOCdevkit\\VOC2012\\SegmentationClassAug\\" + os.path.basename(img_path)
        img = cv2.imread(image)
        gray = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))
        cv2.imwrite(save_path, gray)
        txt_path = r"./test/data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
        with open(txt_path, "w") as file:
            file.write(os.path.basename(img_path).replace(".png", ""))
        a = subprocess.Popen(".\\test\\test.exe")
        a.wait()
        root = tkinter.Tk()
        root.withdraw()
        result_path = r'.\\test\\result\\'
        pix = QPixmap(result_path + os.path.basename(img_path))
        self.lbl2.setPixmap(pix)
        self.lbl2.setScaledContents(True)

    def pinggu(self):            # 功能结果评估
        global result_path
        image = result_path + os.path.basename(img_path)
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        binary[binary == 255] = 1
        skeleton0 = morphology.skeletonize(binary)
        skeleton = skeleton0.astype(np.uint8) * 255
        aa = area2(skeleton)
        m = tongji(image)
        a = area1(image)

        picture1 = cv2.imread(img_path)
        img = cv2.imread(image, 0)
        img = cv2.Canny(img, 50, 150)
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        b = np.zeros_like(img)
        g = np.zeros_like(img)
        r = img.copy()
        r = np.where(r > 0, 255, 0).astype(np.uint8)
        res = cv2.merge((b, g, r))
        img = cv2.addWeighted(picture1, 0.8, res, 1.2, 0)
        cv2.imwrite(r'.\\test\\huancun\\' + '1.png', img)
        pix = QPixmap(r'.\\test\\huancun\\' + '1.png')

        self.lbl5.setText(
            '该区域的土地沟壑侵蚀条数为' + str(m - 1) + '条' + '\n' + '总的沟壑侵蚀面积为' + '' + str(
                a) + '平方米''\n' + '沟壑侵蚀总长度为' + str(
                aa) + '米')

        self.lbl10.setPixmap(pix)
        self.lbl10.resetView()
        self.add_history_record(img_path, self.lbl5.text())

    def add_history_record(self, image_path, result_text):
        """添加历史记录，新记录显示在顶部"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.historyListWidget.count() == 0 and hasattr(self, 'history_placeholder'):
            self.history_placeholder.hide()

        record_widget = HistoryRecordWidget(image_path, result_text, timestamp)
        item = QListWidgetItem()
        item.setSizeHint(record_widget.sizeHint())  # 设置项大小

        self.historyListWidget.insertItem(0, item)
        self.historyListWidget.setItemWidget(item, record_widget)
        self.historyListWidget.scrollToTop()
        self.history_records.insert(0, (item, record_widget))

        if len(self.history_records) > self.max_history:
            oldest_item, oldest_widget = self.history_records.pop()
            row = self.historyListWidget.row(oldest_item)
            self.historyListWidget.takeItem(row)

            if self.historyListWidget.count() == 0 and hasattr(self, 'history_placeholder'):
                self.history_placeholder.show()


    def edit_mask_page1(self):                   # 功能掩模调整
        """编辑第一部分的掩模图"""
        global img_path, result_path
        # 检查是否已分割
        if not img_path or not result_path:
            QMessageBox.warning(self.ui, "警告", "请先进行图像分割")
            return

        mask_path = os.path.join(result_path, os.path.basename(img_path))

        try:
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                QMessageBox.warning(self.ui, "警告", "无法读取分割结果图像")
                return

            self.mask_editor = MaskEditorDialog(img_path, mask_img, self)
            self.mask_editor.exec_()

            if self.mask_editor.result_mask is not None:
                cv2.imwrite(mask_path, self.mask_editor.result_mask)

                pix = QPixmap(mask_path)
                self.lbl2.setPixmap(pix)

                QMessageBox.information(self.ui, "完成", "掩模已更新")
        except Exception as e:
            QMessageBox.critical(self.ui, "错误", f"编辑掩模失败:\n{str(e)}")

    def saveonepath(self):
        global pathsaveone
        pathsaveone = QFileDialog.getExistingDirectory(None, "选取文件夹", "./")
        if pathsaveone == "":
            root = tkinter.Tk()
            root.withdraw()
            messagebox.showinfo(title='侵蚀沟检测', message='请正确选择文件夹')
            print("未选取文件夹")
        else:
            root = tkinter.Tk()
            root.withdraw()
            messagebox.showinfo(title='侵蚀沟检测', message='当前保存路径为' + pathsaveone)
            print('当前保存路径为' + pathsaveone)

    def saveone(self):
        global pathsaveone
        global result_path
        if pathsaveone == "":
            root = tkinter.Tk()
            root.withdraw()
            messagebox.showinfo(title='侵蚀沟检测', message='请事先设置保存路径')
        else:
            image = result_path + os.path.basename(img_path)
            img = cv2.imread(image)
            cv2.imwrite(pathsaveone + '/' + os.path.basename(img_path), img)
            root = tkinter.Tk()
            root.withdraw()
            messagebox.showinfo(title='侵蚀沟检测', message='保存成功')
        self.clear_test_result_directory()



    # 功能2函数
    def rgb_image_selected(self):
        """选取RGB图像并直接显示"""
        path = QFileDialog.getOpenFileName(self.ui, "选择RGB图像", "./", "TIFF Files (*.tif *.tiff);;All Files (*)")
        self.rgb_path = path[0]

        if not self.rgb_path:
            QMessageBox.warning(self.ui, "警告", "未选择RGB图像")
            return
        output_dir = ".\\eval\\data\\metrics\\1"
        try:
            if os.path.exists(output_dir):

                for filename in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"删除文件 {file_path} 时出错: {e}")
            else:
                os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self.ui, "警告", f"清理文件夹时出错: {e}")
            return

        filename = os.path.basename(self.rgb_path)
        display_path = os.path.join(output_dir, "display_" + os.path.splitext(filename)[0] + ".png")

        try:
            from PIL import Image
            img = Image.open(self.rgb_path)

            if img.mode == 'I;16B' or img.mode == 'I;16':
                img = img.convert('I')
                img_array = np.array(img)
                img_array = (img_array / 256).astype(np.uint8)
                img = Image.fromarray(img_array)

            img = img.convert('RGB')
            img.save(display_path, format='PNG')
            width, height = img.size
            bytes_per_line = 3 * width
            q_img = QImage(img.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            self.lbl21.setPixmap(pixmap)
            self.lbl21.setScaledContents(True)
            self.B13.setText("重新选择RGB")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self.ui, "错误", f"加载图像失败:\n{str(e)}\n\n详细错误信息:\n{error_details}")

    def dsm_image_selected(self):
        """选取DSM图像并转换为8位显示（投影坐标系）"""
        path = QFileDialog.getOpenFileName(self.ui, "选择DSM图像", "./", "TIFF Files (*.tif *.tiff)")
        self.dsm_path = path[0]  # 保存原始32位图像路径

        if not self.dsm_path:
            QMessageBox.warning(self.ui, "警告", "未选择DSM图像")
            return

        output_dir = ".\\eval\\data\\metrics\\2"

        try:
            if os.path.exists(output_dir):
                for filename in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"删除文件 {file_path} 时出错: {e}")
            else:
                os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self.ui, "警告", f"清理文件夹时出错: {e}")
            return

        filename = os.path.basename(self.dsm_path)
        display_path = os.path.join(output_dir, "display_" + filename)
        success = self.convert_32bit_to_8bit(
            self.dsm_path,
            display_path,
            coord_sys="projected"
        )

        if not success:
            QMessageBox.critical(self.ui, "错误", "DSM图像转换失败")
            return

        pix = QPixmap(display_path)
        self.lbl19.setPixmap(pix)
        self.lbl19.setScaledContents(True)
        self.B8.setText("重新选择DSM")

    def generate_mask(self):
        """调用eval.exe生成侵蚀沟掩模图，支持8/16/32位DSM图像自动转换为8位，每次执行前清空mask文件夹"""
        if not self.rgb_path or not self.dsm_path:
            QMessageBox.warning(self.ui, "警告", "请先选择RGB和DSM图像")
            return

        temp_dsm_path = None

        try:
            output_dir = ".\\eval\\data\\mask"

            if os.path.exists(output_dir):
                try:
                    shutil.rmtree(output_dir)
                    print(f"已清空目录: {output_dir}")
                except Exception as e:
                    raise Exception(f"清空目录失败: {output_dir}\n{str(e)}")

            os.makedirs(output_dir, exist_ok=True)
            print(f"已创建新目录: {output_dir}")

            dsm_img = cv2.imread(self.dsm_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            if dsm_img is None:
                raise Exception(f"无法读取DSM图像: {self.dsm_path}")

            if dsm_img.dtype == np.uint8:
                dsm_8bit = dsm_img
                print(f"DSM位深度: {self.dsm_path} (8bit) - 无需转换")
            elif dsm_img.dtype == np.uint16:
                dsm_8bit = cv2.convertScaleAbs(dsm_img, alpha=(255.0/65535.0))
                print(f"DSM位深度转换: {self.dsm_path} (16bit) -> 8bit")
            elif dsm_img.dtype == np.float32:
                min_val = np.min(dsm_img)
                max_val = np.max(dsm_img)
                if max_val == min_val:
                    dsm_8bit = np.zeros(dsm_img.shape, dtype=np.uint8)
                else:
                    dsm_8bit = np.clip((dsm_img - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
                print(f"DSM位深度转换: {self.dsm_path} (32bit) -> 8bit")
            else:
                raise Exception(f"DSM图像位深度不支持: {dsm_img.dtype} (必须是8/16/32位)")

            temp_dsm_path = os.path.join(output_dir, "temp_dsm_8bit.tif")
            cv2.imwrite(temp_dsm_path, dsm_8bit)
            print(f"临时8位DSM文件: {temp_dsm_path}")

            cmd = [
                ".\\eval\\eval.exe",
                "--rgb", self.rgb_path,
                "--modal_x", temp_dsm_path,  # 使用8位TIFF
                "--model", "log_NYUDepthv2_mit_b2/epoch-last.pth",
                "--output", output_dir
            ]

            print("执行命令:", " ".join(cmd))

            if not os.path.exists(".\\eval\\eval.exe"):
                raise Exception("eval.exe不存在，请确保路径正确: .\\eval\\eval.exe")

            if not os.path.exists("log_NYUDepthv2_mit_b2/epoch-last.pth"):
                raise Exception("模型权重文件不存在: log_NYUDepthv2_mit_b2/epoch-last.pth")

            if not os.path.exists(self.rgb_path):
                raise Exception(f"RGB图像不存在: {self.rgb_path}")

            # 运行外部程序
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "未知错误"
                print(f"错误输出: {error_msg}")
                raise Exception(f"eval.exe执行失败 (返回码: {process.returncode}): {error_msg}")

            if stdout:
                print(f"标准输出: {stdout.decode('utf-8', errors='ignore')}")

            filename = os.path.basename(self.rgb_path)
            mask_filename = os.path.splitext(filename)[0] + ".png"
            mask_path = os.path.join(output_dir, mask_filename)

            if not os.path.exists(mask_path):
                img_files = [f for f in os.listdir(output_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
                if img_files:
                    mask_path = os.path.join(output_dir, img_files[0])
                    print(f"使用找到的掩模文件: {mask_path}")
                else:
                    raise Exception("eval.exe未在输出目录中生成掩模文件")

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise Exception("无法读取生成的掩模文件")
            self.mask_image = mask
            self.edited_mask = mask.copy()
            pix = QPixmap(mask_path)
            self.lbl20.setPixmap(pix)
            self.lbl20.resetView()
            self.B9.setText("重新预测")

            try:
                os.remove(temp_dsm_path)
                print(f"已删除临时DSM文件: {temp_dsm_path}")
            except Exception as e:
                print(f"警告: 无法删除临时DSM文件 {temp_dsm_path}: {str(e)}")

            QMessageBox.information(self.ui, "成功", f"掩模生成完成！\n保存路径: {mask_path}")

        except Exception as e:
            if temp_dsm_path and os.path.exists(temp_dsm_path):
                try:
                    os.remove(temp_dsm_path)
                    print(f"清理: 已删除临时DSM文件 {temp_dsm_path}")
                except:
                    pass
            QMessageBox.critical(self.ui, "错误", f"生成掩模失败:\n{str(e)}")


    def calculate_parameters(self):  # 功能2：参数检测并填写参数结果
        """计算侵蚀沟参数并显示结果，每次计算前清空所有相关目录"""
        # 修改条件判断方式
        if not self.dsm_path or self.edited_mask is None:
            QMessageBox.warning(self.ui, "警告", "请先选择DSM图像并生成掩模图")
            return
        try:
            base_dir = ".\\eval\\data\\metrics"
            for i in range(3, 4):  # 清理 metrics\\1, metrics\\2, metrics\\3
                dir_to_clean = os.path.join(base_dir, str(i))
                if os.path.exists(dir_to_clean):
                    try:
                        shutil.rmtree(dir_to_clean)
                        print(f"已清空目录: {dir_to_clean}")
                    except Exception as e:
                        print(f"警告: 清空目录失败 ({dir_to_clean}): {str(e)}")

            output_dir = ".\\eval\\data\\metrics\\3"
            os.makedirs(output_dir, exist_ok=True)
            print(f"已创建新输出目录: {output_dir}")
            mask_path = os.path.join(output_dir, "temp_mask.png")
            cv2.imwrite(mask_path, self.edited_mask)

            # 计算参数
            self.metrics_result = metricsv5.calculate_metrics(
                mask_path,
                self.dsm_path,
                output_dir
            )

            # 更新 UI 参数显示
            self.ui.LengthLineEdit_2.setText(f"{self.metrics_result.get('骨架长度 (m)', 0):.2f}")
            self.ui.WidthLineEdit_2.setText(f"{self.metrics_result.get('平均宽度 (m)', 0):.2f}")
            self.ui.CircumferenceLineEdit_2.setText(f"{self.metrics_result.get('周长 (m)', 0):.2f}")
            self.ui.AreaLineEdit_2.setText(f"{self.metrics_result.get('面积 (平方米)', 0):.2f}")
            self.ui.VolumeLineEdit_2.setText(f"{self.metrics_result.get('体积 (立方米)', 0):.2f}")
            self.ui.DepthLineEdit_2.setText(f"{self.metrics_result.get('平均深度 (m)', 0):.2f}")
            self.ui.SlopeLineEdit_2.setText(f"{self.metrics_result.get('坡度(°)', 0):.2f}")

            self.add_history_record_2(
                self.rgb_path,
                self.metrics_result
            )
            QMessageBox.information(self.ui, "完成", "参数计算完成！")

        except Exception as e:
            QMessageBox.critical(self.ui, "错误", f"计算过程中发生错误:\n{str(e)}")

    def set_save_path_metrics(self):
        """设置参数保存路径"""
        self.metrics_save_path = QFileDialog.getExistingDirectory(None, "选择保存路径", "./")
        if self.metrics_save_path:
            QMessageBox.information(self.ui, "信息", f"保存路径设置为: {self.metrics_save_path}")

    def save_metrics(self):
        """保存参数结果"""
        if not self.metrics_result:
            QMessageBox.warning(self.ui, "警告", "没有可保存的结果")
            return

        if not self.metrics_save_path:
            save_dir = ".\\eval\\data\\metrics\\4"
            os.makedirs(save_dir, exist_ok=True)
            self.metrics_save_path = save_dir

        filename = os.path.basename(self.dsm_path).split('.')[0] + "_metrics.csv"
        save_path = os.path.join(self.metrics_save_path, filename)

        try:
            with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['参数名称', '值'])
                for key, value in self.metrics_result.items():
                    writer.writerow([key, value])

            QMessageBox.information(self.ui, "保存成功", f"参数已保存至:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self.ui, "保存失败", f"保存过程中发生错误:\n{str(e)}")

    def edit_mask_page2(self):
        """启动掩模图编辑功能"""
        if self.mask_image is None:
            QMessageBox.warning(self.ui, "警告", "请先生成掩模图")
            return

        self.mask_editor = MaskEditorDialog(self.rgb_path, self.edited_mask, self)
        self.mask_editor.exec_()

        # 更新掩模
        if self.mask_editor.result_mask is not None:
            self.edited_mask = self.mask_editor.result_mask
            # 更新界面显示
            h, w = self.edited_mask.shape
            qimg = QImage(self.edited_mask.data, w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)
            self.lbl20.setPixmap(pixmap)

    def add_history_record_2(self, image_path, params):
        """添加第二页面的历史记录"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.historyListWidget_2.count() == 0 and self.history_placeholder_2:
            self.history_placeholder_2.hide()

        record_widget = HistoryRecordWidget(
            image_path,
            params,
            timestamp,
            record_type="measurement"
        )

        # 创建QListWidgetItem
        item = QListWidgetItem()
        item.setSizeHint(record_widget.sizeHint())

        self.historyListWidget_2.insertItem(0, item)
        self.historyListWidget_2.setItemWidget(item, record_widget)
        self.historyListWidget_2.scrollToTop()
        self.history_records_2.insert(0, (item, record_widget))

        if len(self.history_records_2) > self.max_history:
            oldest_item, oldest_widget = self.history_records_2.pop()
            row = self.historyListWidget_2.row(oldest_item)
            self.historyListWidget_2.takeItem(row)

            if self.historyListWidget_2.count() == 0 and self.history_placeholder_2:
                self.history_placeholder_2.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)                # 创建QApplication实例，sys.argv包含命令行参数
    w = MyWindow()                              # 创建应用程序主窗口实例
    # 展示窗口（设置窗口固定大小为1600x1050像素）
    w.ui.setFixedSize(1600, 1050)
    w.ui.show()                                 # 显示主窗口
    w.ui.setWindowTitle("Intelligent Monitoring System")  # 设置窗口名称
    app.exec_()                                 # 进入应用程序的主事件循环