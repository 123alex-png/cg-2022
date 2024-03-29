#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import cg_algorithms as alg
from typing import Optional
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    qApp,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QListWidget,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QGraphicsRectItem,
    QStyleOptionGraphicsItem,
    QColorDialog)
from PyQt5.QtGui import QPainter, QMouseEvent, QColor
from PyQt5.QtCore import QRectF
from PyQt5.QtCore import Qt
from PyQt5.Qt import QPen
from PyQt5.Qt import (
    QInputDialog,
    QFileDialog,
    QImage
)

import math

class MyCanvas(QGraphicsView):
    """
    画布窗体类，继承自QGraphicsView，采用QGraphicsView、QGraphicsScene、QGraphicsItem的绘图框架
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.main_window = None
        self.list_widget = None
        self.item_dict = {}
        self.selected_id = ''

        self.status = ''
        self.temp_algorithm = ''
        self.temp_id = ''
        self.temp_color = QColor(0, 0, 0)
        self.temp_item = None

        self.origin_pos = None
        self.origin_p_list = None
        self.center = None
        self.edge = None

    def start_draw_line(self, algorithm, item_id):
        self.status = 'line'
        self.temp_algorithm = algorithm
        self.temp_id = item_id
        self.temp_item = None

    def start_draw_polygon(self, algorithm, item_id):
        self.status = 'polygon'
        self.temp_algorithm = algorithm
        self.temp_id = item_id
        self.temp_item = None

    def start_draw_ellipse(self, item_id):
        self.status = 'ellipse'
        self.temp_id = item_id
        self.temp_item = None

    def start_draw_curve(self, algorithm, item_id):
        self.status = 'curve'
        self.temp_algorithm = algorithm
        self.temp_id = item_id
        self.temp_item = None

    def start_translate(self):
        self.status = 'translate'
        self.temp_item = None

    def start_rotate(self):
        self.status = 'rotate'
        self.temp_item = None
        self.center = None
        self.origin_p_list = None

    def start_scale(self):
        self.status = 'scale'
        self.temp_item = None
        self.center = None
        self.origin_p_list = None

    def start_clip(self, algorithm):
        self.status = 'clip'
        self.temp_algorithm = algorithm
        self.temp_item = None
        self.origin_pos = None
        self.origin_p_list = None


    def finish_draw(self):
        self.temp_id = self.main_window.nxt_id()

    def clear_selection(self):
        if self.selected_id != '':
            self.item_dict[self.selected_id].selected = False
            self.selected_id = ''

    def selection_changed(self, selected):
        if self.status == 'polygon' or self.status == 'curve':
            self.finish_draw()
        self.main_window.statusBar().showMessage('图元选择： %s' % selected)
        if self.selected_id != '':
            self.item_dict[self.selected_id].selected = False
            self.item_dict[self.selected_id].update()
        if selected != '':
            self.selected_id = selected
            self.item_dict[selected].selected = True
            self.item_dict[selected].update()
            self.status = ''
        self.updateScene([self.sceneRect()])

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = self.mapToScene(event.localPos().toPoint())
        x = int(pos.x())
        y = int(pos.y())
        if self.status == 'line' or self.status == 'ellipse':
            self.temp_item = MyItem(self.temp_id, self.status, [[x, y], [x, y]], self.temp_algorithm, self.temp_color)
            self.scene().addItem(self.temp_item)
        elif self.status == 'polygon' or self.status == 'curve':
            if self.temp_item is None:
                self.temp_item = MyItem(self.temp_id, self.status, [[x, y]], self.temp_algorithm, self.temp_color)
                self.scene().addItem(self.temp_item)
            else:
                self.temp_item.p_list.append([x, y])
        elif self.status == 'translate':
            if self.selected_id != '':
                self.temp_item = self.item_dict[self.selected_id]
                self.origin_pos = pos
                self.origin_p_list = self.temp_item.p_list
        elif self.status == 'rotate':
            if self.selected_id != '':
                self.temp_item = self.item_dict[self.selected_id]
                self.origin_p_list = self.temp_item.p_list
                if self.center is None:
                    self.center = pos
                else:
                    self.origin_pos = pos
        elif self.status == 'scale':
            if self.selected_id != '':
                self.temp_item = self.item_dict[self.selected_id]
                self.origin_p_list = self.temp_item.p_list
                if self.center is None:
                    self.center = pos
                else:
                    self.origin_pos = pos
        elif self.status == 'clip':
            if self.selected_id != '':
                self.temp_item = self.item_dict[self.selected_id]
                if self.selected_id != '' and self.temp_item.item_type == 'line':
                        self.origin_pos = pos
                        self.origin_p_list = self.temp_item.p_list
        self.updateScene([self.sceneRect()])
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.temp_item is None:
            return
        pos = self.mapToScene(event.localPos().toPoint())
        x = int(pos.x())
        y = int(pos.y())
        if self.status == 'line' or self.status == 'ellipse':
            self.temp_item.p_list[1] = [x, y]
        elif self.status == 'polygon':
            self.temp_item.p_list[-1] = [x, y]
        elif self.status == 'translate':
            dx, dy = int(x - self.origin_pos.x()), int(y - self.origin_pos.y())
            self.temp_item.p_list = alg.translate(self.origin_p_list, dx, dy)
        elif self.status == 'rotate':
                x_a, y_a = int(self.origin_pos.x() - self.center.x()), int(
                    self.origin_pos.y() - self.center.y())
                len_a = (x_a ** 2 + y_a ** 2) ** 0.5
                x_b, y_b = int(x - self.center.x()), int(y - self.center.y())
                len_b = (x_b ** 2 + y_b ** 2) ** 0.5
                if len_a != 0 and len_b != 0:
                    sin_a = y_a / len_a
                    cos_a = x_a / len_a
                    sin_b = y_b / len_b
                    cos_b = x_b / len_b
                    cos_theta = cos_a * cos_b + sin_a * sin_b
                    sin_theta = sin_b * cos_a - cos_b * sin_a
                if cos_theta >= 0:
                    r = math.asin(sin_theta)
                else:
                    r = math.pi - math.asin(sin_theta)
                r = r / math.pi * 180
                # print(r)
                self.temp_item.p_list = alg.rotate(self.origin_p_list, int(self.center.x()), int(self.center.y()), r)
        elif self.status == 'scale':
            if self.selected_id != '' and self.center is not None and self.origin_pos is not None:
                x_a, y_a = int(self.origin_pos.x() - self.center.x()), int(
                    self.origin_pos.y() - self.center.y())
                len_a = (x_a ** 2 + y_a ** 2) ** 0.5
                x_b, y_b = int(x - self.center.x()), int(y - self.center.y())
                len_b = (x_b ** 2 + y_b ** 2) ** 0.5
                if len_a != 0:
                    k = len_b / len_a
                    self.temp_item.p_list = alg.scale(self.origin_p_list, int(self.center.x()), int(self.center.y()), k)
        elif self.status == 'clip':
            if self.selected_id != '' and self.origin_pos is not None and self.temp_item.item_type == 'line':
                x_min = min(int(self.origin_pos.x()), x)
                x_max = max(int(self.origin_pos.x()), x)
                y_min = min(int(self.origin_pos.y()), y)
                y_max = max(int(self.origin_pos.y()), y)
                if self.edge is None:
                    self.edge = QGraphicsRectItem(x_min - 1, y_min - 1, x_max - x_min + 2, y_max - y_min + 2)
                    self.scene().addItem(self.edge)
                    self.edge.setPen(QColor(0, 255, 255))
                else:
                    self.edge.setRect(x_min - 1, y_min - 1, x_max - x_min + 2, y_max - y_min + 2)
        self.updateScene([self.sceneRect()])
        super().mouseMoveEvent(event)


    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.temp_item is None:
            return
        pos = self.mapToScene(event.localPos().toPoint())
        x = int(pos.x())
        y = int(pos.y())
        if self.status == 'line' or self.status == 'ellipse':
            self.item_dict[self.temp_id] = self.temp_item
            self.list_widget.addItem(self.temp_id)
            self.finish_draw()
        elif self.status == 'polygon' or self.status == 'curve':
            self.item_dict[self.temp_id] = self.temp_item
            if not self.list_widget.findItems(self.temp_id, Qt.MatchContains):
                self.list_widget.addItem(self.temp_id)#防止重复加入
        elif self.status == 'clip':
            if self.selected_id != '' and self.temp_item.item_type == 'line':
                x_min = min(int(self.origin_pos.x()), x)
                x_max = max(int(self.origin_pos.x()), x)
                y_min = min(int(self.origin_pos.y()), y)
                y_max = max(int(self.origin_pos.y()), y)
                temp_p_list = alg.clip(self.origin_p_list, x_min, y_min, x_max, y_max, self.temp_algorithm)
                if len(temp_p_list) == 0:#若整个线段全部被裁剪掉，则直接删除该线段对于的图元
                    no = self.list_widget.findItems(self.selected_id, Qt.MatchContains)
                    row = self.list_widget.row(no[0])
                    self.list_widget.takeItem(row)
                    temp_id = self.selected_id
                    self.clear_selection()
                    self.list_widget.clearSelection()
                    self.scene().removeItem(self.temp_item)
                    self.temp_item = None
                    del self.item_dict[temp_id]
                if self.temp_item is not None:
                    self.temp_item.p_list = temp_p_list
                if self.edge is not None:
                    self.scene().removeItem(self.edge)
                    self.edge = None
                self.updateScene([self.sceneRect()])
        super().mouseReleaseEvent(event)


class MyItem(QGraphicsItem):
    """
    自定义图元类，继承自QGraphicsItem
    """
    def __init__(self, item_id: str, item_type: str, p_list: list, algorithm: str = '', color = QColor(0, 0, 0), parent: QGraphicsItem = None):
        """

        :param item_id: 图元ID
        :param item_type: 图元类型，'line'、'polygon'、'ellipse'、'curve'等
        :param p_list: 图元参数
        :param algorithm: 绘制算法，'DDA'、'Bresenham'、'Bezier'、'B-spline'等
        :param parent:
        """
        super().__init__(parent)
        self.id = item_id           # 图元ID
        self.item_type = item_type  # 图元类型，'line'、'polygon'、'ellipse'、'curve'等
        self.p_list = p_list        # 图元参数
        self.algorithm = algorithm  # 绘制算法，'DDA'、'Bresenham'、'Bezier'、'B-spline'等
        self.selected = False
        self.color = color

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = ...) -> None:
        painter.setPen(self.color)
        if self.item_type == 'line':
            item_pixels = alg.draw_line(self.p_list, self.algorithm)
            for p in item_pixels:
                painter.drawPoint(*p)
            if self.selected:
                painter.setPen(self.color)
                painter.setPen(QPen(Qt.DashLine))
                painter.drawRect(self.boundingRect())
        elif self.item_type == 'polygon':
            item_pixels = alg.draw_polygon(self.p_list, self.algorithm)
            for p in item_pixels:
                painter.drawPoint(*p)
            if self.selected:
                painter.setPen(self.color)
                painter.setPen(QPen(Qt.DashLine))
                painter.drawRect(self.boundingRect())
        elif self.item_type == 'ellipse':
            item_pixels = alg.draw_ellipse(self.p_list)
            for p in item_pixels:
                painter.drawPoint(*p)
            if self.selected:
                painter.setPen(self.color)
                painter.setPen(QPen(Qt.DashLine))
                painter.drawRect(self.boundingRect())
        elif self.item_type == 'curve':
            item_pixels = alg.draw_curve(self.p_list, self.algorithm)
            assist_pixels = alg.draw_polygon(self.p_list, 'Bresenham')
            for p in item_pixels:
                painter.drawPoint(*p)
            painter.setPen(QPen(Qt.DashLine))
            painter.setPen(QColor(170, 255, 255))
            for p in assist_pixels:
                painter.drawPoint(*p)
            if self.selected:
                painter.setPen(self.color)
                painter.setPen(QPen(Qt.DashLine))
                painter.drawRect(self.boundingRect())

    def boundingRect(self) -> QRectF:
        if self.item_type == 'line' or self.item_type == 'ellipse':
            x0, y0 = self.p_list[0]
            x1, y1 = self.p_list[1]
            x = min(x0, x1)
            y = min(y0, y1)
            w = max(x0, x1) - x
            h = max(y0, y1) - y
            return QRectF(x - 1, y - 1, w + 2, h + 2)
        elif self.item_type == 'polygon' or self.item_type == 'curve':
            x_max, y_max = 0, 0
            x_min, y_min = 999999, 999999
            for p in self.p_list:
                x_min = min(x_min, p[0])
                x_max = max(x_max, p[0])
                y_min = min(y_min, p[1])
                y_max = max(y_max, p[1])
            w = x_max - x_min
            h = y_max - y_min
            return QRectF(x_min-1, y_min-1, w+2, h+2)


class MainWindow(QMainWindow):
    """
    主窗口类
    """
    def __init__(self):
        super().__init__()
        self.item_cnt = 0

        # 使用QListWidget来记录已有的图元，并用于选择图元。注：这是图元选择的简单实现方法，更好的实现是在画布中直接用鼠标选择图元
        self.list_widget = QListWidget(self)
        self.list_widget.setMinimumWidth(200)

        # 使用QGraphicsView作为画布
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 600, 600)
        self.canvas_widget = MyCanvas(self.scene, self)
        self.canvas_widget.setFixedSize(600, 600)
        self.canvas_widget.main_window = self
        self.canvas_widget.list_widget = self.list_widget
        self.height = 0
        self.width = 0
        self.default_input_dir = None

        # self.rect = None
        # self.pixmap = None
        # self.painter = None
        # self.rectf = None

        # 设置菜单栏
        menubar = self.menuBar()
        file_menu = menubar.addMenu('文件')
        set_pen_act = file_menu.addAction('设置画笔')
        reset_canvas_act = file_menu.addAction('重置画布')
        save_canvas_act = file_menu.addAction('保存画布')
        exit_act = file_menu.addAction('退出')
        draw_menu = menubar.addMenu('绘制')
        line_menu = draw_menu.addMenu('线段')
        line_naive_act = line_menu.addAction('Naive')
        line_dda_act = line_menu.addAction('DDA')
        line_bresenham_act = line_menu.addAction('Bresenham')
        polygon_menu = draw_menu.addMenu('多边形')
        polygon_dda_act = polygon_menu.addAction('DDA')
        polygon_bresenham_act = polygon_menu.addAction('Bresenham')
        ellipse_act = draw_menu.addAction('椭圆')
        curve_menu = draw_menu.addMenu('曲线')
        curve_bezier_act = curve_menu.addAction('Bezier')
        curve_b_spline_act = curve_menu.addAction('B-spline')
        edit_menu = menubar.addMenu('编辑')
        translate_act = edit_menu.addAction('平移')
        rotate_act = edit_menu.addAction('旋转')
        scale_act = edit_menu.addAction('缩放')
        clip_menu = edit_menu.addMenu('裁剪')
        clip_cohen_sutherland_act = clip_menu.addAction('Cohen-Sutherland')
        clip_liang_barsky_act = clip_menu.addAction('Liang-Barsky')

        # 连接信号和槽函数
        set_pen_act.triggered.connect(self.set_pen_action)
        reset_canvas_act.triggered.connect(self.reset_canvas_action)
        save_canvas_act.triggered.connect(self.save_canvas_action)
        exit_act.triggered.connect(qApp.quit)
        line_naive_act.triggered.connect(self.line_naive_action)
        line_dda_act.triggered.connect(self.line_dda_action)
        line_bresenham_act.triggered.connect(self.line_bresenham_action)
        polygon_dda_act.triggered.connect(self.polygon_dda_action)
        polygon_bresenham_act.triggered.connect(self.polygon_bresenham_action)
        ellipse_act.triggered.connect(self.ellipse_action)
        curve_bezier_act.triggered.connect(self.curve_bezier_action)
        curve_b_spline_act.triggered.connect(self.curve_b_spline_action)
        translate_act.triggered.connect(self.translate_action)
        rotate_act.triggered.connect(self.rotate_action)
        scale_act.triggered.connect(self.scale_action)
        clip_cohen_sutherland_act.triggered.connect(self.clip_cohen_sutherland_action)
        clip_liang_barsky_act.triggered.connect(self.clip_liang_barsky_action)

        self.list_widget.currentTextChanged.connect(self.canvas_widget.selection_changed)

        # 设置主窗口的布局
        self.hbox_layout = QHBoxLayout()
        self.hbox_layout.addWidget(self.canvas_widget)
        self.hbox_layout.addWidget(self.list_widget, stretch=1)
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.hbox_layout)
        self.setCentralWidget(self.central_widget)
        self.statusBar().showMessage('空闲')
        self.resize(600, 600)
        self.setWindowTitle('CG Demo')

    def nxt_id(self):
        self.item_cnt += 1
        _id = str(self.item_cnt)
        return _id

    def get_id(self):
        _id = str(self.item_cnt)
        return _id

    def set_pen_action(self):
        color = QColorDialog.getColor()
        self.canvas_widget.temp_color = color

    def reset_canvas_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_widget.finish_draw()
        self.list_widget.clearSelection()
        self.list_widget.clear()
        self.canvas_widget.clear_selection()
        self.canvas_widget.item_dict.clear()
        self.canvas_widget.scene().clear()
        self.item_cnt = 0
        self.canvas_widget.status = 0
        self.canvas_widget.selected_id = ''
        self.width = QInputDialog.getInt(self, '重置画布尺寸', '输入宽度')[0]
        self.height = QInputDialog.getInt(self, '重置画布尺寸','输入高度')[0]
        self.scene.setSceneRect(0, 0, self.width, self.height)
        self.canvas_widget.setFixedSize(self.width, self.height)

    def save_canvas_action(self, painter: QPainter):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_widget.finish_draw()
        filePath, ok = QFileDialog.getSaveFileName(caption='保存画布', filter='PNG (*.png)')
        rect = self.scene.sceneRect()
        pixmap = QImage(int(rect.height()), int(rect.width()), QImage.Format_ARGB32_Premultiplied)
        painter = QPainter(pixmap)
        self.scene.render(painter)
        pixmap.save(filePath)

    def line_naive_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_widget.finish_draw()
        self.canvas_widget.start_draw_line('Naive', self.get_id())
        self.statusBar().showMessage('Naive算法绘制线段')
        self.list_widget.clearSelection()
        self.canvas_widget.clear_selection()

    def line_dda_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_widget.finish_draw()
        self.canvas_widget.start_draw_line('DDA', self.get_id())
        self.statusBar().showMessage('DDA算法绘制线段')
        self.list_widget.clearSelection()
        self.canvas_widget.clear_selection()

    def line_bresenham_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_widget.finish_draw()
        self.canvas_widget.start_draw_line('Bresenham', self.get_id())
        self.statusBar().showMessage('Bresenham算法绘制线段')
        self.list_widget.clearSelection()
        self.canvas_widget.clear_selection()

    def polygon_dda_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_widget.finish_draw()
        self.canvas_widget.start_draw_polygon('DDA', self.get_id())
        self.statusBar().showMessage('DDA算法绘制多边形')
        self.list_widget.clearSelection()
        self.canvas_widget.clear_selection()

    def polygon_bresenham_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_widget.finish_draw()
        self.canvas_widget.start_draw_polygon('Bresenham', self.get_id())
        self.statusBar().showMessage('Bresenham算法绘制多边形')
        self.list_widget.clearSelection()
        self.canvas_widget.clear_selection()

    def ellipse_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_widget.finish_draw()
        self.canvas_widget.start_draw_ellipse(self.get_id())
        self.statusBar().showMessage('中点圆算法绘制椭圆')
        self.list_widget.clearSelection()
        self.canvas_widget.clear_selection()

    def curve_bezier_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_widget.finish_draw()
        self.canvas_widget.start_draw_curve('Bezier', self.get_id())
        self.statusBar().showMessage('Bezier曲线')
        self.list_widget.clearSelection()
        self.canvas_widget.clear_selection()

    def curve_b_spline_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_widget.finish_draw()
        self.canvas_widget.start_draw_curve('B-spline', self.get_id())
        self.statusBar().showMessage('B-spline曲线')
        self.list_widget.clearSelection()
        self.canvas_widget.clear_selection()

    def translate_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve' :
            self.canvas_widget.finish_draw()
        self.canvas_widget.start_translate()
        self.statusBar().showMessage('平移')

    def rotate_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_widget.finish_draw()
        self.canvas_widget.start_rotate()
        self.statusBar().showMessage('旋转')

    def scale_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_widget.finish_draw()
        self.canvas_widget.start_scale()
        self.statusBar().showMessage('缩放')

    def clip_cohen_sutherland_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_wdget.finish_draw()
        self.canvas_widget.start_clip('Cohen-Sutherland')
        self.statusBar().showMessage('Cohen-Sutherland算法裁剪线段')

    def clip_liang_barsky_action(self):
        if self.canvas_widget.status == 'polygon' or self.canvas_widget.status == 'curve':
            self.canvas_widget.finish_draw()
        self.canvas_widget.start_clip('Liang-Barsky')
        self.statusBar().showMessage('Liang-Barsky算法裁剪线段')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
