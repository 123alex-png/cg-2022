#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 本文件只允许依赖math库
import math


def draw_line(p_list, algorithm):
    """绘制线段

    :param p_list: (list of list of int: [[x0, y0], [x1, y1]]) 线段的起点和终点坐标
    :param algorithm: (string) 绘制使用的算法，包括'DDA'和'Bresenham'，此处的'Naive'仅作为示例，测试时不会出现
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 绘制结果的像素点坐标列表
    """
    x0, y0 = p_list[0]
    x1, y1 = p_list[1]
    result = []
    if algorithm == 'Naive':
        if x0 == x1:
            for y in range(y0, y1 + 1):
                result.append((x0, y))
        else:
            if x0 > x1:
                x0, y0, x1, y1 = x1, y1, x0, y0
            k = (y1 - y0) / (x1 - x0)
            for x in range(x0, x1 + 1):
                result.append((x, int(y0 + k * (x - x0))))
    elif algorithm == 'DDA':
        if x0 == x1:
            if y0 > y1:
                y0, y1 = y1, y0
            for y in range(y0, y1 + 1):
                result.append((x0, y))
        else:
            k = (y1 - y0) / (x1 - x0)
            if abs(k) <= 1:
                if x0 > x1:
                    x0, y0, x1, y1 = x1, y1, x0, y0
                y = y0
                for x in range(x0, x1 + 1):
                    result.append((x, int(y + 0.5)))
                    y += k
            else:
                if y0 > y1:
                    x0, y0, x1, y1 = x1, y1, x0, y0
                x = x0
                for y in range(y0, y1 + 1):
                    result.append((int(x + 0.5), y))
                    x += 1/k
    # elif algorithm == 'Bresenham':
    #     if x0 == x1:
    #         if y0 > y1:
    #             y0, y1 = y1, y0
    #         for y in range(y0, y1 + 1):
    #             result.append((x0, y))
    #     else:
    #         steep = abs(y1 - y0) > abs(x1 - x0)
    #         if steep:
    #             x0, y0 = y0, x0
    #             x1, y1 = y1, x1
    #         if x0 > x1:
    #             x0, x1 = x1, x0
    #             y0, y1 = y1, y0
    #         delta_x = x1 - x0
    #         delta_y = abs(y1 - y0)
    #         error = 0
    #         delta_error = delta_y / delta_x
    #         y = y0
    #         if y0 < y1:
    #             y_step = 1
    #         else:
    #             y_step = -1
    #         for x in range(x0, x1 + 1):
    #             if steep:
    #                 result.append((y, x))
    #             else:
    #                 result.append((x, y))
    #             error += delta_error
    #             if error >= 0.5:
    #                y += y_step
    #                error -= 1.0
    elif algorithm == 'Bresenham':
        if x0 == x1:
            if y0 > y1:
                y0, y1 = y1, y0
            for y in range(y0, y1 + 1):
                result.append((x0, y))
        else:
            steep = abs(y1 - y0) > abs(x1 - x0)
            if steep:
                x0, y0 = y0, x0
                x1, y1 = y1, x1
            if x0 > x1:
                x0, x1 = x1, x0
                y0, y1 = y1, y0
            delta_x = x1 - x0
            delta_y = y1 - y0
            k = delta_y / delta_x
            b = y0 - k * x0
            c = 2 * delta_y + delta_x * (2 * b - 1)
            p = int(2 * delta_y * x0 - 2 * delta_x * y0 + c)
            y = y0
            if y0 < y1:
                y_step = 1
            else:
                y_step = -1
            for x in range(x0, x1 + 1):
                if steep:
                    result.append((y, x))
                else:
                    result.append((x, y))
                if p > 0:
                    p += 2 * (abs(delta_y) - delta_x)
                else:
                    p += 2 * abs(delta_y)
                if p > 0:
                    y += y_step
    return result


def draw_polygon(p_list, algorithm):
    """绘制多边形

    :param p_list: (list of list of int: [[x0, y0], [x1, y1], [x2, y2], ...]) 多边形的顶点坐标列表
    :param algorithm: (string) 绘制使用的算法，包括'DDA'和'Bresenham'
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 绘制结果的像素点坐标列表
    """
    result = []
    for i in range(len(p_list)):
        line = draw_line([p_list[i - 1], p_list[i]], algorithm)
        result += line
    return result

def draw_ellipse(p_list):
    """绘制椭圆（采用中点圆生成算法）

    :param p_list: (list of list of int: [[x0, y0], [x1, y1]]) 椭圆的矩形包围框左上角和右下角顶点坐标
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 绘制结果的像素点坐标列表
    """
    x0, y0 = p_list[0]
    x1, y1 = p_list[1]
    result = []
    s, t = (x0 + x1)//2, (y0 + y1)//2
    a, b = abs(x1 - x0)//2, abs(y1 - y0)//2
    #上半部分
    result.extend([(s, t + b), (s, t - b)])
    x, y = 0, b
    J = (1/4 - b) * a * a + b * b
    while b * b * x < a * a * y:
        if J <= 0:
            J += b * b * (2 * x+ 3)
            x += 1
        else:
            J += b * b * (2 * x + 3) + a * a * (2 - 2 * y)
            x += 1
            y -= 1
        result.extend([(s + x, t + y), (s - x, t + y),
                       (s + x, t - y), (s - x, t - y)])
    #下半部分
    result.extend([(s + a, t), (s - a, t)])
    x, y = a, 0
    K = (1 / 4 - a) * b * b + a * a
    while b * b * x > a * a * y:
        if K <= 0:
            K += a * a * (2 * y + 3)
            y += 1
        else:
            K += a * a * (2 * y + 3) + b * b * (2 - 2 * x)
            y += 1
            x -= 1
        result.extend([(s + x, t + y), (s - x, t + y),
                       (s + x, t - y), (s - x, t - y)])
    return result

class Bspline:
    result = []
    node_vector = []
    def Bspline_basis(self, k, d, u):
        if d == 1:
            if u > self.node_vector[k] and u <= self.node_vector[k+1]:
                ret = 1
            else:
                ret = 0
        else:
            length1 = self.node_vector[k + d - 1] - self.node_vector[k]
            length2 = self.node_vector[k + d] - self.node_vector[k + 1]
            if length1 == 0:
                alpha = 0
            else:
                alpha = (u - self.node_vector[k]) / length1
            if length2 == 0:
                beta = 0
            else:
                beta = (self.node_vector[k + d] - u) / length2
            t1 = self.Bspline_basis(k, d - 1, u)
            t2 = self.Bspline_basis(k + 1, d - 1, u)
            ret = alpha * t1 + beta * t2
        return ret
    
    def solve(self, p_list):
        n = len(p_list)
        if n <= 3:
            return self.result
        point_num = 2000
        rx = [0 for i in range(point_num)]
        ry = [0 for i in range(point_num)]
        basis = [0 for i in range(point_num)]
        for i in range(n + point_num + 2):
            self.node_vector.append(i)
        for i in range(3, n):
            cur = self.node_vector[3]
            U = []
            for j in range(point_num):
                U.append(cur)
                cur += (self.node_vector[n] - self.node_vector[3])/(point_num)
            j = 0
            for u in U:
                basis[j] = self.Bspline_basis(i, 4, u)
                j += 1
            for j in range(point_num):
                rx[j] += p_list[i][0] * basis[j]
                ry[j] += p_list[i][1] * basis[j]
        self.result = [[int(rx[i]+0.5), int(ry[i]+0.5)] for i in range(point_num)]

        return self.result

def draw_curve(p_list, algorithm):
    """绘制曲线

    :param p_list: (list of list of int: [[x0, y0], [x1, y1], [x2, y2], ...]) 曲线的控制点坐标列表
    :param algorithm: (string) 绘制使用的算法，包括'Bezier'和'B-spline'（三次均匀B样条曲线，曲线不必经过首末控制点）
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 绘制结果的像素点坐标列表
    """
    result = []
    if algorithm == 'Bezier':
        n = len(p_list) - 1
        dk = 0.001
        k = 0
        while k <= 1:
            task_list = p_list.copy()
            for i in range(n):
                tmp = []
                for j in range(len(task_list) - 1):
                    x1, y1 = task_list[j]
                    x2, y2 = task_list[j+1]
                    tmp.append([(1 - k) * x1 + k * x2, (1 - k) * y1 + k * y2])
                task_list = tmp.copy()
            result.append([int(task_list[0][0] + 0.5), int(task_list[0][1] + 0.5)])
            k += dk
    if algorithm == 'B-spline':
        dt = 0.001
        t = 0
        n = len(p_list)

        while t <= 1:
            coef = [1 - 3 * t + 3 * t ** 2 - t ** 3, 4 - 6 * t ** 2 + 3 * t ** 3, 1 + 3 * t + 3 * t ** 2 - 3 * t ** 3,
                    t ** 3]
            for i in range(n - 3):
                s_x, s_y = 0, 0
                for j in range(0, 4):
                    s_x += coef[j] * p_list[i + j][0]
                    s_y += coef[j] * p_list[i + j][1]
                s_x /= 6
                s_y /= 6
                result.append([int(s_x + 0.5), int(s_y + 0.5)])
            t += dt
    return result

def translate(p_list, dx, dy):
    """平移变换

    :param p_list: (list of list of int: [[x0, y0], [x1, y1], [x2, y2], ...]) 图元参数
    :param dx: (int) 水平方向平移量
    :param dy: (int) 垂直方向平移量
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 变换后的图元参数
    """
    result = []
    for x, y in p_list:
        x += dx
        y += dy
        result.append((x, y))
    return result


def rotate(p_list, x, y, r):
    """旋转变换（除椭圆外）

    :param p_list: (list of list of int: [[x0, y0], [x1, y1], [x2, y2], ...]) 图元参数
    :param x: (int) 旋转中心x坐标
    :param y: (int) 旋转中心y坐标
    :param r: (int) 顺时针旋转角度（°）
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 变换后的图元参数
    """
    result = []
    t = r / 180 * math.pi
    for p in p_list:
        a = (p[0] - x) * math.cos(t) - (p[1] - y) * math.sin(t)
        b = (p[0] - x) * math.sin(t) + (p[1] - y) * math.cos(t)
        result.append((int(a + 0.5) + x, int(b + 0.5) + y))
    return result


def scale(p_list, x, y, s):
    """缩放变换

    :param p_list: (list of list of int: [[x0, y0], [x1, y1], [x2, y2], ...]) 图元参数
    :param x: (int) 缩放中心x坐标
    :param y: (int) 缩放中心y坐标
    :param s: (float) 缩放倍数
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 变换后的图元参数
    """
    result = []
    for p in p_list:
        a = (p[0] - x) * s
        b = (p[1] - y) * s
        result.append((int(a + 0.5) + x, int(b + 0.5) + y))
    return result



# def Cohen_Sutherland(p_list, x_min, y_min, x_max, y_max)
class Cohen_Sutherland:
    result = []

    def Cohen_code(self, x, y, x_min, y_min, x_max, y_max):
        ret = 0
        if y > y_max:
            ret |= 8
        if y < y_min:
            ret |= 4
        if x > x_max:
            ret |= 2
        if x < x_min:
            ret |= 1
        return ret

    def on_line(self, p_list, x, y):
        x0, y0 = p_list[0]
        x1, y1 = p_list[1]
        return (x - x0) * (x - x1) < 0

    def interset(self, p_list, x_min, y_min, x_max, y_max):
        ret = []
        x0, y0 = p_list[0]
        x1, y1 = p_list[1]
        if x0 ==x1:
            if (y_min - y0) * (y_min - y1) < 0:
                return [x0, y_min]
            else:
                return [x0, y_max]
        else:
            k = (y1 - y0) / (x1 - x0)
            b = y0 - k * x0
            tmp_x, tmp_y = int(x_min), int(k * x_min + b)
            if self.on_line(p_list, tmp_x, tmp_y) and tmp_y >= y_min and tmp_y <= y_max:
                return [tmp_x, tmp_y]
            tmp_x, tmp_y = int(x_max), int(k * x_max + b)
            if self.on_line(p_list, tmp_x, tmp_y) and tmp_y >= y_min and tmp_y <= y_max:
                return [tmp_x, tmp_y]
            tmp_x, tmp_y = int((y_min - b) / k), int(y_min)
            if self.on_line(p_list, tmp_x, tmp_y) and tmp_x >= x_min and tmp_x <= x_max:
                return [tmp_x, tmp_y]
            tmp_x, tmp_y = int((y_max - b) / k), int(y_max)
            if self.on_line(p_list, tmp_x, tmp_y) and tmp_x >= x_min and tmp_x <= x_max:
                return [tmp_x, tmp_y]
            return ret
    def func(self, p_list, x_min, y_min, x_max, y_max):
        x0, y0 = p_list[0][0], p_list[0][1]
        x1, y1 = p_list[1][0], p_list[1][1]
        code0 = self.Cohen_code(x0, y0, x_min, y_min, x_max, y_max)
        code1 = self.Cohen_code(x1, y1, x_min, y_min, x_max, y_max)
        if code0 == 0 and code1 == 0:
            self.result = p_list
            return
        else:
            t = code0 & code1
            if t != 0:
                return
            else:
                intersec = self.interset(p_list, x_min, y_min, x_max, y_max)
                if len(intersec) == 0:
                    return
                else:
                    p_list1 = [p_list[0], intersec]
                    p_list2 = [intersec, p_list[1]]
                    self.func(p_list1, x_min, y_min, x_max, y_max)
                    self.func(p_list2, x_min, y_min, x_max, y_max)
    def solve(self, p_list, x_min, y_min, x_max, y_max):
        self.func(p_list, x_min, y_min, x_max, y_max)
        return self.result

class Liang_Barsky:
    result = []
    x0, y0, x1, y1 = 0, 0, 0, 0
    x_min, y_min, x_max, y_max = 0, 0, 0, 0

    def param_equation(self, u):
        x = self.x0 + u * (self.x1 - self.x0)
        y = self.y0 + u * (self.y1 - self.y0)
        return [int(x+0.5), int(y+0.5)]

    def func(self):
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        p = [-dx, dx, -dy, dy]
        q = [self.x0 - self.x_min, self.x_max - self.x0, self.y0 - self.y_min, self.y_max - self.y0]
        u1, u2 = 0, 1
        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return []
            elif p[i] < 0:
                u1 = max(u1, q[i] / p[i])
            else:
                u2 = min(u2, q[i] / p[i])
            if u1 > u2:
                return []
        self.result = [self.param_equation(u1), self.param_equation(u2)]

    def solve(self, p_list, x_min, y_min, x_max, y_max):
        self.x0, self.y0 = p_list[0]
        self.x1, self.y1 = p_list[1]
        self.x_min, self.y_min, self.x_max, self.y_max =  x_min, y_min, x_max, y_max
        self.func()
        return self.result

def clip(p_list, x_min, y_min, x_max, y_max, algorithm):
    """线段裁剪

    :param p_list: (list of list of int: [[x0, y0], [x1, y1]]) 线段的起点和终点坐标
    :param x_min: 裁剪窗口左上角x坐标
    :param y_min: 裁剪窗口左上角y坐标
    :param x_max: 裁剪窗口右下角x坐标
    :param y_max: 裁剪窗口右下角y坐标
    :param algorithm: (string) 使用的裁剪算法，包括'Cohen-Sutherland'和'Liang-Barsky'
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1]]) 裁剪后线段的起点和终点坐标
    """
    if len(p_list) != 2:
        return p_list
    if algorithm == 'Cohen-Sutherland':
        return Cohen_Sutherland().solve(p_list, x_min, y_min, x_max, y_max)
    if algorithm == 'Liang-Barsky':
        return Liang_Barsky().solve(p_list, x_min, y_min, x_max, y_max)

