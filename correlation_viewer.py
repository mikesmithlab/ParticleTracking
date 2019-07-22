import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QApplication, QHBoxLayout, QWidget,
                             QVBoxLayout, QFileDialog, QLabel)
from scipy import signal, optimize

from Generic import pyqt5_widgets
from ParticleTracking import correlations


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.data = Data()
        self.setup_main_window()
        self.setup_main_widget()
        self.initial_plot()
        self.show()

    def setup_main_window(self):
        self.setWindowTitle(self.data.file)
        self.resize(1280, 720)

    def setup_main_widget(self):
        self.main_widget = QWidget(self)
        vbox = QVBoxLayout(self.main_widget)
        duty_slider = pyqt5_widgets.ArraySlider(
            self.main_widget, 'Duty', self.duty_changed, self.data.duty)
        vbox.addWidget(duty_slider)
        hbox = QHBoxLayout()
        self.g_graph = GGraph(self.main_widget)
        self.g6_graph = G6Graph(self.main_widget)
        g_box = GraphBox(self.main_widget, self.g_graph)
        g6_box = GraphBox(self.main_widget, self.g6_graph)
        hbox.addWidget(g_box)
        hbox.addWidget(g6_box)
        vbox.addLayout(hbox)
        self.setCentralWidget(self.main_widget)

    def duty_changed(self, duty):
        r, g, g6 = self.data.get(duty)
        self.g_graph.set_data(r, g, g6)
        self.g6_graph.set_data(r, g, g6)

    def initial_plot(self):
        r, g, g6 = self.data.get(self.data.duty[0])
        self.g_graph.set_data(r, g, g6)
        self.g6_graph.set_data(r, g, g6)


class GraphBox(QWidget):
    def __init__(self, parent, graph):
        QWidget.__init__(self, parent=parent)
        self.setLayout(QVBoxLayout())

        height_slider = pyqt5_widgets.CheckedSlider(
            parent, 'line height', graph.set_offset,
            start=-1, end=1, dpi=100, initial=0)

        projection_combo = pyqt5_widgets.ComboBox(
            parent, 'projection',
            ['linear', 'logx', 'logy', 'loglog'],
            graph.change_projection)

        autoscale_checkbox = pyqt5_widgets.CheckBox(
            parent,
            'autoscale',
            graph.set_autoscale,
            'on')

        peak_finder = PeakFinder(parent, graph)

        self.layout().addWidget(graph)
        self.layout().addWidget(height_slider)
        self.layout().addWidget(projection_combo)
        self.layout().addWidget(autoscale_checkbox)
        self.layout().addWidget(peak_finder)


class PeakFinder(QWidget):
    def __init__(self, parent, graph):
        self.graph = graph
        self.show_fit = False
        self.show_peaks = False
        self.height = None
        self.threshold = None
        self.distance = None
        QWidget.__init__(self, parent)
        self.setLayout(QVBoxLayout())
        self.create_widgets()

    def create_widgets(self):
        # checkboxes
        show_peaks = pyqt5_widgets.CheckBox(
            self,
            'show peaks',
            self.show_peaks_changed)
        show_exp_fit = pyqt5_widgets.CheckBox(
            self,
            'show exp fit',
            self.show_exp_fit_changed)
        show_power_fit = pyqt5_widgets.CheckBox(
            self,
            'show power fit',
            self.show_power_fit_changed)

        hbox = QHBoxLayout()
        hbox.addWidget(show_peaks)
        hbox.addWidget(show_exp_fit)
        hbox.addWidget(show_power_fit)
        self.layout().addLayout(hbox)

        # sliders
        height_slider = pyqt5_widgets.CheckedSlider(
            self, 'height', self.height_changed,
            start=-2, end=2, dpi=100)

        threshold_slider = pyqt5_widgets.CheckedSlider(
            self, 'threshold', self.threshold_changed,
            start=0, end=1, dpi=100)

        distance_slider = pyqt5_widgets.CheckedSlider(
            self, 'distance', self.distance_changed,
            start=1, end=50, dpi=1, initial=1)

        self.layout().addWidget(height_slider)
        self.layout().addWidget(threshold_slider)
        self.layout().addWidget(distance_slider)

    def show_peaks_changed(self, state):
        show = True if state == Qt.Checked else False
        self.graph.show_peaks = show
        self.update_peaks()

    def show_exp_fit_changed(self, state):
        show = True if state == Qt.Checked else False
        self.graph.show_exp_fit = show
        self.update_peaks()

    def show_power_fit_changed(self, state):
        show = True if state == Qt.Checked else False
        self.graph.show_power_fit = show
        self.update_peaks()

    def height_changed(self, value):
        self.graph.peak_height = value
        self.update_peaks()

    def threshold_changed(self, value):
        self.graph.peak_threshold = value
        self.update_peaks()

    def distance_changed(self, value):
        self.graph.peak_distance = value
        self.update_peaks()

    def update_peaks(self):
        self.graph.update_peaks()


class Graph(pyqt5_widgets.MatplotlibFigure):
    def __init__(self, parent, power):
        self.power = power
        pyqt5_widgets.MatplotlibFigure.__init__(self, parent)
        self.setup_axes()
        self.initial_plot()
        self.setup_variables()
        self.add_fit_labels()

    def setup_axes(self):
        self.ax = self.fig.add_subplot(111)

    def initial_plot(self):
        self.line, = self.ax.plot([], [], label='data')
        self.power_line, = self.ax.plot([], [], label='power')
        self.peak_handle, = self.ax.plot([], [], 'r.', label='peaks')
        self.exp_fit_handle, = self.ax.plot([], [], 'r-', label='exp fit')
        self.power_fit_handle, = self.ax.plot([], [], 'g-', label='power fit')
        self.draw()

    def setup_variables(self):
        self.offset = None
        self.autoscale = True

        self.show_peaks = False
        self.show_power = False
        self.show_exp_fit = False
        self.show_power_fit = False

        self.peak_distance = None
        self.peak_threshold = None
        self.peak_height = None

    def set_labels(self, xlabel, ylabel):
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.fig.tight_layout()

    def update(self):
        self.line.set_xdata(self.xdata)
        self.line.set_ydata(self.ydata)
        if self.autoscale:
            self.ax.relim()
            self.ax.autoscale_view()
        self.draw()
        self.update_power_line()
        self.update_peaks()

    def update_power_line(self):
        if self.offset is None:
            self.power_line.set_xdata([])
            self.power_line.set_ydata([])
        else:
            self.power_line.set_xdata(self.xdata)
            self.power_line.set_ydata(self.xdata ** self.power + self.offset)
        self.draw()

    def set_offset(self, offset):
        self.offset = offset
        self.update_power_line()

    def change_projection(self, choice):
        if choice == 'linear':
            self.ax.set_xscale('linear')
            self.ax.set_yscale('linear')
        elif choice == 'logx':
            self.ax.set_xscale('log')
            self.ax.set_yscale('linear')
        elif choice == 'logy':
            self.ax.set_xscale('linear')
            self.ax.set_yscale('log')
        else:
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
        self.draw()

    def set_autoscale(self, state):
        if state == Qt.Checked:
            self.autoscale = True
        else:
            self.autoscale = False

    def update_peaks(self):
        peaks, props = signal.find_peaks(
            self.ydata,
            height=self.peak_height,
            threshold=self.peak_threshold,
            distance=self.peak_distance)
        if self.show_peaks:
            self.peak_handle.set_ydata(self.ydata[peaks])
            self.peak_handle.set_xdata(self.xdata[peaks])
        self.draw()

        if self.show_exp_fit:
            self.update_exp_fit(self.xdata[peaks], self.ydata[peaks])
        else:
            self.exp_fit_handle.set_xdata([])
            self.exp_fit_handle.set_ydata([])
            self.draw()

        if self.show_power_fit:
            self.update_power_fit(self.xdata[peaks], self.ydata[peaks])
        else:
            self.power_fit_handle.set_xdata([])
            self.power_fit_handle.set_ydata([])
            self.draw()

    def update_exp_fit(self, x, y):
        try:
            popt, pcov = optimize.curve_fit(self.exp, x, y, p0=(1, 25, -1))
            self.update_exp_label(*popt)
            yfit = self.exp(self.xdata, *popt)
            self.exp_fit_handle.set_xdata(self.xdata)
            self.exp_fit_handle.set_ydata(yfit)
        except:
            self.exp_fit_handle.set_xdata([])
            self.exp_fit_handle.set_ydata([])
            self.update_exp_label(None, None, None)
        self.draw()

    @staticmethod
    def exp(x, a, b, c):
        return a * np.exp(-x / b) + c

    def update_power_fit(self, x, y):
        try:
            popt, pcov = optimize.curve_fit(self.power_eq, x, y,
                                            p0=(1, -(1 / 3), -1))
            self.update_power_label(*popt)
            yfit = self.power_eq(self.xdata, *popt)
            self.power_fit_handle.set_xdata(self.xdata)
            self.power_fit_handle.set_ydata(yfit)
        except:
            self.exp_fit_handle.set_xdata([])
            self.exp_fit_handle.set_ydata([])
            self.update_power_label(None, None, None)
        self.draw()

    def power_eq(self, x, a, b, c):
        return a * x ** b + c

    def add_fit_labels(self):
        self.exp_label = QLabel(self)
        self.power_label = QLabel(self)
        self.layout().addWidget(self.exp_label)
        self.layout().addWidget(self.power_label)

    def update_exp_label(self, a, b, c):
        if a is not None:
            string = "exp fit: a * exp(-x/b) + c, a = {:.2f}, b = {:.2f}, c = {:.2f}".format(
                a, b, c)
            self.exp_label.setText(string)
        else:
            self.exp_label.setText('')

    def update_power_label(self, a, b, c):
        if a is not None:
            string = "power fit: a * x **b + c, a = {:.2f}, b = {:.2f}, c = {:.2f}".format(
                a, b, c)
            self.power_label.setText(string)
        else:
            self.power_label.setText('')


class GGraph(Graph):
    def __init__(self, parent=None):
        super().__init__(parent, -1 / 3)
        self.set_labels('r', '$G(r)$')

    def set_data(self, r, g, g6):
        self.xdata = r
        self.ydata = g - 1
        self.update()


class G6Graph(Graph):
    def __init__(self, parent=None):
        super().__init__(parent, -1 / 4)
        self.set_labels('r', '$G_6(r)$')

    def set_data(self, r, g, g6):
        self.xdata = r
        self.ydata = g6 / g
        self.update()


class Data:

    def __init__(self):
        self.open(None)

    def open(self, event):
        self.file = QFileDialog.getOpenFileName()[0]
        self.df = correlations.load_corr_data(self.file)
        self.duty = self.df.d.values

    def get(self, d):
        data = self.df.loc[self.df.d == d, ['r', 'g', 'g6']].values
        r, g, g6 = data.T
        r = r[0]
        g = g[0]
        g6 = g6[0]
        g6 = np.real(g6)
        return r, g, g6

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    sys.exit(app.exec_())
