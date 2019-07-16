import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QApplication, QHBoxLayout, QWidget,
                             QVBoxLayout, QFileDialog)

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
        duty_slider = pyqt5_widgets.Slider(
            self.main_widget, 'Duty', self.duty_changed, 0,
            len(self.data.duty) - 1, 1, 0)
        # vbox.addLayout(self.setup_slider_box())
        vbox.addWidget(duty_slider)
        hbox = QHBoxLayout()
        g_box = self.create_g_box()
        g6_box = self.create_g6_box()
        hbox.addLayout(g_box)
        hbox.addLayout(g6_box)
        vbox.addLayout(hbox)
        self.setCentralWidget(self.main_widget)

    def duty_changed(self, value):
        duty = self.data.duty[int(value)]
        # self.change_duty(duty)
        r, g, g6 = self.data.get(duty)
        self.g_graph.set_data(r, g, g6)
        self.g6_graph.set_data(r, g, g6)

    def create_g_box(self):
        self.g_graph = GGraph(self.main_widget)

        height_slider = pyqt5_widgets.Slider(
            self.main_widget, 'line height', self.g_graph.set_offset,
            -1, 1, 100, 0)

        projection_combo = pyqt5_widgets.ComboBox(
            self.main_widget, 'projection',
            ['linear', 'logx', 'logy', 'loglog'],
            self.g_graph.change_projection)

        autoscale_checkbox = pyqt5_widgets.CheckBox(
            self.main_widget,
            'autoscale',
            self.g_graph.set_autoscale,
            'on')

        peak_finder = PeakFinder(self.main_widget, self.g_graph)

        vbox = QVBoxLayout()
        vbox.addWidget(self.g_graph)
        vbox.addWidget(height_slider)
        vbox.addWidget(projection_combo)
        vbox.addWidget(autoscale_checkbox)
        vbox.addWidget(peak_finder)
        return vbox

    def create_g6_box(self):
        self.g6_graph = G6Graph(self.main_widget)

        height_slider = pyqt5_widgets.Slider(
            self.main_widget, 'line height', self.g6_graph.set_offset,
            -1, 1, 100, 0)

        projection_combo = pyqt5_widgets.ComboBox(
            self.main_widget, 'projection',
            ['linear', 'logx', 'logy', 'loglog'],
            self.g6_graph.change_projection)

        autoscale_checkbox = pyqt5_widgets.CheckBox(
            self.main_widget,
            'autoscale',
            self.g6_graph.set_autoscale,
            'on')

        vbox = QVBoxLayout()
        vbox.addWidget(self.g6_graph)
        vbox.addWidget(height_slider)
        vbox.addWidget(projection_combo)
        vbox.addWidget(autoscale_checkbox)
        return vbox

    def initial_plot(self):
        r, g, g6 = self.data.get(self.data.duty[0])
        self.g_graph.set_data(r, g, g6)
        self.g6_graph.set_data(r, g, g6)


class PeakFinder(QWidget):
    def __init__(self, parent, graph):
        self.show_fit = False
        self.show_peaks = False
        QWidget.__init__(self, parent)
        self.setLayout(QVBoxLayout())
        self.create_widgets()

    def create_widgets(self):
        # checkboxes
        show_peaks = pyqt5_widgets.CheckBox(
            self,
            'show peaks',
            self.show_peaks_changed)
        show_fit = pyqt5_widgets.CheckBox(
            self,
            'show fit',
            self.show_fit_changed)
        hbox = QHBoxLayout()
        hbox.addWidget(show_peaks)
        hbox.addWidget(show_fit)
        self.layout().addLayout(hbox)

        # sliders
        height_slider = pyqt5_widgets.CheckedSlider(
            self, 'height', self.height_changed, 0, 50, 1, 0)
        threshold_slider = pyqt5_widgets.CheckedSlider(
            self, 'threshold', self.threshold_changed, 0, 1, 10, 0)

        self.layout().addWidget(height_slider)
        self.layout().addWidget(threshold_slider)

    def show_peaks_changed(self, state):
        self.show_peaks = True if state == Qt.Checked else False

    def show_fit_changed(self, state):
        self.show_fit = True if state == Qt.Checked else False

    def height_changed(self, value):
        self.height = value

    def threshold_changed(self, value):
        self.threshold = value


class Graph(pyqt5_widgets.MatplotlibFigure):
    def __init__(self, parent, power):
        self.power = power
        pyqt5_widgets.MatplotlibFigure.__init__(self, parent)
        self.setup_axes()
        self.initial_plot()
        self.setup_variables()

    def setup_axes(self):
        self.ax = self.fig.add_subplot(111)

    def initial_plot(self):
        self.line, = self.ax.plot([], [])
        self.power_line, = self.ax.plot([], [])
        self.draw()

    def setup_variables(self):
        self.offset = 0
        self.autoscale = True
        self.peaks_on = False

    def set_labels(self, xlabel, ylabel):
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.fig.tight_layout()

    def update(self):
        self.line.set_xdata(self.x)
        self.line.set_ydata(self.y)
        if self.autoscale:
            self.ax.relim()
            self.ax.autoscale_view()
        self.draw()
        self.update_power_line()

    def update_power_line(self):
        self.power_line.set_xdata(self.x)
        self.power_line.set_ydata(self.x ** self.power + self.offset)
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


class GGraph(Graph):
    def __init__(self, parent=None):
        super().__init__(parent, -1 / 3)
        self.set_labels('r', '$G(r)$')

    def set_data(self, r, g, g6):
        self.x = r
        self.y = g - 1
        self.update()


class G6Graph(Graph):
    def __init__(self, parent=None):
        super().__init__(parent, -1 / 4)
        self.set_labels('r', '$G_6(r)$')

    def set_data(self, r, g, g6):
        self.x = r
        self.y = g6 / g
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
        return r[0], g[0], g6[0]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    sys.exit(app.exec_())
