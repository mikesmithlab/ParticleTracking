from ParticleTracking import correlations
from PyQt5.QtWidgets import (QMainWindow, QApplication, QHBoxLayout, QWidget,
                             QVBoxLayout, QSlider, QSizePolicy, QFileDialog,
                             QLabel, QComboBox)
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, \
    NavigationToolbar2QT
from matplotlib.figure import Figure

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

        vbox.addLayout(self.setup_slider_box())
        hbox = QHBoxLayout()
        g_box = self.create_g_box()
        g6_box = self.create_g6_box()
        hbox.addLayout(g_box)
        hbox.addLayout(g6_box)
        vbox.addLayout(hbox)
        self.setCentralWidget(self.main_widget)

    def setup_slider_box(self):
        label = QLabel(self.main_widget)
        label.setText('Duty')

        hbox = QHBoxLayout()
        duty_slider = QSlider(Qt.Horizontal)
        duty_slider.setRange(0, len(self.data.duty))

        duty_label = QLabel(self.main_widget)
        duty_label.setText(str(self.data.duty[0]))

        def slider_changed(value):
            duty_label.setText(str(self.data.duty[value]))
            self.change_duty(value)

        duty_slider.valueChanged[int].connect(slider_changed)
        hbox.addWidget(label)
        hbox.addWidget(duty_slider)
        hbox.addWidget(duty_label)
        return hbox

    def create_g_box(self):
        self.g_graph = GGraph(self.main_widget)

        height_slider = hbox_slider(
            self.main_widget, 'line height', -1, 1, 10, 0,
            self.g_graph.change_line_height)

        projection_combo = combo_box(
            self.main_widget, 'projection',
            ['linear', 'logx', 'logy', 'loglog'],
            self.g_graph.change_projection)

        vbox = QVBoxLayout()
        vbox.addWidget(self.g_graph)
        vbox.addLayout(height_slider)
        vbox.addLayout(projection_combo)
        return vbox

    def create_g6_box(self):
        vbox = QVBoxLayout()
        self.g6_graph = G6Graph(self.main_widget)
        height_slider = hbox_slider(
            self.main_widget, 'line height', -1, 1, 10, 0,
            self.g6_graph.change_line_height)
        projection_combo = combo_box(
            self.main_widget, 'projection',
            ['linear', 'logx', 'logy', 'loglog'],
            self.g6_graph.change_projection)
        vbox.addWidget(self.g6_graph)
        vbox.addLayout(height_slider)
        vbox.addLayout(projection_combo)
        return vbox

    def change_duty(self, value):
        r, g, g6 = self.data.get(self.data.duty[value])
        self.g_graph.set_data(r, g, g6)
        self.g6_graph.set_data(r, g, g6)

    def initial_plot(self):
        r, g, g6 = self.data.get(self.data.duty[0])
        self.g_graph.plot(r, g, g6)
        self.g6_graph.plot(r, g, g6)


def combo_box(parent, label, items, function):
    lbl = QLabel(label, parent)
    combo = QComboBox(parent)
    combo.addItems(items)
    combo.activated[str].connect(function)

    hbox = QHBoxLayout()
    hbox.addWidget(lbl)
    hbox.addWidget(combo)
    hbox.addStretch(1)
    return hbox


def hbox_slider(parent, label, start, end, dpi, initial, function):
    start *= dpi
    end *= dpi
    initial *= dpi

    hbox = QHBoxLayout()
    lab = QLabel(parent)
    lab.setText(label)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(start, end)
    slider.setSliderPosition(initial)

    val_label = QLabel(parent)
    val_label.setText(str(initial / dpi))

    def slider_changed(value):
        val_label.setText(str(value / dpi))
        function(value / dpi)

    slider.valueChanged[int].connect(slider_changed)
    hbox.addWidget(lab)
    hbox.addWidget(slider)
    hbox.addWidget(val_label)
    return hbox


class Graph(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.type = type
        self.fig = Figure()

        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(self,
                                        QSizePolicy.Expanding,
                                        QSizePolicy.Expanding)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.setup_axes()

    def on_key_press(self, event):
        key_press_handler(event, self.canvas, self.toolbar)

    def setup_axes(self):
        self.ax = self.fig.add_subplot(111)

    def set_labels(self, xlabel, ylabel):
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def change_projection(self, key):
        if key == 'linear':
            self.ax.set_xscale('linear')
            self.ax.set_yscale('linear')
        elif key == 'logx':
            self.ax.set_xscale('log')
            self.ax.set_yscale('linear')
        elif key == 'logy':
            self.ax.set_xscale('linear')
            self.ax.set_yscale('log')
        else:
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
        self.draw()


class GGraph(Graph):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_labels('r', '$G(r)$')

    def plot(self, r, g, g6):
        self.r = r
        choice = {'G': g - 1, 'G6': g6 / g}
        self.line, = self.ax.plot(r, g - 1)
        self.power_line, = self.ax.plot(r, r ** (-1 / 3))

    def set_data(self, r, g, g6):
        self.r = r
        self.line.set_xdata(r)
        self.line.set_ydata(g - 1)
        self.draw()

    def change_line_height(self, value):
        self.power_line.set_ydata(self.r ** (-1 / 3) + value)
        self.draw()


class G6Graph(Graph):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_labels('r', '$G_6(r)$')

    def plot(self, r, g, g6):
        self.r = r
        self.line, = self.ax.plot(r, g6 / g)
        self.power_line, = self.ax.plot(r, r ** (-1 / 4))

    def set_data(self, r, g, g6):
        self.r = r
        self.line.set_xdata(r)
        self.line.set_ydata(g6 / g)
        self.draw()

    def change_line_height(self, value):
        self.power_line.set_ydata(self.r ** (-1 / 4) + value)
        self.draw()


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
