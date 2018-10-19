import sys
import os
from PyQt5.QtWidgets import (
    QAction, QWidget, QLabel, QDesktopWidget,
    QApplication, QComboBox, QPushButton, QGridLayout,
    QMainWindow, qApp, QVBoxLayout, QSlider,
    QHBoxLayout, QLineEdit, QListView, QAbstractItemView
    )
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtGui
import Generic.video as vid
import cv2
import preprocessing as pp
import configuration as con
import particle_tracking as pt
import numpy as np
import pyperclip


class MainWindow(QMainWindow):

    def __init__(self, video):
        super().__init__()
        self.video = video
        self.frame = self.video.read_next_frame()
        cv2.imwrite('frame.png', self.frame)
        self.methods = con.GLASS_BEAD_PROCESS_LIST
        self.options = con.GLASS_BEAD_OPTIONS_DICT
        self.pp = pp.ImagePreprocessor(self.video, self.methods, self.options)
        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.statusBar().showMessage('Ready')
        self.resize(1280, 720)
        self.center()
        self.setWindowTitle('Particle Tracker')
        self.show()

        self.layout = QGridLayout(self.central_widget)

        self.setup_file_menu()
        self.create_main_image()
        self.create_main_buttons()
        self.create_process_options()
        self.create_track_options()

        self.add_widgets_to_layout()

    def create_process_options(self):
        self.process_options_layout = QVBoxLayout()
        self.create_tray_choice_combo()
        self.process_options_layout.addWidget(self.tray_choice_combo)

        self.create_grayscale_threshold_slider()
        self.process_options_layout.addWidget(self.grayscale_label)
        self.process_options_layout.addWidget(self.grayscale_threshold_slider)

        self.create_blur_kernel_slider()
        self.process_options_layout.addWidget(self.blur_kernel_label)
        self.process_options_layout.addWidget(self.blur_kernel_slider)

        self.create_adaptive_block_size_slider()
        self.process_options_layout.addWidget(self.adaptive_block_size_label)
        self.process_options_layout.addWidget(self.adaptive_block_size_slider)

        self.create_adaptive_constant_slider()
        self.process_options_layout.addWidget(self.adaptive_constant_label)
        self.process_options_layout.addWidget(self.adaptive_constant_slider)

        self.create_methods_list()
        self.process_options_layout.addWidget(self.methods_list)

        self.create_save_config_button()
        self.process_options_layout.addWidget(self.save_config_button)

        self.process_options_layout.addStretch()

    def create_save_config_button(self):
        self.save_config_button = QPushButton("Save config to clipboard")
        self.save_config_button.clicked.connect(self.save_config_button_clicked)

    def save_config_button_clicked(self):
        pyperclip.copy(str(self.options))

    def create_methods_list(self):
        self.methods_list = QListView()
        self.methods_list.setDragDropMode(QListView.InternalMove)
        self.methods_list.setDefaultDropAction(Qt.MoveAction)
        self.methods_list.setDragDropMode(False)
        self.methods_list.setAcceptDrops(True)
        self.methods_list.setDropIndicatorShown(True)
        self.methods_list.setDragEnabled(True)
        self.methods_list.setWindowTitle('Method Order')

        self.methods_model = QtGui.QStandardItemModel(self.methods_list)
        for method in self.methods:
            item = QtGui.QStandardItem(method)
            item.setData(method)
            item.setCheckable(True)
            item.setDragEnabled(True)
            item.setDropEnabled(False)
            item.setCheckState(2)

            self.methods_model.appendRow(item)

        self.methods_list.setModel(self.methods_model)
        self.methods_list.setFixedHeight(
            self.methods_list.sizeHintForRow(0)
            * (self.methods_model.rowCount() + 1))

    def check_method_list(self):
        new_methods = []
        for i in range(self.methods_model.rowCount()):
            it = self.methods_model.item(i)
            if it is not None and it.checkState() == Qt.Checked:
                new_methods.append(it.text())
        self.methods = new_methods

    def create_adaptive_constant_slider(self):
        self.adaptive_constant_label = QLabel()
        self.adaptive_constant_label.setText(
            'Adaptive threshold constant: '
            + str(self.options['adaptive threshold C']))
        self.adaptive_constant_slider = QSlider(Qt.Horizontal)
        self.adaptive_constant_slider.setRange(-20, 20)
        self.adaptive_constant_slider.\
            setValue(self.options['adaptive threshold C'])
        self.adaptive_constant_slider.\
            valueChanged[int].connect(self.adaptive_constant_slider_changed)

    def adaptive_constant_slider_changed(self, val):
        self.options['adaptive threshold C'] = val
        self.adaptive_constant_label.setText(
            'Adaptive threshold constant: '
            + str(self.options['adaptive threshold C']))

    def create_adaptive_block_size_slider(self):
        self.adaptive_block_size_label = QLabel()
        self.adaptive_block_size_label.setText(
            'Adaptive Threshold kernel size: '
            + str(self.options['adaptive threshold block size']))

        self.adaptive_block_size_slider = QSlider(Qt.Horizontal)
        self.adaptive_block_size_slider.setRange(1, 20)
        self.adaptive_block_size_slider.setValue(
            self.options['adaptive threshold block size'])
        self.adaptive_block_size_slider.valueChanged[int].\
            connect(self.adaptive_block_size_slider_changed)

    def adaptive_block_size_slider_changed(self, val):
        self.options['adaptive threshold block size'] = val*2+1
        self.adaptive_block_size_label.setText(
            'Adaptive Threshold kernel size: '
            + str(val*2+1))

    def create_blur_kernel_slider(self):
        self.blur_kernel_label = QLabel()
        self.blur_kernel_label.setText('Blur kernel size: ' +
                                       str(self.options['blur kernel'][0]))
        self.blur_kernel_slider = QSlider(Qt.Horizontal)
        self.blur_kernel_slider.setRange(0, 5)
        self.blur_kernel_slider.setValue((self.options['blur kernel'][0]-1)/2)
        self.blur_kernel_slider.valueChanged[int].\
            connect(self.blur_kernel_slider_changed)

    def blur_kernel_slider_changed(self, val):
        self.options['blur kernel'] = (val*2+1, val*2+1)
        self.blur_kernel_label.setText('Blur kernel size: ' + str(val*2+1))

    def create_grayscale_threshold_slider(self):
        self.grayscale_label = QLabel()
        self.grayscale_label.setText('Grayscale Threshold: ' +
                                     str(self.options['grayscale threshold']))
        self.grayscale_threshold_slider = QSlider(Qt.Horizontal)
        self.grayscale_threshold_slider.setRange(0, 255)
        self.grayscale_threshold_slider.\
            setValue(self.options['grayscale threshold'])
        self.grayscale_threshold_slider.valueChanged[int].\
            connect(self.grayscale_threshold_slider_changed)

    def grayscale_threshold_slider_changed(self, val):
        self.options['grayscale threshold'] = val
        self.grayscale_label.setText('Grayscale Threshold: ' + str(val))

    def create_tray_choice_combo(self):
        self.tray_choice_combo = QComboBox(self.central_widget)
        tray_choices = 'Circular', 'Hexagonal', 'Square'
        self.tray_choice_combo.addItems(tray_choices)
        self.tray_choice_combo.activated[str].connect(self.tray_choice_changed)

    def tray_choice_changed(self, text):
        tray = {'Circular': 1, 'Hexagonal': 6, 'Square': 4}
        self.options['number of tray sides'] = tray[text]

    def create_main_buttons(self):
        self.main_button_layout = QVBoxLayout()
        self.create_process_button()
        self.create_detect_button()
        self.main_button_layout.addWidget(self.process_button)
        self.main_button_layout.addWidget(self.detect_button)

    def setup_file_menu(self):
        exit_action = QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(qApp.quit)

        loadvid_action = QAction(QtGui.QIcon('load.png'), '&Load', self)
        loadvid_action.setShortcut('Ctrl+l')
        loadvid_action.setStatusTip('Load Video')
        loadvid_action.triggered.connect(self.load_vid)

        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(exit_action)
        file_menu.addAction(loadvid_action)

    def load_vid(self):
        pass

    def create_detect_button(self):
        self.detect_button = QPushButton("Detect Circles")
        self.detect_button.clicked.connect(self.detect_button_clicked)

    def detect_button_clicked(self):
        self.check_method_list()
        self.pt = pt.ParticleTracker(self.video, None, self.pp, self.options,
                                     self.methods)
        circles = self.pt.find_circles(self.new_frame)
        circles = np.array(circles).squeeze()
        annotated_frame = \
            self.pt.annotate_frame_with_circles(self.cropped_frame, circles)
        cv2.imwrite('frame.png', annotated_frame)
        self.update_main_image()

    def add_widgets_to_layout(self):
        self.layout.addWidget(self.main_image, 0, 0, 2, 2)
        self.layout.addLayout(self.main_button_layout, 0, 3, 1, 1)
        self.layout.addLayout(self.process_options_layout, 1, 3, 1, 1)
        self.layout.addLayout(self.track_options_layout, 2, 3, 1, 1)

    def create_track_options(self):
        self.track_options_layout = QVBoxLayout()
        self.create_min_dist_slider()
        self.track_options_layout.addWidget(self.min_dist_label)
        self.track_options_layout.addWidget(self.min_dist_slider)
        self.create_min_rad_slider()
        self.track_options_layout.addWidget(self.min_rad_label)
        self.track_options_layout.addWidget(self.min_rad_slider)
        self.create_max_rad_slider()
        self.track_options_layout.addWidget(self.max_rad_label)
        self.track_options_layout.addWidget(self.max_rad_slider)
        self.track_options_layout.addStretch()

    def create_min_dist_slider(self):
        self.min_dist_label = QLabel()
        self.min_dist_label.setText('Minimum distance: '
                                    + str(self.options['min_dist']))
        self.min_dist_slider = QSlider(Qt.Horizontal)
        self.min_dist_slider.setRange(1, 50)
        self.min_dist_slider.setValue(self.options['min_dist'])
        self.min_dist_slider.valueChanged[int].\
            connect(self.min_dist_slider_changed)

    def min_dist_slider_changed(self, val):
        self.options['min_dist'] = val
        self.min_dist_label.setText('Minimum distance: ' + str(val))

    def create_min_rad_slider(self):
        self.min_rad_label = QLabel()
        self.min_rad_label.setText('Minimum radius: '
                                   + str(self.options['min_rad']))
        self.min_rad_slider = QSlider(Qt.Horizontal)
        self.min_rad_slider.setRange(1, 50)
        self.min_rad_slider.setValue(self.options['min_rad'])
        self.min_rad_slider.valueChanged[int].\
            connect(self.min_rad_slider_changed)

    def min_rad_slider_changed(self, val):
        self.options['min_rad'] = val
        self.min_rad_label.setText('Minimum radius: ' + str(val))

    def create_max_rad_slider(self):
        self.max_rad_label = QLabel()
        self.max_rad_label.setText('Maximum radius: '
                                   + str(self.options['max_rad']))
        self.max_rad_slider = QSlider(Qt.Horizontal)
        self.max_rad_slider.setRange(1, 50)
        self.max_rad_slider.setValue(self.options['max_rad'])
        self.max_rad_slider.valueChanged[int].\
            connect(self.max_rad_slider_changed)

    def max_rad_slider_changed(self, val):
        self.options['max_rad'] = val
        self.max_rad_label.setText('Maximum radius: ' + str(val))

    def create_process_button(self):
        self.process_button = QPushButton("Process Image")
        self.process_button.clicked.connect(self.process_button_clicked)

    def process_button_clicked(self):
        self.check_method_list()
        self.pp.update_options(self.options, self.methods)
        self.new_frame, self.cropped_frame, _ = \
            self.pp.process_image(self.frame)
        cv2.imwrite('frame.png', self.new_frame)
        self.update_main_image()

    def update_main_image(self):
        pixmap = QtGui.QPixmap('frame.png')
        pixmap = pixmap.scaled(self.main_image.size(), Qt.KeepAspectRatio)
        self.main_image.setPixmap(pixmap)

    def create_main_image(self):
        self.main_image = QLabel()
        pixmap = QtGui.QPixmap('frame.png')
        self.main_image.setFixedHeight(int(pixmap.height()/2))
        self.main_image.setFixedWidth(int(pixmap.width()/2))
        pixmap = pixmap.scaled(self.main_image.size(), Qt.KeepAspectRatio)
        self.main_image.setPixmap(pixmap)

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width()-size.width())/2),
                  int((screen.height()-size.height())/2))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    in_video = vid.ReadVideo("/home/ppxjd3/Code/ParticleTracking/test_data/"
                             "test_video_EDIT.avi")
    ex = MainWindow(in_video)
    ex.show()
    sys.exit(app.exec_())
