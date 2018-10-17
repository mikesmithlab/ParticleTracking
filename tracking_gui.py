import sys
import os
from PyQt5.QtWidgets import (
    QAction, QWidget, QLabel, QDesktopWidget,
    QApplication, QComboBox, QPushButton, QGridLayout,
    QMainWindow, qApp, QVBoxLayout, QSlider,
    QHBoxLayout, QLineEdit
    )
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
import Generic.video as vid
import cv2
import preprocessing as pp
import configuration as con
import particle_tracking as pt
import numpy as np


class MainWindow(QMainWindow):

    def __init__(self, video):
        super().__init__()
        self.video = video
        self.frame = self.video.read_next_frame()
        cv2.imwrite('frame.png', self.frame)
        self.methods = con.GLASS_BEAD_PROCESS_LIST
        self.options = con.GLASS_BEAD_OPTIONS_DICT
        self.pp = pp.ImagePreprocessor(self.video, self.methods, self.options)
        self.initUI()

    def initUI(self):
        self.central_widget = QWidget(self)
        self.statusBar().showMessage('Ready')
        self.setup_file_menu()

        self.layout = QGridLayout(self.central_widget)

        self.create_main_image()
        self.create_main_buttons()
        self.create_process_options()

        self.add_widgets_to_layout()

        self.central_widget.setFocus()
        self.setCentralWidget(self.central_widget)
        self.resize(1280, 720)
        self.center()
        self.setWindowTitle('Particle Tracker')
        self.show()

    def create_process_options(self):
        self.process_options_layout = QVBoxLayout(self.central_widget)
        self.create_tray_choice_combo()
        self.process_options_layout.addWidget(self.tray_choice_combo)

        self.create_grayscale_threshold_slider()
        self.process_options_layout.addWidget(self.grayscale_label)
        self.process_options_layout.addWidget(self.grayscale_threshold_slider)

        self.create_blur_kernel_slider()
        self.process_options_layout.addWidget(self.blur_kernel_label)
        self.process_options_layout.addWidget(self.blur_kernel_slider)

        self.process_options_layout.addStretch()

    def create_blur_kernel_slider(self):
        self.blur_kernel_label = QLabel(self.central_widget)
        self.blur_kernel_label.setText('Blur kernel size: ' +
                                       str(self.options['blur kernel'][0]))
        self.blur_kernel_slider = QSlider(Qt.Horizontal, self.central_widget)
        self.blur_kernel_slider.setRange(0, 5)
        self.blur_kernel_slider.setValue((self.options['blur kernel'][0]-1)/2)
        self.blur_kernel_slider.valueChanged[int].connect(self.blur_kernel_slider_changed)

    def blur_kernel_slider_changed(self, val):
        self.options['blur kernel'] = (val*2+1, val*2+1)
        self.blur_kernel_label.setText('Blur kernel size: ' + str(val*2+1))


    def create_grayscale_threshold_slider(self):
        self.grayscale_label = QLabel(self.central_widget)
        self.grayscale_label.setText('Grayscale Threshold: ' +
                                     str(self.options['grayscale threshold']))
        self.grayscale_threshold_slider = QSlider(Qt.Horizontal,
                                                  self.central_widget)
        self.grayscale_threshold_slider.setRange(0, 255)
        self.grayscale_threshold_slider.setValue(self.options['grayscale threshold'])
        self.grayscale_threshold_slider.valueChanged[int].connect(self.grayscale_threshold_slider_changed)




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
        self.main_button_layout = QVBoxLayout(self.central_widget)
        self.create_process_button()
        self.create_detect_button()
        self.main_button_layout.addWidget(self.process_button)
        self.main_button_layout.addWidget(self.detect_button)

    def setup_file_menu(self):
        exitAction = QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        loadvidAction = QAction(QtGui.QIcon('load.png'), '&Load', self)
        loadvidAction.setShortcut('Ctrl+l')
        loadvidAction.setStatusTip('Load Video')
        loadvidAction.triggered.connect(self.load_vid)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        fileMenu.addAction(loadvidAction)

    def load_vid(self):
        pass

    def create_detect_button(self):
        self.detect_button = QPushButton("Detect Circles", self.central_widget)
        self.detect_button.clicked.connect(self.detect_button_clicked)

    def detect_button_clicked(self):
        self.pt = pt.ParticleTracker(self.video, None, self.pp, self.options,
                                     self.methods)
        circles = self.pt.find_circles(self.new_frame)
        circles = np.array(circles).squeeze()
        annotated_frame = self.pt.annotate_frame_with_circles(self.cropped_frame, circles)
        cv2.imwrite('frame.png', self.cropped_frame)
        self.update_main_image()

    def add_widgets_to_layout(self):
        self.layout.addWidget(self.main_image, 0, 0, 2, 2)
        self.layout.addLayout(self.main_button_layout, 0, 3, 1, 1)
        self.layout.addLayout(self.process_options_layout, 1, 3, 1, 1)

    def create_process_button(self):
        self.process_button = QPushButton("Process Image", self.central_widget)
        self.process_button.clicked.connect(self.process_button_clicked)

    def process_button_clicked(self):
        self.pp.update_options(self.options, self.methods)
        self.new_frame, self.cropped_frame, _ = self.pp.process_image(self.frame)
        cv2.imwrite('frame.png', self.new_frame)
        self.update_main_image()

    def update_main_image(self):
        pixmap = QtGui.QPixmap('frame.png')
        pixmap = pixmap.scaled(self.main_image.size(), Qt.KeepAspectRatio)
        self.main_image.setPixmap(pixmap)

    def create_main_image(self):
        self.main_image = QLabel(self)
        pixmap = QtGui.QPixmap('frame.png')
        self.main_image.setFixedHeight(pixmap.height()/2)
        self.main_image.setFixedWidth(pixmap.width()/2)
        pixmap = pixmap.scaled(self.main_image.size(), Qt.KeepAspectRatio)
        self.main_image.setPixmap(pixmap)

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width()-size.width())/2,
                  (screen.height()-size.height())/2)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    in_video = vid.ReadVideo("/home/ppxjd3/Code/ParticleTracking/test_data/test_video_EDIT.avi")
    ex = MainWindow(in_video)
    ex.show()
    sys.exit(app.exec_())
