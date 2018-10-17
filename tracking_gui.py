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
        methods = con.GLASS_BEAD_PROCESS_LIST
        options = con.GLASS_BEAD_OPTIONS_DICT
        self.pp = pp.ImagePreprocessor(self.video, methods, options)
        self.pt = pt.ParticleTracker(self.video, None, self.pp, options, methods)
        self.initUI()

    def initUI(self):
        self.central_widget = QWidget(self)
        self.layout = QGridLayout(self.central_widget)

        self.create_main_image()
        self.create_process_button()
        self.create_detect_button()

        self.add_widgets_to_layout()

        self.central_widget.setFocus()
        self.setCentralWidget(self.central_widget)
        self.resize(1280, 720)
        self.center()
        self.setWindowTitle('Particle Tracker')
        self.show()

    def create_detect_button(self):
        self.detect_button = QPushButton("Detect Circles", self.central_widget)
        self.detect_button.clicked.connect(self.detect_button_clicked)

    def detect_button_clicked(self):
        circles = self.pt.find_circles(self.new_frame)
        circles = np.array(circles).squeeze()
        annotated_frame = self.pt.annotate_frame_with_circles(self.cropped_frame, circles)
        cv2.imwrite('frame.png', self.cropped_frame)
        self.update_main_image()

    def add_widgets_to_layout(self):
        self.layout.addWidget(self.main_image, 0, 0, 2, 2)
        self.layout.addWidget(self.process_button, 0, 3, 1, 1)
        self.layout.addWidget(self.detect_button, 0, 3, 2, 1)

    def create_process_button(self):
        self.process_button = QPushButton("Process Image", self.central_widget)
        self.process_button.clicked.connect(self.process_button_clicked)

    def process_button_clicked(self):
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
