import sys
import os
from PyQt5.QtWidgets import (
    QAction, QWidget, QLabel, QDesktopWidget,
    QApplication, QComboBox, QPushButton, QGridLayout,
    QMainWindow, qApp, QVBoxLayout, QSlider,
    QHBoxLayout, QLineEdit, QListView, QFileDialog
    )
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
import Generic.video as vid
import Generic.images as im
import cv2
import ParticleTracking.preprocessing as pp
import ParticleTracking.configuration as con
import ParticleTracking.tracking as pt
import numpy as np
import ParticleTracking.dataframes as df


class MainWindow(QMainWindow):

    def __init__(self, video=None):
        super().__init__()
        if video:
            self.video = video
        else:
            self.load_vid()
        self.frame = self.video.read_next_frame()
        cv2.imwrite('frame.png', self.frame)
        self.methods = con.MethodsList('Glass_Bead', load=True)
        self.config_dataframe = con.ConfigDataframe()
        self.options = self.config_dataframe.get_options('Glass_Bead')
        self.pp = pp.ImagePreprocessor(self.methods.extract_methods(),
                                       self.options)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.statusBar().showMessage('Ready')
        self.resize(1280, 720)
        self.center()
        self.setWindowTitle('Particle Tracker')
        self.show()

        self.layout = QGridLayout(central_widget)

        self.setup_file_menu()
        self.create_main_image()
        self.create_main_buttons()
        self.create_config_controls()
        self.create_process_options()
        self.create_track_options()

        self.add_widgets_to_layout()

    def setup_file_menu(self):
        exit_action = QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(qApp.quit)

        loadvid_action = QAction(QtGui.QIcon('load.png'), '&Load', self)
        loadvid_action.setShortcut('Ctrl+l')
        loadvid_action.setStatusTip('Load Video')
        loadvid_action.triggered.connect(self.load_vid_clicked)

        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(exit_action)
        file_menu.addAction(loadvid_action)

    def create_main_image(self):
        self.main_image = QLabel()
        pixmap = QtGui.QPixmap('frame.png')
        self.main_image.setFixedHeight(600)
        self.main_image.setFixedWidth(600)
        pixmap = pixmap.scaled(self.main_image.size(), Qt.KeepAspectRatio)
        self.main_image.setPixmap(pixmap)

    """Main buttons and their callback functions"""

    def create_main_buttons(self):
        self.main_button_layout = QVBoxLayout()
        self.create_process_button()
        self.create_detect_button()
        self.create_save_config_button()
        self.create_track_button()
        self.main_button_layout.addWidget(self.process_button)
        self.main_button_layout.addWidget(self.detect_button)
        self.main_button_layout.addWidget(self.save_config_button)
        self.main_button_layout.addWidget(self.track_button)

    def create_process_button(self):
        self.process_button = QPushButton("Process Image")
        self.process_button.clicked.connect(self.process_button_clicked)

    def process_button_clicked(self):
        self.check_method_list()
        self.pp.update(self.options, self.methods.extract_methods())
        self.new_frame, self.cropped_frame, _ = \
            self.pp.process(self.frame)
        cv2.imwrite('frame.png', self.new_frame)
        self.update_main_image()

    def create_detect_button(self):
        self.detect_button = QPushButton("Detect Circles")
        self.detect_button.clicked.connect(self.detect_button_clicked)

    def detect_button_clicked(self):
        self.check_method_list()
        self.pt = pt.ParticleTracker(self.filename[0],
                                     self.options,
                                     self.methods.extract_methods())
        circles = self.pt.find_circles(self.new_frame)
        circles = np.array(circles).squeeze()
        annotated_frame = im.draw_circles(self.cropped_frame, circles)
        cv2.imwrite('frame.png', annotated_frame)
        self.update_main_image()

    def create_save_config_button(self):
        self.save_config_button = QPushButton("Save configs")
        self.save_config_button.clicked.connect(self.save_config_button_clicked)

    def save_config_button_clicked(self):
        self.check_method_list()
        pyperclip.copy(str(self.options))
        text = self.config_choice_combo.currentText()
        self.config_dataframe.replace_row(self.options, text)
        self.methods.write_list()

    def create_track_button(self):
        self.track_button = QPushButton("Start Track")
        self.track_button.clicked.connect(self.track_button_clicked)

    def track_button_clicked(self):
        self.begin_track()

    def create_config_controls(self):
        self.config_controls_layout = QHBoxLayout()
        self.create_config_choice_combo()
        self.config_controls_layout.addWidget(self.config_choice_combo)

    def create_config_choice_combo(self):
        self.config_choice_combo = QComboBox()
        config_choices = con.PARTICLE_LIST
        self.config_choice_combo.addItems(config_choices)
        self.config_choice_combo.activated.connect(self.config_choice_combo_changed)

    def config_choice_combo_changed(self):
        self.reload_configs(self.config_choice_combo.currentText())

    """Process options and their callbacks"""

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

        self.process_options_layout.addStretch()

    def create_tray_choice_combo(self):
        self.tray_choice_combo = QComboBox()
        tray_choices = 'Circular', 'Hexagonal', 'Square'
        self.tray_choice_combo.addItems(tray_choices)
        self.tray_choice_combo.activated.connect(self.tray_choice_changed)

    def tray_choice_changed(self):
        text = self.tray_choice_combo.currentText()
        tray = {'Circular': 1, 'Hexagonal': 6, 'Square': 4}
        self.options['number of tray sides'] = tray[text]

    def create_grayscale_threshold_slider(self):
        self.grayscale_label = QLabel()
        self.grayscale_label.setText('Grayscale Threshold: ' +
                                     str(self.options['grayscale threshold']))
        self.grayscale_threshold_slider = QSlider(Qt.Horizontal)
        self.grayscale_threshold_slider.setRange(0, 255)
        self.grayscale_threshold_slider.\
            setValue(self.options['grayscale threshold'])
        self.grayscale_threshold_slider.valueChanged.\
            connect(self.grayscale_threshold_slider_changed)

    def grayscale_threshold_slider_changed(self):
        val = self.grayscale_threshold_slider.value()
        self.options['grayscale threshold'] = val
        self.grayscale_label.setText('Grayscale Threshold: ' + str(val))

    def create_blur_kernel_slider(self):
        self.blur_kernel_label = QLabel()
        self.blur_kernel_label.setText('Blur kernel size: ' +
                                       str(self.options['blur kernel']))
        self.blur_kernel_slider = QSlider(Qt.Horizontal)
        self.blur_kernel_slider.setRange(0, 5)
        self.blur_kernel_slider.setValue((self.options['blur kernel']-1)/2)
        self.blur_kernel_slider.valueChanged.\
            connect(self.blur_kernel_slider_changed)

    def blur_kernel_slider_changed(self):
        val = self.blur_kernel_slider.value()
        self.options['blur kernel'] = val*2+1
        self.blur_kernel_label.setText('Blur kernel size: ' + str(val*2+1))

    def create_adaptive_block_size_slider(self):
        self.adaptive_block_size_label = QLabel()
        self.adaptive_block_size_label.setText(
            'Adaptive Threshold kernel size: '
            + str(self.options['adaptive threshold block size']))

        self.adaptive_block_size_slider = QSlider(Qt.Horizontal)
        self.adaptive_block_size_slider.setRange(1, 20)
        self.adaptive_block_size_slider.setValue(
            self.options['adaptive threshold block size'])
        self.adaptive_block_size_slider.valueChanged.\
            connect(self.adaptive_block_size_slider_changed)

    def adaptive_block_size_slider_changed(self):
        val = self.adaptive_block_size_slider.value()
        self.options['adaptive threshold block size'] = val*2+1
        self.adaptive_block_size_label.setText(
            'Adaptive Threshold kernel size: '
            + str(val*2+1))

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
            valueChanged.connect(self.adaptive_constant_slider_changed)

    def adaptive_constant_slider_changed(self):
        val = self.adaptive_constant_slider.value()
        self.options['adaptive threshold C'] = val
        self.adaptive_constant_label.setText(
            'Adaptive threshold constant: '
            + str(self.options['adaptive threshold C']))

    def create_methods_list(self):
        self.methods_list = QListView()
        self.methods_list.setDragDropMode(QListView.InternalMove)
        self.methods_list.setDefaultDropAction(Qt.MoveAction)
        self.methods_list.setAcceptDrops(True)
        self.methods_list.setDropIndicatorShown(True)
        self.methods_list.setDragEnabled(True)
        self.methods_list.setWindowTitle('Method Order')

        self.methods_model = QtGui.QStandardItemModel(self.methods_list)
        for method, check in self.methods.methods_list:
            item = QtGui.QStandardItem(method)
            item.setData(method)
            item.setCheckable(True)
            item.setDragEnabled(True)
            item.setDropEnabled(False)
            if check:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

            self.methods_model.appendRow(item)

        self.methods_list.setModel(self.methods_model)
        self.methods_list.setFixedHeight(
            self.methods_list.sizeHintForRow(0)
            * (self.methods_model.rowCount() + 1))

    def check_method_list(self):
        new_methods = []
        for i in range(self.methods_model.rowCount()):
            it = self.methods_model.item(i)
            if it is not None:
                if it.checkState() == Qt.Checked:
                    check = True
                else:
                    check = False
                new_methods.append([it.text(), check])
        self.methods.methods_list = new_methods

    """Track options and their callbacks"""

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

        self.create_p1_slider()
        self.track_options_layout.addWidget(self.p1_label)
        self.track_options_layout.addWidget(self.p1_slider)

        self.create_p2_slider()
        self.track_options_layout.addWidget(self.p2_label)
        self.track_options_layout.addWidget(self.p2_slider)

        self.track_options_layout.addStretch()

    def create_min_dist_slider(self):
        self.min_dist_label = QLabel()
        self.min_dist_label.setText('Minimum distance: '
                                    + str(self.options['min_dist']))
        self.min_dist_slider = QSlider(Qt.Horizontal)
        self.min_dist_slider.setRange(1, 50)
        self.min_dist_slider.setValue(self.options['min_dist'])
        self.min_dist_slider.valueChanged.\
            connect(self.min_dist_slider_changed)

    def min_dist_slider_changed(self):
        val = self.min_dist_slider.value()
        self.options['min_dist'] = val
        self.min_dist_label.setText('Minimum distance: ' + str(val))

    def create_min_rad_slider(self):
        self.min_rad_label = QLabel()
        self.min_rad_label.setText('Minimum radius: '
                                   + str(self.options['min_rad']))
        self.min_rad_slider = QSlider(Qt.Horizontal)
        self.min_rad_slider.setRange(1, 50)
        self.min_rad_slider.setValue(self.options['min_rad'])
        self.min_rad_slider.valueChanged.\
            connect(self.min_rad_slider_changed)

    def min_rad_slider_changed(self):
        val = self.min_rad_slider.value()
        self.options['min_rad'] = val
        self.min_rad_label.setText('Minimum radius: ' + str(val))

    def create_max_rad_slider(self):
        self.max_rad_label = QLabel()
        self.max_rad_label.setText('Maximum radius: '
                                   + str(self.options['max_rad']))
        self.max_rad_slider = QSlider(Qt.Horizontal)
        self.max_rad_slider.setRange(1, 50)
        self.max_rad_slider.setValue(self.options['max_rad'])
        self.max_rad_slider.valueChanged.\
            connect(self.max_rad_slider_changed)

    def max_rad_slider_changed(self):
        val = self.max_rad_slider.value()
        self.options['max_rad'] = val
        self.max_rad_label.setText('Maximum radius: ' + str(val))

    def create_p1_slider(self):
        self.p1_label = QLabel()
        self.p1_label.setText('p1: ' + str(self.options['p_1']))
        self.p1_slider = QSlider(Qt.Horizontal)
        self.p1_slider.setRange(0, 255)
        self.p1_slider.setValue(self.options['p_1'])
        self.p1_slider.valueChanged.connect(self.p1_slider_changed)

    def p1_slider_changed(self):
        val = self.p1_slider.value()
        self.options['p_1'] = val
        self.p1_label.setText('p1: ' + str(val))

    def create_p2_slider(self):
        self.p2_label = QLabel()
        self.p2_label.setText('p2: ' + str(self.options['p_2']))
        self.p2_slider = QSlider(Qt.Horizontal)
        self.p2_slider.setRange(0, 255)
        self.p2_slider.setValue(self.options['p_2'])
        self.p2_slider.valueChanged.connect(self.p2_slider_changed)

    def p2_slider_changed(self):
        val = self.p2_slider.value()
        self.options['p_2'] = val
        self.p2_label.setText('p2: ' + str(val))

        '''
    def create_max_displacement_slider(self):
        self.max_displacement_label = QLabel()
        self.max_displacement_label.setText(
            'max frame displacement: '
            + str(self.options['max frame displacement'])
        self.max_displacement_slider = QSlider(qt.Horizontal)
        '''
    """Other methods"""

    def load_vid_clicked(self):
        self.load_vid()
        self.frame = self.video.read_next_frame()
        cv2.imwrite('frame.png', self.frame)
        self.update_main_image()

    def load_vid(self):
        self.filename = QFileDialog.getOpenFileName(self, 'Open video', '/home')
        self.video = vid.ReadVideo(self.filename[0])
        self.name = os.path.split(self.filename[0])[0]

    def add_widgets_to_layout(self):
        self.layout.addWidget(self.main_image, 0, 0, 4, 2)
        self.layout.addLayout(self.main_button_layout, 0, 3, 1, 1)
        self.layout.addLayout(self.config_controls_layout, 1, 3, 1, 1)
        self.layout.addLayout(self.process_options_layout, 2, 3, 1, 1)
        self.layout.addLayout(self.track_options_layout, 3, 3, 1, 1)

    def update_main_image(self):
        pixmap = QtGui.QPixmap('frame.png')
        pixmap = pixmap.scaled(self.main_image.size(), Qt.KeepAspectRatio)
        self.main_image.setPixmap(pixmap)

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width()-size.width())/2),
                  int((screen.height()-size.height())/2))

    def begin_track(self):
        # reload video
        self.video = vid.ReadVideo(self.filename[0])
        # create dataframe
        dataframe = df.DataStore(self.name + '.hdf5')
        new_pt = pt.ParticleTracker(self.filename[0],
                                    self.options,
                                    self.methods.extract_methods(),
                                    multiprocess=True,
                                    save_crop_video=True,
                                    save_check_video=True)
        qApp.quit()
        new_pt.track()

    def reload_configs(self, config):
        self.options = self.config_dataframe.get_options(config)
        self.methods = con.MethodsList(config)
        self.update_all_sliders()
        self.update_method_list()

    def update_all_sliders(self):
        self.grayscale_threshold_slider.\
            setValue(self.options['grayscale threshold'])
        self.grayscale_label.setText('Grayscale Threshold: ' +
                                     str(self.options['grayscale threshold']))
        self.blur_kernel_label.setText('Blur kernel size: ' +
                                       str(self.options['blur kernel']))
        self.blur_kernel_slider.setValue(
            (self.options['blur kernel'] - 1) / 2)
        self.adaptive_block_size_label.setText(
            'Adaptive Threshold kernel size: '
            + str(self.options['adaptive threshold block size']))
        self.adaptive_block_size_slider.setValue(
            self.options['adaptive threshold block size'])
        self.adaptive_constant_label.setText(
            'Adaptive threshold constant: '
            + str(self.options['adaptive threshold C']))
        self.adaptive_constant_slider. \
            setValue(self.options['adaptive threshold C'])
        self.min_dist_label.setText('Minimum distance: '
                                    + str(self.options['min_dist']))
        self.min_dist_slider.setValue(self.options['min_dist'])
        self.min_rad_label.setText('Minimum radius: '
                                   + str(self.options['min_rad']))
        self.min_rad_slider.setValue(self.options['min_rad'])
        self.max_rad_label.setText('Maximum radius: '
                                   + str(self.options['max_rad']))
        self.max_rad_slider.setValue(self.options['max_rad'])
        self.p1_label.setText('p1: ' + str(self.options['p_1']))
        self.p1_slider.setValue(self.options['p_1'])
        self.p2_label.setText('p2: ' + str(self.options['p_2']))
        self.p2_slider.setValue(self.options['p_2'])

    def update_method_list(self):
        self.methods_model = QtGui.QStandardItemModel(self.methods_list)
        for method, check in self.methods.methods_list:
            item = QtGui.QStandardItem(method)
            item.setData(method)
            item.setCheckable(True)
            item.setDragEnabled(True)
            item.setDropEnabled(False)
            if check:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

            self.methods_model.appendRow(item)
        self.methods_list.setModel(self.methods_model)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    in_video = vid.ReadVideo("/home/ppxjd3/Code/ParticleTracking/test_data/"
                             "test_video_EDIT.avi")
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
