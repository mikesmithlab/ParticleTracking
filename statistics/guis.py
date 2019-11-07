import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from Generic import images
from ParticleTracking import statistics
from ParticleTracking.general import dataframes


class OrderGui(images.ParamGui):

    def __init__(self, img, data):
        self.data = data
        self.grayscale = False
        self.param_dict = {
            'rad_t': [3, 1, 5, 1]
        }
        images.ParamGui.__init__(self, img)

    def update(self):
        info = self.data.df.loc[self.frame_no, ['x', 'y', 'r']]
        features = statistics.order.order_process(info, self.param_dict[
            'rad_t'][0])
        self.im0 = images.crop_img(self.im0, self.data.metadata['crop'])
        self._display_img(
            images.add_colorbar(
                images.draw_circles(
                    self.im0, features[['x', 'y', 'r', 'order_mag']].values)))


class OrderHistogramViewer:

    def __init__(self, file):
        self.data = dataframes.DataStore(file)

        self.calc = statistics.PropertyCalculator(self.data)
        self.duty = self.calc.duty()
        self.duty_unique = np.sort(np.unique(self.duty))
        self.setup_figure()
        plt.show()

    def setup_figure(self):
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.25)
        ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.05])
        self.duty_slider = Slider(ax, 'Duty', 0, 1000, valinit=0, valstep=1)
        self.duty_slider.on_changed(self.update)
        freq, bins = self.get_data(self.duty_unique[0])
        self.plot, = self.ax.plot(bins, freq)

    def get_data(self, val):
        data = self.data.df.loc[
            self.data.df.Duty == val, 'shape_factor'].values
        # data = np.sum(data, axis=1)
        # data = np.abs(data)
        freq, bins = np.histogram(data, bins=100,
                                  density=True)
        return freq, bins[:-1]

    def update(self, val):
        val = self.duty_slider.val
        if val in self.duty_unique:
            freq, bins = self.get_data(val)
            self.plot.set_ydata(freq)
            self.ax.set_title(str(val))
            self.fig.canvas.draw_idle()
