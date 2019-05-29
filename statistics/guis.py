from Generic import video, images, filedialogs
from ParticleTracking import statistics, dataframes
import os

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

if __name__ == "__main__":
    file = filedialogs.load_filename()
    vid_name = os.path.splitext(file)[0] + '.MP4'
    data_name = os.path.splitext(file)[0] + '.hdf5'
    vid = video.ReadVideo(vid_name)
    data = dataframes.DataStore(data_name)
    OrderGui(vid, data)