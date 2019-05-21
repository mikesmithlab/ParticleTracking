from Generic import video, images
import os
import numpy as np


class BacteriaAnnotator(video.Annotator):

    def __init__(self, filename, data, parameter, crop=False):
        self.data = data
        self.parameter = parameter
        self.crop = crop
        in_name = filename
        out_name = os.path.splitext(filename)[0]+'_'+parameter+'.mp4'
        # frames_as_surface - if True returns pygames_surface else returns numpy array.
        video.Annotator.__init__(self, in_name, out_name, frames_as_surface=False)

    def process_frame(self, frame, f):
        print(np.shape(frame))

        if self.parameter == 'box':
            info = self.data.get_info(f, ['box','classifier'])

            for bacterium in info:
                if bacterium[1] == 1:
                    annotated_frame = images.draw_contours(frame, [
                        bacterium[0]], col=(0, 0, 255))
                elif bacterium[1] == 2:
                    annotated_frame = images.draw_contours(frame, [
                        bacterium[0]], col=(255, 0, 0))
                elif bacterium[1] == 3:
                    annotated_frame = images.draw_contours(frame, [
                        bacterium[0]], col=(0, 255, 0))


        return annotated_frame

    def check_crop(self, filename):
        return filename


if __name__ == "__main__":
    from ParticleTracking import dataframes
    dataframe = dataframes.DataStore(
        '/media/ppzmis/data/ActiveMatter/bacteria_plastic/bacteria.hdf5', load=True)
    input_video = '/media/ppzmis/data/ActiveMatter/bacteria_plastic/bacteria.mp4'
    annotator = BacteriaAnnotator(input_video, dataframe, 'box')
    annotator.annotate()