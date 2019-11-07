#Do we need this?

from Generic import video, images
import os


class CircleAnnotator(video.Annotator):

    def __init__(self, filename, data, parameter, crop=False):
        self.data = data
        self.parameter = parameter
        self.crop = crop
        in_name = filename
        out_name = os.path.splitext(filename)[0]+'_'+parameter+'.mp4'
        video.Annotator.__init__(self, in_name, out_name, frames_as_surface=True)

    def process_frame(self, frame, f):
        info = self.data.get_info(f, ['x', 'y', 'r', self.parameter])
        info = info[:, :3] if self.parameter == 'particle' else info
        frame = images.pygame_draw_circles(frame, info)
        return frame

    def check_crop(self, filename):
        if self.crop:
            crop_name = os.path.splitext(filename)[0]+'_crop.mp4'
            if not os.path.exists(crop_name):
                    crop = self.data.metadata['crop']
                    print(crop)
                    video.crop_video(filename,
                                     crop[0][0],
                                     crop[1][0],
                                     crop[0][1],
                                     crop[1][1])
            return crop_name
        else:
            return filename


if __name__ == "__main__":
    from ParticleTracking.general import dataframes

    dataframe = dataframes.DataStore(
        "/home/ppxjd3/Videos/short.hdf5", load=True)
    input_video = "/home/ppxjd3/Videos/short.MP4"
    annotator = CircleAnnotator(input_video, dataframe, 'particle', crop=True)
    annotator.annotate()