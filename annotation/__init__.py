from Generic import video, images
import cv2
from cv2 import FONT_HERSHEY_PLAIN as font
import numpy as np


class TrackingAnnotator(video.Annotator):

    def __init__(self, in_name, out_name, params):
        self.params = params
        video.Annotator.__init__(self, in_name, out_name)

    def _draw_boxes(self, frame, f):
        box = self.data.get_info(f, 'box')
        classifiers = self.data.get_info(f,'classifier')#particle
        for index, classifier in enumerate(classifiers):
            annotated_frame = images.draw_contours(frame, [
                    box[index]], col=self.params['colors'][classifier], thickness=self.params['contour thickness'])
        return annotated_frame
    

    def _add_number(self, frame, f):
        x = self.data.get_info(f, 'x')
        y = self.data.get_info(f, 'y')
        particles = self.data.get_info(f, 'particle')
        classifiers = self.data.get_info(f, 'classifier')

        for index, classifier in enumerate(classifiers):
                frame = cv2.putText(frame, str(int(particles[index])), (int(x[index]), int(y[index])), font, self.params['font size'], self.params['colors'][classifier], 1, cv2.LINE_AA)
        return frame

    def _draw_trajs(self, frame, f):
        particle_ids = self.data.df[self.data.df.index == f]['particle'].unique()
        classifiers = self.data.get_info(f, 'classifier')
        for index, particle in enumerate(particle_ids):
            single_traj = self.data.df[self.data.df['particle'] == particle]
            traj_pts = single_traj[single_traj.index <= f][['x','y']].values
            if np.shape(traj_pts)[0] > 3:
                col_id = classifiers[index]
                frame = cv2.polylines(frame, np.int32([traj_pts]), False, self.params['colors'][col_id], self.params['trajectory thickness'])
        return frame

