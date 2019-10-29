
from Generic import images
import cv2


def _draw_circles(frame, data, f, param=None, default_r=15):
    if 'r' not in list(data.df.columns):
        data.add_particle_property('r', default_r)
    info = data.get_info(f, ['x', 'y', 'r'])
    annotated_frame = images.pygame_draw_circles(frame, info)
    return annotated_frame

def _draw_boxes(self, frame, f):
    #Requires a column classifying traj with corresponding colour
    box = self.data.get_info(f, 'box')
    classifiers = self.data.get_info(f,'classifier')
    for index, classifier in enumerate(classifiers):
       annotated_frame = images.draw_contours(frame, [
        box[index]], col=self.params['colors'][classifier], thickness=self.params['contour thickness'])
    return annotated_frame

'''
def _add_number(self, frame, f, colx='x', coly='y'):
    #This can only be run on a linked trajectory
    box = self.data.get_info(f, 'box')

    x = self.data.get_info(f, colx)
    y = self.data.get_info(f, coly)
    particles = self.data.get_info(f, 'particle')
    classifiers = self.data.get_info(f, 'classifier')

    for index, classifier in enumerate(classifiers):
        frame = cv2.putText(frame, str(int(particles[index])), (int(x[index]), int(y[index])), font, self.params['font size'], self.params['colors'][classifier], 1, cv2.LINE_AA)

    return frame

def _draw_trajs(self, frame, f, colx='x',coly='y'):
df = self.data.df
#This can only be run on a linked trajectory
particle_ids = self.data.get_info(f, 'particle')

df_temp=df[df['particle'].isin(particle_ids)]
df_temp2 = df_temp[df_temp.index <= f]
for index, particle in enumerate(particle_ids):
traj_pts= df_temp2[df_temp2['particle'] == particle][[colx, coly,'classifier']]
frame = cv2.polylines(frame, np.int32([traj_pts[[colx,coly]].values]), False,
                     self.params['colors'][traj_pts['classifier'].median()],
                     self.params['trajectory thickness'])
return frame

'''