from ParticleTracking.general.parameters import get_param_val
from ParticleTracking.general.dataframes import DataStore
from Generic import images
import cv2


def draw_circles(frame, data, f, parameters=None):
    if 'r' not in list(data.df.columns):
        data.add_particle_property('r', get_param_val(parameters['circle:radius']))
    colour = parameters['circle:cmap']
    thickness = get_param_val(parameters['circle:thickness'])
    circles = data.get_info(f, ['x', 'y', 'r'])
    for circle in circles:
        frame = cv2.circle(frame, (int(circle[0]), int(circle[1])), int(circle[2]), colour, thickness)
    return frame

#Not yet working
def _draw_boxes(frame, data, f, parameters=None):
    #Requires a column classifying traj with corresponding colour
    box = data.get_info(f, 'box')
    classifiers = data.get_info(f,'classifier')
    for index, classifier in enumerate(classifiers):
       annotated_frame = images.draw_contours(frame, [
        box[index]], col=get_param_val(parameters['colors'])[classifier], thickness=get_param_val(parameters['contour thickness']))
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