from ParticleTracking.general.parameters import get_param_val
from ParticleTracking.general.dataframes import DataStore
from Generic import images
import cv2


def circles(frame, data, f, parameters=None):
    '''
    Function draws circles on an image at x,y locations. If data.df['r'] exists
    circles have this radius, else 'r' col is created with value set from annotation
    sub dictionary.

    :param frame: frame to be annotated should be 3 colour channel
    :param data: datastore with particle information
    :param f: frame number
    :param parameters: annotation sub dictionary

    :return: annotated frame
    '''

    if 'r' not in list(data.df.columns):
        data.add_particle_property('r', get_param_val(parameters['circle:radius']))
    colour = parameters['circle:cmap']
    thickness = get_param_val(parameters['circle:thickness'])
    circles = data.get_info(f, ['x', 'y', 'r'])
    for circle in circles:
        frame = cv2.circle(frame, (int(circle[0]), int(circle[1])), int(circle[2]), colour, thickness)
    return frame

#Not yet working
def _boxes(frame, data, f, parameters=None):
    #Requires a column classifying traj with corresponding colour
    box = data.get_info(f, 'box')
    classifiers = data.get_info(f,'classifier')
    for index, classifier in enumerate(classifiers):
       annotated_frame = images.draw_contours(frame, [
        box[index]], col=get_param_val(parameters['colors'])[classifier], thickness=get_param_val(parameters['contour thickness']))
    return annotated_frame

def add_label(frame, data, f, parameters=None):
    '''
    Function puts text on an image at specific location.
    This function is for adding metadata or info not labelling
    particles with their ids.

    :param frame: frame to be annotated should be 3 colour channel
    :param data: datastore with particle information
    :param f: frame number
    :param parameters: annotation sub dictionary

    :return: annotated frame
    '''
    text=''
    x=1
    y=1
    annotated_frame=cv2.putText(frame, text ,(x, y), parameters['font'], parameters['font size'], parameters['font colour'],1, cv2.LINE_AA)

    return annotated_frame

def add_particle_numbers(frame, data, f, parameters=None):
    '''
        Function annotates image with particle ids
        This function only makes sense if run on linked trajectories

        :param frame: frame to be annotated should be 3 colour channel
        :param data: datastore with particle information
        :param f: frame number
        :param parameters: annotation sub dictionary

        :return: annotated frame
        '''

    x = data.get_info(f, 'x')
    y = data.get_info(f, 'y')
    particles = data.get_info(f, 'particle')

    for index in enumerate(particles):
        frame = cv2.putText(frame, str(int(particles[index])), (int(x[index]), int(y[index])), parameters['font'], parameters['font size'], params['font colours'], 1, cv2.LINE_AA)

    return frame

'''
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