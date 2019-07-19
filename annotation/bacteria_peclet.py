from ParticleTracking import dataframes
import matplotlib.pyplot as plt
import numpy as np


# magnitude of a vector
def magvect(v):
  magnitude = np.sqrt(v[0]*v[0] + v[1]*v[1])
  return magnitude

# angle between vectors
def angle(v1,v2):
  ang = np.arccos(np.dot(v1,v2)/(magvect(v1)*magvect(v2)))
  ang = ang*180/np.pi
  if np.cross(v1,v2) > 0:
      ang = -ang
  return ang

def angle_calc(dx,dy,step=2):
    dtheta = []
    v1 = np.array([0, 0])
    v2 = np.array([0, 0])

    time = []
    for i in range(np.size(dy)):
        v1[0] = dx[0]
        v1[1] = dy[0]
        v2[0] = dx[i]
        v2[1] = dy[i]

        dtheta.append(angle(v1, v2))
        time.append(i)
        #dr.append(magvect(v2-v1))
    dtheta = np.array(dtheta)
    plt.figure(4)
    plt.plot(time, dtheta)
    plt.show()

    #return dtheta

if __name__=='__main__':

    dataframe = dataframes.DataStore('/media/ppzmis/data/ActiveMatter/Microscopy/videosexample/videoDIC.hdf5', load=True)
    df = dataframe.df

    #Find all trajectories where end is > 45 pixels ~ 3 bacterium lengths. Excludes those that are stuck to surface
    distance_threshold = 45

    particle_ids = df['particle'].unique()
    print(particle_ids)
    moving = []
    dtheta = []
    dr = []
    #plt.figure(1)
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.title('Trajectories')

    for id in particle_ids:
        try:
            xvals = df[df['particle'] == id].x.to_numpy()
            yvals = df[df['particle'] == id].y.to_numpy()

            dx = np.diff(xvals)
            dy = np.diff(yvals)

            if np.size(xvals) > 20:
                dist_moved = ((xvals[0] - xvals[-1])**2 + (yvals[0] - yvals[-1])**2)**0.5

                if dist_moved > distance_threshold:
                    moving.append(id)
                    #dtheta, dr=angle_calc(dx, dy, dr)
                    angle_calc(dx,dy)
                    #plt.figure(1)
                    #plt.plot(xvals, yvals, '.')
        except:
            print('key error')
    print(len(moving))
    plt.show()
    '''
    dtheta = np.array(dtheta)
    dr = np.array(dr)


    [freq, binedges]=np.histogram(dtheta[~np.isnan(dtheta)], bins=11)
    [freqdr, binedgesdr] = np.histogram(dr[~np.isnan(dtheta)], bins=11)
    bins = 0.5*(binedges[:-1]+binedges[1:])
    binsdr = 0.5*(binedgesdr[:-1]+binedgesdr[1:])

    plt.figure(2)
    plt.title('Histogram dtheta')
    plt.xlabel('dtheta')
    plt.ylabel('freq')
    plt.plot(bins, freq, 'rx-')


    plt.figure(3)
    plt.title('Histogram dr')
    plt.xlabel('dr')
    plt.ylabel('freq')
    plt.plot(binsdr, freqdr)
    plt.show()

    '''
