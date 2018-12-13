from ParticleTracking import dataframes
from Generic import video, images, filedialogs
import time
import pygame

vid_name = "/home/ppxjd3/Videos/solid_crop.mp4"
vid = video.ReadVideo(vid_name)
frame = vid.read_next_frame()

data_name = "/home/ppxjd3/Videos/solid_data.hdf5"
data = dataframes.DataStore(data_name, load=True)

surface = pygame.pixelcopy.make_surface(frame)

for f in range(1):
    circles = data.get_info(f, include_size=True)
    for x, y, rad in circles:
        pygame.draw.circle(surface, (255, 0, 255), (y, x), rad, 3)

image = pygame.surfarray.array3d(surface)
images.display(image)