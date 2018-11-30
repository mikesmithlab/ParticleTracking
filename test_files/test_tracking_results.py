import ParticleTracking.dataframes as dataframes
import Generic.video as video
import Generic.images as images
import matplotlib.pyplot as plt
import Generic.filedialogs as fd
import numpy as np


dataframe_name = fd.load_filename('Choose a dataframe')
dataframe_store = dataframes.TrackingDataframe(dataframe_name, True)
vid_name = fd.load_filename('Choose a video')
vid = video.ReadVideo(vid_name)
"""Test number detected"""
# number_of_particles = input('Enter the number of particles in the tray'
#                             ' for the selected dataframe: ')
detected_particles = np.zeros(dataframe_store.num_frames)
frames = np.arange(0, dataframe_store.num_frames)
for f in frames:
    info = dataframe_store.return_property_and_circles_for_frame(
        f,
        parameter='particle')
    detected_particles[f] = np.shape(info)[0]
    if f == 0:
        frame = vid.read_next_frame()
        frame = images.draw_circles(frame, info[:, :3])
        images.display(frame)
        false_negatives = input('Enter the number of missed detections: ')
        false_positives = input('Enter the number of false positives: ')
        number_of_particles = np.shape(info)[0] + int(false_negatives) - int(false_positives)
    if detected_particles[f] > number_of_particles:
        images.display(frame, str(number_of_particles - detected_particles[f]))
print(detected_particles)
mean = np.mean(detected_particles)
err = np.std(detected_particles)/np.sqrt(len(detected_particles))
data_range = np.max(detected_particles) - np.min(detected_particles)
print('The number of particles detected is {} +/- {} with a range of {}'.format(mean, err, data_range))
plt.figure()
plt.plot(frames, detected_particles, 'o')
plt.plot([frames[0], frames[-1]], [number_of_particles, number_of_particles])
plt.show()

"""Check percentage of circles with dark average colours"""
# vid_fname = fd.load_filename('Load the same video')
# vid = video.ReadVideo(vid_fname)
# for f in np.arange(0, 100, 10):
#     frame = vid.find_frame(f)
#     images.display(frame)
#     info = dataframe_store.return_property_and_circles_for_frame(f, 'particle')
#     particles = []
#     for x, y, r, part in info:
#         section = frame[int(y-r/2):int(y+r/2), int(x-r/2):int(x+r/2), :]
#         color = np.mean(section[:])
#         if color < 100:
#             particles.append(True)
#         else:
#             images.display(section)
#             particles.append(False)
#     print(np.mean(particles))

"""Check circles which are outside the boundary"""
# import matplotlib.path as mpath
# import cv2
# vid = video.ReadVideo(vid_fname)
# for f in np.arange(0, 100, 10):
#     info = dataframe_store.return_property_and_circles_for_frame(f, 'particle')
#     boundary = dataframe_store.extract_boundary_for_frame(f)
#     particles = []
#     points = np.vstack((info[:, 0], info[:, 1])).transpose()
#     if len(np.shape(boundary)) == 1:
#         vertices_from_centre = points - boundary[0:2]
#         points_inside_index = np.linalg.norm(vertices_from_centre, axis=1) < \
#             boundary[2]
#     else:
#         path = mpath.Path(boundary)
#         points_inside_index = path.contains_points(points)
#
#     frame = vid.find_frame(f)
#     points_inside = info[points_inside_index, 0:3]
#     points_outside = info[~points_inside_index, 0:3]
#     images.draw_circles(frame, points_inside, color=images.RED)
#     images.draw_circles(frame, points_outside, color=images.GREEN, thickness=6)
#     cv2.polylines(frame, np.int32([boundary]), isClosed=True, color=images.PINK, thickness=5)
#     images.display(frame)

