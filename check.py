import os

import numpy as np

from Generic import video, images
from ParticleTracking import dataframes


def tracking(filename):
    core_name = os.path.splitext(filename)[0]
    vid_name = filename
    data_name = core_name + '.hdf5'
    out_name = core_name + '_check.png'
    data = dataframes.DataStore(data_name)
    crop = data.crop
    vid = video.ReadVideo(vid_name)
    print(vid_name)
    frames = np.arange(4)*vid.num_frames//4
    ims = [images.crop_img(vid.find_frame(f), crop) for f in frames]
    circles = [data.get_info(f, ['x', 'y', 'r']) for f in frames]
    new_ims = [images.draw_circles(im, c) for im, c in zip(ims, circles)]
    out = images.vstack(images.hstack(new_ims[0], new_ims[1]),
                        images.hstack(new_ims[2], new_ims[3]))
    images.save(out, out_name)


if __name__ == "__main__":
    from Generic import filedialogs
    name = filedialogs.load_filename()
    tracking(name)