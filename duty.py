from Generic import audio
from ParticleTracking import dataframes
import numpy as np


def duty(video_filename, data_filename):
    with dataframes.DataStore(data_filename) as data:
        num_frames = data.num_frames
        duty_cycle = read_audio_file(video_filename, num_frames)
        duty_cycle = np.uint16(duty_cycle)
        data.add_frame_property('Duty', duty_cycle)
        data.save()


def read_audio_file(file, frames):
    wav = audio.extract_wav(file)
    wav_l = wav[:, 0]
    # wav = audio.digitise(wav)
    freqs = audio.frame_frequency(wav_l, frames, 48000)
    d = (freqs - 1000) / 15
    return d

if __name__ == "__main__":
    from Generic import filedialogs
    import os
    file = filedialogs.load_filename()
    data_name = os.path.splitext(file)[0]+'.hdf5'
    vid_name = os.path.splitext(file)[0]+'.MP4'
    duty(vid_name, data_name)
    