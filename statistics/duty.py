from Generic import audio
from ParticleTracking import dataframes
import numpy as np


def duty(video_filename, num_frames):
        duty_cycle = read_audio_file(video_filename, num_frames)
        duty_cycle = np.uint16(duty_cycle)
        return duty_cycle


def read_audio_file(file, frames):
    wav = audio.extract_wav(file)
    wav_l = wav[:, 0]
    # wav = audio.digitise(wav)
    freqs = audio.frame_frequency(wav_l, frames, 48000)
    d = (freqs - 1000) / 15
    return d

