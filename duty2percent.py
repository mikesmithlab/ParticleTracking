import numpy as np


def convert(duty):
    duty = duty.astype(np.int32)
    percent = np.loadtxt('ParticleTracking/duty2percent.txt')
    return percent[duty]
