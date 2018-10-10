import cv2


class ParticleTracker:
    """Class to track the locations of the particles in a video for each frame."""

    def __init__(self, video, dataframe):
        self.video = video
        self.dataframe = dataframe

    def find_circles(self):
        pass


if __name__ == "__main__":
    video1 = 1
    dataframe1 = 2
    PT = ParticleTracker(video1, dataframe1)
