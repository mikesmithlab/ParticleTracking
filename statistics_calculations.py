import numpy as np
import dataframes
import particle_tracking


class PropertyCalculator:
    """Class to calculate the properties associated with tracking"""
    def __init__(self, tracking_dataframe):
        self.td = tracking_dataframe
        self.calculate_order_parameter()

    def calculate_order_parameter(self):
        num_frames = np.max(self.td.dataframe['frame'])
        print(num_frames)


if __name__ == "__main__":
    dataframe = dataframes.TrackingDataframe(
            "/home/ppxjd3/Code/ParticleTracking/test_data/test_video.hdf5",
            load=True)
    PC = PropertyCalculator(dataframe)
