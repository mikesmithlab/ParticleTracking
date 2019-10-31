from Generic.images import ParamGui
from Generic.images.basics import stack_3
import numpy as np

class TrackingGui(ParamGui):

    def __init__(self, tracker):
        self.grayscale = True
        self.tracker = tracker
        self.read_slideable_parameters()
        ParamGui.__init__(self, self.tracker.frame)

    def read_slideable_parameters(self):
        parameters = self.tracker.parameters
        self.param_dict = {}
        for key in parameters:
            value = parameters[key]
            if type(value) == list:
                self.param_dict[key] = value
        self.param_dict['frame'] = [0, 0, self.tracker.cap.num_frames-1, 1]
        self.update_slideable_parameters()
        return self.param_dict

    def update_slideable_parameters(self):
        parameters = self.tracker.parameters
        for key in self.param_dict:
            parameters[key] = self.param_dict[key]
        #self.tracker.update_parameters(parameters)

    def update(self):
        self.update_slideable_parameters()
        new_frame, annotated_frame = self.tracker.process_frame(self.param_dict['frame'][0])
        self._display_img(stack_3(new_frame), annotated_frame)


if __name__ == "__main__":
    from ParticleTracking.tracking.example_child import ExampleChild
    from ParticleTracking.tracking.james_nitrile import JamesPT
    from Generic import filedialogs

    file = filedialogs.load_filename('Load a video')
    choice = input('Enter 1 for example, 2 for James')
    if int(choice) == 1:
        tracker = ExampleChild(file)
    else:
        tracker = JamesPT(file)
    gui = TrackingGui(tracker)

