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
            paramsubset = parameters[key]

            for subkey in paramsubset:
                if type(paramsubset[subkey]) == dict:
                    paramsubsubset = paramsubset[subkey]
                    for subsubkey in paramsubsubset:
                        value = paramsubsubset[subsubkey]
                        if type(value) == list:
                            self.param_dict[subsubkey] = value
        self.param_dict['frame'] = [0, 0, self.tracker.cap.num_frames-1, 1]
        self.update_slideable_parameters()
        return self.param_dict

    def update_slideable_parameters(self):
        parameters = self.tracker.parameters
        for key in parameters:
            paramsubset = parameters[key]
            for subkey in paramsubset:
                if type(paramsubset[subkey]) == dict:
                    paramsubsubset=paramsubset[subkey]
                    for subsubkey in paramsubsubset:
                        if subsubkey in self.param_dict.keys():
                            paramsubsubset[subsubkey] = self.param_dict[subsubkey]


    def update(self):
        self.update_slideable_parameters()
        new_frame, annotated_frame = self.tracker.process_frame(self.param_dict['frame'][0])
        if np.size(np.shape(new_frame)) == 2:
            new_frame = stack_3(new_frame)
        if np.size(np.shape(annotated_frame)) == 2:
            annotated_frame = stack_3(annotated_frame)
        self._display_img(new_frame, annotated_frame)


if __name__ == "__main__":
    from ParticleTracking.other_legacy.example_child import ExampleChild
    from ParticleTracking.other_legacy.james_nitrile import JamesPT
    from Generic import filedialogs

    file = filedialogs.load_filename('Load a video')
    choice = input('Enter 1 for example, 2 for James')
    if int(choice) == 1:
        tracker = ExampleChild(file)
    else:
        tracker = JamesPT(file)
    gui = TrackingGui(tracker)

