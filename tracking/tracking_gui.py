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
        for paramsubsetkey in parameters:
            paramsubset = parameters[paramsubsetkey]
            for key in paramsubset:
                value = paramsubset[key]
                if type(value) == list:
                    self.param_dict[key] = value
        self.param_dict['frame'] = [0, 0, self.tracker.cap.num_frames-1, 1]
        self.update_slideable_parameters()
        return self.param_dict

    def update_slideable_parameters(self):
        parameters = self.tracker.parameters
        print('start')
        for paramsubsetkey in parameters:
            print(paramsubsetkey)
            paramsubset = parameters[paramsubsetkey]
            print(paramsubset)
            for key in paramsubset:
                print(key)
                print(paramsubset[key])
                print(self.param_dict)
                if key in self.param_dict.keys():
                    paramsubset[key] = self.param_dict[key]
        #self.tracker.update_parameters(parameters)

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

