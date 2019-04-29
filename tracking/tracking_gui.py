from Generic.images import ParamGui


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
        self.param_dict['frame'] = [0, 0, self.tracker.cap.num_frames, 1]
        self.update_slideable_parameters()
        return self.param_dict

    def update_slideable_parameters(self):
        parameters = self.tracker.parameters
        for key in self.param_dict:
            parameters[key] = self.param_dict[key]
        self.tracker.update_parameters(parameters)

    def update(self):
        self.update_slideable_parameters()
        new_frame, annotated_frame = self.tracker.analyse_frame()
        self._display_img(new_frame, annotated_frame)




if __name__ == "__main__":
    from ParticleTracking.tracking_methods.example_child import ExampleChild, JamesPT
    from Generic import filedialogs

    #file = filedialogs.load_filename('Load a video')
    file='/home/ppzmis/Documents/PythonScripts/ParticleTracking/test_video.mp4'
    tracker = JamesPT(file)
    gui = TrackingGui(tracker)