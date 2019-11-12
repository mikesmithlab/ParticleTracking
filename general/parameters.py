
def get_param_val(param):
    '''
    Simple function to determine if parameter is a list or something else
    Lists are 4 long an indicate an a parameter adjusted by the slider in TrackGui

    :param param: parameter to be tested
    :return: value from position zero of list or returns the parameter as is if not a list
    '''
    type_param = type(param)
    if type_param == type([]):
        return param[0]
    else:
        return param
