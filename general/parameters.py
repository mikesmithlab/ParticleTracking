
def get_param_val(param):
    type_param = type(param)
    if type_param == type([]):
        return param[0]
    else:
        return param
