



def cmap(data, f, col_name, parameters=None):
    '''
    cmap could have 3 different use cases:
    1: want all annotation to have a colour or colours which is unrelated to data
    2: want annotations to have colours defined by discrete data ranges
    3: want the cmap to change continuously over the range of the data.

    :param data: Datastore object containing column upon which to perform the mapping
    :param f: integer specifying frame number
    :param col_name: string specifying column in dataframe of Datastore
    :param parameters: Dictionary specifying parameters

    :return: a
    '''
    for

    data_column = data.get_info(f, col_name)
