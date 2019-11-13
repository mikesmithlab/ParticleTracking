import matplotlib.pyplot as plt



def cmap(data, f, col_name, parameters=None):
    '''
    cmap could have different use cases:
    1: 'discrete' data is colour coded according to some classifier
    2: 'continuous' data is colour coded according to continuous scale

    Colormap definitions: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    case 1 colourmap is Set1 of Matplotlibs Qualitative color maps
    case 2 colourmap is seismic from diverging colormaps


    :param data: Datastore object containing column upon which to perform the mapping
    :param f: integer specifying frame number
    :param col_name: string specifying column in dataframe of Datastore which supplies values for colour coding
    :param parameters: Dictionary specifying parameters

    :return: a
    '''



    data_column = data.get_info(f, col_name)


if __name__=='__main__':
    cmap = plt.get_cmap('Set1')
    print(cmap)