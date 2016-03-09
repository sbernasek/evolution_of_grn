__author__ = 'Sebi'


import matplotlib.pyplot as plt


def create_subplot_figure(dim=(1, 2), size=(20, 6)):
    """
    Creates figure of specified size and returns axes object.

    Parameters:
        dim (tuple) - subplot dimensions
        size (tuple) - figure dimensions

    Returns:
        ax - axes object
    """
    _ = plt.figure(figsize=size)
    plt.rc('lines', linewidth=1)
    plt.rcParams.update({'axes.titlesize': 'x-large'})

    # add grey grid
    r = 256
    g = 256
    b = 256

    axes = ()
    for i in range(1, dim[0]*dim[1]+1):

        # create subplot
        ax = plt.subplot(dim[0], dim[1], i)

        # format grid
        ax.set_axis_bgcolor((r/256, g/256, b/256))
        ax.grid(b=True, which='major', axis='both')
        axes += (ax,)

    return axes