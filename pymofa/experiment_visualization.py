"""Visualize the outcome of computer experiment.

TODOs
-----
Explore Parameterspace3D
    see: http://matplotlib.org/examples/mplot3d/contour3d_demo3.html
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(X, Y, values, rstride=1, cstride=1, alpha=0.3,
                    cmap=cm.PiYG)
    cset = ax.contour(X, Y, values, zdir='y', offset=0.9, cmap=cm.Blues)
    cset = ax.contour(X, Y, values, zdir='z', offset=0.0, cmap=cm.Blues)
    ax.set_xlabel("tau")
    ax.set_ylabel("sigma")
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('Agg')
# mpl.style.use("ggplot")


def explore_Parameterspace(TwoDFrame,
                           title="",
                           cmap='viridis',
                           colorbar=True,
                           norm=None,
                           vmin=None,
                           vmax=None,
                           ax=None):
    """
    Explore variables in a 2-dim Parameterspace.

    Parameters
    ----------
    TwoDFrame : 2D pandas.DataFrame with index and column names
        The data to plot
    title : string
        Title of the plot (Default: "")
    cmap : string
        Colormap to use (Default: "RdBu")
    vmin : float
        Minimum value of the colormap (Default: None)
    vmax : float
        Maximum vlaue of the colormap (Defualt: None)
    ax : mpl axis
        optional

    Examples
    --------
    >>> import init_data
    >>> data = init_data.get_Data("phi")
    >>> explore_Parameterspace(data.unstack(level="deltaT")["<safe>",0.5].
    >>>                        unstack(level="phi"))

    """
    xparams = TwoDFrame.columns.values
    yparams = TwoDFrame.index.values

    assert type(yparams[0]) != tuple, "Be aware of multi indicies"
    assert type(xparams[0]) != tuple, "Be aware of multi indicies"

    values = TwoDFrame.values

    X, Y = _create_meshgrid(xparams, yparams)

    if ax is  None:
        fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, values, cmap=cmap, norm=norm, vmin=vmin,
                       vmax=vmax)
    if colorbar:
        plt.colorbar(c, ax=ax, orientation="vertical")
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_ylim(np.min(Y), np.max(Y))
    ax.set_xlabel(TwoDFrame.columns.name)
    ax.set_ylabel(TwoDFrame.index.name)

    return c


def _create_meshgrid(x, y):
    """
    Proper spaced meshgrid.

    Create a meshgrid out of the array-like types x and y. We assume that x and
    y are equally spaced. The funciton positions the values of x and y into the
    middle of the return value.

    Parameters
    ----------
    x : 1D array like
        The x values
    y : 1D array like
        The y values

    Returns
    -------
    meshgrid : 2D np.array
        widened meshgrid of x and y

    Example
    -------
    >>> _create_meshgrid([1,2], [10, 12])
    >>> [array([[ 0.5,  1.5,  2.5],
                [ 0.5,  1.5,  2.5],
                [ 0.5,  1.5,  2.5]]),
         array([[  9.,   9.,   9.],
                [ 11.,  11.,  11.],
                [ 13.,  13.,  13.]])]

    """
    x = np.array(x)
    y = np.array(y)

    def broaden_grid(x):
        """Extend the x,y grid for proper spacing."""
        dx = x[1:] - x[:-1]
        xx = np.concatenate(([x[0]], x))
        dxx = np.concatenate(([-dx[0]], dx, [dx[-1]]))
        return xx + dxx/2.

    X = broaden_grid(x)
    Y = broaden_grid(y)

    return np.meshgrid(X, Y)
