# Copyright (C) [2025] [Claus Naves Eikmeier]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License, version 3, as published
# by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# ===========================================================================
# Imports
# ===========================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler
# ===========================================================================

# ===========================================================================
# Exportable functions
# ===========================================================================
__all__ = [
    'plot_cs_seismogram',
    'plot_dt_amp_scatter',
    'plot_dt_histogram',
    'plot_dtw_connections',
    'plot_seismogram',
    'plot_seismogram_as_matrix',
    'plot_signal',
    'plot_signals',
    'plot_src',
    'plot_xy']
# ===========================================================================

# ===========================================================================
# Seismogram plot with indication of cycle-skipping
# ===========================================================================


def plot_cs_seismogram(seismograms_target, seismograms_curr, dt, dt_cs,
                       best_path, shot_num, windows=None, wp_marker_size=200,
                       cs_marker_size=8, seismogram_to_plot='target', gain=1,
                       title=None, ylimit=None, figsize=None, dpi=150,
                       save=None, show=True):
    """
    Seismogram plot with indication of cycle-skipping segments, i.e. with
    dt > dt_cs.

    Parameters
    ----------
    seismograms_target : numpy.ndarray
        Target seismograms with shape (shots, receivers, data).
    seismograms_curr : numpy.ndarray
        Current seismograms with shape (shots, receivers, data).
    dt : float
        Time step in milliseconds.
    dt_cs : float
        Cycle-skipping dt in milliseconds.
    best_path : numpy.ndarray
        Best paths with shape (shots, receivers, connections).
        If `windows` is provided, only paths inside the windows are
        considered.
    shot_num : int
        Shot number to consider for the plot.
    windows : list, optional
        List of indexes defining the boundaries of each window,
        with shape (shots, receivers, boundaries).
        Default is None.
    wp_marker_size : float, optional
        Windows marker size. Default is 200.
    cs_marker_size : float, optional
        Cycle-skipping marker size. Default is 8.
    seismogram_to_plot : str
        Seismogram to plot ('target' or 'curr'). Default is 'target'.
    gain : float, optional
        Signal gain (amplitude*gain). Default is 1.
    title : str, optinal
        Title of the plot. Default is None.
    ylimit : 2-tuple, optional
        Limits for the y-axis as (from, to). Default is None.
    figsize : 2-tuple, optional
        Size of the figure as (width, height). Default is None.
    dpi : int, optional
        Dots per inch (resolution) of the image. Default is 150.
    save : str, optional
        Path, filename, and format for saving the figure.
        Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    This function does not return a value.
    """

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams.update({'font.size': 12})

    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()

    for j in range(len(best_path[shot_num])):
        signal_target = seismograms_target[shot_num][j][:]\
            .astype(np.float64)
        signal_curr = seismograms_curr[shot_num][j][:]\
            .astype(np.float64)

        y_target = np.arange(0, len(signal_target))*dt/1000
        y_curr = np.arange(0, len(signal_curr))*dt/1000

        if seismogram_to_plot == 'target':
            x = (j+1)+(signal_target*gain)
            ax.plot(x, y_target, 'k-')
        else:
            x = (j+1)+(signal_curr*gain)
            ax.plot(x, y_curr, 'k-')

        if windows is not None:
            start = windows[shot_num][j][0]
            stop = windows[shot_num][j][1]
            ax.scatter(
                j+1, start*dt/1000, s=wp_marker_size, c='k', marker='_')
            ax.scatter(
                j+1, stop*dt/1000, s=wp_marker_size, c='k', marker='_')

    if ylimit is None:
        ylimit = (y_target[0], y_target[-1])

    for j in range(len(best_path[shot_num])):
        y_target = []
        y_curr = []
        amplitude_target = []
        amplitude_curr = []
        delta_t = []
        bp = best_path[shot_num][j]
        for k in range(len(bp)):
            delta_t.append((bp[k][1] - bp[k][0])*dt)

        signal_target = seismograms_target[shot_num][j][:]\
            .astype(np.float64)
        signal_curr = seismograms_curr[shot_num][j][:]\
            .astype(np.float64)

        for m in range(len(bp)):
            index_bp_target = bp[m][1]
            index_bp_curr = bp[m][0]

            if windows is not None:
                start = windows[shot_num][j][0]
                stop = windows[shot_num][j][1]
                index_target = index_bp_target + start
                index_curr = index_bp_curr + start
            else:
                index_target = index_bp_target
                index_curr = index_bp_curr

            dt_curr = abs(index_target*dt - index_curr*dt)
            if dt_curr > dt_cs:
                y_target.append(index_target*dt/1000)
                y_curr.append(index_curr*dt/1000)
                amplitude_target.append(
                    signal_target[index_target]*gain + (j+1))
                amplitude_curr.append(signal_curr[index_curr]*gain + (j+1))

        if seismogram_to_plot == 'target':
            ax.scatter(amplitude_target, y_target, s=cs_marker_size, c='r')
        else:
            ax.scatter(amplitude_curr, y_curr, s=cs_marker_size, c='r')

    ax.set_xlabel('Trace number')
    ax.set_ylabel('Time (s)')

    plt.ylim(ylimit)

    ax.grid(axis='y')
    plt.gca().invert_yaxis()
    plt.xlim(0, len(seismograms_target[shot_num])+1)

    if title is not None:
        plt.title(title, fontsize='medium')

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Plot scatter of amplitude vs DTW dt values
# ===========================================================================


def plot_dt_amp_scatter(seismograms_target, seismograms_curr, dt, dt_cs,
                        best_path, shot_num=None, rec_num=None, windows=None,
                        title=None, xlimit=None, ylimit=None, figsize=None,
                        markersize=1.5, dpi=150, save=None, show=True):
    """
    Plot scatter of amplitude vs DTW dt values.

    Parameters
    ----------
    seismograms_target : numpy.ndarray
        Target seismograms with shape (shots, receivers, data).
    seismograms_curr : numpy.ndarray
        Current seismograms with shape (shots, receivers, data).
    dt : float
        Time step in milliseconds.
    dt_cs : float
        Cycle-skipping dt in milliseconds.
    best_path : numpy.ndarray
        Best paths with shape (shots, receivers, connections).
        If `windows` is provided, only paths inside the windows are
        considered.
    shot_num : int, optional
        Shot number to consider for the plot. Default is None.
        If "shot_num" is None or "rec_num" is None,
        both are considered None.
    rec_num : int, optional
        Receiver number to consider for the plot. Default is None.
        If "shot_num" is None or "rec_num" is None,
        both are considered None.
    windows : list, optional
        List of indexes defining the boundaries of each window,
        with shape (shots, receivers, boundaries).
        Default is None.
    title : str, optinal
        Title of the plot. Default is None.
    xlimit : 2-tuple, optional
        Limits for the x-axis as (from, to). Default is None.
    ylimit : 2-tuple, optional
        Limits for the y-axis as (from, to). Default is None.
    figsize : 2-tuple, optional
        Size of the figure as (width, height). Default is None.
    markersize : float
        Marker size of the plot. Default is 1.0.
    dpi : int, optional
        Dots per inch (resolution) of the image. Default is 150.
    save : str, optional
        Path, filename, and format for saving the figure.
        Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    This function does not return a value.
    """

    amplitude_target = []
    amplitude_curr = []
    delta_t = []
    if shot_num is None or rec_num is None:
        for i in range(len(best_path)):
            for j in range(len(best_path[i])):
                bp = best_path[i][j]
                for k in range(len(bp)):
                    delta_t.append((bp[k][1] - bp[k][0])*dt)

                if windows is None:
                    signal_target = seismograms_target[i][j][:]\
                        .astype(np.float64)
                    signal_curr = seismograms_curr[i][j][:]\
                        .astype(np.float64)
                else:
                    start = windows[i][j][0]
                    stop = windows[i][j][1]
                    signal_target = seismograms_target[i][j][start:stop]\
                        .astype(np.float64)
                    signal_curr = seismograms_curr[i][j][start:stop]\
                        .astype(np.float64)

                for m in range(len(bp)):
                    index_target = bp[m][1]
                    index_curr = bp[m][0]
                    amplitude_target.append(signal_target[index_target])
                    amplitude_curr.append(signal_curr[index_curr])
    else:
        bp = best_path[shot_num][rec_num]
        for i in range(len(bp)):
            delta_t.append((bp[i][1] - bp[i][0])*dt)

        if windows is None:
            signal_target = seismograms_target[shot_num][rec_num][:]\
                .astype(np.float64)
            signal_curr = seismograms_curr[shot_num][rec_num][:]\
                .astype(np.float64)
        else:
            start = windows[shot_num][rec_num][0]
            stop = windows[shot_num][rec_num][1]
            signal_target = seismograms_target[shot_num][rec_num][:]\
                .astype(np.float64)
            signal_curr = seismograms_curr[shot_num][rec_num][:]\
                .astype(np.float64)

        amplitude_target = []
        amplitude_curr = []
        for i in range(len(bp)):
            index_target = bp[i][1]
            index_curr = bp[i][0]
            amplitude_target.append(signal_target[index_target])
            amplitude_curr.append(signal_curr[index_curr])

    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})

    if figsize is not None:
        plt.figure(figsize=figsize)

    plt.plot(
        delta_t, amplitude_target, 'o', markersize=markersize, color='green',
        label='Target signal')
    plt.plot(
        delta_t, amplitude_curr, 'o', markersize=markersize, color='orange',
        label='Current signal')

    plt.axvline(
        x=dt_cs, color='k', linestyle='--',
        label='$|\Delta t_{cs}| = $' + f'{round(dt_cs, 2)} ms')
    plt.axvline(
        x=-dt_cs, color='k', linestyle='--')

    if title is not None:
        plt.title(title, fontsize='medium')

    plt.xlabel('$\Delta t$ (ms)')

    plt.ylabel('Amplitude')

    if xlimit is not None:
        plt.xlim(xlimit)

    if ylimit is not None:
        plt.ylim(ylimit)

    plt.legend()

    # plt.box(on=False)

    plt.grid()

    if save is not None:
        plt.savefig(save, dpi=dpi, facecolor='white', edgecolor='none')

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Histogram (cumulative frequency) of DTW dt values
# ===========================================================================


def plot_dt_histogram(dt, dt_cs, best_path, shot_num=None, rec_num=None,
                      title=None, xlimit=None, ylimit=None, color='black',
                      figsize=None, dpi=150, save=None, show=True):
    """
    Plot histogram (cumulative frequency) of DTW dt values.

    Parameters
    ----------
    dt : float
        Time step in milliseconds.
    dt_cs : float
        Cycle-skipping dt in milliseconds.
    best_path : numpy.ndarray
        Best paths with shape (shots, receivers, connections).
        If `windows` is provided, only paths inside the windows are
        considered.
    shot_num : int, optional
        Shot number to consider for the plot. Default is None.
        If "shot_num" is None or "rec_num" is None,
        both are considered None.
    rec_num : int, optional
        Receiver number to consider for the plot. Default is None.
        If "shot_num" is None or "rec_num" is None,
        both are considered None.
    title : str, optinal
        Title of the plot. Default is None.
    xlimit : 2-tuple, optional
        Limits for the x-axis as (from, to). Default is None.
    ylimit : 2-tuple, optional
        Limits for the y-axis as (from, to). Default is None.
    color : str, optional
        Plot color. Default is 'k' (black).
    figsize : 2-tuple, optional
        Size of the figure as (width, height). Default is None.
    dpi : int, optional
        Dots per inch (resolution) of the image. Default is 150.
    save : str, optional
        Path, filename, and format for saving the figure.
        Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    This function does not return a value.
    """

    delta_t = []
    if shot_num is None or rec_num is None:
        for i in range(len(best_path)):
            for j in range(len(best_path[i])):
                bp = best_path[i][j]
                for k in range(len(bp)):
                    delta_t.append(abs((bp[k][1] - bp[k][0])*dt))
    else:
        bp = best_path[shot_num][rec_num]
        for i in range(len(bp)):
            delta_t.append(abs((bp[i][1] - bp[i][0])*dt))

    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})

    if figsize is not None:
        plt.figure(figsize=figsize)

    binsCnt = len(delta_t)
    bins = np.append(
        np.linspace(min(delta_t), max(delta_t), binsCnt), [np.inf])

    y, x = np.histogram(delta_t, bins=bins)
    y_cum = np.cumsum(y)
    y_cum_norm = (y_cum-np.min(y_cum))/(np.max(y_cum)-np.min(y_cum))
    plt.plot(x[:-1], y_cum_norm, color=color, linewidth=1.5)

    plt.axvline(
        x=dt_cs, color='k', linestyle='--',
        label='$|\Delta t_{cs}| = $' + f'{round(dt_cs, 2)} ms')

    index = (np.abs(x - dt_cs)).argmin()
    plt.plot(
        dt_cs, y_cum_norm[index], color=color, marker='o', markersize=7.0,
        markeredgecolor="black", markerfacecolor="black",
        label=f'{round(y_cum_norm[index]*100, 2)}%')

    if title is not None:
        plt.title(title, fontsize='medium')

    plt.xlabel('$\Delta t$ (ms)')

    plt.ylabel('Cumulative frequency')

    if xlimit is not None:
        plt.xlim(xlimit)

    if ylimit is not None:
        plt.ylim(ylimit)

    plt.legend()

    # plt.box(on=False)

    plt.grid()

    if save is not None:
        plt.savefig(save, dpi=dpi, facecolor='white', edgecolor='none')

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# DTW connections plot
# ===========================================================================


def plot_dtw_connections(seismograms_target, seismograms_curr, dt, dt_cs,
                         best_path, shot_num, rec_num, windows=None,
                         connections='all', title=None, xlimit=None,
                         ylimit=None, figsize=None, dpi=150, save=None,
                         show=True):
    """
    Plot DTW connections between two signals.

    Parameters
    ----------
    seismograms_target : numpy.ndarray
        Target seismograms with shape (shots, receivers, data).
    seismograms_curr : numpy.ndarray
        Current seismograms with shape (shots, receivers, data).
    dt : float
        Time step in milliseconds.
    dt_cs : float
        Cycle-skipping dt in milliseconds.
    best_path : numpy.ndarray
        Best paths with shape (shots, receivers, connections).
        If `windows` is provided, only paths inside the windows are
        considered.
    shot_num : int
        Shot number to consider for the plot.
    rec_num : int
        Receiver number to consider for the plot.
    windows : list, optional
        List of indexes defining the boundaries of each window,
        with shape (shots, receivers, boundaries).
        Default is None.
    connections : str, optinal
        Type of connections to display. Options are:
        'all': Show all connections.
        'higher': Show connections higher than "dt_cs".
        'lower': Show connections lower than "dt_cs".
        Default is 'all'.
    title : str, optinal
        Title of the plot. Default is None.
    xlimit : 2-tuple, optional
        Limits for the x-axis as (from, to). Default is None.
    ylimit : 2-tuple, optional
        Limits for the y-axis as (from, to). Default is None.
    figsize : 2-tuple, optional
        Size of the figure as (width, height). Default is None.
    dpi : int, optional
        Dots per inch (resolution) of the image. Default is 150.
    save : str, optional
        Path, filename, and format for saving the figure.
        Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    This function does not return a value.
    """

    bp = best_path[shot_num][rec_num]

    if windows is None:
        signal_target = seismograms_target[shot_num][rec_num][:]\
            .astype(np.float64)
        signal_curr = seismograms_curr[shot_num][rec_num][:]\
            .astype(np.float64)
    else:
        start = windows[shot_num][rec_num][0]
        stop = windows[shot_num][rec_num][1]
        signal_target = seismograms_target[shot_num][rec_num][start:stop]\
            .astype(np.float64)
        signal_curr = seismograms_curr[shot_num][rec_num][start:stop]\
            .astype(np.float64)

    x_target = np.arange(0, len(signal_target))*dt/1000
    x_curr = np.arange(0, len(signal_curr))*dt/1000

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams['axes.prop_cycle'] = cycler(
        'color', plt.get_cmap('tab20').colors)
    plt.rcParams['lines.linewidth'] = 3.0

    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    axs[0].plot(x_target, signal_target, label='Target signal', color='green')
    axs[1].plot(x_curr, signal_curr, label='Current signal', color='orange')

    for i in range(len(bp)):
        index_target = bp[i][1]
        index_curr = bp[i][0]
        xy_target = (x_target[index_target], signal_target[index_target])
        xy_curr = (x_curr[index_curr], signal_curr[index_curr])
        dt_curr = abs(xy_target[0] - xy_curr[0])*1000
        condition = (xlimit is not None and (xy_target[0] < xlimit[0] or
                     xy_target[0] > xlimit[1] or xy_curr[0] < xlimit[0] or
                     xy_curr[0] > xlimit[1]))
        if condition:
            pass
        else:
            if dt_curr > dt_cs and\
                    (connections == 'all' or connections == 'higher'):
                con = ConnectionPatch(
                    xyA=xy_target, xyB=xy_curr, coordsA=axs[0].transData,
                    coordsB=axs[1].transData, axesA=axs[0], axesB=axs[1],
                    color="r", label='$Delta t > |\Delta t_cs|$')
                fig.add_artist(con)
            elif dt_curr <= dt_cs and\
                    (connections == 'all' or connections == 'lower'):
                con = ConnectionPatch(
                    xyA=xy_target, xyB=xy_curr, coordsA=axs[0].transData,
                    coordsB=axs[1].transData, axesA=axs[0], axesB=axs[1],
                    color="k", label='$Delta t <= |\Delta t_cs|$')
                fig.add_artist(con)

    axs[0].legend(loc='upper right')
    axs[1].legend(loc='lower right')
    plt.legend

    plt.setp(axs, xlim=xlimit, ylim=ylimit)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                    right=False)

    if title is not None:
        plt.title(title, fontsize='medium')

    plt.xlabel('Time (s)')

    plt.ylabel('Amplitude \n \n')

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Seismogram plot in time domain
# ===========================================================================


def plot_seismogram(seismograms, dt, shot_num, gain=1, title=None, ylimit=None,
                    figsize=None, dpi=150, save=None, show=True):
    """
    Seismogram plot function based on matplotlib.

    Parameters
    ----------
    seismograms : numpy.ndarray
        Seismograms with shape (shots, receivers, data).
    dt : float
        Time step in milliseconds.
    shot_num : int
        Shot number to consider for the plot.
    gain : float, optional
        Signal gain (amplitude*gain). Default is 1.
    title : str, optinal
        Title of the plot. Default is None.
    ylimit : 2-tuple, optional
        Limits for the y-axis as (from, to). Default is None.
    figsize : 2-tuple, optional
        Size of the figure as (width, height). Default is None.
    dpi : int, optional
        Dots per inch (resolution) of the image. Default is 150.
    save : str, optional
        Path, filename, and format for saving the figure.
        Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    This function does not return a value.
    """

    seismogram = seismograms[shot_num]

    time_range_values = np.arange(0, len(seismogram[0])*dt, dt)/1000

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.autolayout': True})

    if figsize is not None:
        ax = plt.subplots(figsize=figsize)[1]
    else:
        ax = plt.subplots()[1]

    for i in range(len(seismogram)):

        x = (i+1)+(seismogram[i, :]*gain)

        ax.plot(x, time_range_values, 'k-')
        ax.fill_betweenx(time_range_values, i+1, x, where=(x > (i+1)),
                         color='k')
        ax.fill_betweenx(time_range_values, i+1, x, where=(x < (i+1)),
                         color='white')

    if ylimit is None:
        ylimit = (time_range_values[0], time_range_values[-1])

    ax.set_xlabel('Trace number')
    ax.set_ylabel('Time (s)')
    plt.ylim(ylimit)
    ax.grid(axis='y')
    plt.gca().invert_yaxis()
    plt.xlim(0, len(seismogram)+1)

    if title is not None:
        plt.title(title, fontsize='medium')

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Seismogram as matrix plot in time domain
# ===========================================================================


def plot_seismogram_as_matrix(seismograms, dt, shot_num, pseudo_gain=10,
                              windows=None, title=None, cmap='gray',
                              figsize=None, dpi=150, save=None, show=True):
    """
    Seismogram as matix plot, based on matplotlib.

    Parameters
    ----------
    seismograms : numpy.ndarray
        Seismograms with shape (shots, receivers, data).
    dt : float
        Time step in milliseconds.
    shot_num : int
        Shot number to consider for the plot.
    pseudo_gain : float, optional
        Signals pseudo gain. val = max_value/pseudo_gain,
        vmin=-val and vmax=val.
        Default is 10.
    windows : list, optional
        List of indexes defining the boundaries of each window,
        with shape (shots, receivers, boundaries).
        Default is None.
    title : str, optinal
        Title of the plot. Default is None.
    cmap : str, optional
        Colormap. Default is 'gray'.
    figsize : 2-tuple, optional
        Size of the figure as (width, height). Default is None.
    dpi : int, optional
        Dots per inch (resolution) of the image. Default is 150.
    save : str, optional
        Path, filename, and format for saving the figure.
        Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    This function does not return a value.
    """

    seismogram = seismograms[shot_num]

    t0 = 0
    tn = len(seismogram[0])*dt

    xi = 1
    xf = len(seismogram)

    val = np.amax(np.abs(seismogram))/pseudo_gain

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.autolayout': True})

    if figsize is not None:
        ax = plt.subplots(figsize=figsize)[1]
    else:
        ax = plt.subplots()[1]

    if title is not None:
        plt.title(title, fontsize='medium')

    imag = plt.imshow(seismogram.transpose(1, 0), vmin=-val, vmax=val,
                      aspect='auto', extent=[xi, xf, (tn-t0)/1000, 0],
                      cmap=cmap)
    ax.set_xlabel('Trace number')
    ax.set_ylabel('Time (s)')

    # Plot min/max cutoff amplitude
    if windows is not None:
        for i in range(len(seismogram[0])):
            plt.scatter(i+1, windows[i][0]*dt/1000, s=10, c='red', marker='o')
            plt.scatter(i+1, windows[i][1]*dt/1000, s=10, c='red', marker='o')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(imag, cax=cax, label='Amplitude')

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Plot target, initial, current and intermediate (interpolated) signals
# ===========================================================================


def plot_signal(shot_num, rec_num, dt, seismograms_target, seismograms_curr,
                seismograms_initial=None, seismograms_inter=None,
                windows=None, title=None, xlimit=None, ylimit=None,
                figsize=None, dpi=150, save=None, show=True):
    """
    Plot target, initial, current and intermediate (interpolated) signals,
    based on matplotlib.

    Parameters
    ----------
    shot_num : int
        Shot number to consider for the plot. Default is None.
        If "shot_num" is None or "rec_num" is None,
        both are considered None.
    rec_num : int
        Receiver number to consider for the plot. Default is None.
        If "shot_num" is None or "rec_num" is None,
        both are considered None.
    dt : float
        Time step in milliseconds.
    seismograms_target : numpy.ndarray
        Target seismograms with shape (shots, receivers, data).
    seismograms_curr : numpy.ndarray
        Current seismograms with shape (shots, receivers, data).
    seismograms_initial : numpy.ndarray, optional
        Initial seismograms with shape (shots, receivers, data).
        Default is None.
    seismograms_inter : numpy.ndarray, optional
        Intermediate (interpolated) seismograms with shape
        (shots, receivers, data). Default is None.
    windows : list, optional
        List of indexes defining the boundaries of each window,
        with shape (shots, receivers, boundaries).
        Default is None.
    title : str, optinal
        Title of the plot. Default is None.
    xlimit : 2-tuple, optional
        Limits for the x-axis as (from, to). Default is None.
    ylimit : 2-tuple, optional
        Limits for the y-axis as (from, to). Default is None.
    figsize : 2-tuple, optional
        Size of the figure as (width, height). Default is None.
    dpi : int, optional
        Dots per inch (resolution) of the image. Default is 150.
    save : str, optional
        Path, filename, and format for saving the figure.
        Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    This function does not return a value.
    """

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.autolayout': True})

    if figsize is not None:
        plt.figure(figsize=figsize)

    signal_target = seismograms_target[shot_num][rec_num]
    x_target = np.arange(
        0, len(seismograms_target[shot_num][rec_num])*dt/1000, dt/1000)
    plt.plot(x_target, signal_target, label='Target signal', color='green',
             linestyle='solid', linewidth=1.0)

    signal_curr = seismograms_curr[shot_num][rec_num]
    x_curr = np.arange(
        0, len(seismograms_curr[shot_num][rec_num])*dt/1000, dt/1000)
    plt.plot(x_curr, signal_curr, label='Current signal', color='orange',
             linestyle='solid', linewidth=1.0)

    if seismograms_initial is not None:
        signal_0 = seismograms_initial[shot_num][rec_num]
        x_0 = np.arange(
            0, len(seismograms_initial[shot_num][rec_num])*dt/1000, dt/1000)
        plt.plot(x_0, signal_0, label='Initial signal', color='red',
                 linestyle='solid', linewidth=1.0)

    if seismograms_inter is not None:
        signal_inter = seismograms_inter[shot_num][rec_num]
        x_i = np.arange(
            0, len(seismograms_inter[shot_num][rec_num])*dt/1000, dt/1000)
        plt.plot(x_i, signal_inter, label='Intermediate signal', color='black',
                 linestyle='dashed', linewidth=1.0)

    if windows is not None:
        plt.axvline(windows[shot_num][rec_num][0]*dt/1000, ymin=0.25,
                    ymax=0.75, color='black', lw=1.5, ls='--')
        plt.axvline(windows[shot_num][rec_num][1]*dt/1000, ymin=0.25,
                    ymax=0.75, color='black', lw=1.5, ls='--')

    if title is not None:
        plt.title(title, fontsize='medium')

    if xlimit is not None:
        plt.xlim(xlimit)

    if ylimit is not None:
        plt.ylim(ylimit)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close('all')
# ===========================================================================

# ===========================================================================
# Plot sets of target, initial, current and intermediate (interpo.) signals
# ===========================================================================


def plot_signals(shot_num, dt, seismograms_target, seismograms_curr,
                 rec_num_list=None, seismograms_initial=None,
                 seismograms_inter=None, windows=None, title=None,
                 xlimit=None, figsize=None, dpi=150, save=None,
                 show=True):
    """
    Plot sets of target, initial, current and intermediate (interpolated)
    signals, based on matplotlib.

    Parameters
    ----------
    shot_num : int
        Shot number to consider for the plot. Default is None.
        If "shot_num" is None or "rec_num" is None,
        both are considered None.
    dt : float
        Time step in milliseconds.
    seismograms_target : numpy.ndarray
        Target seismograms with shape (shots, receivers, data).
    seismograms_curr : numpy.ndarray
        Current seismograms with shape (shots, receivers, data).
    rec_num_list : list, optional
        List with receivers number to be considered for the plot.
        Default is None (all receivers will be considered).
    seismograms_initial : numpy.ndarray, optional
        Initial seismograms with shape (shots, receivers, data).
        Default is None.
    seismograms_inter : numpy.ndarray, optional
        Intermediate (interpolated) seismograms with shape
        (shots, receivers, data). Default is None.
    windows : list, optional
        List of indexes defining the boundaries of each window,
        with shape (shots, receivers, boundaries).
        Default is None.
    title : str, optinal
        Title of the plot. Default is None.
    xlimit : 2-tuple, optional
        Limits for the x-axis as (from, to). Default is None.
    figsize : 2-tuple, optional
        Size of the figure as (width, height). Default is None.
    dpi : int, optional
        Dots per inch (resolution) of the image. Default is 150.
    save : str, optional
        Path, filename, and format for saving the figure.
        Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    This function does not return a value.
    """

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.autolayout': True})

    if figsize is None and rec_num_list is None:
        fig, ax = plt.subplots(
            len(seismograms_target[shot_num]), 1, sharex=True)
    elif figsize is not None and rec_num_list is None:
        fig, ax = plt.subplots(
            len(seismograms_target[shot_num]), 1, sharex=True,
            figsize=figsize)
    elif figsize is None and rec_num_list is not None:
        fig, ax = plt.subplots(len(rec_num_list), 1, sharex=True)
    elif figsize is not None and rec_num_list is not None:
        fig, ax = plt.subplots(
            len(rec_num_list), 1, sharex=True, figsize=figsize)

    x = np.arange(
        0, len(seismograms_target[shot_num][0])*dt/1000, dt/1000)

    if seismograms_initial is None and seismograms_inter is None:
        l1 = 0
        l2 = 0

        line_labels = ["Target signal", "Current signal"]

        for n, i in enumerate(
            range(len(seismograms_target[shot_num])))\
                if rec_num_list is None else enumerate(rec_num_list):
            signal_target = seismograms_target[shot_num][i][:]
            signal_curr = seismograms_curr[shot_num][i][:]

            if len(seismograms_target[shot_num]) != 1:
                l1 = ax[n].plot(x, signal_target, label='Target signal',
                                color='green', linestyle='solid',
                                linewidth=0.5)[0]
                l2 = ax[n].plot(x, signal_curr, label='Current signal',
                                color='orange', linestyle='solid',
                                linewidth=0.5)[0]
                ax[n].set_yticks([])
                ax[n].set_ylabel(f'{i+1}   ', rotation=0)

                if windows is not None:
                    ax[n].axvline(windows[shot_num][i][0]*dt/1000, ymin=0.25,
                                  ymax=0.75, color='black', lw=1.0, ls='--')
                    ax[n].axvline(windows[shot_num][i][1]*dt/1000, ymin=0.25,
                                  ymax=0.75, color='black', lw=1.0, ls='--')

            else:
                l1 = ax.plot(x, signal_target, label='Target signal',
                             color='green', linestyle='solid',
                             linewidth=0.5)[0]
                l2 = ax.plot(x, signal_curr, label='Current signal',
                             color='orange', linestyle='solid',
                             linewidth=0.5)[0]
                ax.set_yticks([])
                ax.set_ylabel(f'{i+1}   ', rotation=0)

                if windows is not None:
                    ax.axvline(windows[shot_num][i][0]*dt/1000, ymin=0.25,
                               ymax=0.75, color='black', lw=1.0, ls='--')
                    ax.axvline(windows[shot_num][i][1]*dt/1000, ymin=0.25,
                               ymax=0.75, color='black', lw=1.0, ls='--')

    if seismograms_initial is not None and seismograms_inter is None:
        l1 = 0
        l2 = 0
        l3 = 0

        line_labels = ["Target signal", "Current signal", "Initial signal"]

        for n, i in enumerate(
            range(len(seismograms_target[shot_num])))\
                if rec_num_list is None else enumerate(rec_num_list):
            signal_target = seismograms_target[shot_num][i][:]
            signal_curr = seismograms_curr[shot_num][i][:]
            signal_initial = seismograms_initial[shot_num][i][:]

            if len(seismograms_target[shot_num]) != 1:
                l1 = ax[n].plot(x, signal_target, label='Target signal',
                                color='green', linestyle='solid',
                                linewidth=0.5)[0]
                l2 = ax[n].plot(x, signal_curr, label='Current signal',
                                color='orange', linestyle='solid',
                                linewidth=0.5)[0]
                l3 = ax[n].plot(x, signal_initial, label='Initial signal',
                                color='red', linestyle='solid',
                                linewidth=0.5)[0]
                ax[n].set_yticks([])
                ax[n].set_ylabel(f'{i+1}   ', rotation=0)

                if windows is not None:
                    ax[n].axvline(windows[shot_num][i][0]*dt/1000, ymin=0.25,
                                  ymax=0.75, color='black', lw=1.0, ls='--')
                    ax[n].axvline(windows[shot_num][i][1]*dt/1000, ymin=0.25,
                                  ymax=0.75, color='black', lw=1.0, ls='--')

            else:
                l1 = ax.plot(x, signal_target, label='Target signal',
                             color='green', linestyle='solid',
                             linewidth=0.5)[0]
                l2 = ax.plot(x, signal_curr, label='Current signal',
                             color='orange', linestyle='solid',
                             linewidth=0.5)[0]
                l3 = ax.plot(x, signal_initial, label='Initial signal',
                             color='red', linestyle='solid',
                             linewidth=0.5)[0]
                ax.set_yticks([])
                ax.set_ylabel(f'{i+1}   ', rotation=0)

                if windows is not None:
                    ax.axvline(windows[shot_num][i][0]*dt/1000, ymin=0.25,
                               ymax=0.75, color='black', lw=1.0, ls='--')
                    ax.axvline(windows[shot_num][i][1]*dt/1000, ymin=0.25,
                               ymax=0.75, color='black', lw=1.0, ls='--')

    if seismograms_initial is None and seismograms_inter is not None:
        l1 = 0
        l2 = 0
        l4 = 0

        line_labels = [
            "Target signal", "Current signal", "Intermediate signal"]

        for n, i in enumerate(
            range(len(seismograms_target[shot_num])))\
                if rec_num_list is None else enumerate(rec_num_list):
            signal_target = seismograms_target[shot_num][i][:]
            signal_curr = seismograms_curr[shot_num][i][:]
            signal_inter = seismograms_inter[shot_num][i][:]

            if len(seismograms_target[shot_num]) != 1:
                l1 = ax[n].plot(x, signal_target, label='Target signal',
                                color='green', linestyle='solid',
                                linewidth=0.5)[0]
                l2 = ax[n].plot(x, signal_curr, label='Current signal',
                                color='orange', linestyle='solid',
                                linewidth=0.5)[0]
                l4 = ax[n].plot(x, signal_inter, label='Intermediate signal',
                                color='black', linestyle='dashed',
                                linewidth=0.5)[0]
                ax[n].set_yticks([])
                ax[n].set_ylabel(f'{i+1}   ', rotation=0)

                if windows is not None:
                    ax[n].axvline(windows[shot_num][i][0]*dt/1000,
                                  ymin=0.25, ymax=0.75, color='black',
                                  lw=1.0, ls='--')
                    ax[n].axvline(windows[shot_num][i][1]*dt/1000,
                                  ymin=0.25, ymax=0.75, color='black',
                                  lw=1.0, ls='--')

            else:
                l1 = ax.plot(x, signal_target, label='Target signal',
                             color='green', linestyle='solid',
                             linewidth=0.5)[0]
                l2 = ax.plot(x, signal_curr, label='Current signal',
                             color='orange', linestyle='solid',
                             linewidth=0.5)[0]
                l4 = ax.plot(x, signal_inter, label='Intermediate signal',
                             color='black', linestyle='dashed',
                             linewidth=0.5)[0]
                ax.set_yticks([])
                ax.set_ylabel(f'{i+1}   ', rotation=0)

                if windows is not None:
                    ax.axvline(windows[shot_num][i][0]*dt/1000, ymin=0.25,
                               ymax=0.75, color='black', lw=1.0, ls='--')
                    ax.axvline(windows[shot_num][i][1]*dt/1000, ymin=0.25,
                               ymax=0.75, color='black', lw=1.0, ls='--')

    if seismograms_initial is not None and seismograms_inter is not None:
        l1 = 0
        l2 = 0
        l3 = 0
        l4 = 0

        line_labels = ["Target signal", "Current signal", "Initial signal",
                       "Intermediate signal"]

        for n, i in enumerate(range(len(seismograms_target[shot_num])))\
                if rec_num_list is None else enumerate(rec_num_list):
            signal_target = seismograms_target[shot_num][i][:]
            signal_curr = seismograms_curr[shot_num][i][:]
            signal_initial = seismograms_initial[shot_num][i][:]
            signal_inter = seismograms_inter[shot_num][i][:]

            if len(seismograms_target[shot_num]) != 1:
                l1 = ax[n].plot(x, signal_target, label='Target signal',
                                color='green', linestyle='solid',
                                linewidth=0.5)[0]
                l2 = ax[n].plot(x, signal_curr, label='Current signal',
                                color='orange', linestyle='solid',
                                linewidth=0.5)[0]
                l3 = ax[n].plot(x, signal_initial, label='Initial signal',
                                color='red', linestyle='solid',
                                linewidth=0.5)[0]
                l4 = ax[n].plot(x, signal_inter, label='Intermediate signal',
                                color='black', linestyle='dashed',
                                linewidth=0.5)[0]
                ax[n].set_yticks([])
                ax[n].set_ylabel(f'{i+1}   ', rotation=0)

                if windows is not None:
                    ax[n].axvline(windows[shot_num][i][0]*dt/1000, ymin=0.25,
                                  ymax=0.75, color='black', lw=1.0, ls='--')
                    ax[n].axvline(windows[shot_num][i][1]*dt/1000, ymin=0.25,
                                  ymax=0.75, color='black', lw=1.0, ls='--')

            else:
                l1 = ax.plot(x, signal_target, label='Target signal',
                             color='green', linestyle='solid',
                             linewidth=0.5)[0]
                l2 = ax.plot(x, signal_curr, label='Current signal',
                             color='red', linestyle='solid', linewidth=0.5)[0]
                l3 = ax.plot(x, signal_initial, label='Initial signal',
                             color='red', linestyle='solid',
                             linewidth=0.5)[0]
                l4 = ax.plot(x, signal_inter, label='Intermediate signal',
                             color='black', linestyle='dashed',
                             linewidth=0.5)[0]
                ax.set_yticks([])
                ax.set_ylabel(f'{i+1}   ', rotation=0)

                if windows is not None:
                    ax.axvline(windows[shot_num][i][0]*dt/1000, ymin=0.25,
                               ymax=0.75, color='black', lw=1.0, ls='--')
                    ax.axvline(windows[shot_num][i][1]*dt/1000, ymin=0.25,
                               ymax=0.75, color='black', lw=1.0, ls='--')

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                    right=False)

    if title is not None:
        fig.suptitle(title, fontsize='medium')

    if xlimit is not None:
        plt.xlim(xlimit)

    plt.xlabel('Time (s)')
    plt.ylabel('Trace number')

    if seismograms_initial is None and seismograms_inter is None:
        fig.legend([l1, l2], line_labels, fontsize='medium',
                   loc='upper right', ncol=1)
    if seismograms_initial is not None and seismograms_inter is None:
        fig.legend([l1, l2, l3], line_labels, fontsize='medium',
                   loc='upper right', ncol=1)
    if seismograms_initial is None and seismograms_inter is not None:
        fig.legend([l1, l2, l4], line_labels, fontsize='medium',
                   loc='upper right', ncol=1)
    if seismograms_initial is not None and seismograms_inter is not None:
        fig.legend([l1, l2, l3, l4], line_labels, fontsize='medium',
                   loc='upper right', ncol=1)

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close('all')
# ===========================================================================

# ===========================================================================
# Source plot in time domain
# ===========================================================================


def plot_src(src, dt, title=None, xlimit=None, ylimit=None, color='k',
             figsize=None, dpi=150, save=None, show=True):
    """
    Function for plotting the source wavelet in the time domain
    (using matplotlib).

    Parameters
    ----------
    src : numpy.ndarray
        Array representing the source wavelet.
    dt : float
        Time step in milliseconds.
    title : str, optinal
        Title of the plot. Default is None.
    xlimit : 2-tuple, optional
        Limits for the x-axis as (from, to). Default is None.
    ylimit : 2-tuple, optional
        Limits for the y-axis as (from, to). Default is None.
    color : str, optional
        Plot color. Default is 'k' (black).
    figsize : 2-tuple, optional
        Size of the figure as (width, height). Default is None.
    dpi : int, optional
        Dots per inch (resolution) of the image. Default is 150.
    save : str, optional
        Path, filename, and format for saving the figure.
        Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    This function does not return a value.
    """

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.autolayout': True})

    if figsize is not None:
        plt.figure(figsize=figsize)

    time_range_values = np.arange(0, len(src)*dt, dt)
    plt.plot(time_range_values, src[:], color=color)

    if title is not None:
        plt.title(title, fontsize='medium')

    if xlimit is not None:
        plt.xlim(xlimit)

    if ylimit is not None:
        plt.ylim(ylimit)

    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Standard y versus x plot
# ===========================================================================


def plot_xy(y, title=None, xlabel=None, ylabel=None, xlimit=None, ylimit=None,
            yscale='linear', xrearrange=None, xticks=None, color='k',
            figsize=None, dpi=150, save=None, show=True):
    """
    Standard y versus x plot based on matplotlib.

    Parameters
    ----------
    y : numpy.ndarray
        Data.
    title : str, optinal
        Title of the plot. Default is None.
    xlabel : str, optional
        Label for the x-axis. Default is None.
    ylabel : str, optional
        Label for the x-axis. Default is None.
    xlimit : 2-tuple, optional
        Limits for the x-axis as (from, to). Default is None.
    ylimit : 2-tuple, optional
        Limits for the y-axis as (from, to). Default is None.
    yscale : str, optional
        Scale for the y-axis ('linear', 'log', 'symlog', 'logit').
        Default is 'linear'.
    xrearrange : list of float, optional
        Create a new x array. List with x-axis start and step values.
        Default is None.
    xticks : list of float, optional
        Set the tick locations and labels of the x-axis.
        List with x-axis start and step values.
        Default is None.
    color : str, optional
        Plot color. Default is 'k' (black).
    figsize : 2-tuple, optional
        Size of the figure as (width, height). Default is None.
    dpi : int, optional
        Dots per inch (resolution) of the image. Default is 150.
    save : str, optional
        Path, filename, and format for saving the figure.
        Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    This function does not return a value.
    """

    plt.rcParams.update(plt.rcParamsDefault)

    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})

    if xrearrange is not None:
        xr = np.arange(
            xrearrange[0], len(y)+xrearrange[0], xrearrange[1])
        plt.plot(xr, y, color=color)
    else:
        plt.plot(y, color=color)

    if figsize is not None:
        plt.figure(figsize=figsize)

    if xticks is not None:
        xt = np.arange(xticks[0], len(y)+xticks[0], xticks[1])
        plt.xticks(xt)

    if title is not None:
        plt.title(title, fontsize='medium')

    if xlimit is not None:
        plt.xlim(xlimit)

    if ylimit is not None:
        plt.ylim(ylimit)

    if yscale != 'linear':
        plt.yscale(yscale)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plt.margins(0, 0)

    # plt.box(on=False)

    plt.grid()

    if save is not None:
        plt.savefig(save, dpi=dpi, facecolor='white', edgecolor='none')

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================


