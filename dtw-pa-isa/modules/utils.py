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
from scipy import interpolate, signal, stats, spatial
import copy
from dtaidistance import dtw
# ===========================================================================

# ===========================================================================
# Exportable functions
# ===========================================================================
__all__ = [
    'freq_cs_dt',
    'get_windows',
    'is_mtli',
    'is_pi',
    'normalize_seismograms',
    'sgs_dtw_best_paths_and_distances',
    'src_cs_dt']
# ===========================================================================

# ===========================================================================
# Calculate dt for mitigating cycle-skipping based on the half period
# ===========================================================================


def freq_cs_dt(cs_freq, pcent_befor=10):
    """
    This function returns the dt required to avoid cycle-skipping based on
    the half-period.

    Parameters
    ----------
    cs_freq : float
        Frequency (in Hz) to be considered for avoiding cycle-skipping.
    pcent_before : float, optional
        cs_dt = (100% - pcent_before%) * critical_cs_dt, where cs_dt is the
        dt required to avoid cycle-skipping, and critical_cs_dt is,
        theoretically, the dt between the global minimum and the first local
        maximum. Default is 10.

    Returns
    -------
    cs_dt : float
        The dt required to avoid cycle-skipping, in milliseconds.
    """

    # Cycle-skipping dt in ms
    critical_cs_dt = (0.5/cs_freq)/0.001
    cs_dt = ((100 - pcent_befor)/100)*critical_cs_dt

    return cs_dt
# ===========================================================================

# ===========================================================================
# Function for defining windows for two sets of seismograms
# ===========================================================================


def get_windows(seismograms_target, seismograms_curr, cutoff_amp, ppet=0):
    """
    This function defines windows on pairs of signal (target and
    initial/current) based on the specified cutoff amplitude.

    Parameters
    ----------
    seismograms_target : numpy.ndarray
        Target seismograms with shape (shots, receivers, data).
    seismograms_curr : numpy.ndarray
        Initial or current seismograms with shape (shots, receivers, data).
    cutoff_amp : float
        Cutoff amplitude.
    ppet : int, optional
        Pre- and post-event time. This parameter specifies how many points
        around the cutoff amplitude are included in the window for event
        picking. The distance between points corresponds to the critical dt.
        Default is 0.

    Returns
    -------
    windows : list
        List of indexes defining the boundaries of each window.
        The structure of "windows" is:
        windows[shot number][receiver number][i],
        where i = 0 for the minimum index or
        1 for the maximum index of the windows.
    """

    nshots = len(seismograms_target)
    nreceivers = len(seismograms_target[0])
    npoints = len(seismograms_target[0][0])

    windows = [None for _ in range(nshots)]

    for i in range(nshots):
        windows[i] = []

        for j in range(nreceivers):
            windows[i].append([])

            # Here the seismograms are scanned forward
            # until the cutoff amplitude is found
            for k in range(npoints):
                if np.abs(seismograms_target[i][j][k]) >= \
                        cutoff_amp or \
                        np.abs(seismograms_curr[i][j][k]) >= \
                        cutoff_amp:
                    if k-ppet < 0:
                        windows[i][j].append(0)
                    else:
                        windows[i][j].append(k-ppet)
                    break

            # Here the seismograms are scanned backward
            # until the cutoff amplitude is found
            for m in range(npoints-1, 0, -1):
                if np.absolute(seismograms_target[i][j][m]) >= \
                        cutoff_amp or \
                        np.absolute(
                            seismograms_curr[i][j][m] >= cutoff_amp):
                    if m+ppet > npoints-1:
                        windows[i][j].append(npoints-1)
                    else:
                        windows[i][j].append(m+ppet)
                    break

    return windows
# ===========================================================================

# ===========================================================================
# Intermediate seismograms with Maximum Time Lag Interpolation (MTLI)
# ===========================================================================


def is_mtli(seismograms_target, seismograms_curr, dt, dt_cs, dt_cs_percent=90,
            smooth=False, f_nyquist=None, windows=None):
    """
    Perform Maximum Time Lag Interpolation (MTLI) based on DTW to
    generate intermediate seismograms for mitigating cycle-skipping in FWI.

    Parameters
    ----------
    seismograms_target : numpy.ndarray
        Target seismograms with shape (shots, receivers, data).
    seismograms_curr : numpy.ndarray
        Initial or current seismograms with shape (shots, receivers, data).
    dt : float
        Time step in milliseconds.
    dt_cs : float
        Cycle-skipping dt in milliseconds.
    dt_cs_percent : float, optional
        The percentage of dt_cs to be considered for interpolation.
        Default is 90.
    smooth : bool, optional
        Whether to apply the Savitzky–Golay smoothing filter to the
        intermediate (interpolated) seismograms.
        If "True", "f_nyquist" id required. Default is False.
    f_nyquist : float, optional
        Nyquist frequency used in the Savitzky–Golay smoothing filter.
        Required if "smooth=True". Default is None.
    windows : list, optional
        List of indexes defining the boundaries of each window, with shape
        (shots, receivers, boundaries).
        Default is None.

    Returns
    -------
    seismograms_interpolated : numpy.ndarray
        The intermediate (interpolated) seismograms. The structure of
        the numpy.ndarray is:
        seismograms_interpolated[shot number][receiver number][data].
    sgs_interpolated_are_sgs_obs : bool
        Indicates whether "seismograms_interpolated==seismograms_target".
        If True, it implies that "seismograms_curr" is already sufficiently
        close to "seismograms_target".
    dtw_best_paths : numpy.ndarray
        Array containing the DTW optimal paths, one for each pair of signals,
        considering all seismograms.
        Each path describes the correspondence between elements of the
        target and initial/current signals. The structure of the array is:
        dtw_best_paths[shot number][receiver number][connection].
        If "windows=None", DTW is applied to the entire signal.
        If "windows" is provided, DTW is constrained within the specified
        window, while outside the relationship remains between equivalent
        elements.
    dtw_distances : numpy.ndarray
        Array containing the DTW best path distances, one for each pair of
        signals, considering all seismograms.
        The structure of the array is:
        dtw_distances[shot number][receiver number].
        If "windows=None", DTW is applied to the entire signal.
        If "windows" is provided, DTW is constrained within the specified
        window, while outside the relationship remains between equivalent
        elements (distance equal to zero).
    sgs_distance : float
        It is the normalized distance for the set of seismograms.
        sgs_distance = sum(dtw_distances)/(nshots*nreceivers).
    """

    dtw_best_paths, _, dtw_distances, sgs_distance = \
        sgs_dtw_best_paths_and_distances(
            seismograms_target, seismograms_curr, windows=windows)

    # Dimensionless cycle-skipping dt
    dt_cs_dl = dt_cs*(dt_cs_percent/100)/dt

    nshots = len(seismograms_curr)
    nreceivers = len(seismograms_curr[0])

    seismograms_interpolated = []

    # Variable that will informs if "seismograms_curr" is already
    # sufficiently close to "seismograms_target", that is,
    # "seismograms_interpolated==seismograms_target".
    sgs_interpolated_are_sgs_obs = True

    for i in range(nshots):
        seismograms_interpolated.append([])
        for j in range(nreceivers):
            seismograms_interpolated[i].append([])

            signal_obs = seismograms_target[i][j][:].astype(np.float64)
            signal_curr = seismograms_curr[i][j][:].astype(np.float64)
            dtw_best_path_current = dtw_best_paths[i][j]

            x = []  # Time axis for the interpolated signal
            y = []  # Amplitude axis for the interpolated signal

            for k in range(len(dtw_best_path_current)):
                # DTW dimensionless current time diffrences
                dtw_best_path_tdiff = (
                    dtw_best_path_current[k][1] -
                    dtw_best_path_current[k][0])

                # If the DTW time differences exceeds the critical 
                # cycle-skipping dt (dt_cs), the interpolation is performed.
                # Otherwise, the intermediate signal will match the target
                # one.
                if np.abs(dtw_best_path_tdiff) > dt_cs_dl:
                    sgs_interpolated_are_sgs_obs = False
                    frac = dt_cs_dl/np.abs(dtw_best_path_tdiff)
                else:
                    frac = 1

                x.append(dtw_best_path_current[k][0] +
                         dtw_best_path_tdiff*frac)
                y.append(signal_curr[dtw_best_path_current[k][0]] +
                         (signal_obs[dtw_best_path_current[k][1]] -
                         signal_curr[dtw_best_path_current[k][0]])*frac)

            f = interpolate.interp1d(x, y)
            x = np.arange(0, len(signal_curr), 1)
            y = f(x)

            # Savitzky–Golay smoothing filter
            if smooth is True:
                npoints = 1/(2*(f_nyquist*1.25)*(dt/1000))
                npoints_odd = int(np.ceil(npoints) // 2 * 2 + 1)
                y = signal.savgol_filter(y, npoints_odd, 2)

            seismograms_interpolated[i][j].append(y.tolist())

    seismograms_interpolated = np.array(seismograms_interpolated)
    seismograms_interpolated = np.squeeze(seismograms_interpolated, axis=2)

    return seismograms_interpolated, sgs_interpolated_are_sgs_obs, \
        dtw_best_paths, dtw_distances, sgs_distance
# ===========================================================================

# ===========================================================================
# Intermediate seismograms with Proportional Interpolation (PI)
# ===========================================================================


def is_pi(seismograms_target, seismograms_curr, dt, dt_cs, dt_cs_percent=90,
          smooth=False, f_nyquist=None, windows=None):
    """
    Perform Proportional Interpolation (PI) based on DTW to
    generate intermediate seismograms for mitigating cycle-skipping in FWI.

    Parameters
    ----------
    seismograms_target : numpy.ndarray
        Target seismograms with shape (shots, receivers, data).
    seismograms_curr : numpy.ndarray
        Initial or current seismograms with shape (shots, receivers, data).
    dt : float
        Time step in milliseconds.
    dt_cs : float
        Cycle-skipping dt in milliseconds.
    dt_cs_percent : float, optional
        The percentage of dt_cs to be considered for interpolation.
        Default is 90.
    smooth : bool, optional
        Whether to apply the Savitzky–Golay smoothing filter to the
        intermediate (interpolated) seismograms.
        If "True", "f_nyquist" id required. Default is False.
    f_nyquist : float, optional
        Nyquist frequency used in the Savitzky–Golay smoothing filter.
        Required if "smooth=True". Default is None.
    windows : list, optional
        List of indexes defining the boundaries of each window, with shape
        (shots, receivers, boundaries).
        Default is None.

    Returns
    -------
    seismograms_interpolated : numpy.ndarray
        The intermediate (interpolated) seismograms. The structure of
        the numpy.ndarray is:
        seismograms_interpolated[shot number][receiver number][data].
    sgs_interpolated_are_sgs_obs : bool
        Indicates whether "seismograms_interpolated==seismograms_target".
        If True, it implies that "seismograms_curr" is already sufficiently
        close to "seismograms_target".
    dtw_best_paths : numpy.ndarray
        Array containing the DTW optimal paths, one for each pair of signals,
        considering all seismograms.
        Each path describes the correspondence between elements of the
        target and initial/current signals. The structure of the array is:
        dtw_best_paths[shot number][receiver number][connection].
        If "windows=None", DTW is applied to the entire signal.
        If "windows" is provided, DTW is constrained within the specified
        window, while outside the relationship remains between equivalent
        elements.
    dtw_distances : numpy.ndarray
        Array containing the DTW best path distances, one for each pair of
        signals, considering all seismograms.
        The structure of the array is:
        dtw_distances[shot number][receiver number].
        If "windows=None", DTW is applied to the entire signal.
        If "windows" is provided, DTW is constrained within the specified
        window, while outside the relationship remains between equivalent
        elements (distance equal to zero).
    sgs_distance : float
        It is the normalized distance for the set of seismograms.
        sgs_distance = sum(dtw_distances)/(nshots*nreceivers).
    """

    dtw_best_paths, _, dtw_distances, sgs_distance = \
        sgs_dtw_best_paths_and_distances(
            seismograms_target, seismograms_curr, windows=windows)

    # Dimensionless cycle-skipping dt
    dt_cs_dl = dt_cs*(dt_cs_percent/100)/dt

    nshots = len(seismograms_curr)
    nreceivers = len(seismograms_curr[0])

    seismograms_interpolated = []

    # Variable that will informs if "seismograms_curr" is already
    # sufficiently close to "seismograms_target", that is,
    # "seismograms_interpolated==seismograms_target".
    sgs_interpolated_are_sgs_obs = True

    for i in range(nshots):
        seismograms_interpolated.append([])
        for j in range(nreceivers):
            seismograms_interpolated[i].append([])

            signal_obs = seismograms_target[i][j][:].astype(np.float64)
            signal_curr = seismograms_curr[i][j][:].astype(np.float64)
            dtw_best_path_current = dtw_best_paths[i][j]

            dtw_best_path_tdiff = []  # DTW dimensionless time diffrences
            x = []  # Time axis for the interpolated signal
            y = []  # Amplitude axis for the interpolated signal

            # All DTW dimensionless time differences
            for k in range(len(dtw_best_path_current)):

                dtw_best_path_tdiff.append(
                    dtw_best_path_current[k][1] -
                    dtw_best_path_current[k][0])

            # Maximum DTW time differences and related fraction
            dt_dtw_max = np.amax(np.abs(dtw_best_path_tdiff))
            frac = dt_cs_dl/dt_dtw_max

            # DTW-based interpolation for intermediate signal generation.
            # If the maximum DTW time differences exceeds the critical 
            # cycle-skipping dt (dt_cs), the interpolation is performed.
            # Otherwise, the intermediate signal will match the target one.
            if frac < 1:

                sgs_interpolated_are_sgs_obs = False

                for k in range(len(dtw_best_path_current)):

                    x.append(dtw_best_path_current[k][0] +
                             dtw_best_path_tdiff[k]*frac)
                    y.append(signal_curr[dtw_best_path_current[k][0]] +
                             (signal_obs[dtw_best_path_current[k][1]] -
                             signal_curr[dtw_best_path_current[k][0]])*frac)

                f = interpolate.interp1d(x, y)
                x = np.arange(0, len(signal_curr), 1)
                y = f(x)

                # Savitzky–Golay smoothing filter
                if smooth is True:
                    if f_nyquist is None:
                        print('The "f_nyquist" parameter is required for'
                              'the smoothing filter.')
                        return
                    else:
                        npoints = 1/(2*(f_nyquist*1.25)*(dt/1000))
                        npoints_odd = int(np.ceil(npoints) // 2 * 2 + 1)
                        y = signal.savgol_filter(y, npoints_odd, 2)

                seismograms_interpolated[i][j].append(y.tolist())

            else:

                seismograms_interpolated[i][j].append(signal_obs.tolist())

    seismograms_interpolated = np.array(seismograms_interpolated)
    seismograms_interpolated = np.squeeze(seismograms_interpolated, axis=2)

    return seismograms_interpolated, sgs_interpolated_are_sgs_obs, \
        dtw_best_paths, dtw_distances, sgs_distance
# ===========================================================================

# ===========================================================================
# Time domain seismograms normalization
# ===========================================================================


def normalize_seismograms(seismograms, norm_type='z_score'):
    """
    Function for normalizing seismograms in the time domain.

    Parameters
    ----------
    seismograms : numpy.ndarray
        Seismograms with shape (shots, receivers, data).
    norm_type : str, optional
        Type of normalization to apply. The available options are:
        'min_max' or 'z_score'. Default is 'z_score'.

    Returns
    -------
    sgs : numpy.ndarray
        Normalized seismograms. The structure of "sgs" is:
        sgs[shot number][receiver number][data].
    """

    sgs = copy.deepcopy(seismograms)

    nshots = len(sgs)
    nreceivers = len(sgs[0])

    for i in range(nshots):
        if norm_type == 'min_max':
            for j in range(nreceivers):
                min = np.amin(sgs[i][j][:])
                max = np.amax(sgs[i][j][:])
                sgs[i][j][:] = (
                    sgs[i][j][:]-min)/(max-min)

        if norm_type == 'z_score':
            for j in range(nreceivers):
                sgs[i][j][:] = stats.zscore(sgs[i][j][:])

    return sgs
# ===========================================================================

# ===========================================================================
# DTW best paths and distances between each two signals of two sets of
# seismograms
# ===========================================================================


def sgs_dtw_best_paths_and_distances(seismograms_target, seismograms_curr,
                                     windows=None):
    """
    Calculate the optimal paths between each two signals of two sets of
    seismograms using the Dynamic Time Warping (DTW) method.

    Parameters
    ----------
    seismograms_target : numpy.ndarray
        Target seismograms with shape (shots, receivers, data).
    seismograms_curr : numpy.ndarray
        Initial or current seismograms with shape (shots, receivers, data).
    windows : list, optional
        List of cutoff amplitude index for all signals of the seismograms.
        Default is None.

    Returns
    -------
    dtw_best_paths : numpy.ndarray
        Array containing the DTW optimal paths, one for each pair of signals,
        considering all seismograms.
        Each path describes the correspondence between elements of the
        target and initial/current signals. The structure of the array is:
        dtw_best_paths[shot number][receiver number][connection].
        If "windows=None", DTW is applied to the entire signal.
        If "windows" is provided, DTW is constrained within the specified
        window, while outside the relationship remains between equivalent
        elements.
    dtw_short_best_paths : numpy.ndarray
        Array containing the DTW short best paths, one for each pair of
        signals, considering all seismograms, restricted to the regions
        defined by "windows". The structure of the array is:
        dtw_short_best_paths[shot number][receiver number][connection].
    dtw_distances : numpy.ndarray
        Array containing the DTW best path distances, one for each pair of
        signals, considering all seismograms.
        The structure of the array is:
        dtw_distances[shot number][receiver number].
        If "windows=None", DTW is applied to the entire signal.
        If "windows" is provided, DTW is constrained within the specified
        window, while outside the relationship remains between equivalent
        elements (distance equal to zero).
    sgs_distance : float
        It is the normalized distance for the set of seismograms.
        sgs_distance = sum(dtw_distances)/(nshots*nreceivers).
    """

    nshots = len(seismograms_target)
    nreceivers = len(seismograms_target[0])
    npoints = len(seismograms_target[0][0])

    dtw_short_best_paths = [None for _ in range(nshots)]
    dtw_best_paths = [None for _ in range(nshots)]
    dtw_distances = [None for _ in range(nshots)]
    sgs_distance = 0.0

    def with_win(i):
        dtw_short_best_paths[i] = []
        dtw_best_paths[i] = []
        dtw_distances[i] = []
        nonlocal sgs_distance

        for j in range(nreceivers):
            dtw_best_paths[i].append([])

            for k in range(windows[i][j][0]):
                dtw_best_paths[i][j].append((k, k))

            start = windows[i][j][0]
            stop = windows[i][j][1]

            signal_obs = seismograms_target[i][j][start:stop].astype(np.float64)
            signal_curr = seismograms_curr[i][j][start:stop].astype(np.float64)

            # DTW best path
            best_path, dist = dtw.warping_path_fast(
                signal_curr, signal_obs, include_distance=True)

            dtw_short_best_paths[i].append(best_path)
            dtw_distances[i].append(dist)
            sgs_distance += dist

            for k in range(len(best_path)):
                dtw_best_paths[i][j].append(
                    (best_path[k][0] + start, best_path[k][1] + start))

            for k in range(stop, npoints):
                dtw_best_paths[i][j].append((k, k))

    def without_win(i):
        dtw_best_paths[i] = []
        dtw_distances[i] = []
        nonlocal sgs_distance

        for j in range(nreceivers):
            signal_obs = seismograms_target[i][j][:].astype(np.float64)
            signal_curr = seismograms_curr[i][j][:].astype(np.float64)

            # DTW best path
            best_path, dist = dtw.warping_path_fast(
                signal_curr, signal_obs, include_distance=True)

            dtw_best_paths[i].append(best_path)
            dtw_distances[i].append(dist)
            sgs_distance += dist

    if windows is not None:
        for i in range(nshots):
            with_win(i)
        dtw_short_best_paths = np.asanyarray(
            dtw_short_best_paths, dtype=object)
        dtw_best_paths = np.asanyarray(
            dtw_best_paths, dtype=object)
        dtw_distances = np.asanyarray(
            dtw_distances, dtype=object)
    else:
        for i in range(nshots):
            without_win(i)
        dtw_short_best_paths = None
        dtw_best_paths = np.asanyarray(
            dtw_best_paths, dtype=object)
        dtw_distances = np.asanyarray(
            dtw_distances, dtype=object)

    sgs_distance = sgs_distance/(nshots*nreceivers)

    return dtw_best_paths, dtw_short_best_paths, dtw_distances, \
        sgs_distance
# ===========================================================================

# ===========================================================================
# Calculate dt for mitigating cycle-skipping based on the source wavelet
# ===========================================================================


def src_cs_dt(src, dt):
    """
    This function returns the dt required to mitigate cycle-skipping based on
    the source wavelet.

    Parameters
    ----------
    src : numpy.ndarray
        Array representing the source wavelet.
    dt : float
        Time step in milliseconds.

    Returns
    -------
    cs_dt : float
        The dt required to avoid cycle-skipping, in milliseconds.
    objective_function : list
        The objective function computed by shifting two source wavelets
        relative to each other.
    min_index : int
        The index of the minimum value in the objective function.
    max_index : int
        The index of the closest local maximum relative to the global
        minimum in the objective function.
    """

    length = len(src)
    objective_function = []
    src_a = np.zeros(2*length, dtype=np.float64).tolist()
    src_a[length:length] = src[:].tolist()
    for i in range(0, 2*length):
        src_b = np.zeros(2*length, dtype=np.float64).tolist()
        src_b[i:i] = src[:].tolist()
        obj_value = 0.5*(spatial.distance.euclidean(src_a, src_b))**2
        objective_function.append(obj_value)

    obj_min = min(objective_function)
    min_index = objective_function.index(obj_min)

    for i in range(min_index-1):
        max1 = objective_function[min_index - (i+1)]
        max2 = objective_function[min_index - (i+2)]
        if max2 < max1:
            max_a_index = min_index - (i+1)
            break

    for i in range(len(objective_function)-min_index-1):
        max1 = objective_function[min_index + (i+1)]
        max2 = objective_function[min_index + (i+2)]
        if max2 < max1:
            max_b_index = min_index + (i+1)
            break

    if (min_index-max_a_index) < (max_b_index-min_index):
        max_index = max_a_index
    else:
        max_index = max_b_index

    cs_dt = abs(min_index-max_index)*dt

    return cs_dt, objective_function, min_index, max_index
# ===========================================================================