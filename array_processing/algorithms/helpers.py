import sys
import numpy as np
from ..algorithms.wlsqva import wlsqva
from obspy.geodetics import gps2dist_azimuth


def wlsqva_proc(stf, rij, windur, winover):
    r"""
    Module to run :func:`~array_processing.algorithms.wlsqva.wlsqva` array
    processing.

    Args:
        stf (:class:`~obspy.core.stream.Stream`): Filtered data for ``n``
            sensors
        rij: ``(d,n)`` array of ``n`` sensor coordinates as [easting, northing,
            {elevation}] column vectors in ``d`` dimension, units are km
        windur (int or float): Array processing window length [s]
        winover (int or float): Array processing window overlap

    Returns:
        tuple: Tuple containing:

        - **vel** – Array of trace velocities (km/s)
        - **az** – Array of back-azimuths (deg. from N)
        - **sig_tau** – Array of plane wave model assumption violation
          estimates for non-planar arrivals (0 = planar)
        - **mdccm** – Array of median of the xcorr max between sensor pairs,
          defined on :math:`[0,1]`
        - **t** – Vector of time windows as `Matplotlib dates
          <https://matplotlib.org/3.1.3/api/dates_api.html>`__
        - **data** – `stf` as a :class:`numpy.ndarray`

    Warning:
        `tau`, `s`, and `xij` are not currently returned!
    """

    tvec = stf[0].times('matplotlib')

    nchans = len(stf)
    npts = len(stf[0].data)
    fs = stf[0].stats.sampling_rate

    # set up windows
    winlensamp = windur*fs
    sampinc = int((1-winover) * winlensamp)
    its = np.arange(0, npts, sampinc)
    nits = len(its)-1

    vel = np.zeros(nits)
    az = np.zeros(nits)
    mdccm = np.zeros(nits)
    sig_tau = np.zeros(nits)
    t = np.zeros(nits)

    #put stream data into matrix
    data = np.empty((npts, nchans))
    for i, tr in enumerate(stf):
        data[:, i] = tr.data

    #run wlsqva for each data window and save median of peak xcorr
    print('Running wlsqva for %d windows' % nits)
    for j in range(nits):
        ptr = int(its[j]), int(its[j]+winlensamp)
        vel[j], az[j], tau, cmax, sig_tau[j], s, xij = wlsqva(data[ptr[0]:ptr[1], :], rij, fs)
        mdccm[j] = np.median(cmax)
        # Keep time value from center of window.
        try:
            t[j] = tvec[ptr[0]+int(winlensamp/2)]
        except:
            t[j] = t.max()
        tmp = int((j+1)/nits*100)
        sys.stdout.write("\r%d%%" % tmp)
        sys.stdout.flush()
    print('\nDone\n')

    return vel, az, sig_tau, mdccm, t, data


def getrij(latlist, lonlist):
    r"""
    Calculate ``rij`` from lat-lon. Returns the projected geographic positions
    in :math:`x`–:math:`y` with zero-mean. Typically used for array locations.

    Args:
        latlist (list): List of latitude points
        lonlist (list): List of longitude points

    Returns:
        :class:`numpy.ndarray` with the first row corresponding to Cartesian
        :math:`x`-coordinates and the second row corresponding to Cartesian
        :math:`y`-coordinates, in units of km
    """

    latsize = len(latlist)
    lonsize = len(lonlist)

    # Basic error checking
    if latsize != lonsize:
        raise ValueError('latsize != lonsize')

    xnew = np.zeros((latsize, ))
    ynew = np.zeros((lonsize, ))
    for i in range(1, lonsize):
        # WGS84 ellipsoid
        dist, az, _ = gps2dist_azimuth(latlist[0], lonlist[0],
                                       latlist[i], lonlist[i])
        # Convert azimuth in degrees to angle in radians
        ang = np.deg2rad((450 - az) % 360)
        # Convert from m to km, do trig
        xnew[i] = (dist / 1000) * np.cos(ang)
        ynew[i] = (dist / 1000) * np.sin(ang)

    # Remove the mean
    xnew = xnew - xnew.mean()
    ynew = ynew - ynew.mean()

    # Form rij array
    rij = np.vstack((xnew, ynew))

    return rij
