import sys
import numpy as np
from ..algorithms.wlsqva import wlsqva
from obspy.geodetics import gps2dist_azimuth


def wlsqva_proc(stf, rij, windur, winover):
    """
    Module to run wlsqva array processing

    example: vel, az, mdccm, t = wlsqva_proc(stf, rij, tvec, windur, winover)

    Note that tau, sig_tau, s, and xij are not currently returned (but should be!)

    Args:
        stf : (d,n) obspy stream of filtered data for n sensors
        rij : (d,n) array of `n` sensor coordinates as [easting, northing, {elevation}]
            column vectors in `d` dimension, units are km
        windur : scalar, array processing window length in seconds
        winover : scalar, array processing window overlap

    Returns:
        vel: array of trace velocities (km/s)
        az : array of back-azimuths (deg from N)
        sig_tau: array of plane wave model assumption violation estimates
            for non-planar arrivals (0=planar)
        mdccm : array of median of the xcorr max between sensor pairs (0-1)
        t : vector of time windows (datenum)
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
    """Calculate rij from lat-lon

    Returns the projected geographic positions in X-Y with zero-mean. Typically
    used for array locations.

    Args:
        latlist : list of latitude points
        lonlist : list of longitude points

    Returns:
        rij : numpy array with the first row corresponding to cartesian
            "X"-coordinates and the second row corresponding to cartesian
            "Y"-coordinates, in units of km.
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
