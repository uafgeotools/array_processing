import sys
import numpy as np
from obspy.geodetics.base import calc_vincenty_inverse
from ..algorithms.wlsqva import wlsqva


def wlsqva_proc(stf, rij, tvec, windur, winover):
    """
    Module to run wlsqva array processing

    example: vel, az, mdccm, t = wlsqva_proc(stf, rij, tvec, windur, winover)

    Note that tau, sig_tau, s, and xij are not currently returned (but should be!)

    Args:
        stf : (d,n) obspy stream of filtered data for n sensors
        rij : (d,n) array of `n` sensor coordinates as [easting, northing, {elevation}]
            column vectors in `d` dimension, units are km
        tvec : (n) time vector in datenum format
        windur : scalar, array processing window length in seconds
        winover : scalar, array processing window overlap

    Returns:
        vel: vector of trace velocities (km/s)
        az : vector of back-azimuths (deg from N)
        mdccm : vector of median of the xcorr max between sensor pairs (0-1)
        t : vector of time windows (datenum)
    """

    nchans = len(stf)
    npts = len(stf[0].data)
    fs = stf[0].stats.sampling_rate

    #set up windows
    winlensamp = windur*fs
    sampinc = int((1-winover) * winlensamp)
    its = np.arange(0, npts, sampinc)
    nits = len(its)-1

    vel = np.zeros(nits)
    az = np.zeros(nits)
    mdccm = np.zeros(nits)
    t = np.zeros(nits)

    #put stream data into matrix
    data = np.empty((npts, nchans))
    for i, tr in enumerate(stf):
        data[:, i] = tr.data

    #run wlsqva for each data window and save median of peak xcorr
    print('Running wlsqva for %d windows' % nits)
    for j in range(nits):
        ptr = int(its[j]), int(its[j]+winlensamp)
        vel[j], az[j], tau, cmax, sig_tau, s, xij = wlsqva(data[ptr[0]:ptr[1], :], rij, fs)
        mdccm[j] = np.median(cmax)
        #keep time value from center of window
        try:
            t[j] = tvec[ptr[0]+int(winlensamp/2)]
        except:
            t[j] = t.max()
        tmp = int(j/nits*100)
        sys.stdout.write("\r%d%%" % tmp)
        sys.stdout.flush()
    print('Done\n')

    return vel, az, mdccm, t, data


def getrij(latlist, lonlist):
    """
    Returns the projected geographic positions in X-Y. Points are calculated
    with the Vicenty inverse from obspy and will have a zero-mean. Typically
    used for array locations.

    Args:
        latlist : list of latitude points
        lonlist : list of longitude points

    Returns:
        rij : numpy array with the first row corresponding to cartesian
            "X"-coordinates and the second row corresponding to cartesian
            "Y"-coordinates.
    """

    # Basic error checking
    latsize = len(latlist)
    lonsize = len(lonlist)
    if latsize != lonsize:
        raise ValueError('latsize != lonsize')

    xnew = np.zeros((latsize, ))
    ynew = np.zeros((lonsize, ))

    for i in range(1, lonsize):
        # Obspy defaults are set as: a = 6378137.0, f = 0.0033528106647474805
        # This is apparently the WGS84 ellipsoid.
        delta, az, __ = calc_vincenty_inverse(latlist[0], lonlist[0], latlist[i], lonlist[i])
        # Converting azimuth to radians
        az = (450 - az) % 360

        xnew[i] = delta/1000*np.cos(az*np.pi/180)
        ynew[i] = delta/1000*np.sin(az*np.pi/180)

    # Removing the mean
    xnew = xnew - np.mean(xnew)
    ynew = ynew - np.mean(ynew)

    rij = np.array([xnew.tolist(), ynew.tolist()])

    return rij
