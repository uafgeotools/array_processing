import sys
import numpy as np
from ..algorithms.wlsqva import wlsqva
from obspy.geodetics import gps2dist_azimuth


def wlsqva_proc(stf, rij, tvec, windur, winover):
    """
    Module to run wlsqva array processing
    @ authors: David Fee, Curt Szuberla

    example: vel,az,mdccm,t=wlsqva_proc(stf,rij,tvec,windur,winover)

    stf=(d,n) obspy stream of filtered data for n sensors
    rij=(d,n) array of `n` sensor coordinates as [easting, northing, {elevation}]
        column vectors in `d` dimension, units are km
    tvec=(n) time vector in datenum format
    windur=scalar, array processing window length in seconds
    winover=scalar, array processing window overlap

    vel=vector of trace velocities (km/s)
    az=vector of back-azimuths (deg from N)
    mdccm=median of the xcorr max between sensor pairs (0-1)
    t=vector of time windows (datenum)
    """

    nchans=len(stf)
    npts=len(stf[0].data)
    Fs=stf[0].stats.sampling_rate

    #set up windows
    winlensamp=windur*Fs
    sampinc=int((1-winover)*winlensamp)
    its=np.arange(0,npts,sampinc)
    nits=len(its)-1

    vel=np.zeros(nits)
    az=np.zeros(nits)
    mdccm=np.zeros(nits)
    t=np.zeros(nits)

    #put data from stream into matrix
    data=np.empty((npts,nchans))
    for i,tr in enumerate(stf):
        data[:,i] = tr.data#*tr.stats.calib

    #run wlsqva for each data window and save median of peak xcorr
    print('Running wlsqva for %d windows' % nits)
    for j in range(nits):
        ptr=int(its[j]),int(its[j]+winlensamp)
        vel[j], az[j], tau, cmax, sig_tau, s, xij=wlsqva(data[ptr[0]:ptr[1],:], rij, Fs)
        mdccm[j]=np.median(cmax)
        #keep time value from center of window
        try:
            t[j]=tvec[ptr[0]+int(winlensamp/2)]
        except:
            t[j]=t.max()
        tmp=int(j/nits*100)
        sys.stdout.write("\r%d%%" % tmp)
        sys.stdout.flush()
    print('Done\n')

    return vel,az,mdccm,t,data


def getrij(latlist, lonlist):
    """Calculate rij from lat-lon

    Returns the projected geographic positions in X-Y. Points are calculated
    with the Vicenty inverse and will have a zero-mean.

    @ authors: Jordan W. Bishop, David Fee, and Liam Toney

    Args:
      latlist - a list of latitude points
      lonlist - a list of longitude points

    Outputs:
      rij - a numpy array with the first row corresponding to cartesian
            "X"-coordinates and the second row corresponding to cartesian
            "Y"-coordinates, in units of km.
    """

    latsize = len(latlist)
    lonsize = len(lonlist)

    # Basic error checking
    if latsize != lonsize:
        raise ValueError('latsize != lonsize')

    # Now to calculate
    xnew = np.zeros((latsize, ))
    ynew = np.zeros((lonsize, ))
    for jj in range(1, lonsize):
        # WGS84 ellipsoid
        dist, az, _ = gps2dist_azimuth(latlist[0], lonlist[0],
                                       latlist[jj], lonlist[jj])
        # Convert azimuth in degrees to angle in radians
        ang = np.deg2rad((450 - az) % 360)
        # Convert from m to km, do trig
        xnew[jj] = (dist / 1000) * np.cos(ang)
        ynew[jj] = (dist / 1000) * np.sin(ang)

    # Remove the mean
    xnew = xnew - xnew.mean()
    ynew = ynew - ynew.mean()

    # Form rij array
    rij = np.vstack((xnew, ynew))

    return rij
