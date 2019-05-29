def getrij(latlist, lonlist):
    r'''
    Returns the projected geographic positions in X-Y. Points are calculated with the Vicenty inverse and will have a zero-mean.

    @ authors: Jordan W. Bishop and David Fee

    Inputs:
    1) latarray - a list of latitude points
    2) lonarray - a list of longitude points

    Outputs:
    1) X - a list of cartesian "X"-coordinates
    2) Y - a list of cartesian "Y"-coordinates

    rij - a numpy array with the first row corresponding to cartesian "X"-coordinates and the second row corresponding to cartesian "Y"-coordinates.

    Date Last Modified: 7/11/2018
    Version: 1.0
    '''

    import numpy as np
    from obspy.geodetics.base import calc_vincenty_inverse

    getrij.__version__ = '1.0'

    # Basic error checking
    latsize = len(latlist)
    lonsize = len(lonlist)

    if (latsize != lonsize):
        raise ValueError('latsize != lonsize')

    # Now to calculate
    xnew = np.zeros((latsize, ))
    ynew = np.zeros((lonsize, ))

    # azes = [0]
    for jj in range(1, lonsize):
        # Obspy defaults are set as: a = 6378137.0, f = 0.0033528106647474805
        # This is apparently the WGS84 ellipsoid.
        delta, az, baz = calc_vincenty_inverse(latlist[0], lonlist[0], latlist[jj], lonlist[jj])
        # Converting azimuth to radians
        az = (450 - az) % 360
        # azes.append(az)
        xnew[jj] = delta/1000*np.cos(az*np.pi/180)
        ynew[jj] = delta/1000*np.sin(az*np.pi/180)

    # Removing the mean
    xnew = xnew - np.mean(xnew)
    ynew = ynew - np.mean(ynew)

    # rij
    rij = np.array([xnew.tolist(), ynew.tolist()])

    return rij
