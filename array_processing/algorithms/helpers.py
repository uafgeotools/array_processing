import numpy as np
from obspy.geodetics import gps2dist_azimuth


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
