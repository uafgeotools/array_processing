import numpy as np
from obspy.geodetics import gps2dist_azimuth

_M_PER_KM = 1000  # [m/km]

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
        xnew[i] = (dist / _M_PER_KM) * np.cos(ang)
        ynew[i] = (dist / _M_PER_KM) * np.sin(ang)

    # Remove the mean
    xnew = xnew - xnew.mean()
    ynew = ynew - ynew.mean()

    # Form rij array
    rij = np.vstack((xnew, ynew))

    return rij


def compass2rij(distances, azimuths):
    """Convert tape-and-compass survey data to Cartesian :math:`x`–:math:`y` coordinates.

    The output type is the same as the :func:`getrij` function. Note that typically,
    distances and azimuths will be surveyed from one of the array elements. In this
    case, that array element will have distance 0 and azimuth 0. However, this function
    can handle an arbitrary reference point for the distances and azimuths. This
    function assumes that all array elements lie on the same plane.

    Args:
        distances (array): Distances to each array element, in meters
        azimuths (array): Azimuths to each array element, in degrees from **true** north

    Returns:
        :class:`numpy.ndarray` with the first row corresponding to Cartesian
        :math:`x`-coordinates and the second row corresponding to Cartesian
        :math:`y`-coordinates, in units of km
    """

    # Type conversion and error checking
    distances = np.array(distances)
    azimuths = np.array(azimuths)
    if distances.size != azimuths.size:
        raise ValueError('There must be the same number of distances and azimuths')
    assert (distances >= 0).all(), 'Distances cannot be negative'
    assert ((azimuths >= 0) & (azimuths < 360)).all(), 'Azimuths must be 0–359°'

    # Convert distances and azimuths to Cartesian coordinates in units of km
    x = distances * np.sin(np.deg2rad(azimuths)) / _M_PER_KM
    y = distances * np.cos(np.deg2rad(azimuths)) / _M_PER_KM

    # Remove the mean
    x -= x.mean()
    y -= y.mean()

    # Form rij array
    rij = np.vstack((x, y))

    return rij
