import numpy as np
from scipy import optimize
from ..tools.generic import tauCalcSWxy


def srcLoc(rij, tau, nord=2, seedXY_size=0.05, seedV_size=0.3):
    """
    Estimate a geopgraphical source location and propagation velocity for an
    event recorded on an array of sensors

    Args:
        rij : list|array
            (d, n) `n` array coordinates as [easting, northing, {elevation}]
            column vectors in `d` dimensions
        tau : array
            (n*(n-1)/2, ) unique intersensor TDOA information (delays)
        nord : positive int|inf
            Order of the norm to calculate the cost function (optional, default
            is 2 for the usual Euclidean L2 norm)
        seedXY_size : float
            Geographic seed value (optional, default is 0.05)
        seedXY_size : float
            Propagation velocity seed value (optional, default is 0.3)

    Returns:
        Sxyc : array
            (d+1, ) optimized source location as geographic coordinates (same as
            the columns of `rij`) and propagation speed
        Srtc : array
            (d+1, ) optimized source location as [range, azimuth, {elevation},
            propagation speed]

    Notes
    ~~~~~
    This is a Pythonic method for srcLoc that might've been dubbed srcLocLite.
    It takes a naive approach to the seed, ignoring Dr. Arnoult's spacetime
    approach, but takes into account the quirks of the Nelder-Mead optimization
    and prduces a fairly good (if not great) facsimilie of the MATLAB version.
    """

    # The below line can be removed once we add rij2rTh
    raise NotImplementedError('rij2rTh not available!')

    # cost function
    def minTau(xyv_trial, tau_o, rij):
        tau_trial = tauCalcSWxy(xyv_trial[-1], xyv_trial[:-1], rij)
        return np.linalg.norm(tau_o - tau_trial, nord)
    # copnstruct naive seed
    xyv_seed = [seedXY_size] * len(rij) + [seedV_size]
    for k in range(0, len(xyv_seed)-1, 2):
        xyv_seed[k] = -xyv_seed[k]
    # perform optimization
    costFn = lambda xyv_trial: minTau(xyv_trial, tau, rij)
    xyv_opt = optimize.minimize(costFn, xyv_seed, method='Nelder-Mead')

    return xyv_opt.x, rij2rTh(xyv_opt.x[:len(rij)])
