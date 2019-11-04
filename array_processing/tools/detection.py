from scipy import optimize
import numpy as np
from .other import tauCalcSWxy, phaseAlignIdx, phaseAlignData
from functools import reduce
from itertools import groupby
from operator import itemgetter


def array_thresh(mcthresh, azvolc, azdiff, mdccm, az, vel):

    #find values above threshold...using numpy for now
    mcgood=np.where(mdccm>mcthresh)[0]
    azgood=np.where((az>=azvolc-azdiff) & (az<=azvolc+azdiff))[0]
    velgood=np.where((vel>=.25) & (vel<=.45))[0]
    igood=reduce(np.intersect1d,(mcgood,azgood,velgood))

    ranges=[]
    nconsec=[]
    for k, g in groupby(enumerate(igood), lambda x:x[0]-x[1]):
        group = list(map(itemgetter(1), g))
        ranges.append((group[0], group[-1]))
        nconsec.append(group[-1]-group[0]+1)

    if len(nconsec)>0:
        consecmax=max(nconsec)
    else:
        consecmax=0
    print('%d above trheshold, %d consecutive\n' % (len(igood),consecmax))

    return igood


def MCCMcalc(cmax, wgt=None):
    """
    Weighted mean (and median) of the cross-correlation maxima of wlsqva

    This method calculates `MCCM` as the weighted arithmetic mean of the cross-
    correlation maxima output of `wlsqva`, such that channels given a weight
    identical to zero are excluded from the `MCCM` calculation.  Optionally, it
    will calculate `MdCCM` as the weighted median of same.

    Parameters
    ~~~~~~~~~~
    cmax : array
           (n(n-1)//2, ) cross-correlation maxima for `n` channels of data and
           each channel pairing in `tau` (output of `wlsqva`)
    wgt : array
           (n, ) weights corresponding each of the `n` data channels (input to
           `wlsqva`). Default is an array of ones (all channels equally
           weighted and included).

    Returns
    ~~~~~~~
    MCCM : float
           Weighted arithmetic mean of the cross-correlation maxima
    MdCCM : float
           Weighted median of the cross-correlation maxima

    Notes
    ~~~~~
    A typical use of the weights in `wlsqva` is a simple binary scheme (i.e.,
    zeros to exclude channels & ones to include).  If both `cmax` and `wgt`
    are passed to `MCCMcalc`, then the unweighted channels will not
    contribute to the value `MCCM`.  If, however, only `cmax` is passed in such
    a case, the value of `MCCM` will be lower than expected.  This is since
    `wlsqva` sets identically to zero any `cmax` element corresponding to a
    pairing with an unweighted channel.  If `wgt` was passed to `wlsqva`, use
    it in this method, too, in order to ensure the expected numerical behavior
    of `MCCM`.  Non-binary weights in `wgt` will be used to calculate the
    traditionally-defined, weighted arithmetic mean of `cmax`.

    In the calculation of `MdCCM`, the weights are converted to binary -- i.e,
    channels are simply either included or excluded.
    """

    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #


    if wgt is None:
        # default is to calculate using all channels & unity for weights
        MCCM = np.average(cmax)
        MdCCM = np.median(cmax)
    else:

        # first, standard weighted arithmetic mean, allows for
        # arbitrary weights
        Wgt = np.array([wgt[i] * wgt[j] for i in range(wgt.size - 1)
                           for j in range(i+1,wgt.size)])
        MCCM = np.average(cmax, weights=Wgt)
        # next, weighted median here only allows binary weights, so
        # just use non-zero weighted channels
        idx = [i for i, e in enumerate(Wgt) if e != 0]
        MdCCM = np.median(cmax[idx])
    return MCCM, MdCCM


def fstatbland(dtmp, rij,fs,tau):
    """
    calculates the F-statistic based on Blandford's method.

    @author: David Fee, dfee1@alaska.edu using some WATC/Szuberla codes


    Parameters
    ~~~~~~~~~~
    dtmp : array
        (m, n) time series with `m` samples from `n` traces as columns
    rij : array
        (d, n) `n` sensor coordinates as [northing, easting, {elevation}]
        column vectors in `d` dimensions
    fs : float or int
        sample rate [Hz]
    tau : array
        (n(n-1)//2, ) time delays of relative signal arrivals (TDOA) for all
        unique sensor pairings

    Returns
    ~~~~~~~
    fstat : array
        f-statistic
    snr : float
        SNR

    Reference:
      Blandford, R. R., 1974, Geophysics, vol. 39, no. 5, p. 633-643
    """

    m,n=dtmp.shape
    wgt=np.ones(n)

    #individual trace offsets from arrival model shifts...zero for this
    Moffset=[0 for i in range(n)]

    # calculate beam delays
    beam_delays = phaseAlignIdx(tau, fs, wgt, 0)

    # apply shifts, resulting in a zero-padded array
    beamMatrix = phaseAlignData(dtmp, beam_delays, wgt, 0,m,Moffset)

    fnum = np.sum(np.sum(beamMatrix,axis=1)**2)
    term1 = np.sum(beamMatrix,axis=1)/n
    term1_0 = term1
    for i in range(1,n):
        term1 = np.vstack((term1,term1_0))
    fden = np.sum(np.sum((beamMatrix.T-term1)**2))
    fstat= (n-1)*fnum/(n*fden)

    #calculate snr based on fstat
    snr=np.sqrt((fstat-1)/n)

    return fstat,snr


def srcLoc(rij, tau, nord=2, seedXY_size=0.05, seedV_size=0.3):
    """
    Estimate a geopgraphical source location and propagation velocity for an
    event recorded on an array of sensors

    Parameters
    ~~~~~~~~~~
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

    Returns
    ~~~~~~~
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

    # (c) 2018 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #

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
