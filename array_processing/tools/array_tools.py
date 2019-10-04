import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import dates
from ..algorithms.array_algorithms import wlsqva
from obspy.geodetics.base import calc_vincenty_inverse
from functools import reduce
from itertools import groupby
from operator import itemgetter
    

def wlsqva_proc(stf,rij,tvec,windur,winover):
    '''
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
            
    '''
    
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
    

def array_plot(tvec,data,t,mdccm,vel,az,mcthresh):
    '''
    Module to run plot array processing results
    @ authors: David Fee

    example: array_plot(stf,tvec,t,mdccm,vel,az,mcthresh):
    
    '''
    
    cm='RdYlBu_r'   #colormap
    cax=0.2,1       #colorbar/y-axis for mccm

    fig1, axarr1 = plt.subplots(4, 1,sharex='col')
    fig1.set_size_inches(10,9)
    axs1=axarr1.ravel()
    axs1[0].plot(tvec,data[:,0],'k')
    axs1[0].axis('tight')
    axs1[0].set_ylabel('Pressure [Pa]')
    #set_xlim(st[0].stats.starttime,st[0].stats.endtime)
    
    sc=axs1[1].scatter(t,mdccm,c=mdccm,edgecolors='k',lw=.3,cmap=cm)
    axs1[1].plot([t[0],t[-1]],[mcthresh,mcthresh],'r--')
    axs1[1].axis('tight')
    axs1[1].set_xlim(t[0],t[-1])
    axs1[1].set_ylim(cax)
    sc.set_clim(cax)
    axs1[1].set_ylabel('MdCCM')
    
    sc=axs1[2].scatter(t,vel,c=mdccm,edgecolors='k',lw=.3,cmap=cm)
    axs1[2].set_ylim(.25,.45)
    axs1[2].set_xlim(t[0],t[-1])
    sc.set_clim(cax)
    axs1[2].set_ylabel('Trace Velocity\n [km/s]')
    
    sc=axs1[3].scatter(t,az,c=mdccm,edgecolors='k',lw=.3,cmap=cm)
    #axs1[3].plot([t[0],t[-1]],[azvolc,azvolc],'r--')
    axs1[3].set_ylim(0,360)

    axs1[3].set_xlim(t[0],t[-1])
    sc.set_clim(cax)
    axs1[3].set_ylabel('Back-azimuth\n [deg]')
    
    axs1[3].xaxis_date()
    axs1[3].tick_params(axis='x',labelbottom='on')
    axs1[3].fmt_xdata = dates.DateFormatter('%HH:%MM')
    axs1[3].xaxis.set_major_formatter(dates.DateFormatter("%d-%H:%M"))
    axs1[3].set_xlabel('UTC Time')
    
    cbot=.1
    ctop=axs1[1].get_position().y1
    cbaxes=fig1.add_axes([.92,cbot,.02,ctop-cbot])
    hc=plt.colorbar(sc,cax=cbaxes)
    hc.set_label('MdCCM')

    return fig1,axs1
    

def array_thresh(mcthresh,azvolc,azdiff,mdccm,az,vel):
    
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


def getrij(latlist, lonlist):
    r'''
    Returns the projected geographic positions in X-Y. Points are calculated 
    with the Vicenty inverse and will  have a zero-mean.

    @ authors: Jordan W. Bishop and David Fee

    Inputs:
    1) latarray - a list of latitude points
    2) lonarray - a list of longitude points

    Outputs:
    1) X - a list of cartesian "X"-coordinates
    2) Y - a list of cartesian "Y"-coordinates

    rij - a numpy array with the first row corresponding to cartesian 
    "X"-coordinates and the second row corresponding to cartesian "Y"-coordinates.

    '''

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


def psf(x, p=2, w=3, n=3, window=None):
    r"""
    Pure-state filter a data matrix

    This function uses a generalized coherence estimator to enhance coherent
    channels in an ensemble of time series.

    Parameters
    ~~~~~~~~~~
    x : array
        (m, d) array of real-valued time series data as columns
    p : float, optional
        Level of contrast in filter.  Default is 2.
    w : int, optional
        Width of smoothing window in frequency domain.  Default is 3.
    n : float, optional
        Number of times to smooth in frequency domain.  Default is 3.
    window : function, optional
        Type of smoothing window in frequency domain.  Default is None, which
        results in a triangular window.

    Returns
    ~~~~~~~
    x_ppf : array
        (m, d) real-valued, pure state-filtered version of `x`
    P : array
        (m//2+1, ) degree of polarization (generalized coherence estimate) in
        frequency components of `x` from DC to the Nyquist

    Notes
    ~~~~~
    See any of Samson & Olson's early 1980s papers, or Szuberla's 1997
    PhD thesis, for a full description of the underlying theory.  The code
    implementation's defaults reflect historical values for the smoothing
    window -- a more realistic `w` would be of order :math:`\sqrt{m}\;`
    combined with a smoother window, such as `np.hanning`.  Letting `n=3`
    is a reasonable choice for all window types to ensure confidence in the
    spectral estimates used to construct the filter.

    For `m` samples of `d` channels of data, a (d, d) spectral matrix
    :math:`\mathbf{S}[f]` can be formed at each of the ``m//2+1`` real
    frequency components from DC to the Nyquist.  The generalized coherence
    among all of the `d` channels at each frequency is estimated by

    .. math::

        P[f] = \frac{d \left(\text{tr}\mathbf{S}^2[f]\right) -
        \left(\text{tr}\mathbf{S}[f]\right)^2}
        {\left(d-1\right)\left(\text{tr}\mathbf{S}[f]\right)^2} \;,

    where :math:`\text{tr}\mathbf{S}[f]` is the trace of the spectral
    matrix at frequency :math:`f`.  The filter is constructed by applying
    the following multiplication in the frequency domain

    .. math::

        \hat{\mathbf{X}}[f] = P[f]^p\mathbf{X}[f] \;,

    where :math:`\mathbf{X}[f]` is the Fourier transform component of the all
    channels at frequency :math:`f` and `p` is the level of contrast.  The
    inverse Fourier transform of the matrix :math:`\hat{\mathbf{X}}` gives the
    filtered time series.

    The estimator :math:`\mathbf{P}[f] = 1`, identically, without smoothing in
    the spectral domain (a consequence of the variance in the raw Fourier
    components), but it is bound by :math:`\mathbf{P}[f]\in[0,1]` even with
    smoothing, hence its utility as a multiplicative filter in the frequency
    domain.  Similarly, this bound allows the contrast between channels to be
    enhanced based on their generalized coherence if :math:`p>1\;`.

    Data channels should be pre-processed to have unit-variance, since unlike
    the traditional two-channel magintude squared coherence estimators, the
    generalized coherence estimate can be biased by relative amplitude
    variations among the channels.  To mitigate the effects of smoothing
    complex values into the DC and Nyquist components, they are set to zero
    before computing the inverse transform of :math:`\hat{\mathbf{X}}`.

    """
    psf.__version__ = '1.0.2'
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    # private functions up front
    def Ssmooth(S, w, n, window):
        # smooth special format of spectral matries as vectors
        from scipy.signal import convolve2d
        for k in range(n):
            # f@#$ing MATLAB treats odd/even differently with mode='full'
            # but the behavior below now matches conv2 exactly
            S = convolve2d(S, window(w).reshape(-1,1),
                           mode='full')[w//2:-w//2+1,:]
        return S
    def triang(N):
        # for historical reasons, the default window shape
        from numpy import bartlett
        return bartlett(N+2)[1:-1]
    # main code block
    from numpy import empty, outer, tile, vstack
    # size up input data
    N, d = x.shape
    # Fourier transform of data matrix by time series columns, retain only
    # the diagonal & above (unique spectral components)
    X = ft(x)
    X = X[:N//2+1, :]
    # form spectral matrix stack in reduced vector form (**significant**
    # speed improvement due to memory problem swith 3D tensor format -- what
    # was too slow in 1995 is still too slow in 2017!)
    # -- pre-allocate stack & fill it in
    Sidx = [(i, j) for i in range(d) for j in range(i, d)]
    S = empty((X.shape[0], d*(d+1)//2), dtype=complex)
    for i in range(X.shape[0]):
        # at each freq w, S_w is outer product of raw Fourier transforms
        # of each column at that freq; select unique components to store
        S_w = outer(X[i,:], X[i,:].conj())
        S[i,:] = S_w[[i[0] for i in Sidx], [j[1] for j in Sidx]]
    # smooth each column of S (i,e., in freq. domain)
    if not window:
        # use default window
        window = triang
    S = Ssmooth(S, w, n, window)
    # trace calculations (notation consistent w traceCalc.m in MATLAB), but
    # since results are positive, semi-definite, real -- return as such
    #  -- diagonal elements
    didx = [i for i in range(len(Sidx)) if Sidx[i][0]==Sidx[i][1]]
    #  -- traceS**2 of each flapjack (really a vector here)
    trS = sum(S[:,didx].real.T)**2
    #  -- trace of each flapjack (ditto, vector), here we recognize that
    #     trace(S@S.T) is just sum square magnitudes of all the
    #     non-redundant components of S, doubling squares of the non-diagonal
    #     elements
    S = (S*(S.conj())*2).real
    S[:,didx] /= 2
    trS2 = sum(S.T)
    # estimate samson-esque polarization estimate (if d==2, same as fowler)
    P = (d*trS2 - trS)/((d-1)*trS)
    # a litle trick here to handle odd/even number of samples and zero-out
    # both the DC & Nyquist (they're both complex-contaminated due to Ssmooth)
    P[0] = 0
    if N%2:
        # odd case: Nyquist is 1/2 freq step between highest components
        fudgeIdx = 0
    else:
        # even case: we have a Nyquist component
        fudgeIdx = 1
        P[-1] = 0
    # apply P as contrast agent to frequency series
    X *= tile(P**p, d).reshape(X.shape[::-1]).T
    # inverse transform X and ensure real output
    x_psf = ift(vstack((X[list(range(N//2+1))],
                          X[list(range(N//2-fudgeIdx,0,-1))].conj())),
                            allowComplex=False)
    return x_psf, P


def ft(x, n=None, axis=0, norm=None):
    """
    Sentman-like normalization of numpy's fft

    This function calculates the fast Fourier transform of a time series.

    Parameters
    ~~~~~~~~~~
    x : array
        Array of data, can be complex
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros.  If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT.  Default is 0.
    norm : {None, "ortho"}, optional
        Normalization mode (see `numpy.fft`). Default is None.

    Returns
    ~~~~~~~
    out : complex array
        The truncated or zero-padded input, transformed along the
        indicated axis.

    See Also
    ~~~~~~~~
    ift : Sentman-like normalization of ifft
    numpy.fft : master fft functions

    Notes
    ~~~~~
    This function is just `numpy.fft.fft` with Dr. Davis Sentman
    normalization such that the DC components of the transform are the
    mean of the each column/row in the input time series.

    Version
    ~~~~~~~
    1.0.1 -- 3 Oct 2016

    """
    ft.__version__ = '1.0.1'
    # (c) 2016 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    from numpy.fft import fft
    # get normalization constant
    N = x.shape
    return fft(x, n, axis, norm)/N[axis]


def ift(X, n=None, axis=0, norm=None, allowComplex=False):
    r"""
    Sentman-like normalization of numpy's ifft

    This function calculates the fast Fourier inverse transform of a
    frequency series.

    Parameters
    ~~~~~~~~~~
    X : array
        Array of data, can be complex
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros.  If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT.  Default is 0.
    norm : {None, "ortho"}, optional
        Normalization mode (see `numpy.fft`).  Default is None.
    allowComplex : boolean
        If False, will force real output.  This is useful for suppressing
        cases of spurious (machine precision) imaginary parts where the
        time series are known to be real.  Default is False.

    Returns
    ~~~~~~~
    out : complex array
        The truncated or zero-padded input, transformed along the
        indicated axis.

    See Also
    ~~~~~~~~
    ft : Sentman-like normalization of fft
    numpy.fft : master fft functions

    Notes
    ~~~~~
    This function is just `numpy.fft.ifft` with Dr. Davis Sentman
    normalization such that the DC components of the input are the
    mean of the each column/row in the output time series.

    To check on the small imaginary parts of an inverse transform
    one could insert the code below. ::

        from numpy.linalg import norm as Norm
        print(Norm(x.imag))}``

    Version
    ~~~~~~~
    1.0.1 -- 7 Oct 2016

    """
    ift.__version__ = '1.0.1'
    # (c) 2016 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    from numpy import real
    from numpy.fft import ifft
    # get normalization constant
    N = X.shape
    x = ifft(X, n, axis, norm)*N[axis]
    # force real output, if requested
    if allowComplex:
        return x
    else:
        return real(x)


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
    fstatbland.__version__ = '1.0'

    import numpy as np
    from Z import phaseAlignIdx, phaseAlignData
    
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
    

def beamForm(data, rij, Hz, azPhi, vel=0.340, r=None, wgt=None, refTrace=None,
             M=None, Moffset=None):
    """
    Form a "best beam" from the traces of an array

    Parameters
    ~~~~~~~~~~
    data : array
        (m, n) time series with `m` samples from `n` traces as columns
    rij : array
        (d, n) `n` sensor coordinates as [easting, northing, {elevation}]
        column vectors in `d` dimensions
    Hz : float or int
        sample rate
    azPhi : float or list|array
        back azimuth (float) from co-array coordinate origin (º CW from N);
        back azimuth and elevation angle (list) from co-array coordinate
        origin (º CW from N, º from N-E plane)
    vel : float
        optional estimated signal velocity across array. Default is 0.340.
    r : float
        optional range to source from co-array origin. Default is None
        (use plane wave arrival model), If not None, use spherical wave
        arrival model.
    wgt : list or array
        optional list|vector of relative weights of length `n`
        (0 == exclude trace). Default is None (use all traces with equal
        relative weights ``[1 for i in range(nTraces)]``).
    refTrace : int
        optional reference sensor for TDOA information. Default is None
        (use first non-zero-weighted trace).
    M : int
        optional length of best beam result in samples. Default is None
        (use `m` samples, same as input `data`)
    Moffset : list or array
        optional individual trace offsets from arrival model shifts. Default
        is None (use ``[0 for i in range(nTraces)]``)

    Returns
    ~~~~~~~
    beam : array
        (M, ) vector of summed and weighted shifted traces to form a best beam

    Raises
    ~~~~~~
    IndexError
        If the input argument dimensions are not consistent.

    Notes
    ~~~~~
    This beamformer handles planar- or spherical-model arrivals from
    arbitrarily elevated sources incident on 2- or 3D arrays.  Weights are
    relative and normalized upon beam output.  The default value for `vel`
    assumes rij is in units of km (e.g., the speed is in km/s).

    Version
    ~~~~~~~
    0.2 -- 2 Mar 2018

"""
    beamForm.__version__ = '0.2'
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    from numpy import array
    # size things up
    m, nTraces = data.shape
    # -- default parsing & error checking -----------------------------------
    # default weight is unity weight for all traces
    if wgt is None:
        wgt = [1 for i in range(nTraces)]
    else:
        if len(wgt) != nTraces:
            # catch dimension mismatch between tau & wgt
            raise IndexError('len(wgt) != ' + str(nTraces))
    wgt = array(wgt)    # require array form here for later operations
    # default refTrace is first non-zero wgt
    if refTrace is None:
        from numpy import min as Min, where
        refTrace = Min(where(wgt != 0)) # requires array wgt
    # default Moffset is zero for all traces
    if Moffset is None:
        Moffset = [0 for i in range(nTraces)]
    else:
        if len(Moffset) != nTraces:
            # catch dimension mismatch between tau & Moffset
            raise IndexError('len(Moffset) != ' + str(nTraces))
    # -- end default parsing & error checking -------------------------------
    # planar (far-field) or spherical (near-field) arrival?
    if r is None:
        tau = tauCalcPW(vel, azPhi, rij)
    else:
        from numpy import isscalar
        # need to unpack & repack azPhi with care
        if isscalar(azPhi):
            tau = tauCalcSW(vel, [r, azPhi], rij)
        else:
            tau = tauCalcSW(vel, [r, azPhi[0], azPhi[1]], rij)
    # calculate shifts as samples
    beam_delays = phaseAlignIdx(tau, Hz, wgt, refTrace)
    # apply shifts, resulting in a zero-padded array
    beamMatrix = phaseAlignData(data, beam_delays, wgt, refTrace, M, Moffset)
    # linear algrebra to perform sum & then normalize by weights
    return beamMatrix@wgt / wgt.sum()


def phaseAlignData(data, delays, wgt, refTrace, M, Moffset, plotFlag=False):
    """
    Embeds `n` phase aligned traces in a data matrix

    Parameters
    ~~~~~~~~~~
    data : array
        (m, n) time series with `m` samples from `n` traces as columns
    delays : array
        (n, ) vector of shifts as indicies for embedding traces in an array,
        such that trace `i` will begin at index ``out[i]``
    wgt : list or array
       vector of relative weights of length `n` (0 == exclude trace by setting
       to padding value, see `plotFlag`)
    refTrace : int
        reference sensor for TDOA information
    M : int
        length of best beam result in samples (use `m` to let beam be same
        length as inpout traces)
    Moffset : list or array
        individual trace offsets from arrival model shifts (use
        ``[0 for i in range(nTraces)]`` to skip this effect)
    plotFlag : Boolean
        optional flag to indicate output array will be used for plotting
        purposes.  Default is False (pads shifts with zeros; pads with
        np.nan if True).

    Returns
    ~~~~~~~
    data_align : array
        (M, n) array of shifted traces as columns

    Notes
    ~~~~~
    The output of `phaseAlignIdx` is used to calculate the input `delays`.

    Version
    ~~~~~~~
    0.2 -- 27 Feb 2017

    """
    phaseAlignData.__version__ = '0.0.2'
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    # -- this is low level code w/out error checks or defaults, designed
    # --  to be called by wrappers that make use of the indices provided
    from numpy import zeros, array
    # size up data
    m, nTraces = data.shape
    # if plotting, embed in array of np.nan
    if plotFlag:
        from numpy import nan
        nanOrOne = nan
    else:
        nanOrOne = 1
    # correct for negative Moffset elements
    # subtract this to ensure corrected delays is positive,
    # semi-definite and has (at least) one zero element
    maxNegMoffset = min(array(Moffset)[array(Moffset) <= 0])
    # apply Moffset & correction for negative elements of Moffset
    delays = delays + Moffset - maxNegMoffset
    # start with everything in the window as a default (trim||pad later)
    data_align = zeros((max(delays) + m, nTraces)) * nanOrOne
    # embed shifted traces in array
    for k in range(nTraces):
        if wgt[k]:
            data_align[delays[k]:delays[k]+m, k] = data[:, k] * wgt[k]
    # truncate|| pad data_align if M >< m, centered on refTrace
    mp = data_align.shape[0]  # new value for m
    if M is not None and M is not mp:
        alignBounds = [delays[refTrace] + m//2 - M//2,
                      delays[refTrace] + m//2 + M//2]
        # trap round-off errors and force (M, nTraces) data_align
        if alignBounds[1]-alignBounds[0] != M:
            alignBounds[1] += 1
            if not (alignBounds[1]-alignBounds[0])%2:
                alignBounds[0] -= 1
        #  -- LHS (graphically, but actually topside in array-land!)
        if alignBounds[0] < 0:
            # pad LHS of traces w zeros or np.nans
            from numpy import vstack
            data_align = vstack((zeros((-alignBounds[0], nTraces)) * nanOrOne,
                                data_align))
        elif alignBounds[0] > 0:
            data_align = data_align[alignBounds[0]:]
        #  -- RHS (graphically, but actually bottom in array-land!)
        if alignBounds[1] > mp:
            # pad RHS of traces w zeros or np.nans
            from numpy import vstack
            data_align = vstack( (data_align, zeros((alignBounds[1]-mp,
                                      nTraces) ) * nanOrOne))
        elif alignBounds[1] < mp:
            data_align = data_align[:M]
    return data_align


def phaseAlignIdx(tau, Hz, wgt, refTrace):
    """
    Calculate shifts required to phase align `n` traces in a data matrix

    Parameters
    ~~~~~~~~~~
    tau : array
        (n(n-1)//2, ) time delays of relative signal arrivals (TDOA) for all
        unique sensor pairings
    Hz : float or int
        sample rate
    wgt : list or array
       vector of relative weights of length `n` (0 == exclude trace)
    refTrace : int
        reference sensor for TDOA information

    Returns
    ~~~~~~~
    delays : array
        (n, ) vector of shifts as indicies for embedding traces in an array,
        such that trace `i` will begin at index ``out[i]``

    Notes
    ~~~~~
    The output of this function is compatible with the inputs of
    `phaseAlignData`.

    Version
    ~~~~~~~
    0.2 -- 27 Feb 2017

    """
    phaseAlignIdx.__version__ = '0.2'
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    # -- this is low level code w/out error checks or defaults, designed
    # --  to be called by wrappers that make use of the indices provided
    from numpy import sqrt, hstack
    # solve for number of traces from pairings in tau
    nTraces = int(1+sqrt(1+8*len(tau)))//2
    # calculate delays (samples) relative to refTrace for each trace
    #   -- first pass grabs delays starting with refTrace as i in ij
    delayIdx = (nTraces*refTrace - refTrace*(refTrace+1)//2,
            nTraces*(refTrace+1) - (refTrace+1)*(refTrace+2)//2)
    delays = hstack( (0, (tau[range(delayIdx[0],delayIdx[1])]*Hz))
                    ).astype(int)
    # the std. rij list comprehension for unique inter-trace pairs
    tau_ij = [(i, j) for i in range(nTraces) for j in range(i+1, nTraces)]
    #  -- second pass grabs delays with refTrace as j in ij
    preRefTau_idx = [k for k in range(len(tau)) if tau_ij[k][1] == refTrace]
    delays = hstack( (-tau[preRefTau_idx]*Hz, delays) ).astype(int)
    # re-shift delays such that the first sample of the trace requiring the
    # largest shift left relative to the refTrace (hence, largest positive,
    # semi-definite element of delays) has an index of zero; i.e., all traces
    # have a valid starting index for embedding into an array (no padding)
    return -delays + max(delays)


def tauCalcPW(vel, azPhi, rij):
    """
    Calculates theoretical tau vector for a plane wave moving across an array
    of `n` elements

    Parameters
    ~~~~~~~~~~
    vel : float
        signal velocity across array
    azPhi : float or list|array
        back azimuth (float) from co-array coordinate origin (º CW from N);
        back azimuth and elevation angle (array) from co-array coordinate
        origin (º CW from N, º from N-E plane)
    rij : array
        (d, n) `n` element coordinates as [easting, northing, {elevation}]
        column vectors in `d` dimensions

    Returns
    ~~~~~~~
    tau : array
        (n(n-1)//2, ) time delays of relative signal arrivals (TDOA) for all
        unique sensor pairings

    Version
    ~~~~~~~
    1.0.1 -- 2 Mar 2018

    """
    tauCalcPW.__version__ = '1.0.1'
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    from numpy import vstack, zeros, isscalar, pi, cos, sin, array
    dim, nTraces = rij.shape
    if dim == 2:
        rij = vstack((rij, zeros((1, nTraces))))
    idx = [(i, j) for i in range(rij.shape[1]-1)
                    for j in range(i+1,rij.shape[1])]
    X = rij[:,[i[0] for i in idx]] - rij[:,[j[1] for j in idx]]
    if isscalar(azPhi):
        phi = 0
        az = azPhi
    else:
        phi = azPhi[1]/180*pi
        az = azPhi[0]
    az = pi*(-az/180 + 0.5)
    s = array([cos(az), sin(az), sin(phi)])
    s[:-1] *= cos(phi)
    return X.T@(s/vel)


def tauCalcSW(vel, rAzPhi, rij):
    """
    Calculates theoretical tau vector for a spherical wave moving across an
    array of `n` elements

    Parameters
    ~~~~~~~~~~
    vel : float
        signal velocity across array
    rAzPhi : list|array
        range to source and back azimuth from co-array coordinate origin
        (º CW from N); range to source, back azimuth and elevation angle
        from co-array coordinate origin (º CW from N, º from N-E plane)
    rij : array
        (d, n) `n` element coordinates as [easting, northing, {elevation}]
        column vectors in `d` dimensions

    Returns
    ~~~~~~~
    tau : array
        (n(n-1)//2, ) time delays of relative signal arrivals (TDOA) for all
        unique sensor pairings

    Version
    ~~~~~~~
    1.0.1 -- 2 Mar 2018
    """
    tauCalcSW.__version__ = '1.0.1'
    # (c) 2017 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    from numpy import vstack, zeros, pi, cos, sin, array, tile
    from numpy.linalg import norm as Norm
    dim, nTraces = rij.shape
    if dim == 2:
        rij = vstack((rij, zeros((1, nTraces))))
    if len(rAzPhi) == 3:
        phi = rAzPhi[2]/180*pi
    else:
        phi = 0
    idx = [(i, j) for i in range(rij.shape[1]-1)
                    for j in range(i+1,rij.shape[1])]
    # aw, this is so convolutedly elegant that it must be saved in a
    # comment for posterity!, but the line below it is "simpler"
    # az = -( (rAzPhi[1]/180*pi - 2*pi)%(2*pi) - pi/2  )%(2*pi)
    az = pi*(-rAzPhi[1]/180 + 0.5)
    source = rAzPhi[0] * array([cos(az), sin(az), sin(phi)])
    source[:-1] *= cos(phi)
    tau2sensor = Norm(rij - tile(source, nTraces).reshape(nTraces, 3).T,
                      2, axis=0)/vel
    return tau2sensor[[j[1] for j in idx]] - tau2sensor[[i[0] for i in idx]]


def tauCalcSWxy(vel, xy, rij):
    """
    Calculates theoretical tau vector for a spherical wave moving across an
    array of `n` elements

    Parameters
    ~~~~~~~~~~
    vel : float
        signal velocity across array
    xy : list|array
        (d, ) source location as 2-D [easting, northing] or 3-D [easting,
        northing, elevation] coordinates
    rij : list|array
        (d, n) `n` element coordinates as [easting, northing, {elevation}]
        column vectors in `d` dimensions

    Returns
    ~~~~~~~
    tau : array
        (n(n-1)//2, ) time delays of relative signal arrivals (TDOA) for all
        unique sensor pairings

    Version
    ~~~~~~~
    1.1 -- 16 Mar 2018
    """
    tauCalcSWxy.__version__ = '1.1'
    # (c) 2018 Curt A. L. Szuberla
    # University of Alaska Fairbanks, all rights reserved
    #
    from numpy import vstack, hstack, tile
    from numpy.linalg import norm as Norm
    dim, nTraces = len(rij), len(rij[0])
    if dim == 2:
        rij = vstack((rij, [0]* nTraces))
    else:
        rij = vstack((rij, ))
    if len(xy) == 2:
        xy0 = 0
    else:
        xy0 = []
    source = hstack((xy, xy0))
    idx = [(i, j) for i in range(rij.shape[1]-1)
                    for j in range(i+1,rij.shape[1])]
    tau2sensor = Norm(rij - tile(source, nTraces).reshape(nTraces, 3).T,
                      2, axis=0)/vel
    return tau2sensor[[j[1] for j in idx]] - tau2sensor[[i[0] for i in idx]]
