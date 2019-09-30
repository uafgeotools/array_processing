import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import dates
from array_algorithms import wlsqva
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