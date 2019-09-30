import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import dates
from WATCtools import wlsqva
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


def fk_freq(data, fs, rij, vmin, vmax, fmin, fmax, nvel,ntheta):
    """
    f-k beamforming with loop over frequency bands
    
    @ authors: Jordan W. Bishop and David Fee

    Parameters
    ~~~~~~~~~~
    data : array
        (m, n) time series with `m` samples from `n` traces as columns
    rij : array
        (d, n) `n` sensor coordinates as [northing, easting, {elevation}]
        column vectors in `d` dimensions
    fs : float or int
        sample rate
    vmin: float or int
        min velocity in km/s, suggest 0.25
    vmax:float or int
        max velocity in km/s, suggest 0.45
    fmin: float or int
        minimum frequency in Hz
    fmax:float or int
        maximum frequency in Hz
    nvel: float or int
        number of velocity iterations, suggest 100-200
    ntheta: float or int
        number of azimuth iterations, suggest 100-200 
        
    Returns
    ~~~~~~~
    pow_map : array
        (ntheta, nvel))
        beamformed slowness map, not normalized
        can find max using: ix,iy = np.unravel_index(bmpwr.argmax(), bmpwr.shape)
          

    """
    fk_freq.__version__ = '1.0'

    # (c) 2018 Jordan W. Bishop, David Fee
    # University of Alaska, Fairbanks, all rights reserved
    #

    import numpy as np
    from math import (pow, ceil)
    from obspy.signal.invsim import cosine_taper

    #reshape rij from standard setup
    rij = np.transpose(rij)
    rij[:, 0] = rij[:, 0] - np.mean(rij[:, 0])
    rij[:, 1] = rij[:, 1] - np.mean(rij[:, 1])
    
    # Getting the size of the data
    [m, nsta] = np.shape(data)

    # set up velocity/slowness and theta vectors
    #vits = np.linspace(vmin, vmax, int(nvel))
    sits = np.linspace(1/vmax, 1/vmin, int(nvel))
    theta = np.linspace(0, 2*np.pi, ntheta)

    # Getting initial time shifts
    # x time delay
    cost = np.cos(theta)
    Tx1 = np.outer(sits, np.transpose(cost))
    Txm = Tx1[:, :, None] * np.transpose(rij[:, 1])

    # y time delay
    sint = np.sin(theta)
    Ty1 = np.outer(sits, np.transpose(sint))
    Tym = Ty1[:, :, None] * np.transpose(rij[:, 0])

    # All possible time delays
    TT = Txm + Tym

    # Computing the next power of 2 for fft input
    n2 = ceil(np.log2(m))
    nfft = int(pow(2, n2))

    # Frequency increment
    deltaf = fs / nfft

    # Getting frequency bands
    nlow = int(fmin / float(deltaf) + 0.5)
    nhigh = int(fmax / float(deltaf) + 0.5)
    nlow = max(1, nlow)  # avoid using the offset
    nhigh = min(nfft // 2 - 1, nhigh)  # avoid using Nyquist
    nf = nhigh - nlow + 1  # include upper and lower frequency

    # Apply a 22% Cosine Taper
    taper = cosine_taper(m, p=0.22)

    # Calculated the FFT of each trace
    # ft are complex Fourier coefficients
    # is this faster in scipy?
    ft = np.empty((nsta, nf), dtype=np.complex128)
    for jj in range(nsta):
        data[:, jj] = data[:, jj] - np.mean(data[:, jj])
        dat = data[:, jj] * taper
        ft[jj, :] = np.fft.rfft(dat, nfft)[nlow:nlow + nf]

    # Change data structure for performance boost --> Check this.
    ft = np.ascontiguousarray(ft, np.complex128)

    # Pre-allocating
    freqrange = np.linspace(fmin, fmax, nf)
    pow_mapt = np.zeros((int(nvel), int(ntheta)), dtype=np.float64, order='C')
    pow_mapb = np.zeros((int(nvel), int(ntheta)), dtype=np.float64, order='C')
    flen = len(freqrange)

    # loop entire slowness map over frequencies
    # compute covariance
    for ii in range(flen):
        # Generating the exponentials - steering vectors
        freq = freqrange[ii]
        expo = -1j * 2 * np.pi * freq * TT
        Master = np.exp(expo)
        # Broadcasting the Fourier coefficients at each station
        fcoeff = ft[:, ii]
        Master = Master * fcoeff[None, None, :]
        Top = np.sum(Master, axis=2)
        Top2 = np.real(np.multiply(Top.conj(), Top))
        Bot = np.real(np.multiply(Master.conj(), Master))
        Bot = np.sum(Bot, axis=2)
        pow_mapt += Top2
        pow_mapb += Bot
    pow_map = pow_mapt/pow_mapb

    return pow_map