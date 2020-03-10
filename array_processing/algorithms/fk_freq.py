import numpy as np
from math import pow, ceil
from obspy.signal.invsim import cosine_taper


def fk_freq(data, fs, rij, vmin, vmax, fmin, fmax, nvel, ntheta):
    r"""
    :math:`f`–:math:`k` beamforming with loop over frequency bands.

    Args:
        data: ``(m, n)`` array; time series with ``m`` samples from ``n``
            traces as columns
        rij: ``(d, n)`` array; ``n`` sensor coordinates as [northing, easting,
            {elevation}] column vectors in ``d`` dimensions
        fs (int or float): Sample rate [Hz]
        vmin (int or float): Min velocity in km/s, suggest 0.25
        vmax (int or float): Max velocity in km/s, suggest 0.45
        fmin (int or float): Minimum frequency in Hz
        fmax (int or float): Maximum frequency in Hz
        nvel (int or float): Number of velocity iterations, suggest 100–200
        ntheta (int or float): Number of azimuth iterations, suggest 100–200

    Returns:
        ``(ntheta, nvel)`` array; beamformed slowness map, not normalized. Can
        find max using

        .. code-block:: python

            ix, iy = np.unravel_index(bmpwr.argmax(), bmpwr.shape)
    """

    #reshape rij from standard setup
    rij = np.transpose(rij)
    rij[:, 0] = rij[:, 0] - np.mean(rij[:, 0])
    rij[:, 1] = rij[:, 1] - np.mean(rij[:, 1])

    # Getting the size of the data
    [m, nsta] = np.shape(data)

    # set up velocity/slowness and theta vectors
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
