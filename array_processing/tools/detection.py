import numpy as np
from obspy.core import Stream
from .generic import phaseAlignIdx, phaseAlignData


def MCCMcalc(cmax, wgt=None):
    r"""
    Weighted mean (and median) of the cross-correlation maxima of
    :func:`~array_processing.algorithms.wlsqva.wlsqva`. This method
    calculates MCCM as the weighted arithmetic mean of the cross-
    correlation maxima output of
    :func:`~array_processing.algorithms.wlsqva.wlsqva`, such that channels
    given a weight identical to zero are excluded from the MCCM calculation.
    Optionally, it will calculate MdCCM as the weighted median of same.

    Args:
        cmax: ``(n(n-1)//2, )`` array; cross-correlation maxima for ``n``
            channels of data and each channel pairing in `tau` (output of
            :func:`~array_processing.algorithms.wlsqva.wlsqva`)
        wgt: ``(n, )`` array; weights corresponding each of the ``n`` data
            channels (input to
            :func:`~array_processing.algorithms.wlsqva.wlsqva`). Default is an
            array of ones (all channels equally weighted and included)

    Returns:
        tuple: Tuple containing:

        - **MCCM** – Weighted arithmetic mean of the cross-correlation maxima
        - **MdCCM** – Weighted median of the cross-correlation maxima

    Notes:
        A typical use of the weights in
        :func:`~array_processing.algorithms.wlsqva.wlsqva` is a simple binary
        scheme (i.e., zeros to exclude channels & ones to include). If both
        `cmax` and `wgt` are passed to this function, then the unweighted
        channels will not contribute to the value `MCCM`. If, however, only
        `cmax` is passed in such a case, the value of `MCCM` will be lower
        than expected. This is since
        :func:`~array_processing.algorithms.wlsqva.wlsqva` sets identically to
        zero any `cmax` element corresponding to a pairing with an
        unweighted channel. If `wgt` was passed to
        :func:`~array_processing.algorithms.wlsqva.wlsqva`, use it in this
        method, too, in order to ensure the expected numerical behavior of
        `MCCM`. Non-binary weights in `wgt` will be used to calculate the
        traditionally-defined, weighted arithmetic mean of `cmax`.

        In the calculation of `MdCCM`, the weights are converted to binary —
        i.e, channels are simply either included or excluded.
    """

    if wgt is None:
        # default is to calculate using all channels & unity for weights
        MCCM = np.average(cmax)
        MdCCM = np.median(cmax)
    else:

        # first, standard weighted arithmetic mean, allows for
        # arbitrary weights
        wgt = np.array([wgt[i] * wgt[j] for i in range(wgt.size - 1)
                        for j in range(i+1, wgt.size)])
        MCCM = np.average(cmax, weights=wgt)
        # next, weighted median here only allows binary weights, so
        # just use non-zero weighted channels
        idx = [i for i, e in enumerate(wgt) if e != 0]
        MdCCM = np.median(cmax[idx])

    return MCCM, MdCCM


def fstatbland(dtmp, fs, tau):
    r"""
    Calculates the F-statistic and SNR based on Blandford's method (see Notes).

    Args:
        dtmp: ``(m, n)`` array; time series with ``m`` samples from ``n``
            traces as columns
        fs (int or float): Sample rate [Hz]
        tau: ``(n(n-1)//2)`` array; time delays of relative signal arrivals
            (TDOA) for all unique sensor pairings

    Returns:
        tuple: Tuple containing:

        - **fstat** – F-statistic
        - **snr** – SNR

    Notes:
        Blandford, R. R., 1974, Geophysics, vol. 39, no. 5, p. 633–643.
    """

    m, n = dtmp.shape
    wgt = np.ones(n)

    #individual trace offsets from arrival model shifts. Zeros here
    m_offset = [0 for i in range(n)]

    # calculate beam delays
    beam_delays = phaseAlignIdx(tau, fs, wgt, 0)

    # apply shifts, resulting in a zero-padded array
    beam = phaseAlignData(dtmp, beam_delays, wgt, 0, m, m_offset)

    fnum = np.sum(np.sum(beam, axis=1)**2)
    term1 = np.sum(beam, axis=1)/n
    term1_0 = term1
    for i in range(1, n):
        term1 = np.vstack((term1, term1_0))
    fden = np.sum(np.sum((beam.T - term1)**2))
    fstat = (n-1) * fnum/(n * fden)

    #calculate snr based on fstat
    snr = np.sqrt((fstat-1)/n)

    return fstat, snr


def calculate_semblance(data_in):
    r"""
    Calculates the semblance, a measure of multi-channel coherence, following
    the definition of Neidell & Taner (1971). Assumes data are already
    time-shifted to construct the beam.

    Args:
        data_in: Time-shifted ObsPy Stream or time-shifted NumPy array

    Returns:
        Multi-channel coherence defined on :math:`[0, 1]`
    """

    if isinstance(data_in, Stream):
        # check that all traces have the same length
        if len(set([len(tr) for tr in data_in])) != 1:
            raise ValueError('Traces in stream must have same length!')

        n = len(data_in)

        beam = np.sum([tr.data for tr in data_in], axis=0) / n
        beampower = n * np.sum(beam**2)

        avg_power = np.sum(np.sum([tr.data**2 for tr in data_in], axis=0))

    elif isinstance(data_in, np.ndarray):
        n = data_in.shape[0]

        beam = np.sum(data_in, axis=0) / n
        beampower = n * np.sum(beam**2)

        avg_power = np.sum(np.sum([data_in**2], axis=0))

    semblance = beampower / avg_power

    return semblance
