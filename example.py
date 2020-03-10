#%% User-defined parameters

from waveform_collection import gather_waveforms
from obspy.core import UTCDateTime

# Data collection
SOURCE = 'IRIS'
NETWORK = 'IM'
STATION = 'I53H?'
LOCATION = '*'
CHANNEL = '*'
START = UTCDateTime('2018-12-19T01:45:00')
END = START + 20*60

# Filtering
FMIN = 0.1  # [Hz]
FMAX = 1    # [Hz]

# Array processing
WINLEN = 50  # [s]
WINOVER = 0.5

#%% Grab and filter waveforms

st = gather_waveforms(SOURCE, NETWORK, STATION, LOCATION, CHANNEL, START, END,
                      remove_response=True)

st.filter('bandpass', freqmin=FMIN, freqmax=FMAX, corners=2, zerophase=True)
st.taper(max_percentage=0.01)

#%% Array processing and plotting using least squares

from array_processing.algorithms.helpers import getrij, wlsqva_proc
from array_processing.tools.plotting import array_plot
from lts_array import ltsva

latlist = [tr.stats.latitude for tr in st]
lonlist = [tr.stats.longitude for tr in st]

rij = getrij(latlist, lonlist)

#%% Array processing. ALPHA = 1.0: least squares processing.
# 0.5 <= ALPHA < 1.0: least trimmed squares processing.
ALPHA = 1.0
stdict, t, mdccm, vel, baz, sig_tau = ltsva(st, rij, WINLEN,
                                                      WINOVER, ALPHA)

fig1, axs1 = array_plot(st, t, mdccm, vel, baz,
                        ccmplot=True, mcthresh=0.6, sigma_tau=sig_tau)

#%% Array uncertainty

from array_processing.tools import arraySig
from array_processing.tools.plotting import arraySigPlt, arraySigContourPlt

SIGLEVEL = 1/st[0].stats.sampling_rate
KMAX = 400
TRACE_VELOCITY = 0.33

sigV, sigTh, impResp, vel, th, kvec = arraySig(rij, kmax=KMAX,
                                               sigLevel=SIGLEVEL)

fig2 = arraySigPlt(rij, SIGLEVEL, sigV, sigTh, impResp, vel, th, kvec)

fig3 = arraySigContourPlt(sigV, sigTh, vel, th, trace_v=TRACE_VELOCITY)

#%% Delay and sum beam

from array_processing.tools import beamForm

beam = beamForm(data, rij, st[0].stats.sampling_rate, 50)

#%% Pure state filter

from array_processing.tools import psf

x_psf, P = psf(data, p=2, w=3, n=3, window=None)
