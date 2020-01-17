#%% User-defined parameters

from waveform_collection import gather_waveforms
from obspy.core import UTCDateTime

SOURCE = 'IRIS'
NETWORK = 'IM'
STATION = 'I53H*'
LOCATION = '*'
CHANNEL = '*'

START = UTCDateTime('2018-12-19T01:45:00')
END = START + 20*60

FMIN = 0.1
FMAX = 1

# Array processing parameters
WINLEN = 50
WINOVER = 0.50

#%% Grab and filter waveforms

st = gather_waveforms(SOURCE, NETWORK, STATION, LOCATION, CHANNEL, START, END,
                      time_buffer=0, remove_response=True)

stf = st.copy()
stf.filter('bandpass', freqmin=FMIN, freqmax=FMAX, corners=2, zerophase=True)
stf.taper(max_percentage=0.01)

tvec = stf[0].times('matplotlib')

#%% Array processing and plotting using least squares

from array_processing.algorithms.helpers import getrij, wlsqva_proc
from array_processing.tools.plotting import array_plot

latlist = [tr.stats.latitude for tr in st]
lonlist = [tr.stats.longitude for tr in st]

rij = getrij(latlist, lonlist)

vel, baz, sig_tau, mdccm, t, data = wlsqva_proc(stf, rij, tvec, WINLEN, WINOVER)

fig1, axs1 = array_plot(stf, t, mdccm, vel, baz, ccmplot=True, sigma_tau=sig_tau)

#%% Array uncertainty

from array_processing.tools import arraySig
from array_processing.tools.plotting import arraySigPlt, arraySigContourPlt

SIGLEVEL = 1/st[0].stats.sampling_rate
KMAX = 400
TRACE_V = 0.33

sigV, sigTh, impResp, vel, th, kvec = arraySig(rij, kmax=KMAX, sigLevel=SIGLEVEL)

fig2 = arraySigPlt(rij, SIGLEVEL, sigV, sigTh, impResp, vel, th, kvec)

fig3 = arraySigContourPlt(sigV, sigTh, vel, th, trace_v=TRACE_V)

#%% Delay and sum beam

from array_processing.tools import beamForm

beam = beamForm(data, rij, stf[0].stats.sampling_rate, 50)


#%% Pure state filter

from array_processing.tools import psf

x_psf, P = psf(data, p=2, w=3, n=3, window=None)
