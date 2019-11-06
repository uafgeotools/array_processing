#%% User-defined parameters

from waveform_collection import gather_waveforms
from obspy.core import UTCDateTime
from matplotlib import dates

SOURCE = 'IRIS'
NETWORK = 'AV'
STATION = 'DLL'
LOCATION = '*'
CHANNEL = '*'

START = UTCDateTime('2019-07-15T16:50:00')
END = START + 10*60

FMIN = 0.5
FMAX = 5

# Array processing parameters
WINLEN = 30
WINOVER = 0.50

#%% Grab and filter waveforms

st = gather_waveforms(SOURCE, NETWORK, STATION, LOCATION, CHANNEL, START, END,
                      time_buffer=0, remove_response=True)

stf = st.copy()
stf.filter('bandpass', freqmin=FMIN, freqmax=FMAX, corners=2, zerophase=True)
stf.taper(max_percentage=.01)

tvec = dates.date2num(stf[0].stats.starttime.datetime)+stf[0].times()/86400

#%% Array processing and plotting using least squares

from array_processing.algorithms.helpers import getrij, wlsqva_proc
from array_processing.tools.plotting import array_plot

latlist = [tr.stats.latitude for tr in st]
lonlist = [tr.stats.longitude for tr in st]

rij = getrij(latlist, lonlist)

vel, az, mdccm, t, data = wlsqva_proc(stf, rij, tvec, WINLEN, WINOVER)

fig1, axs1 = array_plot(tvec, data, t, mdccm, vel, az, .6)

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
