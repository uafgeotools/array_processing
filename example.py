#%% user-defined parameters
from waveform_collection import gather_waveforms
from obspy.core import UTCDateTime
from matplotlib import dates

SOURCE = 'IRIS'
NETWORK = 'AV'
STATION = 'DLL'
LOCATION = '*'
CHANNEL = '*'

START = UTCDateTime("2019-07-15T16:50:00")
END = START + 10*60

FMIN = 0.5
FMAX = 5

# Array Processing parameters
WINLEN = 30
WINOVER = 0.50

#%% grab and filter waveforms
st = gather_waveforms(SOURCE, NETWORK, STATION, LOCATION, CHANNEL, START, END,
                     time_buffer=0, remove_response=True,
                     return_failed_stations=False, watc_username=None,
                     watc_password=None)
stf = st.copy()
stf.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=2, zerophase=True)
stf.taper(max_percentage=.01)

tvec=dates.date2num(stf[0].stats.starttime.datetime)+stf[0].times()/86400   #datenum time vector


#%% get element rijs
from array_processing.tools import array_plot, getrij, wlsqva_proc

latlist = [tr.stats.latitude for tr in st]
lonlist = [tr.stats.longitude for tr in st]

rij = getrij(latlist, lonlist)

#%% array processing and plotting using least squares
vel, az, mdccm, t, data = wlsqva_proc(stf, rij, tvec, WINLEN, WINOVER)

fig1, axs1 = array_plot(tvec,data,t,mdccm,vel,az,.6)

#%% array uncertainty
from array_processing.tools import arraySig, arraySigPlt, arraySigContourPlt
SIGLEVEL = 1/st[0].stats.sampling_rate
KMAX = 400
TRACE_V = 0.33

sigV, sigTh, impResp, vel, th, kvec = arraySig(rij, kmax=KMAX, sigLevel=SIGLEVEL)

arraySigPlt(rij, SIGLEVEL, sigV, sigTh, impResp, vel, th, kvec)

fig = arraySigContourPlt(sigV, sigTh, vel, th, trace_v=TRACE_V)

#%% delay and sum beam
from array_processing.tools import beamForm
beam = beamForm(data, rij, stf[0].stats.sampling_rate, 50)


#%% pure state filter
from array_processing.tools import psf
x_psf, P = psf(data, p=2, w=3, n=3, window=None)
