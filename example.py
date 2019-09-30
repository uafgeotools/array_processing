from matplotlib import dates
from obspy.core import UTCDateTime
import sys




#%% user-defined parameters
sys.path.append('/Users/dfee/repos/waveform_collection')
from waveform_collection import gather_waveforms

SOURCE = 'IRIS'
NETWORK = 'AV'
STATION = 'DLL'

START = UTCDateTime("2019-07-15T16:50:00")
END = START + 10*60

FMIN = 0.5
FMAX = 5

# Array Processing parameters
WINLEN = 30
WINOVER = 0.50

#%% grab and filter waveforms
st = gather_waveforms(SOURCE, NETWORK, STATION, START, END,
                     time_buffer=0, remove_response=True,
                     return_failed_stations=False, watc_username=None,
                     watc_password=None)
stf = st.copy()
stf.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=2, zerophase=True)
stf.taper(max_percentage=.01)

tvec=dates.date2num(stf[0].stats.starttime.datetime)+stf[0].times()/86400   #datenum time vector


#%% get element rijs
sys.path.append('/Users/dfee/repos/array_processing')
from array_tools import array_plot, getrij, wlsqva_proc

latlist = []
lonlist = []
[latlist.append(st[i].stats.latitude) for i in range(len(st))] 
[lonlist.append(st[i].stats.longitude) for i in range(len(st))] 

rij=getrij(latlist,lonlist) 


#%% array processing and plotting

vel,az,mdccm,t,data=wlsqva_proc(stf,rij,tvec,WINLEN,WINOVER)

fig1,axs1=array_plot(tvec,data,t,mdccm,vel,az,.6)



#%% delay and sum beam
from array_tools import beamForm 
beam = beamForm(data, rij, stf[0].stats.sampling_rate, 50) 


#%% pure state filter 
from array_tools import psf 
x_psf, P = psf(data, p=2, w=3, n=3, window=None)