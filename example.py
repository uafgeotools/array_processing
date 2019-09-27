#%% import modules
#from obspy.signal import filter
import numpy as np
from array_tools import wlsqva_proc,array_plot
from getrij import getrij
from matplotlib import dates
import matplotlib.pyplot as plt
from matplotlib import rcParams

from obspy.core import UTCDateTime


import sys
sys.path.append('/Users/dfee/repos/waveform_collection')
sys.path.append('/Users/dfee/repos/array_processing')

from waveform_collection import gather_waveforms
from array_tools import wlsqva_proc
from getrij import getrij

#%% user-defined parameters
SOURCE = 'IRIS'
NETWORK = 'AV'
STATION = 'DLL'

START= UTCDateTime("2019-09-20T00:00:00")
END = START + 10*60

WINDUR = 30

# Filter limits
FMIN = 0.5
FMAX = 5

# Array Processing parameters
WINLEN = 30
WINOVER = 0.50


st=gather_waveforms(SOURCE, NETWORK, STATION, START, END,
                     time_buffer=0, remove_response=True,
                     return_failed_stations=False, watc_username=None,
                     watc_password=None)
stf = st.copy()
stf.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=2, zerophase=True)



latlist = []
lonlist = []
[latlist.append(st[i].stats.latitude) for i in range(len(st))] 
[lonlist.append(st[i].stats.longitude) for i in range(len(st))] 

rij=getrij(latlist,lonlist) #get element rijs

tvec=dates.date2num(stf[0].stats.starttime.datetime)+stf[0].times()/86400   #datenum time vector

#plot array geometry as a check
if plotarray:
    fig0=plt.figure(10)
    plt.clf()
    plt.plot(rij[0,:],rij[1,:],'bo')
    plt.text(rij[0,:],rij[1,:],staname)#loc)

#put stream data into matrix for later processing
nchans=len(stf)
npts=len(stf[0].data)
data=np.empty((npts,nchans))
for i,tr in enumerate(stf):
    data[:,i] = tr.data
        

#%% array processing and plotting

vel,az,mdccm,t,data=wlsqva_proc(stf,rij,tvec,windur,winover)

fig1,axs1=array_plot(tvec,data,t,mdccm,vel,az,mcthresh)

#if isave:
#    tmstr1=UTCDateTime.strftime(t1,'%Y%m%d_%H%M')
#    tmstr2=UTCDateTime.strftime(t2,'%Y%m%d_%H%M')
#    foutname='%s_%s-%s1.png' % (sta[0:3],tmstr1,tmstr2)
#    fig1.savefig(svdir + foutname,dpi=200,bbox_inches='tight')
#    
#%% delay and sum beam. replace first channel w/ beamed data for plotting later
    from Z import beamForm 
    beam = beamForm(data, rij, stf[0].stats.sampling_rate, azvolc) 
    data[:,0]=beam[0:len(data)]
    