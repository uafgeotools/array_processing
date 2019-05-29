#%%
#   Array processing script...work in progress
#   
#
#   authors: David Fee, dfee1@alaska.edu
#   Date Last Modified: 5/29/2019

#%% import modules
from obspy.core import Stream, UTCDateTime
#from obspy.signal import filter
import numpy as np
from array_tools import wlsqva_proc,array_plot
from getrij import getrij
from matplotlib import dates
import matplotlib.pyplot as plt
from matplotlib import rcParams

#%% user-defined parameters
rcParams.update({'font.size': 10})

sta='DLLH1','DLLH2','DLLH3','DLL4','DLL5','DLL6'#'I53H1','DLL1',etc.
chan='HDF','HDF','HDF','HDF','HDF','HDF',#BDF, BD1, etc

#net='AV'
#sta='DLL'
#chan = 'HDF'
#loc='*'
#cli='IRIS'
#loc='01','02','03','04','05','06',
#
net='IM'
sta='I53H*'
chan = 'BDF'
loc='*'
cli='IRIS'
#
#net='IM'        #network
#sta='I53'       #array
#chan='HDF'      #infrasound channel name
#loc='*'         #location code
#cli='WATC-int'      #client to read data from (IRIS, AVO, WATC-ext,WATC-int)

watc_user='dfee'        #watc username
watc_pw=''     #watc fdsn password. leave blank before committing!

t1 = UTCDateTime("2018-12-19T01:40:00")
t2 = UTCDateTime("2018-12-19T02:20:00")

rem_resp=1  #1 for full response removal, 0 for just apply sensitivity at 1 Hz

windur=30  #window duration
winover=.5  #window overlap
filt=[.1,5]    #filter band

azvolc=0   #back-azimuth to expected source. If unknown, leave at 0
mcthresh=0.65

dsbeam=0    #delay and sum beam? need azvolc parameter above to be valid

isave=0     #save figure?
svdir='/Users/dfee/Documents/events/krakatau/'     #save directory

plotarray=0     #plot array geometry as a check    


#%% read in and filter data

def read_st(net, sta, chan, loc, t1, t2, cli,watc_user,watc_pw,remov_resp,filt):
    """Read in array data to stream, remove response, filter.

    example: st,stf,latlist,lonlist,staname=read_st(net, sta, chan, loc, t1, t2, cli,watc_user,watc_pw,rem_resp,filt)

    net: SEED network code for desired station
    sta: SEED station code 
    chan: SEED channel code 
    loc: SEED location code
    t1, t2: UTCDateTime objects defining time range of interest
    cli: Which client to use with ObsPy (currently supports 'AV' for winston 'IRIS' for IRIS FDSN, 'WATC-ext' for WATC external FDSN, 'WATC-int' for WATC internal FDSN))
    watc_user,watc_pw=username and password for watc fdsn server
    remove_resp: flag to remove response. 0 for full response removal, 1 for just sensitivity
    filt: Tuple of (fmin, fmax) for bandpass filter (default = (0.5, 8))

    st: output obspy stream object
    stf: filtered stream 
    latlist, lonlist: list of station latitude and longitude 
    staname: list of station names

    """

    # read in data, resample if needed
    st=Stream()

    if cli=='AV':
        print('Reading in data from AVO winston')
        from obspy.clients.earthworm import Client
        client = Client("pubavo1.wr.usgs.gov", 16023)

    elif cli=='IRIS':
        print('Reading in data from IRIS')
        from obspy.clients.fdsn import Client
        client = Client("IRIS")
        
    elif cli=='WATC-ext':
        from obspy.clients.fdsn import Client
        client = Client(base_url='http://10.30.5.10:8080',user=watc_user,password=watc_pw) #external
        
    elif cli=='WATC-int':
        from obspy.clients.fdsn import Client
        client = Client(base_url='http://10.30.6.3:8080',user=watc_user,password=watc_pw) #internal

    st = client.get_waveforms(net,sta,loc,chan,t1,t2,attach_response=True) 
    st.merge(fill_value='latest')
    st.trim(t1,t2)
    st.sort()

    print(st)
    
    fs=st[0].stats.sampling_rate
    
    # remove response and filter
    if net=='AV':
        #kluge for AVO...needs improvement!
        if sta=='SDPI':
            st[0].data=st[0].data*2.3842e-04
        if sta=='OK':
            st[0].data=st[0].data*4.7733e-05
        if sta=='CLES1':
            st[0].data=st[0].data*(1/((419430)*(5e-2)))
        if sta=='CLES2':
            st[0].data=st[0].data*(1/((419430)*(.009)))
        if sta=='CLCO1':
            st[0].data=st[0].data*(1/((419430)*(1e-2)))
            
    elif rem_resp==1:
        print('Removing response...')
        pre_filt = [0.001, 0.005, fs/2-2, fs/2] #pre-filt for response removal
        st.remove_response(pre_filt=pre_filt,output='VEL',water_level=None)#remove response, use plot=True to check
        
    elif rem_resp==0:
        print('Removing sensitivity...')
        st.remove_sensitivity() 

    st.detrend()
    #st.sort(keys=['channel'])#,reverse=True)

    #filter
    stf = st.copy()
    stf.taper(max_percentage=.01)
    stf.filter('bandpass', freqmin=filt[0], freqmax=filt[1],corners=2, zerophase=True)
    
    #get inventory and lat/lon info (if available). Need solution for avo winston!
    inv = client.get_stations(network=net,station=sta,channel=chan,location=loc,starttime=t1,endtime=t2)#, level='response')  #get station information
    
    latlist=[]
    lonlist=[]
    staname=[]
    for network in inv:
        for station in network:
            latlist.append(station.latitude)
            lonlist.append(station.longitude)
            staname.append(station.code)
            
    return st,stf,latlist,lonlist,staname


st,stf,latlist,lonlist,staname=read_st(net, sta, chan, loc, t1, t2, cli,watc_user,watc_pw,rem_resp,filt)
        
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
        
#delay and sum beam? replace first channel w/ beamed data for plotting later
if dsbeam:
    from Z import beamForm 
    beam = beamForm(data, rij, stf[0].stats.sampling_rate, azvolc) 
    data[:,0]=beam[0:len(data)]
    
#%% array processing and plotting

vel,az,mdccm,t,data=wlsqva_proc(stf,rij,tvec,windur,winover)

fig1,axs1=array_plot(tvec,data,t,mdccm,vel,az,mcthresh)

if isave:
    tmstr1=UTCDateTime.strftime(t1,'%Y%m%d_%H%M')
    tmstr2=UTCDateTime.strftime(t2,'%Y%m%d_%H%M')
    foutname='%s_%s-%s1.png' % (sta[0:3],tmstr1,tmstr2)
    fig1.savefig(svdir + foutname,dpi=200,bbox_inches='tight')
    
