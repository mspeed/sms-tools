# functions that implement analysis and synthesis of sounds using the Sinusoidal Model
# (for example usage check the examples models_interface)

import numpy as np
from scipy.signal import blackmanharris, triang, get_window
from scipy.fftpack import ifft, fftshift
import math
import dftModel as DFT
import utilFunctions as UF

def sineTracking(pfreq, pmag, pphase, tfreq, freqDevOffset=20, freqDevSlope=0.01):
	"""
	Tracking sinusoids from one frame to the next
	pfreq, pmag, pphase: frequencies and magnitude of current frame
	tfreq: frequencies of incoming tracks from previous frame
	freqDevOffset: minimum frequency deviation at 0Hz 
	freqDevSlope: slope increase of minimum frequency deviation
	returns tfreqn, tmagn, tphasen: frequency, magnitude and phase of tracks
	"""

	tfreqn = np.zeros(tfreq.size)                              # initialize array for output frequencies
	tmagn = np.zeros(tfreq.size)                               # initialize array for output magnitudes
	tphasen = np.zeros(tfreq.size)                             # initialize array for output phases
	pindexes = np.array(np.nonzero(pfreq), dtype=np.int)[0]    # indexes of current peaks
	incomingTracks = np.array(np.nonzero(tfreq), dtype=np.int)[0] # indexes of incoming tracks
	newTracks = np.zeros(tfreq.size, dtype=np.int) -1           # initialize to -1 new tracks
	magOrder = np.argsort(-pmag[pindexes])                      # order current peaks by magnitude
	pfreqt = np.copy(pfreq)                                     # copy current peaks to temporary array
	pmagt = np.copy(pmag)                                       # copy current peaks to temporary array
	pphaset = np.copy(pphase)                                   # copy current peaks to temporary array

	# continue incoming tracks
	if incomingTracks.size > 0:                                 # if incoming tracks exist
		for i in magOrder:                                        # iterate over current peaks
			if incomingTracks.size == 0:                            # break when no more incoming tracks
				break
			track = np.argmin(abs(pfreqt[i] - tfreq[incomingTracks]))   # closest incoming track to peak
			freqDistance = abs(pfreq[i] - tfreq[incomingTracks[track]]) # measure freq distance
			if freqDistance < (freqDevOffset + freqDevSlope * pfreq[i]):  # choose track if distance is small 
					newTracks[incomingTracks[track]] = i                      # assign peak index to track index
					incomingTracks = np.delete(incomingTracks, track)         # delete index of track in incomming tracks
	indext = np.array(np.nonzero(newTracks != -1), dtype=np.int)[0]   # indexes of assigned tracks
	if indext.size > 0:
		indexp = newTracks[indext]                                    # indexes of assigned peaks
		tfreqn[indext] = pfreqt[indexp]                               # output freq tracks 
		tmagn[indext] = pmagt[indexp]                                 # output mag tracks 
		tphasen[indext] = pphaset[indexp]                             # output phase tracks 
		pfreqt= np.delete(pfreqt, indexp)                             # delete used peaks
		pmagt= np.delete(pmagt, indexp)                               # delete used peaks
		pphaset= np.delete(pphaset, indexp)                           # delete used peaks

	# create new tracks from non used peaks
	emptyt = np.array(np.nonzero(tfreq == 0), dtype=np.int)[0]      # indexes of empty incoming tracks
	peaksleft = np.argsort(-pmagt)                                  # sort left peaks by magnitude
	if ((peaksleft.size > 0) & (emptyt.size >= peaksleft.size)):    # fill empty tracks
			tfreqn[emptyt[:peaksleft.size]] = pfreqt[peaksleft]
			tmagn[emptyt[:peaksleft.size]] = pmagt[peaksleft]
			tphasen[emptyt[:peaksleft.size]] = pphaset[peaksleft]
	elif ((peaksleft.size > 0) & (emptyt.size < peaksleft.size)):   # add more tracks if necessary
			tfreqn[emptyt] = pfreqt[peaksleft[:emptyt.size]]
			tmagn[emptyt] = pmagt[peaksleft[:emptyt.size]]
			tphasen[emptyt] = pphaset[peaksleft[:emptyt.size]]
			tfreqn = np.append(tfreqn, pfreqt[peaksleft[emptyt.size:]])
			tmagn = np.append(tmagn, pmagt[peaksleft[emptyt.size:]])
			tphasen = np.append(tphasen, pphaset[peaksleft[emptyt.size:]])
	return tfreqn, tmagn, tphasen

def cleaningSineTracks(tfreq, minTrackLength=3):
	"""
	Delete short fragments of a collection of sinusoidal tracks 
	tfreq: frequency of tracks
	minTrackLength: minimum duration of tracks in number of frames
	returns tfreqn: output frequency of tracks
	"""

	if tfreq.shape[1] == 0:                                 # if no tracks return input
		return tfreq
	nFrames = tfreq[:,0].size                               # number of frames
	nTracks = tfreq[0,:].size                               # number of tracks in a frame
	for t in range(nTracks):                                # iterate over all tracks
		trackFreqs = tfreq[:,t]                               # frequencies of one track
		trackBegs = np.nonzero((trackFreqs[:nFrames-1] <= 0)  # begining of track contours
								& (trackFreqs[1:]>0))[0] + 1
		if trackFreqs[0]>0:
			trackBegs = np.insert(trackBegs, 0, 0)
		trackEnds = np.nonzero((trackFreqs[:nFrames-1] > 0)   # end of track contours
								& (trackFreqs[1:] <=0))[0] + 1
		if trackFreqs[nFrames-1]>0:
			trackEnds = np.append(trackEnds, nFrames-1)
		trackLengths = 1 + trackEnds - trackBegs              # lengths of trach contours
		for i,j in zip(trackBegs, trackLengths):              # delete short track contours
			if j <= minTrackLength:
				trackFreqs[i:i+j] = 0
	return tfreq
	

def sineModel(x, fs, w, N, t):
	"""
	Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
	x: input array sound, w: analysis window, N: size of complex spectrum, t: threshold in negative dB 
	returns y: output array sound
	"""
		
	hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
	Ns = 512                                                # FFT size for synthesis (even)
	H = Ns/4                                                # Hop size used for analysis and synthesis
	hNs = Ns/2                                              # half of synthesis FFT size
	pin = max(hNs, hM1)                                     # init sound pointer in middle of anal window       
	pend = x.size - max(hNs, hM1)                           # last sample to start a frame
	fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
	yw = np.zeros(Ns)                                       # initialize output sound frame
	y = np.zeros(x.size)                                    # initialize output array
	w = w / sum(w)                                          # normalize analysis window
	sw = np.zeros(Ns)                                       # initialize synthesis window
	ow = triang(2*H)                                        # triangular window
	sw[hNs-H:hNs+H] = ow                                    # add triangular window
	bh = blackmanharris(Ns)                                 # blackmanharris window
	bh = bh / sum(bh)                                       # normalized blackmanharris window
	sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window
	while pin<pend:                                         # while input sound pointer is within sound 
	#-----analysis-----             
		x1 = x[pin-hM1:pin+hM2]                               # select frame
		mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft                
		ploc = UF.peakDetection(mX, t)                        # detect locations of peaks
		iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
		ipfreq = fs*iploc/float(N)                            # convert peak locations to Hertz
	#-----synthesis-----
		Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)   # generate sines in the spectrum         
		fftbuffer = np.real(ifft(Y))                          # compute inverse FFT
		yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
		yw[hNs-1:] = fftbuffer[:hNs+1] 
		y[pin-hNs:pin+hNs] += sw*yw                           # overlap-add and apply a synthesis window
		pin += H                                              # advance sound pointer
	return y



def sineModelMultiRes(x, fs, w_vec, window, N_vec, t, B_vec):
	"""
	Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
	x: input array sound, w: analysis window, N: size of complex spectrum, t: threshold in negative dB 
	returns y: output array sound
	"""
        w1 = get_window(window,w_vec[0])
        w2 = get_window(window,w_vec[1])
        w3 = get_window(window,w_vec[2])
		
	hM1 = int(math.floor((w1.size+1)/2))                     # half analysis window size by rounding
	hM2 = int(math.floor(w1.size/2))                         # half analysis window size by floor

	Ns = 512                                                # FFT size for synthesis (even)
	H = Ns/4                                                # Hop size used for analysis and synthesis
	hNs = Ns/2                                              # half of synthesis FFT size
	pin = max(hNs, hM1)                                     # init sound pointer in middle of anal window       
	pend = x.size - max(hNs, hM1)                           # last sample to start a frame
	fftbuffer = np.zeros(N_vec[0])                          # initialize buffer for FFT
	yw = np.zeros(Ns)                                       # initialize output sound frame
	y1 = np.zeros(x.size)                                   # initialize output array
	y2 = np.zeros(x.size)                                   # initialize output array
	y3 = np.zeros(x.size)                                   # initialize output array
	w1 = w1 / sum(w1)                                       # normalize analysis window
	w2 = w2 / sum(w2)                                       # normalize analysis window
	w3 = w3 / sum(w3)                                       # normalize analysis window
	sw = np.zeros(Ns)                                       # initialize synthesis window
	ow = triang(2*H)                                        # triangular window
	sw[hNs-H:hNs+H] = ow                                    # add triangular window
	bh = blackmanharris(Ns)                                 # blackmanharris window
	bh = bh / sum(bh)                                       # normalized blackmanharris window
	sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window
        band_low = B_vec[0]                                     # lower band frequency limit
        band_mid = B_vec[1]                                     # mid band frequency limit
        band_high = B_vec[2]                                    # upper band frequency limit
        low_band_FFTN = N_vec[0]                                # lower band FFT size
        mid_band_FFTN = N_vec[1]                                # mid band FFT size
        high_band_FFTN = N_vec[2]                               # upper band FFT size

	while pin<pend:                                         # while input sound pointer is within sound 
	#-----analysis-----             
		x1 = x[pin-hM1:pin+hM2]                                 # Frame selection - lowest band
		x2 = x[pin-hM1:pin-hM1+w_vec[1]]                        # Frame selection - mid band
		x3 = x[pin-hM1:pin-hM1+w_vec[2]]                        # Frame selection - upper band

		mX1, pX1 = DFT.dftAnal(x1, w1, low_band_FFTN)           # DFT Analysis
		mX2, pX2 = DFT.dftAnal(x2, w2, mid_band_FFTN)           # DFT Analysis
		mX3, pX3 = DFT.dftAnal(x3, w3, high_band_FFTN)          # DFT Analysis

		ploc1 = UF.peakDetection(mX1, t)                        # Detect locations of peaks
		ploc2 = UF.peakDetection(mX2, t)                        # Detect locations of peaks
		ploc3 = UF.peakDetection(mX3, t)                        # Detect locations of peaks                

                pfreq1 = fs * ploc1 / float(low_band_FFTN)              # Find a frequency for bandlimiting
                pfreq2 = fs * ploc2 / float(mid_band_FFTN)              # Find a frequency for bandlimiting
                pfreq3 = fs * ploc3 / float(high_band_FFTN)             # Find a frequency for bandlimiting

                ploc1[pfreq1 > band_low] = 0                            # Remove peaks found to be outside each band
                ploc2[pfreq2 > band_mid] = 0
                ploc2[pfreq2 <= band_low] = 0
                ploc3[pfreq3 <= band_mid] = 0

                ploc1 = ploc1[ploc1.nonzero()]                          # Trim post-bandlimiting 
                ploc2 = ploc2[ploc2.nonzero()]
                ploc3 = ploc3[ploc3.nonzero()]

		iploc1, ipmag1, ipphase1 = UF.peakInterp(mX1, pX1, ploc1)   # refine peak values by interpolation
		iploc2, ipmag2, ipphase2 = UF.peakInterp(mX2, pX2, ploc2)   # refine peak values by interpolation
		iploc3, ipmag3, ipphase3 = UF.peakInterp(mX3, pX3, ploc3)   # refine peak values by interpolation

		ipfreq1 = fs*iploc1/float(low_band_FFTN)                    # convert peak locations to Hertz
		ipfreq2 = fs*iploc2/float(mid_band_FFTN)                    # convert peak locations to Hertz
		ipfreq3 = fs*iploc3/float(high_band_FFTN)                   # convert peak locations to Hertz

	#-----synthesis-----
		Y1 = UF.genSpecSines(ipfreq1, ipmag1, ipphase1, Ns, fs)   # generate sines in the spectrum         
		Y2 = UF.genSpecSines(ipfreq2, ipmag2, ipphase2, Ns, fs)   # generate sines in the spectrum         
		Y3 = UF.genSpecSines(ipfreq3, ipmag3, ipphase3, Ns, fs)   # generate sines in the spectrum         
                ## Low band
		fftbuffer = np.real(ifft(Y1))                          # compute inverse FFT
		yw[:hNs-1] = fftbuffer[hNs+1:]                         # undo zero-phase window
		yw[hNs-1:] = fftbuffer[:hNs+1] 
		y1[pin-hNs:pin+hNs] += sw*yw                           # overlap-add and apply a synthesis window
                ## Mid band
		fftbuffer = np.real(ifft(Y2))                          # compute inverse FFT
		yw[:hNs-1] = fftbuffer[hNs+1:]                         # undo zero-phase window
		yw[hNs-1:] = fftbuffer[:hNs+1] 
		y2[pin-hNs:pin+hNs] += sw*yw                           # overlap-add and apply a synthesis window
                ## Upper band
		fftbuffer = np.real(ifft(Y3))                          # compute inverse FFT
		yw[:hNs-1] = fftbuffer[hNs+1:]                         # undo zero-phase window
		yw[hNs-1:] = fftbuffer[:hNs+1] 
		y3[pin-hNs:pin+hNs] += sw*yw                           # overlap-add and apply a synthesis window

		pin += H                                              # advance sound pointer
	return y1, y2, y3

def sineModelAnal(x, fs, w, N, H, t, maxnSines = 100, minSineDur=.01, freqDevOffset=20, freqDevSlope=0.01):
	"""
	Analysis of a sound using the sinusoidal model with sine tracking
	x: input array sound, w: analysis window, N: size of complex spectrum, H: hop-size, t: threshold in negative dB
	maxnSines: maximum number of sines per frame, minSineDur: minimum duration of sines in seconds
	freqDevOffset: minimum frequency deviation at 0Hz, freqDevSlope: slope increase of minimum frequency deviation
	returns xtfreq, xtmag, xtphase: frequencies, magnitudes and phases of sinusoidal tracks
	"""
	
	if (minSineDur <0):                          # raise error if minSineDur is smaller than 0
		raise ValueError("Minimum duration of sine tracks smaller than 0")
	
	hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
	x = np.append(np.zeros(hM2),x)                          # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hM2))                          # add zeros at the end to analyze last sample
	pin = hM1                                               # initialize sound pointer in middle of analysis window       
	pend = x.size - hM1                                     # last sample to start a frame
	w = w / sum(w)                                          # normalize analysis window
	tfreq = np.array([])
	while pin<pend:                                         # while input sound pointer is within sound            
		x1 = x[pin-hM1:pin+hM2]                               # select frame
		mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft
		ploc = UF.peakDetection(mX, t)                        # detect locations of peaks
		iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
		ipfreq = fs*iploc/float(N)                            # convert peak locations to Hertz
		# perform sinusoidal tracking by adding peaks to trajectories
		tfreq, tmag, tphase = sineTracking(ipfreq, ipmag, ipphase, tfreq, freqDevOffset, freqDevSlope)
		tfreq = np.resize(tfreq, min(maxnSines, tfreq.size))  # limit number of tracks to maxnSines
		tmag = np.resize(tmag, min(maxnSines, tmag.size))     # limit number of tracks to maxnSines
		tphase = np.resize(tphase, min(maxnSines, tphase.size)) # limit number of tracks to maxnSines
		jtfreq = np.zeros(maxnSines)                          # temporary output array
		jtmag = np.zeros(maxnSines)                           # temporary output array
		jtphase = np.zeros(maxnSines)                         # temporary output array   
		jtfreq[:tfreq.size]=tfreq                             # save track frequencies to temporary array
		jtmag[:tmag.size]=tmag                                # save track magnitudes to temporary array
		jtphase[:tphase.size]=tphase                          # save track magnitudes to temporary array
		if pin == hM1:                                        # if first frame initialize output sine tracks
			xtfreq = jtfreq 
			xtmag = jtmag
			xtphase = jtphase
		else:                                                 # rest of frames append values to sine tracks
			xtfreq = np.vstack((xtfreq, jtfreq))
			xtmag = np.vstack((xtmag, jtmag))
			xtphase = np.vstack((xtphase, jtphase))
		pin += H
	# delete sine tracks shorter than minSineDur
	xtfreq = cleaningSineTracks(xtfreq, round(fs*minSineDur/H))  
	return xtfreq, xtmag, xtphase


def sineModelSynth(tfreq, tmag, tphase, N, H, fs):
	"""
	Synthesis of a sound using the sinusoidal model
	tfreq,tmag,tphase: frequencies, magnitudes and phases of sinusoids
	N: synthesis FFT size, H: hop size, fs: sampling rate
	returns y: output array sound
	"""
	
	hN = N/2                                                # half of FFT size for synthesis
	L = tfreq.shape[0]                                      # number of frames
	pout = 0                                                # initialize output sound pointer         
	ysize = H*(L+3)                                         # output sound size
	y = np.zeros(ysize)                                     # initialize output array
	sw = np.zeros(N)                                        # initialize synthesis window
	ow = triang(2*H)                                        # triangular window
	sw[hN-H:hN+H] = ow                                      # add triangular window
	bh = blackmanharris(N)                                  # blackmanharris window
	bh = bh / sum(bh)                                       # normalized blackmanharris window
	sw[hN-H:hN+H] = sw[hN-H:hN+H]/bh[hN-H:hN+H]             # normalized synthesis window
	lastytfreq = tfreq[0,:]                                 # initialize synthesis frequencies
	ytphase = 2*np.pi*np.random.rand(tfreq[0,:].size)       # initialize synthesis phases 
	for l in range(L):                                      # iterate over all frames
		if (tphase.size > 0):                                 # if no phases generate them
			ytphase = tphase[l,:] 
		else:
			ytphase += (np.pi*(lastytfreq+tfreq[l,:])/fs)*H     # propagate phases
		Y = UF.genSpecSines(tfreq[l,:], tmag[l,:], ytphase, N, fs)  # generate sines in the spectrum         
		lastytfreq = tfreq[l,:]                               # save frequency for phase propagation
		ytphase = ytphase % (2*np.pi)                         # make phase inside 2*pi
		yw = np.real(fftshift(ifft(Y)))                       # compute inverse FFT
		y[pout:pout+N] += sw*yw                               # overlap-add and apply a synthesis window
		pout += H                                             # advance sound pointer
	y = np.delete(y, range(hN))                             # delete half of first window
	y = np.delete(y, range(y.size-hN, y.size))              # delete half of the last window 
	return y
	
