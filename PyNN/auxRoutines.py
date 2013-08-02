# ###########################################
#
# 	Model implementation converted from
#	Brain to PyNN by Vitor Chaud, 
#	Andrew Davison and Padraig Gleeson(2013).
#
# 	Original implementation reference:
#
# 	Inhibitory synaptic plasticity in a
# 	recurrent network model (F. Zenke, 2011)
#
# 	Adapted from: 
# 	Vogels, T. P., H. Sprekeler, F. Zenke,
# 	C. Clopath, and W. Gerstner. 'Inhibitory
#	Plasticity Balances Excitation and
#	Inhibition in Sensory Pathways and
#	Memory Networks.' Science (November 10, 2011). 
#
# ###########################################

########################################################################
##
##	Auxliar Routines
##
########################################################################


import numpy as np
import matplotlib.pyplot as plt
	



def getNeuronSpikeTimes_old (neuronIndex, popSpikes):
	"Gets the spike times of a specific neuron based on population spikes"
	condition = popSpikes[:,0] == neuronIndex
	return np.compress(condition, popSpikes[:,1])


def getNeuronSpikeTimes (neuronIndex, popSpikes):
	"Gets the spike times of a specific neuron based on population spikes"
	seg = popSpikes.segments[0]
	allSpikes = seg.spiketrains
	return allSpikes[neuronIndex]



def getNeuronISIs (neuronIndex, popSpikes):
	"Gets the interspike intervals of a neurons within a population"
	neuronSpikeTimes = getNeuronSpikeTimes(neuronIndex, popSpikes)
	if neuronSpikeTimes.size == 0 or neuronSpikeTimes.size == 1:
		return -1 * np.ones(1)
	else:
		return np.diff(neuronSpikeTimes)


def getNeuronISIs2 (neuronSpikes):
	"Gets the interspike intervals of a neurons within a population"
	if neuronSpikes.size == 0 or neuronSpikes.size == 1:
		return -1 * np.ones(1)
	else:
		return np.diff(neuronSpikes)


def calculateNeuronFiringRate (neuronIndex, popSpikes):
	"In order to return a valid value the simulation time frame must be of 1000 ms"
	return np.size(getNeuronSpikeTimes(neuronIndex, popSpikes))


def calculateNeuronFiringRate2 (neuronIndex, popSpikes):
	"Calculate firing rate based on the inverse of ISIs. Result in Hz."
	neuronISIs = getNeuronISIs(neuronIndex, popSpikes)
	if neuronISIs[0] == -1:
		IFR = 0
	else:
		IFR = 1000 / getNeuronISIs(neuronIndex, popSpikes)
	return np.mean(IFR)


def calculateISICV (neuronIndex, popSpikes):
	neuronISIs = getNeuronISIs(neuronIndex, popSpikes)
	if neuronISIs[0] == -1:
		return -1
	else:
		return np.std(neuronISIs) / np.mean(neuronISIs)



def calculateISICV2 (neuronSpikes):
	neuronISIs = getNeuronISIs2(neuronSpikes)
	if neuronISIs[0] == -1:
		return -1
	else:
		return np.std(neuronISIs) / np.mean(neuronISIs)


def plotRaster(axis, popSpikes, color):
	seg = popSpikes.segments[0]
	for spiketrain in seg.spiketrains:
		y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
		plt.plot(spiketrain, y, '.', c=color)
	plt.xlabel('Time [ms]')
	#axis.set_frame_on(False)
	axis.spines['top'].set_color('none')
	axis.spines['left'].set_color('none')
	axis.spines['right'].set_color('none')
	axis.tick_params(axis='x', top='off')
	axis.tick_params(axis='x', bottom='off')
	axis.tick_params(axis='y', left='off')
	axis.tick_params(axis='y', right='off')
	axis.spines['bottom'].set_linewidth(2)
	axis.spines['left'].set_linewidth(2)



def plotISICVHist(axis, popSpikes, barColor):
	seg = popSpikes.segments[0]
	allSpikes = seg.spiketrains
	isiCVs = np.zeros(0)
	for neuronSpikes in allSpikes:
		neuronISICV = calculateISICV2(neuronSpikes)
		if neuronISICV != -1:
			isiCVs = np.append(isiCVs, neuronISICV)
	if np.size(isiCVs) != 0:	
		plt.hist(isiCVs, histtype='stepfilled', color=barColor, alpha=0.5)
	#plt.ylabel('Percent [%]')
	plt.xlabel('ISI CV')
	#plt.ylim((0, 100))
	plt.xlim((0.0, 3.0))
	#axis.set_frame_on(False)
	axis.spines['top'].set_color('none')
	axis.spines['left'].set_color('none')
	axis.spines['right'].set_color('none')
	axis.tick_params(axis='x', top='off')
	axis.tick_params(axis='x', bottom='off')
	axis.tick_params(axis='y', left='off')
	axis.tick_params(axis='y', right='off')
	axis.spines['bottom'].set_linewidth(2)
	axis.spines['left'].set_linewidth(2)


def plotISICVDoubleHist(axis, popSpikes, barColor, popSpikes2, barColor2):
	

	seg = popSpikes2.segments[0]
	allSpikes = seg.spiketrains
	isiCVs = np.zeros(0)
	for neuronSpikes in allSpikes:
		neuronISICV = calculateISICV2(neuronSpikes)
		if neuronISICV != -1:
			isiCVs = np.append(isiCVs, neuronISICV)
	if np.size(isiCVs) != 0:	
		plt.hist(isiCVs, histtype='stepfilled', color=barColor2, alpha=0.6)

	seg = popSpikes.segments[0]
	allSpikes = seg.spiketrains
	isiCVs = np.zeros(0)
	for neuronSpikes in allSpikes:
		neuronISICV = calculateISICV2(neuronSpikes)
		if neuronISICV != -1:
			isiCVs = np.append(isiCVs, neuronISICV)
	if np.size(isiCVs) != 0:	
		plt.hist(isiCVs, histtype='stepfilled', color=barColor, alpha=0.6)

	#plt.ylabel('Percent [%]')
	plt.xlabel('ISI CV')
	#plt.ylim((0, 100))
	plt.xlim((0.0, 3.0))
	#axis.set_frame_on(False)
	axis.spines['top'].set_color('none')
	axis.spines['left'].set_color('none')
	axis.spines['right'].set_color('none')
	axis.tick_params(axis='x', top='off')
	axis.tick_params(axis='x', bottom='off')
	axis.tick_params(axis='y', left='off')
	axis.tick_params(axis='y', right='off')
	axis.spines['bottom'].set_linewidth(2)
	axis.spines['left'].set_linewidth(2)


def biExponentialKernelFunction (t):
	"Calculate the bi-exponential kernel as in Vogels et al. 2011. The time is specified in ms"
	tau1 = 50 # [ms]
	tau2 = 4 * tau1 # [ms]
	return (1/tau1) * np.exp(-np.absolute(t) / tau1)  +  (1/tau2) * np.exp(-np.absolute(t) / tau2)


def biExponentialKernel (timeStep, timeBoundKernel):
	"Calculate the bi-exponential kernel as in Vogels et al. 2011. The time is specified in ms"
	kernel = np.zeros(0)
	for t in range(-timeBound, timeBound, timeStep):
		kernel = np.append(kernel, biExponentialKernelFunction(t))
	return kernel


def createSpikesTrain (neuronSpikes):
	"Receives the spike times of a neuron and returns an array representing the spike train"
	
	return 1


def filterSpikesTrain (spikeTrain, timeStep, timeBoundKernel):
	"Filter spike train using the bi-exponential kernel"
	filteredSignal = np.convolve(spikeTrain, biExponentialKernel (timeStep, timeBoundKernel))
	return filteredSignal


def calculateCorrCoef (spikeTrain1, spikeTrain2):
	"Calculate the correlation coeficient between spikeTrain1 and spikeTrain2"
	
	return 1



def plotCorrHist(axis, popSpikes, barColor):
	seg = popSpikes.segments[0]
	allSpikes = seg.spiketrains
	correlation = np.zeros(0)
	for neuronSpikes in allSpikes:
		correlation = np.append(correlation, 1)
	if np.size(correlation) != 0:	
		plt.hist(correlation, histtype='stepfilled', color=barColor, alpha=0.7)
	#plt.ylabel('Percent [%]')
	plt.xlabel('Spiking Correlation')
	#plt.ylim((0, 100))
	plt.xlim((0.0, 1.0))
	#axis.set_frame_on(False)
	axis.spines['top'].set_color('none')
	axis.spines['left'].set_color('none')
	axis.spines['right'].set_color('none')
	axis.tick_params(axis='x', top='off')
	axis.tick_params(axis='x', bottom='off')
	axis.tick_params(axis='y', left='off')
	axis.tick_params(axis='y', right='off')
	axis.spines['bottom'].set_linewidth(2)
	axis.spines['left'].set_linewidth(2)


def plotCorrDoubleHist(axis, popSpikes, barColor, popSpikes2, barColor2):
	seg = popSpikes2.segments[0]
	allSpikes = seg.spiketrains
	correlation = np.zeros(0)
	for neuronSpikes in allSpikes:
		correlation = np.append(correlation, 1)
	if np.size(correlation) != 0:	
		plt.hist(correlation, histtype='stepfilled', color=barColor2, alpha=0.6)

	seg = popSpikes.segments[0]
	allSpikes = seg.spiketrains
	correlation = np.zeros(0)
	for neuronSpikes in allSpikes:
		correlation = np.append(correlation, 1)
	if np.size(correlation) != 0:	
		plt.hist(correlation, histtype='stepfilled', color=barColor, alpha=0.6)
	
	#plt.ylabel('Percent [%]')
	plt.xlabel('Spiking Correlation')
	#plt.ylim((0, 100))
	plt.xlim((0.0, 1.0))
	#axis.set_frame_on(False)
	axis.spines['top'].set_color('none')
	axis.spines['left'].set_color('none')
	axis.spines['right'].set_color('none')
	axis.tick_params(axis='x', top='off')
	axis.tick_params(axis='x', bottom='off')
	axis.tick_params(axis='y', left='off')
	axis.tick_params(axis='y', right='off')
	axis.spines['bottom'].set_linewidth(2)
	axis.spines['left'].set_linewidth(2)



def isInSubGrid (x, y, xIni, xFin, yIni, yFin):
	"Checks if an element with coordenates (x, y) in a grid is within a sub-grid with the specified limits"
	return (x >= xIni) and (x <= xFin) and (y >= yIni) and (y <= yFin)



def plotGrid(axis, excSpikes, pattern1Spikes, pattern1_stimSpikes, pattern2Spikes, pattern2_stimSpikes, intersectionSpikes, controlSpikes, inhibSpikes):
	
	axis.get_xaxis().set_visible(False)
	axis.get_yaxis().set_visible(False)


	auxIndexInhib = 0
	auxIndexExc = 0
	auxIndexControl = 0
	auxIndexPattern1 = 0
	auxIndexPattern1_stim = 0
	auxIndexPattern2 = 0
	auxIndexPattern2_stim = 0
	auxIndexPatternIntersection = 0
	
	
	xIniInhib = 0
	xFinInhib = 99
	yIniInhib = 80
	yFinInhib = 99
	
	xIniControl = 7
	xFinControl = 34
	yIniControl = 11
	yFinControl = 38
	
	xIniPattern1 = 42
	xFinPattern1 = 69
	yIniPattern1 = 30
	yFinPattern1 = 57

	xIniPattern1_stim = 56
	xFinPattern1_stim = 69
	yIniPattern1_stim = 30
	yFinPattern1_stim = 43
	
	xIniPattern2 = 22
	xFinPattern2 = 49
	yIniPattern2 = 50
	yFinPattern2 = 77

	xIniPattern2_stim = 22
	xFinPattern2_stim = 35
	yIniPattern2_stim = 64
	yFinPattern2_stim = 77
	
	xIniPatternIntersection = 42
	xFinPatternIntersection = 49
	yIniPatternIntersection = 50
	yFinPatternIntersection = 57
	
	
	
	grid = np.zeros((100, 100))
	
	for x in range(100):

		for y in range(100):

			if isInSubGrid(x, y, xIniInhib, xFinInhib, yIniInhib, yFinInhib):
				grid[x, y] = calculateNeuronFiringRate2(auxIndexInhib, inhibSpikes)
				auxIndexInhib += 1
			
			elif isInSubGrid(x, y, xIniControl, xFinControl, yIniControl, yFinControl):
				grid[x, y] = calculateNeuronFiringRate2(auxIndexControl, controlSpikes)
				auxIndexControl += 1
			
			elif isInSubGrid(x, y, xIniPattern1, xFinPattern1, yIniPattern1, yFinPattern1):
				
				if isInSubGrid(x, y, xIniPatternIntersection, xFinPatternIntersection, yIniPatternIntersection, yFinPatternIntersection):
					
					grid[x, y] = calculateNeuronFiringRate2(auxIndexPatternIntersection, intersectionSpikes)
					auxIndexPatternIntersection += 1

				elif isInSubGrid(x, y, xIniPattern1_stim, xFinPattern1_stim, yIniPattern1_stim, yFinPattern1_stim):
					
					grid[x, y] = calculateNeuronFiringRate2(auxIndexPattern1_stim, pattern1_stimSpikes)
					
					auxIndexPattern1_stim += 1
				else:
					grid[x, y] = calculateNeuronFiringRate2(auxIndexPattern1, pattern1Spikes)
					auxIndexPattern1 += 1
			
			elif isInSubGrid(x, y, xIniPattern2, xFinPattern2, yIniPattern2, yFinPattern2):

				if isInSubGrid(x, y, xIniPattern2_stim, xFinPattern2_stim, yIniPattern2_stim, yFinPattern2_stim):

					grid[x, y] = calculateNeuronFiringRate2(auxIndexPattern2_stim, pattern2_stimSpikes)
					auxIndexPattern2_stim += 1

				else:
					grid[x, y] = calculateNeuronFiringRate2(auxIndexPattern2, pattern2Spikes)
					auxIndexPattern2 += 1
			
			else:
				grid[x, y] = calculateNeuronFiringRate2(auxIndexExc, excSpikes)
				auxIndexExc += 1
	'''
	for i in range(100):
		
		j = range(100)
		plt.scatter(i*np.ones(100), j, c=grid[i, :], hold="true")
	'''
	##plt.imshow(grid, cmap=plt.cm.afmhot)
	im = plt.imshow(grid, vmin=0, vmax=200, interpolation='none', cmap=plt.cm.YlOrBr_r)
	return im




def plotGrid_reduced(axis, excSpikes, pattern1Spikes, pattern2Spikes, intersectionSpikes, controlSpikes, inhibSpikes):

	axis.get_xaxis().set_visible(False)
	axis.get_yaxis().set_visible(False)
	
	auxIndexInhib = 0
	auxIndexExc = 0
	auxIndexControl = 0
	auxIndexPattern1 = 0
	auxIndexPattern2 = 0
	auxIndexPatternIntersection = 0
	
	
	xIniInhib = 0
	xFinInhib = 9
	yIniInhib = 8
	yFinInhib = 9
	
	xIniControl = 1
	xFinControl = 3
	yIniControl = 1
	yFinControl = 3
	
	xIniPattern1 = 6
	xFinPattern1 = 8
	yIniPattern1 = 3
	yFinPattern1 = 5
	
	xIniPattern2 = 5
	xFinPattern2 = 7
	yIniPattern2 = 4
	yFinPattern2 = 6
	
	xIniPatternIntersection = 6
	xFinPatternIntersection = 7
	yIniPatternIntersection = 4
	yFinPatternIntersection = 5
	
	
	
	grid = np.zeros((10, 10))
	
	for x in range(10):

		for y in range(10):

			if isInSubGrid(x, y, xIniInhib, xFinInhib, yIniInhib, yFinInhib):
				grid[x, y] = calculateNeuronFiringRate2(auxIndexInhib, inhibSpikes)
				auxIndexInhib += 1
			
			elif isInSubGrid(x, y, xIniControl, xFinControl, yIniControl, yFinControl):
				grid[x, y] = calculateNeuronFiringRate2(auxIndexControl, controlSpikes)
				auxIndexControl += 1
			
			elif isInSubGrid(x, y, xIniPattern1, xFinPattern1, yIniPattern1, yFinPattern1):
				
				if isInSubGrid(x, y, xIniPatternIntersection, xFinPatternIntersection, yIniPatternIntersection, yFinPatternIntersection):
					
					grid[x, y] = calculateNeuronFiringRate2(auxIndexPatternIntersection, intersectionSpikes)
					auxIndexPatternIntersection += 1
				else:
					grid[x, y] = calculateNeuronFiringRate2(auxIndexPattern1, pattern1Spikes)
					auxIndexPattern1 += 1
			
			elif isInSubGrid(x, y, xIniPattern2, xFinPattern2, yIniPattern2, yFinPattern2):
				grid[x, y] = calculateNeuronFiringRate2(auxIndexPattern2, pattern2Spikes)
				auxIndexPattern2 += 1
			
			else:
				grid[x, y] = calculateNeuronFiringRate2(auxIndexExc, excSpikes)
				auxIndexExc += 1
	'''
	for i in range(100):

		
		j = range(100)
		plt.scatter(i*np.ones(100), j, c=grid[i, :], hold="true")
	'''
	##plt.imshow(grid, cmap=plt.cm.afmhot)
	im = plt.imshow(grid, vmin=0, vmax=200, interpolation='none', cmap=plt.cm.YlOrBr_r)
	return im


def plotGrid_reduced2(axis, excSpikes, pattern1Spikes, pattern2Spikes, intersectionSpikes, controlSpikes, inhibSpikes):
	
	axis.get_xaxis().set_visible(False)
	axis.get_yaxis().set_visible(False)
	
	auxIndexInhib = 0
	auxIndexExc = 0
	auxIndexControl = 0
	auxIndexPattern1 = 0
	auxIndexPattern2 = 0
	auxIndexPatternIntersection = 0
	
	
	xIniInhib = 0
	xFinInhib = 19
	yIniInhib = 16
	yFinInhib = 19
	
	xIniControl = 2
	xFinControl = 7
	yIniControl = 2
	yFinControl = 7
	
	xIniPattern1 = 12
	xFinPattern1 = 17
	yIniPattern1 = 6
	yFinPattern1 = 11
	
	xIniPattern2 = 10
	xFinPattern2 = 15
	yIniPattern2 = 8
	yFinPattern2 = 13
	
	xIniPatternIntersection = 12
	xFinPatternIntersection = 15
	yIniPatternIntersection = 8
	yFinPatternIntersection = 11
	
	
	
	grid = np.zeros((20, 20))
	
	for x in range(20):

		for y in range(20):

			if isInSubGrid(x, y, xIniInhib, xFinInhib, yIniInhib, yFinInhib):
				grid[x, y] = calculateNeuronFiringRate2(auxIndexInhib, inhibSpikes)
				auxIndexInhib += 1
			
			elif isInSubGrid(x, y, xIniControl, xFinControl, yIniControl, yFinControl):
				grid[x, y] = calculateNeuronFiringRate2(auxIndexControl, controlSpikes)
				auxIndexControl += 1
			
			elif isInSubGrid(x, y, xIniPattern1, xFinPattern1, yIniPattern1, yFinPattern1):
				
				if isInSubGrid(x, y, xIniPatternIntersection, xFinPatternIntersection, yIniPatternIntersection, yFinPatternIntersection):
					
					grid[x, y] = calculateNeuronFiringRate2(auxIndexPatternIntersection, intersectionSpikes)
					auxIndexPatternIntersection += 1
				else:
					grid[x, y] = calculateNeuronFiringRate2(auxIndexPattern1, pattern1Spikes)
					auxIndexPattern1 += 1
			
			elif isInSubGrid(x, y, xIniPattern2, xFinPattern2, yIniPattern2, yFinPattern2):
				grid[x, y] = calculateNeuronFiringRate2(auxIndexPattern2, pattern2Spikes)
				auxIndexPattern2 += 1
			
			else:
				grid[x, y] = calculateNeuronFiringRate2(auxIndexExc, excSpikes)
				auxIndexExc += 1
	'''
	for i in range(100):
		
		j = range(100)
		plt.scatter(i*np.ones(100), j, c=grid[i, :], hold="true")
	'''
	##plt.imshow(grid, cmap=plt.cm.afmhot)
	im = plt.imshow(grid, vmin=0, vmax=200, interpolation='none')
	return im




