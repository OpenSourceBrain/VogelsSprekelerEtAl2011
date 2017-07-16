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
import quantities as pq
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
		return np.diff(neuronSpikeTimes.view(pq.Quantity))


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


def plotRaster2(popSpikes, color):
	seg = popSpikes.segments[0]
	for spiketrain in seg.spiketrains:
		y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
		plt.plot(spiketrain, y, '.', c=color)
	plt.xlabel('Time [ms]')


def plotRaster3(axis, popSpikes, color, yAxisFactor):
	seg = popSpikes.segments[0]
	i = 0
	for spiketrain in seg.spiketrains:
		i = i + 1
		y = yAxisFactor * np.ones_like(spiketrain) * (i)
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


def plotSpikeTrains(popSpikes, timeStep, simTimeIni, simTimeFin):
	seg = popSpikes.segments[0]
	for neuronSpikeTimes in seg.spiketrains:
		y = createSpikeTrain(neuronSpikeTimes, timeStep, simTimeIni, simTimeFin)
		x = np.linspace(simTimeIni, simTimeFin, num = int((simTimeFin-simTimeIni)/timeStep))
		plt.plot(x, y)
	plt.xlabel('Time [ms]')



def plotFilteredSpikeTrains(popSpikes, timeStep, simTimeIni, simTimeFin, timeBoundKernel):
	seg = popSpikes.segments[0]
	for neuronSpikeTimes in seg.spiketrains:
		y = filterSpikesTrain (neuronSpikeTimes, timeStep, simTimeIni, simTimeFin, timeBoundKernel)
		numPoints = int((simTimeFin - simTimeIni)/timeStep)
		if int(2*timeBoundKernel/timeStep) + 1 > numPoints:
			numPoints = int(2*timeBoundKernel/timeStep) + 1
		x = np.linspace(simTimeIni, simTimeFin, num = numPoints)
		plt.plot(x, y)
	plt.xlabel('Time [ms]')



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
	return (1./tau1) * np.exp(-np.absolute(t) / tau1)  -  (1./tau2) * np.exp(-np.absolute(t) / tau2)





def biExponentialKernel (timeStep, timeBoundKernel):
	"Calculate the bi-exponential kernel as in Vogels et al. 2011. The time is specified in ms"
	numOfPoints = int(2*timeBoundKernel/timeStep) + 1
	kernel = np.zeros(numOfPoints)
	index = 0
	for t in np.linspace(-timeBoundKernel, timeBoundKernel, num = numOfPoints):
		kernel[index] = biExponentialKernelFunction (t)
		index += 1
	return kernel


def createSpikeTrain (neuronSpikes, timeStep, simTimeIni, simTimeFin):
	"Receives the spike times of a neuron and returns an array representing the spike train"
	spikeIndex = 0
	spikeTrain = np.zeros(int((simTimeFin - simTimeIni)/timeStep))
	for t in np.linspace(simTimeIni, simTimeFin, num = int((simTimeFin-simTimeIni)/timeStep)):
		if spikeIndex < np.size(neuronSpikes):
			spikeTime = neuronSpikes[spikeIndex]
			#print (t)
			#print (spikeTime)
			if int(t) == int(spikeTime): #rounding to 1ms timeStep
				index = int((t-simTimeIni)/timeStep)
				spikeTrain[index] = 1
				spikeIndex += 1
	return spikeTrain


def filterSpikesTrain (neuronSpikes, timeStep, simTimeIni, simTimeFin, timeBoundKernel):
	"Filter spike train using the bi-exponential kernel"
	spikeTrain = createSpikeTrain (neuronSpikes, timeStep, simTimeIni, simTimeFin)
	filteredSignal = np.convolve(spikeTrain, biExponentialKernel (timeStep, timeBoundKernel), 'same')
	return filteredSignal


def calculateCorrCoef (allSpikes, neuronIndex1, neuronIndex2, autoCov, timeStep, simTimeIni, simTimeFin, timeBoundKernel):
	"Calculate the correlation coeficient between spikeTrain1 and spikeTrain2"
	
	if neuronIndex1 == neuronIndex2:
		return 1
	else:
		Fi = filterSpikesTrain(allSpikes[neuronIndex1], timeStep, simTimeIni, simTimeFin, timeBoundKernel)
		Fj = filterSpikesTrain(allSpikes[neuronIndex2], timeStep, simTimeIni, simTimeFin, timeBoundKernel)
		Vij = np.sum(Fi*Fj)
		Vii = autoCov[neuronIndex1]
		Vjj = autoCov[neuronIndex2]
		#print "Fi: %f\t Fj: %f\t Vij: %f\t Vii: %f\t Vjj: %f" %(np.mean(Fi), np.mean(Fj), Vij, Vii, Vjj) 
		return Vij/np.sqrt(Vii*Vjj)


def createAutoCov(numNeuronsPop, popSpikes, timeStep, simTimeIni, simTimeFin, timeBoundKernel): # to speed up processing
	seg = popSpikes.segments[0]
	allSpikes = seg.spiketrains
	
	cov = np.zeros(numNeuronsPop)
	index = 0
	for neuronSpikes in allSpikes:
		if np.size(neuronSpikes) > 0:
			V = filterSpikesTrain(neuronSpikes, timeStep, simTimeIni, simTimeFin, timeBoundKernel)
			cov[index] = np.sum(V*V)
		else:
			cov[index] = -1 # To sinalize that there is no spike in this neuron
		index += 1
	return cov



def plotCorrHist(axis, numNeuronsPop, popSpikes, timeStep, simTimeIni, simTimeFin, timeBoundKernel, barColor):
	seg = popSpikes.segments[0]
	allSpikes = seg.spiketrains
	numSpikingNeurons = 0
	indexesSpikingNeurons = np.zeros(numNeuronsPop)
	
	autoCov = createAutoCov(numNeuronsPop, popSpikes, timeStep, simTimeIni, simTimeFin, timeBoundKernel)
	
	corrCoefs = np.zeros(0)
	for k in range(0, numNeuronsPop): 	# Considering only spiking neurons
			#print "k: %d" %k
			if np.size(allSpikes[k]) > 0:
				numSpikingNeurons += 1
				indexesSpikingNeurons[k] = 1
	
	
	
	for i in range(0, numNeuronsPop):
		for j in range(i, numNeuronsPop):
			if indexesSpikingNeurons[i] == 1 and indexesSpikingNeurons[j] == 1 and i != j:
				corrCoef = calculateCorrCoef(allSpikes,i,j, autoCov, timeStep, simTimeIni, simTimeFin, timeBoundKernel)
				corrCoefs = np.append(corrCoefs, corrCoef)
				print("i: %d\tj: %d\tcorrCoef: %f" % (i, j, corrCoef))
	
	
	if np.size(corrCoefs) != 0:	
		plt.hist(corrCoefs, histtype='stepfilled', color=barColor, alpha=0.7)
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


def plotCorrDoubleHist(axis, numNeuronsPop, popSpikes, barColor, numNeuronsPop2, popSpikes2, barColor2, timeStep, simTimeIni, simTimeFin, timeBoundKernel):
	seg2 = popSpikes2.segments[0]
	allSpikes2 = seg2.spiketrains
	#print "allSpikes2: "
	#print allSpikes2
	#print("\n\n")
	numSpikingNeurons2 = 0
	indexesSpikingNeurons2 = np.zeros(numNeuronsPop2)
	autoCov = createAutoCov(numNeuronsPop2, popSpikes2, timeStep, simTimeIni, simTimeFin, timeBoundKernel)
	
	corrCoefs2 = np.zeros(0)
	
	for k in range(0, numNeuronsPop2): 	# Considering only spiking neurons
			#print "allSpikes2[%d]: " %k
			#print allSpikes2[k]
			#print("\n\n")
			if np.size(allSpikes2[k]) > 0:
				numSpikingNeurons2 += 1
				indexesSpikingNeurons2[k] = 1
	
	print("\n")
	for i in range(0, numNeuronsPop2):
		for j in range(i, numNeuronsPop2):
			if indexesSpikingNeurons2[i] == 1 and indexesSpikingNeurons2[j] == 1 and i != j:
				corrCoef = calculateCorrCoef(allSpikes2,i,j, autoCov, timeStep, simTimeIni, simTimeFin, timeBoundKernel)
				corrCoefs2 = np.append(corrCoefs2, corrCoef)
				print("control: i: %d\tj: %d\tcorrCoef: %f" % (i, j, corrCoef))
	
	if np.size(corrCoefs2) != 0:	
		plt.hist(corrCoefs2, histtype='stepfilled', color=barColor2, alpha=0.7)


	seg = popSpikes.segments[0]
	allSpikes = seg.spiketrains
	numSpikeTrains = np.size(allSpikes)
	#print "allSpikes: "
	#print allSpikes
	#print("\n\n")
	numSpikingNeurons = 0
	indexesSpikingNeurons = np.zeros(numNeuronsPop)
	autoCov = createAutoCov(numNeuronsPop, popSpikes, timeStep, simTimeIni, simTimeFin, timeBoundKernel)
	
	corrCoefs = np.zeros(0)
	
	for k in range(0, numNeuronsPop): 	# Considering only spiking neurons
			#print "allSpikes[%d]: " %k
			#print allSpikes[k]
			#print("\n\n")
			if np.size(allSpikes[k]) > 0:
				numSpikingNeurons += 1
				indexesSpikingNeurons[k] = 1
	
	print("\n")
	for i in range(0, numNeuronsPop):
		for j in range(i, numNeuronsPop):
			if indexesSpikingNeurons[i] == 1 and indexesSpikingNeurons[j] == 1 and i != j:
				corrCoef = calculateCorrCoef(allSpikes,i,j, autoCov, timeStep, simTimeIni, simTimeFin, timeBoundKernel)
				corrCoefs = np.append(corrCoefs, corrCoef)
				print("pattern1: i: %d\tj: %d\tcorrCoef: %f" %(i, j, corrCoef))
	
	if np.size(corrCoefs) != 0:	
		plt.hist(corrCoefs, histtype='stepfilled', color=barColor, alpha=0.7)

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


def plotFig4Column(fig, column, timeStep, simTimeIni, simTimeFin, timeBoundKernel,
		excSpikes, pattern1Spikes, pattern1_stimSpikes, pattern2Spikes, pattern2_stimSpikes, patternIntersectionSpikes, controlSpikes, inhibSpikes,
		numOfSampledNeuronsPattern1, sampledPopPattern1Spikes, numOfSampledNeuronsControl, sampledPopControlSpikes):

	ax1 = fig.add_subplot(4, 6, column)
	#ax1.set_position([0.1 + (column - 1) * 0.15, 0.8, 0.15, 0.15])
	
	if column == 1:
		ax1.set_title('pre')
	elif column == 2:
		ax1.set_title('A')
	elif column == 3:
		ax1.set_title('B')
	elif column == 4:
		ax1.set_title('C')
	elif column == 5:
		ax1.set_title('D')
	elif column == 6:
		ax1.set_title('E')
	
	im = plotGrid(ax1, excSpikes, pattern1Spikes, pattern1_stimSpikes, pattern2Spikes, pattern2_stimSpikes, patternIntersectionSpikes, controlSpikes, inhibSpikes)


	ax2 = fig.add_subplot(4, 6, column + 6)
	if column == 1:
		plt.ylabel('Cell no.')
		#plt.xlim((0.0, 40.0))
		ax2.spines['left'].set_color('black')
	else:
		ax2.get_yaxis().set_visible(False)
		#plt.xlim((0.0, 200.0))
	
	plt.ylim((-15, 15))
	#plotRaster(ax2, pattern1Spikes, 'red')
	plotRaster3(ax2, sampledPopPattern1Spikes, 'red', 1)
	plotRaster3(ax2, sampledPopControlSpikes, 'black', -1)
	ax2.tick_params(axis='y', left='on')
	#ax2.spines['left'].set_color('black')

	ax3 = fig.add_subplot(4, 6, column + 12)
	if column == 1:
		#plt.ylabel('Percent [%]')
		plt.ylabel('Counts')
		ax3.spines['left'].set_color('black')
	else:
		ax3.get_yaxis().set_visible(False)

	#plotISICVHist(ax3, pattern1Spikes, 'red')
	plotISICVDoubleHist(ax3, pattern1Spikes, 'red', controlSpikes, 'black')
	ax3.tick_params(axis='y', left='on')
	ax3.spines['left'].set_color('black')
	#ax3.spines['bottom'].set_position(('axes',0.5))


	ax4 = fig.add_subplot(4, 6, column + 18)
	if column == 1:
		#plt.ylabel('Percent [%]')
		plt.ylabel('Counts')
		ax4.spines['left'].set_color('black')
	else:
		ax4.get_yaxis().set_visible(False)
	#plotISICVHist(ax4, pattern1Spikes, 'red')
	plotCorrDoubleHist(ax4, numOfSampledNeuronsPattern1, sampledPopPattern1Spikes, 'red', 
				numOfSampledNeuronsControl, sampledPopControlSpikes, 'black', 
				timeStep, simTimeIni, simTimeFin, timeBoundKernel)
	
	ax4.spines['left'].set_color('black')
	ax4.tick_params(axis='y', left='on')
	plt.xlim((-0.2, 1.0))

	return im






'''
print("\n\nWEIGHT: connections['e_to_e']: ")
print(connections['e_to_e'].get('weight', format='list'))

print("\n\nWEIGHT: connections['p1_to_p1']: ")
print(connections['p1_to_p1'].get('weight', format='list'))

print("\n\nWEIGHT: connections['p2_to_p2']: ")
print(connections['p2_to_p2'].get('weight', format='list'))

print("\n\nETA: connections['i_to_e']:")
print(connections['i_to_e'].get('eta', format='list'))

print("\n\nWEIGHT: connections['i_to_e']: ")
print(connections['i_to_e'].get('weight', format='list'))

print("\n\nETA: connections['i_to_i']:")
print(connections['i_to_i'].get('eta', format='list'))

print("\n\nWEIGHT: connections['i_to_i']: ")
print(connections['i_to_i'].get('weight', format='list'))
'''
