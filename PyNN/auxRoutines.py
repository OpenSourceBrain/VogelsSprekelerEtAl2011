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


def plotRaster(popSpikes, color):
	seg = popSpikes.segments[0]
	for spiketrain in seg.spiketrains:
		y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
		plt.plot(spiketrain, y, '.')


def plotISICVHist(popSpikes, barColor):
	seg = popSpikes.segments[0]
	allSpikes = seg.spiketrains
	isiCVs = np.zeros(0)
	for neuronSpikes in allSpikes:
		neuronISICV = calculateISICV2(neuronSpikes)
		if neuronISICV != -1:
			isiCVs = np.append(isiCVs, neuronISICV)
	if np.size(isiCVs) != 0:	
		plt.hist(isiCVs, color=barColor)
	plt.ylabel('Percent [%]')
	plt.xlabel('ISI CV')
	plt.ylim((0, 100))
	plt.xlim((0.0, 3.0))


def isInSubGrid (x, y, xIni, xFin, yIni, yFin):
	"Checks if an element with coordenates (x, y) in a grid is within a sub-grid with the specified limits"
	return (x >= xIni) and (x <= xFin) and (y >= yIni) and (y <= yFin)



def plotGrid(excSpikes, pattern1Spikes, pattern2Spikes, intersectionSpikes, controlSpikes, inhibSpikes):
	
	auxIndexInhib = 0
	auxIndexExc = 0
	auxIndexControl = 0
	auxIndexPattern1 = 0
	auxIndexPattern2 = 0
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
	
	xIniPattern2 = 22
	xFinPattern2 = 49
	yIniPattern2 = 50
	yFinPattern2 = 77
	
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
	im = plt.imshow(grid, vmin=0, vmax=300)
	return im









