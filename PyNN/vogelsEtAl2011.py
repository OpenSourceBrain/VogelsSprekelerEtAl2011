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

#############################################
##
##	VERSION 0.2 - Using PyNN 0.8
##
#############################################

from pyNN.random import RandomDistribution, NumpyRNG
import matplotlib.pyplot as plt
from auxRoutines import *
from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]  

exec("from pyNN.%s import *" % simulator_name)

print "Starting PyNN with simulator: %s"%simulator_name


# Total of 8000 excitatory neurons and 2000 inhibitory neurons


numOfNeuronsExcPopulation = 5712
numOfNeuronsPattern1 = 720
numOfNeuronsPattern2 = 720
numOfNeuronsPatternIntersection = 64
numOfNeuronsControl = 784

numOfNeuronsInhibPopulation = 2000


connectivity = 0.02
weightExcSynapses = 0.003 	# [uS]
weightInhibSynapses = 0.03 	# [uS]
potentiationFactor = 5

setup(timestep=0.1, min_delay=0.5)

neuronParameters = 	{
			'tau_m':	20.0,	# [ms]
			'cm':		0.2, 	# [nF]
			'v_rest':	-60.0,	# [mV]
			'v_thresh':	-50.0, 	# [mV]
			'tau_syn_E':	5.0,	# [ms]
			'tau_syn_I':	10.0,	# [ms]
			'e_rev_E':	0.0,	# [mV]
			'e_rev_I':	-80.0,	# [mV]
			'v_reset':	-60.0,	# [mV]
			'tau_refrac':	5.0,	# [ms]
			'i_offset': 	0.2	# [nA]
			}


cell_type = IF_cond_exp(**neuronParameters)

print("----------Creating the network-----------")

print("Creating neuron populations")

excPopulation 		= Population(numOfNeuronsExcPopulation		, cell_type, label='excPop')

inhibPopulation 	= Population(numOfNeuronsInhibPopulation	, cell_type, label='inhibPop')

pattern1 		= Population(numOfNeuronsPattern1		, cell_type, label='pattern1')

pattern2 		= Population(numOfNeuronsPattern2		, cell_type, label='pattern2')

patternIntersection 	= Population(numOfNeuronsPatternIntersection	, cell_type, label='patternIntersection')

controlPopulation	= Population(numOfNeuronsControl		, cell_type, label='controlPop')


rand_distr = RandomDistribution('normal', (-60.0, 5.0), rng=NumpyRNG(seed=85524))

excPopulation.initialize(v=rand_distr)
inhibPopulation.initialize(v=rand_distr)
pattern1.initialize(v=rand_distr)
pattern2.initialize(v=rand_distr)
patternIntersection.initialize(v=rand_distr)
controlPopulation.initialize(v=rand_distr)


# allow self-connections??

# what are the initial conditions??

# what are the synaptic delays?



fpc 	= FixedProbabilityConnector(connectivity)



# Check stdp model!

# Need to correct the max weight of the stdp to 0.3 uS


exc_synapse_type 		= StaticSynapse(weight = weightExcSynapses, delay=0.5)

pattern_synapse_type 		= StaticSynapse(weight = potentiationFactor, delay=0.5)

inhibitory_static_synapse_type 	= StaticSynapse(weight = weightInhibSynapses, delay=0.5)



# Setting

inhibitory_stdp_synapse_type 	= STDPMechanism(weight_dependence = AdditiveWeightDependence(w_max=0.3),
                         		timing_dependence = Vogels2011Rule(eta=1e-1, rho=1e-3),
                         		weight=0.0, delay=0.5)


print("Creating excitatory projections")


# Excitatory projections


e_to_e 	= Projection(excPopulation, 	excPopulation, 		fpc, 	exc_synapse_type)
e_to_p1 = Projection(excPopulation, 	pattern1, 		fpc, 	exc_synapse_type)
e_to_p2	= Projection(excPopulation, 	pattern2, 		fpc, 	exc_synapse_type)
e_to_pi = Projection(excPopulation, 	patternIntersection, 	fpc, 	exc_synapse_type)
e_to_c 	= Projection(excPopulation, 	controlPopulation, 	fpc, 	exc_synapse_type)
e_to_i 	= Projection(excPopulation, 	inhibPopulation, 	fpc, 	exc_synapse_type)


p1_to_e = Projection(pattern1, 		excPopulation, 		fpc, 	exc_synapse_type)
p1_to_p1= Projection(pattern1, 		pattern1, 		fpc, 	pattern_synapse_type)
p1_to_p2= Projection(pattern1, 		pattern2, 		fpc, 	exc_synapse_type)
p1_to_pi= Projection(pattern1, 		patternIntersection, 	fpc, 	pattern_synapse_type)
p1_to_c = Projection(pattern1, 		controlPopulation, 	fpc, 	exc_synapse_type)
p1_to_i = Projection(pattern1, 		inhibPopulation, 	fpc, 	exc_synapse_type)


p2_to_e = Projection(pattern2, 		excPopulation, 		fpc, 	exc_synapse_type)
p2_to_p1= Projection(pattern2, 		pattern1, 		fpc, 	exc_synapse_type)
p2_to_p2= Projection(pattern2, 		pattern2, 		fpc, 	pattern_synapse_type)
p2_to_pi= Projection(pattern2, 		patternIntersection, 	fpc, 	pattern_synapse_type)
p2_to_c = Projection(pattern2, 		controlPopulation, 	fpc, 	exc_synapse_type)
p2_to_i = Projection(pattern2, 		inhibPopulation, 	fpc, 	exc_synapse_type)


pi_to_e = Projection(pattern2, 		excPopulation, 		fpc, 	exc_synapse_type)
pi_to_p1= Projection(pattern2, 		pattern1, 		fpc, 	pattern_synapse_type)
pi_to_p2= Projection(pattern2, 		pattern2, 		fpc, 	pattern_synapse_type)
pi_to_pi= Projection(pattern2, 		patternIntersection, 	fpc, 	pattern_synapse_type)
pi_to_c = Projection(pattern2, 		controlPopulation, 	fpc, 	exc_synapse_type)
pi_to_i = Projection(pattern2, 		inhibPopulation, 	fpc, 	exc_synapse_type)


c_to_e 	= Projection(controlPopulation, excPopulation, 		fpc, 	exc_synapse_type)
c_to_p1 = Projection(controlPopulation, pattern1, 		fpc, 	exc_synapse_type)
c_to_p2 = Projection(controlPopulation, pattern2, 		fpc, 	exc_synapse_type)
c_to_pi = Projection(controlPopulation, patternIntersection, 	fpc, 	exc_synapse_type)
c_to_c 	= Projection(controlPopulation, controlPopulation, 	fpc, 	exc_synapse_type)
c_to_i 	= Projection(controlPopulation, inhibPopulation, 	fpc, 	exc_synapse_type)





''' THIS INITIAL PART IS COMMENTED DUE TO PROBLEMS IN CHANGING FROM STATIC FROM DYNAMIC SYNAPSE MODEL

print("Creating inhibitory projections")

# Inhibitory projections

# Initializing without stdp mechanism

i_to_e = Projection(inhibPopulation, 	excPopulation, 		fpc, 	inhibitory_static_synapse_type)
i_to_p1 = Projection(inhibPopulation, 	pattern1, 		fpc, 	inhibitory_static_synapse_type)
i_to_p2 = Projection(inhibPopulation, 	pattern2, 		fpc, 	inhibitory_static_synapse_type)
i_to_pi = Projection(inhibPopulation, 	patternIntersection, 	fpc, 	inhibitory_static_synapse_type)
i_to_c = Projection(inhibPopulation, 	controlPopulation, 	fpc, 	inhibitory_static_synapse_type)
i_to_i = Projection(inhibPopulation, 	inhibPopulation, 	fpc, 	inhibitory_static_synapse_type)




### SIMULATION TIMES WERE DOWNSCALED 10000 TIMES FOR TESTING PURPOSES


## Start
## 	The asynchronous irregular network dynamics of the model published in
## 	Vogels and Abbott (2005) without inhibitory plasticity.
## 	Original simulation time: 1 min (60000 ms)


excPopulation.record('spikes')
pattern1.record('spikes')
pattern2.record('spikes')
patternIntersection.record('spikes')
controlPopulation.record('spikes')
inhibPopulation.record('spikes')

print("Pre-simulation")

run(60)

excSpikes			= 	excPopulation.get_data('spikes')
pattern1Spikes 			=	pattern1.get_data('spikes')
pattern2Spikes 			=	pattern2.get_data('spikes')
patternIntersectionSpikes 	=	patternIntersection.get_data('spikes')
controlSpikes 			=	controlPopulation.get_data('spikes')
inhibSpikes 			= 	inhibPopulation.get_data('spikes')


plt.subplot(4, 5, 1)
plotGrid(excSpikes, pattern1Spikes, pattern2Spikes, patternIntersectionSpikes, controlSpikes, inhibSpikes)

plt.subplot(4, 5, 6)
plotISICVHist(pattern1Spikes, 13, 'red')

'''

## Fig. 4, A
##
## 	Inhibitory to excitatory synapses are turned to 0 efficacy
##	The network is forced out of the AI regime and begins to fire at high rates
## 	Inhibitory plasticity is turned on.
## 	Original simulation time: 

print("Fig. 4, A")

subPopPattern1 = pattern1.sample(392)
subPopControl = controlPopulation.sample(392)

excPopulation.record('spikes')
pattern1.record('spikes')
pattern2.record('spikes')
patternIntersection.record('spikes')
controlPopulation.record('spikes')
inhibPopulation.record('spikes')



i_to_e = Projection(inhibPopulation, 	excPopulation, 		fpc, 	inhibitory_stdp_synapse_type)
i_to_p1 = Projection(inhibPopulation, 	pattern1, 		fpc, 	inhibitory_stdp_synapse_type)
i_to_p2 = Projection(inhibPopulation, 	pattern2, 		fpc, 	inhibitory_stdp_synapse_type)
i_to_pi = Projection(inhibPopulation, 	patternIntersection, 	fpc, 	inhibitory_stdp_synapse_type)
i_to_c = Projection(inhibPopulation, 	controlPopulation, 	fpc, 	inhibitory_stdp_synapse_type)
i_to_i = Projection(inhibPopulation, 	inhibPopulation, 	fpc, 	inhibitory_static_synapse_type)



run(10)
print("10%")
run(10)
print("20%")
run(10)
print("30%")
run(10)
print("40%")
run(10)
print("50%")
run(10)
print("60%")
run(10)
print("70%")
run(10)
print("80%")
run(10)
print("90%")
run(10)
print("100%")


excSpikes			= 	excPopulation.get_data('spikes')
pattern1Spikes 			=	pattern1.get_data('spikes')
pattern2Spikes 			=	pattern2.get_data('spikes')
patternIntersectionSpikes 	=	patternIntersection.get_data('spikes')
controlSpikes 			=	controlPopulation.get_data('spikes')
inhibSpikes 			= 	inhibPopulation.get_data('spikes')

subPopPattern1Spikes 		=	subPopPattern1.get_data('spikes')
subPopControlSpikes 		= 	subPopControl.get_data('spikes')


print("ploting Fig. 4A")

plt.subplot(4, 5, 2)
plotGrid(excSpikes, pattern1Spikes, pattern2Spikes, patternIntersectionSpikes, controlSpikes, inhibSpikes)

plt.subplot(4, 5, 7)
plotISICVHist(subPopPattern1Spikes, 13, 'red')

plt.subplot(4, 5, 12)
plotISICVHist(subPopControlSpikes, 13, 'black')



## Fig. 4, B
##
## 	Inhibitory plasticity has restored asynchronous irregular dynamics
##	Original simulation time: 60 min (60 * 60 * 1000 ms)


print("Continuing simulation...")

run(26)
print("10%")
run(26)
print("20%")
run(26)
print("30%")
run(26)
print("40%")
run(26)
print("50%")
run(26)
print("60%")
run(26)
print("70%")
run(26)
print("80%")
run(26)
print("90%")
run(26)
print("100%")


excSpikes			= 	excPopulation.get_data('spikes')
pattern1Spikes 			=	pattern1.get_data('spikes')
pattern2Spikes 			=	pattern2.get_data('spikes')
patternIntersectionSpikes 	=	patternIntersection.get_data('spikes')
controlSpikes 			=	controlPopulation.get_data('spikes')
inhibSpikes 			= 	inhibPopulation.get_data('spikes')

subPopPattern1Spikes 		=	subPopPattern1.get_data('spikes')
subPopControlSpikes 		= 	subPopControl.get_data('spikes')


print("ploting Fig. 4B")

plt.subplot(4, 5, 3)
plotGrid(excSpikes, pattern1Spikes, pattern2Spikes, patternIntersectionSpikes, controlSpikes, inhibSpikes)

plt.subplot(4, 5, 8)
plotISICVHist(subPopPattern1Spikes, 13, 'red')

plt.subplot(4, 5, 13)
plotISICVHist(subPopControlSpikes, 13, 'black')





## Fig. 4, C
##
## 	The excitatory non-zero weights of the two designated memory patterns 
## 	are increased ad-hoc by a factor of 5. The neurons of hte subset begin
## 	to exhibit elevated and more sychronized activity
##	Original simulation time: 5 sec (5000 ms)





pattern_synapse_type 		= StaticSynapse(weight = potentiationFactor * weightExcSynapses, delay=0.1)

p1_to_e = Projection(pattern1, 		excPopulation, 		fpc, 	exc_synapse_type)
p1_to_p1= Projection(pattern1, 		pattern1, 		fpc, 	pattern_synapse_type)
p1_to_p2= Projection(pattern1, 		pattern2, 		fpc, 	exc_synapse_type)
p1_to_pi= Projection(pattern1, 		patternIntersection, 	fpc, 	pattern_synapse_type)

p1_to_c = Projection(pattern1, 		controlPopulation, 	fpc, 	exc_synapse_type)
p1_to_i = Projection(pattern1, 		inhibPopulation, 	fpc, 	exc_synapse_type)


p2_to_e = Projection(pattern2, 		excPopulation, 		fpc, 	exc_synapse_type)
p2_to_p1= Projection(pattern2, 		pattern1, 		fpc, 	exc_synapse_type)
p2_to_p2= Projection(pattern2, 		pattern2, 		fpc, 	pattern_synapse_type)
p2_to_pi= Projection(pattern2, 		patternIntersection, 	fpc, 	pattern_synapse_type)
p2_to_c = Projection(pattern2, 		controlPopulation, 	fpc, 	exc_synapse_type)
p2_to_i = Projection(pattern2, 		inhibPopulation, 	fpc, 	exc_synapse_type)


pi_to_e = Projection(pattern2, 		excPopulation, 		fpc, 	exc_synapse_type)
pi_to_p1= Projection(pattern2, 		pattern1, 		fpc, 	pattern_synapse_type)
pi_to_p2= Projection(pattern2, 		pattern2, 		fpc, 	pattern_synapse_type)
pi_to_pi= Projection(pattern2, 		patternIntersection, 	fpc, 	pattern_synapse_type)
pi_to_c = Projection(pattern2, 		controlPopulation, 	fpc, 	exc_synapse_type)
pi_to_i = Projection(pattern2, 		inhibPopulation, 	fpc, 	exc_synapse_type)


run(5)

excSpikes			= 	excPopulation.get_data('spikes')
pattern1Spikes 			=	pattern1.get_data('spikes')
pattern2Spikes 			=	pattern2.get_data('spikes')
patternIntersectionSpikes 	=	patternIntersection.get_data('spikes')
controlSpikes 			=	controlPopulation.get_data('spikes')
inhibSpikes 			= 	inhibPopulation.get_data('spikes')

subPopPattern1Spikes 		=	subPopPattern1.get_data('spikes')
subPopControlSpikes 		= 	subPopControl.get_data('spikes')


print("ploting Fig. 4C")

plt.subplot(4, 5, 4)
plotGrid(excSpikes, pattern1Spikes, pattern2Spikes, patternIntersectionSpikes, controlSpikes, inhibSpikes)

plt.subplot(4, 5, 9)
plotISICVHist(subPopPattern1Spikes, 13, 'red')

plt.subplot(4, 5, 14)
plotISICVHist(subPopControlSpikes, 13, 'black')





## Fig. 4, D
##
## 	Inhibitory plasticity has succesfully suppressed any elevated activity
## 	from the pattern and restored the global background state
##	Original simulation time: 5 sec (5000 ms)



print("Continuing simulation...")

run(35)
print("10%")
run(36)
print("20%")
run(36)
print("30%")
run(36)
print("40%")
run(36)
print("50%")
run(36)
print("60%")
run(36)
print("70%")
run(36)
print("80%")
run(36)
print("90%")
run(36)
print("100%")

print("ploting Fig. 4D")


excSpikes			= 	excPopulation.get_data('spikes')
pattern1Spikes 			=	pattern1.get_data('spikes')
pattern2Spikes 			=	pattern2.get_data('spikes')
patternIntersectionSpikes 	=	patternIntersection.get_data('spikes')
controlSpikes 			=	controlPopulation.get_data('spikes')
inhibSpikes 			= 	inhibPopulation.get_data('spikes')

subPopPattern1Spikes 		=	subPopPattern1.get_data('spikes')
subPopControlSpikes 		= 	subPopControl.get_data('spikes')


plt.subplot(4, 5, 5)
plotGrid(excSpikes, pattern1Spikes, pattern2Spikes, patternIntersectionSpikes, controlSpikes, inhibSpikes)

plt.subplot(4, 5, 10)
plotISICVHist(subPopPattern1Spikes, 13, 'red')

plt.subplot(4, 5, 15)
plotISICVHist(subPopControlSpikes, 13, 'black')


plt.show()



