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
##		VERSION 0.1
##
#############################################

from pyNN.neuron import *


# Total of 8000 excitatory neurons and 2000 inhibitory neurons

numOfNeuronsExcPopulation = 5712
numOfNeuronsPattern1 = 720
numOfNeuronsPattern2 = 720
numOfNeuronsPatternIntersection = 64
numOfNeuronsControl = 784

numOfNeuronsInhibPopulation = 2000


# testing

numOfNeuronsExcPopulation = 571
numOfNeuronsPattern1 = 72
numOfNeuronsPattern2 = 72
numOfNeuronsPatternIntersection = 6
numOfNeuronsControl = 78

numOfNeuronsInhibPopulation = 200


connectivity = 0.02
weightExcSynapses = 0.003 	# [uS]
weightInhibSynapses = 0.03 	# [uS]
potentiationFactor = 5


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


setup(timestep=0.1)



excPopulation 		= Population(numOfNeuronsExcPopulation		, IF_cond_exp, neuronParameters, label='excPop')
pattern1 		= Population(numOfNeuronsPattern1		, IF_cond_exp, neuronParameters, label='pattern1')
pattern2 		= Population(numOfNeuronsPattern2		, IF_cond_exp, neuronParameters, label='pattern2')
patternIntersection 	= Population(numOfNeuronsPatternIntersection	, IF_cond_exp, neuronParameters, label='patternIntersection')
controlPopulation	= Population(numOfNeuronsControl		, IF_cond_exp, neuronParameters, label='controlPop')

inhibPopulation 	= Population(numOfNeuronsInhibPopulation, IF_cond_exp, neuronParameters, label='inhibPop')





# allow self-connections??

# what are the synaptic delays?

fpc = FixedProbabilityConnector


excConn 	= fpc(connectivity, weights = weightExcSynapses, 			delays = 0.1)	
patternConn 	= fpc(connectivity, weights = potentiationFactor * weightExcSynapses, 	delays = 0.1)
inhibConn 	= fpc(connectivity, weights = weightInhibSynapses, 			delays = 0.1)	






# Excitatory projections

e_to_e = Projection(excPopulation, 	excPopulation, 		excConn, 	target = 'excitatory')
e_to_p1 = Projection(excPopulation, 	pattern1, 		excConn, 	target = 'excitatory')
e_to_p2 = Projection(excPopulation, 	pattern2, 		excConn, 	target = 'excitatory')
e_to_pi = Projection(excPopulation, 	patternIntersection, 	excConn, 	target = 'excitatory')
e_to_c = Projection(excPopulation, 	controlPopulation, 	excConn, 	target = 'excitatory')
e_to_i = Projection(excPopulation, 	inhibPopulation, 	excConn, 	target = 'excitatory')


p1_to_e = Projection(pattern1, 		excPopulation, 		excConn, 	target = 'excitatory')
p1_to_p1 = Projection(pattern1, 	pattern1, 		patternConn, 	target = 'excitatory')
p1_to_p2 = Projection(pattern1, 	pattern2, 		excConn, 	target = 'excitatory')
p1_to_pi = Projection(pattern1, 	patternIntersection, 	patternConn, 	target = 'excitatory')
p1_to_c = Projection(pattern1, 		controlPopulation, 	excConn, 	target = 'excitatory')
p1_to_i = Projection(pattern1, 		inhibPopulation, 	excConn, 	target = 'excitatory')


p2_to_e = Projection(pattern2, 		excPopulation, 		excConn, 	target = 'excitatory')
p2_to_p1 = Projection(pattern2, 	pattern1, 		excConn, 	target = 'excitatory')
p2_to_p2 = Projection(pattern2, 	pattern2, 		patternConn, 	target = 'excitatory')
p2_to_pi = Projection(pattern2, 	patternIntersection, 	patternConn, 	target = 'excitatory')
p2_to_c = Projection(pattern2, 		controlPopulation, 	excConn, 	target = 'excitatory')
p2_to_i = Projection(pattern2, 		inhibPopulation, 	excConn, 	target = 'excitatory')


pi_to_e = Projection(pattern2, 		excPopulation, 		excConn, 	target = 'excitatory')
pi_to_p1 = Projection(pattern2, 	pattern1, 		patternConn, 	target = 'excitatory')
pi_to_p2 = Projection(pattern2, 	pattern2, 		patternConn, 	target = 'excitatory')
pi_to_pi = Projection(pattern2, 	patternIntersection, 	patternConn, 	target = 'excitatory')
pi_to_c = Projection(pattern2, 		controlPopulation, 	excConn, 	target = 'excitatory')
pi_to_i = Projection(pattern2, 		inhibPopulation, 	excConn, 	target = 'excitatory')


c_to_e = Projection(controlPopulation, 	excPopulation, 		excConn, 	target = 'excitatory')
c_to_p1 = Projection(controlPopulation, pattern1, 		excConn, 	target = 'excitatory')
c_to_p2 = Projection(controlPopulation, pattern2, 		excConn, 	target = 'excitatory')
c_to_pi = Projection(controlPopulation, patternIntersection, 	excConn, 	target = 'excitatory')
c_to_c = Projection(controlPopulation, 	controlPopulation, 	excConn, 	target = 'excitatory')
c_to_i = Projection(controlPopulation, 	inhibPopulation, 	excConn, 	target = 'excitatory')




# Inhibitory projections


i_to_e = Projection(inhibPopulation, 	excPopulation, 		inhibConn, 	target = 'inhibitory')
i_to_p1 = Projection(inhibPopulation, 	pattern1, 		inhibConn, 	target = 'inhibitory')
i_to_p2 = Projection(inhibPopulation, 	pattern2, 		inhibConn, 	target = 'inhibitory')
i_to_pi = Projection(inhibPopulation, 	patternIntersection, 	inhibConn, 	target = 'inhibitory')
i_to_c = Projection(inhibPopulation, 	controlPopulation, 	inhibConn, 	target = 'inhibitory')
i_to_i = Projection(inhibPopulation, 	inhibPopulation, 	inhibConn, 	target = 'inhibitory')














