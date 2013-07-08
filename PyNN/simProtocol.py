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
##	Simulation protocol to generate Fig. 4 of Vogels et al 2011
##
########################################################################


import matplotlib.pyplot as plt


### SIMULATION TIMES WERE DOWNSCALED 1000 TIMES FOR TESTING PURPOSES


## Start
## 	The asynchronous irregular network dynamics of the model published in
## 	Vogels and Abbott (2005) without inhibitory plasticity.

print("Start")


excPopulation.record()
pattern1.record()
pattern2.record()
patternIntersection.record()
controlPopulation.record()
inhibPopulation.record()

## run(60)

run(100)

excSpikes			= 	excPopulation.getSpikes()
pattern1Spikes 			=	pattern1.getSpikes()
pattern2Spikes 			=	pattern2.getSpikes()
patternIntersectionSpikes 	=	patternIntersection.getSpikes()
controlSpikes 			=	controlPopulation.getSpikes()
inhibSpikes 			= 	inhibPopulation.getSpikes()






## Fig. 4, A
##
## 	Inhibitory to excitatory synapses are turned to 0 efficacy
##	The network is forced out of the AI regime and begins to fire at high rates
## 	Inhibitory plasticity is turned on.

print("Fig. 4, A")


i_to_e.setWeights(0.0)
i_to_p1.setWeights(0.0)
i_to_p2.setWeights(0.0)
i_to_pi.setWeights(0.0)
i_to_c.setWeights(0.0)
i_to_i.setWeights(0.0)



## run(60*60)
run(100)

excSpikes			= 	excPopulation.getSpikes()
pattern1Spikes 			=	pattern1.getSpikes()
pattern2Spikes 			=	pattern2.getSpikes()
patternIntersectionSpikes 	=	patternIntersecton.getSpikes()
controlSpikes 			=	controlPopulation.getSpikes()
inhibSpikes 			= 	inhibPopulation.getSpikes()



plt.subplot(611)
plt.scatter(excSpikes[:,1], excSpikes[:,0], c=excSpikes[:,0], cmap=plt.cm.afmhot)
plt.colorbar()
plt.ylabel('Spikes of excitatory neurons')


plt.subplot(612)
plt.scatter(pattern1Spikes[:,1], pattern1Spikes[:,0], c=pattern1Spikes[:,0], cmap=plt.cm.afmhot)
plt.colorbar()
plt.ylabel('Spikes of pattern1 neurons')


plt.subplot(613)
plt.scatter(pattern2Spikes[:,1], pattern2Spikes[:,0], c=pattern2Spikes[:,0], cmap=plt.cm.afmhot)
plt.colorbar()
plt.ylabel('Spikes of pattern2 neurons')


plt.subplot(614)
plt.scatter(patternIntersectionSpikes[:,1], patternIntersectionSpikes[:,0], c=patternIntersectionSpikes[:,0], cmap=plt.cm.afmhot)
plt.colorbar()
plt.ylabel('Spikes of patternIntersection neurons')


plt.subplot(615)
plt.scatter(controlSpikes[:,1], controlSpikes[:,0], c=controlSpikes[:,0], cmap=plt.cm.afmhot)
plt.colorbar()
plt.ylabel('Spikes of control neurons')


plt.subplot(616)
plt.scatter(inhibSpikes[:,1], inhibSpikes[:,0], c=inhibSpikes[:,0], cmap=plt.cm.afmhot)
plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Spikes of inhibitory neurons')


## Fig. 4, B
##
## 	Inhibitory plasticity has restored asynchronous irregular dynamics

print("Fig. 4, B")


stdpModel = STDPMechanism(timing_dependence=SpikePairRule(tau_plus=20.0, tau_minus=20.0), weight_dependence=AdditiveWeightDependence(w_min=0, w_max=0.02, A_plus=0.01, A_minus=0.012))




run(5)

## Fig. 4, C
##
## 	The excitatory non-zero weights of the two designated memory patterns 
## 	are increased ad-hoc by a factor of 5. The neurons of hte subset begin
## 	to exhibit elevated and more sychronized activity

print("Fig. 4, C")






## run(60*60 - 5)

run(30)


## Fig. 4, D
##
## 	Inhibitory plasticity has succesfully suppressed any elevated activity
## 	from the pattern and restored the global background state

print("Fig. 4, D")






run(5)

## Fig. 4, E
##
## 	By delivering an additional, 1 s long stimulus as described above to
## 	25% of the cells within one memory pattern, the whole pattern is activated.
## 	Activity inside the pattern stays asynchronous and irregular, and the rest
## 	of the network, including the other pattern, ramains nearly unaffected

print("Fig. 4, E")






plt.show()


print("fim")




