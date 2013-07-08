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
##		Simple script to run short simulation
##      and save some traces
##
##      Author: Padraig Gleeson
##
#############################################

from pyNN.utility import get_script_args
simulator_name = get_script_args(1)[0]  
exec("from pyNN.%s import *" % simulator_name)

print "Starting PyNN with simulator: %s"%simulator_name

from networkModel import *
someCells = excPopulation.sample(12)
someCells.record_v()

tstop = 500
run(tstop)

someCells.print_v("someCells_%s.v" % simulator_name)


print "Finished simulation of %g ms"%tstop

end()
