#!/usr/bin/env python3
"""
NEST implementation of Vogels-Sprekeler network model.

File: VogelsSprekeler.py

This implementation adds location information to the neurons to make it easier
to plot network snapshots. The original model does not incorporate location
information. Please note that the inclusion of geometrical information does not
affect the network in any way.

*** NOTE: Requires NEST >= 2.12 ***
https://github.com/nest/nest-simulator/releases/tag/v2.12.0

Copyright 2017 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import nest
import numpy
import math
# use random.sample instead of numpy.random - faster
import random
from scipy.spatial import cKDTree
from mpi4py import MPI
import logging


class VogelsSprekeler:

    """VogelsSprekeler 2016"""

    def __init__(self):
        """Initialise variables."""
        self.comm = MPI.COMM_WORLD
        self.step = False
        # default resolution in nest is 0.1ms. Using the same value
        # http://www.nest-simulator.org/scheduling-and-simulation-flow/
        self.dt = 0.1
        # time to stabilise network after pattern storage etc.
        self.stabilisation_time = 12000.  # seconds
        self.recording_interval = 500.  # seconds

        # populations
        self.populations = {'E': 8000, 'I': 2000, 'STIM': 1000, 'Poisson': 1}
        # pattern percent of E neurons
        self.pattern_percent = .1
        # recall percent of pattern
        self.recall_percent = .25

        self.populations['P'] = self.pattern_percent * self.populations['E']
        self.populations['R'] = self.recall_percent * self.populations['P']

        # location bits
        self.colsE = 80
        self.colsI = 40
        self.neuronal_distE = 150  # micro metres
        self.neuronal_distI = 300  # micro metres
        self.location_sd = 15  # micro metres
        self.location_tree = None

        # time recall stimulus is enabled for
        self.recall_duration = 1000.  # ms

        self.rank = nest.Rank()

        self.patterns = []
        self.recall_neurons = []
        self.sdP = []
        self.sdB = []
        self.pattern_spike_count_file_names = []
        self.pattern_spike_count_files = []
        self.pattern_count = 0

        # Supplementary material documents how this affects the network
        self.wbar = 3.0
        self.weightEE = self.wbar
        self.weightII = self.wbar * -10.
        self.weightEI = self.wbar
        self.weightPatternEE = self.wbar * 5.
        self.weightExt = 50.

        # used to track how many comma separated values each line will have
        # when I store synaptic conductances.
        # Required in post processing, so that I know what the size of my
        # dataframe should be. Pandas cannot figure this out on its own. See
        # postprocessing scripts for more information.
        self.num_synapses_EE = 0
        self.num_synapses_EI = 0
        self.num_synapses_II = 0
        self.num_synapses_IE = 0
        random.seed(42)

    def __setup_neurons(self):
        """Setup properties of neurons."""
        # see the aif source for symbol definitions
        self.neuronDict = {'V_m': -60.,
                           't_ref': 5.0, 'V_reset': -60.,
                           'V_th': -50., 'C_m': 200.,
                           'E_L': -60., 'g_L': 10.,
                           'E_ex': 0., 'E_in': -80.,
                           'tau_syn_ex': 5., 'tau_syn_in': 10.,
                           }
        # Set up TIF neurons
        # Setting up two models because then it makes it easier for me to get
        # them when I need to set up patterns
        nest.CopyModel("iaf_cond_exp", "tif_neuronE")
        nest.SetDefaults("tif_neuronE", self.neuronDict)
        nest.CopyModel("iaf_cond_exp", "tif_neuronI")
        nest.SetDefaults("tif_neuronI", self.neuronDict)

        # external stimulus
        self.poissonExtDict = {'rate': 10., 'origin': 0., 'start': 0.}

    def __create_neurons(self):
        """Create our neurons."""
        self.neuronsE = nest.Create('tif_neuronE', self.populations['E'])
        self.neuronsI = nest.Create('tif_neuronI', self.populations['I'])

        # Generate a grid and construct a cKDTree
        locations = []
        if self.rank == 0:
            loc_file = open("00-neuron-locations-E.txt", 'w')
        for neuron in self.neuronsE:
            row = int((neuron - self.neuronsE[0])/self.colsE)
            y = random.gauss(row * self.neuronal_distE, self.location_sd)
            col = ((neuron - self.neuronsE[0]) % self.colsE)
            x = random.gauss(col * self.neuronal_distE, self.location_sd)
            locations.append([x, y])
            if self.rank == 0:
                print("{}\t{}\t{}\t{}\t{}".format(neuron, col, row, x, y),
                      file=loc_file)
        if self.rank == 0:
            loc_file.close()

        # I neurons have an intiail offset to distribute them evenly between E
        # neurons
        if self.rank == 0:
            loc_file = open("00-neuron-locations-I.txt", 'w')
        for neuron in self.neuronsI:
            row = int((neuron - self.neuronsI[0])/self.colsI)
            y = self.neuronal_distI/4 + random.gauss(
                row * self.neuronal_distI, self.location_sd)
            col = ((neuron - self.neuronsI[0]) % self.colsI)
            x = self.neuronal_distI/4 + random.gauss(
                col * self.neuronal_distI, self.location_sd)
            locations.append([x, y])
            if self.rank == 0:
                print("{}\t{}\t{}\t{}\t{}".format(neuron, col, row, x, y),
                      file=loc_file)
        if self.rank == 0:
            loc_file.close()
        self.location_tree = cKDTree(locations)

        self.poissonExt = nest.Create('poisson_generator',
                                      self.populations['Poisson'],
                                      params=self.poissonExtDict)

    def __setup_initial_connection_params(self):
        """Setup connections."""
        # Global sparsity
        self.sparsity = 0.02
        self.sparsityStim = 0.05
        # Other connection numbers
        self.connectionNumberStim = int((self.populations['STIM'] *
                                         self.populations['R']) *
                                        self.sparsityStim)
        # each neuron gets a single input
        self.connDictExt = {'rule': 'fixed_indegree',
                            'indegree': 1}
        # recall stimulus
        self.connDictStim = {'rule': 'fixed_total_number',
                             'N': self.connectionNumberStim}

        # Documentation says things are normalised in the iaf neuron so that
        # weight of 1 translates to 1nS
        nest.CopyModel('static_synapse', 'static_synapse_ex')
        nest.CopyModel('static_synapse', 'static_synapse_in')
        nest.CopyModel('vogels_sprekeler_synapse', 'stdp_synapse_in')
        self.synDictEE = {'model': 'static_synapse_ex',
                          'weight': self.weightEE}
        self.synDictEI = {'model': 'static_synapse_ex',
                          'weight': self.weightEI}
        self.synDictII = {'model': 'static_synapse_in',
                          'weight': self.weightII}
        self.synDictIE = {'model': 'stdp_synapse_in',
                          'weight': -0.0000001, 'Wmax': -30000.,
                          'alpha': .12, 'eta': 0.01,
                          'tau': 20.}

    def __create_initial_connections(self):
        """Initially connect various neuron sets."""
        nest.Connect(self.poissonExt, self.neuronsE,
                     conn_spec=self.connDictExt,
                     syn_spec={'model': 'static_synapse',
                               'weight': self.weightExt})
        nest.Connect(self.poissonExt, self.neuronsI,
                     conn_spec=self.connDictExt,
                     syn_spec={'model': 'static_synapse',
                               'weight': self.weightExt})

        conndict = {'rule': 'pairwise_bernoulli',
                    'p': self.sparsity}
        logging.debug("Setting up EE connections.")
        nest.Connect(self.neuronsE, self.neuronsE,
                     syn_spec=self.synDictEE,
                     conn_spec=conndict)
        logging.debug("EE connections set up.")

        logging.debug("Setting up EI connections.")
        nest.Connect(self.neuronsE, self.neuronsI,
                     syn_spec=self.synDictEI,
                     conn_spec=conndict)
        logging.debug("EI connections set up.")

        logging.debug("Setting up II connections.")
        nest.Connect(self.neuronsI, self.neuronsI,
                     syn_spec=self.synDictII,
                     conn_spec=conndict)
        logging.debug("II connections set up.")

        logging.debug("Setting up IE connections.")
        nest.Connect(self.neuronsI, self.neuronsE,
                     syn_spec=self.synDictIE,
                     conn_spec=conndict)
        logging.debug("IE connections set up.")

    def __setup_detectors(self):
        """Setup spike detectors."""
        # E neurons
        self.sd_paramsE = {
            'to_file': True,
            'label': 'spikes-E'
        }
        # I neurons
        self.sd_paramsI = {
            'to_file': True,
            'label': 'spikes-I'
        }
        # pattern neurons
        self.sd_paramsP = {
            'to_file': True,
            'label': 'spikes-pattern'
        }
        # background neurons
        self.sd_paramsB = {
            'to_file': True,
            'label': 'spikes-background'
        }

        self.sdE = nest.Create('spike_detector',
                               params=self.sd_paramsE)
        self.sdI = nest.Create('spike_detector',
                               params=self.sd_paramsI)

        nest.Connect(self.neuronsE, self.sdE)
        nest.Connect(self.neuronsI, self.sdI)

    def __setup_files(self):
        """Set up the filenames and handles."""
        # Get the number of spikes in these files and then post-process them to
        # get the firing rate and so on

        self.synaptic_p_weights_file_name_EE = (
            "00-synaptic-weights-EE-" + str(self.rank) + ".txt")
        self.weights_file_handle_EE = open(
            self.synaptic_p_weights_file_name_EE, 'w')
        print("{},{}".format(
            "time(ms)", "EE(nS)"),
            file=self.weights_file_handle_EE)

        self.synaptic_p_weights_file_name_EI = (
            "00-synaptic-weights-EI-" + str(self.rank) + ".txt")
        self.weights_file_handle_EI = open(
            self.synaptic_p_weights_file_name_EI, 'w')
        print("{},{}".format(
            "time(ms)", "EI(nS)"),
            file=self.weights_file_handle_EI)

        self.synaptic_p_weights_file_name_II = (
            "00-synaptic-weights-II-" + str(self.rank) + ".txt")
        self.weights_file_handle_II = open(
            self.synaptic_p_weights_file_name_II, 'w')
        print("{},{}".format(
            "time(ms)", "II(nS)"),
            file=self.weights_file_handle_II)

        self.synaptic_p_weights_file_name_IE = (
            "00-synaptic-weights-IE-" + str(self.rank) + ".txt")
        self.weights_file_handle_IE = open(
            self.synaptic_p_weights_file_name_IE, 'w')
        print("{},{}".format(
            "time(ms)", "IE(nS)"),
            file=self.weights_file_handle_IE)

    def prerun_setup(self, step=False,
                     stabilisation_time=None,
                     recording_interval=None):
        """Pre reun configuration."""
        # Cannot be changed mid simulation
        if step:
            self.step = step
        self.update_time_windows(stabilisation_time, recording_interval)
        self.__setup_simulation()
        self.comm.Barrier()

    def print_simulation_parameters(self):
        """Print the parameters of the simulation to a file."""
        if self.rank == 0:
            with open("00-simulation_params.txt", 'w') as pfile:
                print("{}: {} milli seconds".format("dt", self.dt),
                      file=pfile)
                print("{}: {} seconds".format("stabilisation_time",
                                              self.stabilisation_time),
                      file=pfile)
                print("{}: {} seconds".format("recording_interval",
                                              self.recording_interval),
                      file=pfile)
                print("{}: {}".format("num_E", self.populations['E']),
                      file=pfile)
                print("{}: {}".format("num_I", self.populations['I']),
                      file=pfile)
                print("{}: {}".format("num_P", self.populations['P']),
                      file=pfile)
                print("{}: {}".format("num_R", self.populations['R']),
                      file=pfile)
                print("{}: {}".format("pattern_percent", self.pattern_percent),
                      file=pfile)
                print("{}: {}".format("recall_percent", self.recall_percent),
                      file=pfile)
                print("{}: {}".format("num_colsE", self.colsE),
                      file=pfile)
                print("{}: {}".format("num_colsI", self.colsI),
                      file=pfile)
                print("{}: {}".format("dist_neuronsE", self.neuronal_distE),
                      file=pfile)
                print("{}: {}".format("dist_neuronsI", self.neuronal_distI),
                      file=pfile)
                print("{}: {} micro metres".format(
                    "grid_size_E",
                    self.location_tree.data[len(self.neuronsE) - 1]),
                    file=pfile)
                print("{}: {} micro metres".format("sd_dist", self.location_sd),
                      file=pfile)
                print("{}: {} seconds".format("recording_interval",
                                              self.recording_interval),
                      file=pfile)
                print("{}: {} nS".format("wbar", self.wbar),
                      file=pfile)
                print("{}: {} nS".format("weightEE", self.weightEE),
                      file=pfile)
                print("{}: {} ns".format("weightPatternEE",
                                         self.weightPatternEE),
                      file=pfile)
                print("{}: {} nS".format("weightEI", self.weightEI),
                      file=pfile)
                print("{}: {} nS".format("weightII", self.weightII),
                      file=pfile)
                print("{}: {} nS".format("weightExt", self.weightExt),
                      file=pfile)
                print("{}: {}".format("sparsity", self.sparsity),
                      file=pfile)

    def update_time_windows(self,
                            stabilisation_time=None,
                            recording_interval=None):
        """Set up stabilisation time."""
        if stabilisation_time:
            self.stabilisation_time = stabilisation_time
        if recording_interval:
            self.recording_interval = recording_interval

    def __setup_simulation(self):
        """Setup the common simulation things."""
        # Nest stuff
        nest.ResetKernel()
        # http://www.nest-simulator.org/sli/setverbosity/
        nest.set_verbosity('M_INFO')
        nest.SetKernelStatus(
            {
                'resolution': self.dt,
                'local_num_threads': 1,
                'overwrite_files': True
            }
        )
        self.__setup_neurons()
        self.__create_neurons()
        self.__setup_detectors()
        self.__setup_initial_connection_params()
        self.__create_initial_connections()
        self.__setup_files()

        self.dump_data()

    def stabilise(self):
        """Stabilise network."""
        logging.info("SIMULATION: STABILISING for {} seconds".format(
            self.stabilisation_time))
        update_steps = numpy.arange(0, self.stabilisation_time,
                                    self.recording_interval)
        for j, k in enumerate(update_steps):
            self.run_simulation(self.recording_interval)

    def run_simulation(self, simtime=2000):
        """Run the simulation."""
        if self.step:
            sim_steps = numpy.arange(0, simtime)
            for i, step in enumerate(sim_steps):
                nest.Simulate(1000)
        else:
            nest.Simulate(simtime*1000)
            self.dump_data()
            current_simtime = (
                str(nest.GetKernelStatus()['time']) + "msec")
            logging.info("Simulation time: " "{}".format(current_simtime))

    def __get_neurons_from_region(self, num_neurons, first_point, last_point):
        """Get neurons in the centre of the grid."""
        mid_point = [(x + y)/2 for x, y in zip(last_point, first_point)]
        neurons = self.location_tree.query(
            mid_point, k=num_neurons)[1]
        logging.info("Got {}/{} neurons".format(len(neurons), num_neurons))
        return neurons

    def __strengthen_pattern_connections(self, pattern_neurons):
        """Strengthen connections that make up the pattern."""
        connections = nest.GetConnections(source=pattern_neurons,
                                          target=pattern_neurons)
        nest.SetStatus(connections, {"weight": self.weightPatternEE})
        logging.debug("ANKUR>> Number of connections strengthened: "
                      "{}".format(len(connections)))

    def __track_pattern(self, pattern_neurons):
        """Track the pattern."""
        logging.debug("Tracking this pattern")
        self.patterns.append(pattern_neurons)
        background_neurons = list(
            set(self.neuronsE) - set(pattern_neurons))
        # print to file
        # NOTE: since these are E neurons, the indices match in the location
        # tree. No need to subtract self.neuronsE[0] to get the right indices at
        # the moment. But keep in mind in case something changes in the future.
        if self.rank == 0:
            file_name = "00-pattern-neurons-{}.txt".format(
                self.pattern_count)
            with open(file_name, 'w') as file_handle:
                for neuron in pattern_neurons:
                    print("{}\t{}\t{}".format(
                        neuron,
                        self.location_tree.data[neuron - 1][0],
                        self.location_tree.data[neuron - 1][1]),
                        file=file_handle)

            # background neurons
            file_name = "00-background-neurons-{}.txt".format(
                self.pattern_count)

            with open(file_name, 'w') as file_handle:
                for neuron in background_neurons:
                    print("{}\t{}\t{}".format(
                        neuron,
                        self.location_tree.data[neuron - 1][0],
                        self.location_tree.data[neuron - 1][1]),
                        file=file_handle)

        # set up spike detectors
        sd_params = self.sd_paramsP.copy()
        sd_params['label'] = (sd_params['label'] + "-{}".format(
            self.pattern_count))
        # pattern
        pattern_spike_detector = nest.Create(
            'spike_detector', params=sd_params)
        nest.Connect(pattern_neurons, pattern_spike_detector)
        # save the detector
        self.sdP.append(pattern_spike_detector)

        # background
        sd_params = self.sd_paramsB.copy()
        sd_params['label'] = (sd_params['label'] + "-{}".format(
            self.pattern_count))
        background_spike_detector = nest.Create(
            'spike_detector', params=sd_params)
        nest.Connect(background_neurons, background_spike_detector)
        # save the detector
        self.sdB.append(background_spike_detector)

    def store_pattern_off_centre(self, offset=[0., 0.], track=False):
        """Store a pattern in the centre of network."""
        logging.debug(
            "SIMULATION: Storing pattern {} in centre of network".format(
                self.pattern_count + 1))
        # first E neuron
        first_point = numpy.array(self.location_tree.data[0])
        # last E neuron
        # I neurons are spread among the E neurons and therefore do not make it
        # to the extremeties
        last_point = numpy.array(
            self.location_tree.data[len(self.neuronsE) - 1])
        centre_point = numpy.array(offset) + (first_point + last_point)/2
        self.store_pattern_with_centre(centre_point,
                                       (1.25 * self.populations['P']),
                                       track=True)

    def store_pattern_with_centre(self, centre_point, num_neurons, track=False):
        """Store a pattern by specifying area extent."""
        logging.debug(
            "SIMULATION: Storing pattern {} centred at:".format(
                self.pattern_count + 1, centre_point))
        self.pattern_count += 1
        # get 1000 neurons - 800 will be E and 200 will be I
        # we only need the 800 I neurons
        all_neurons = self.location_tree.query(
            centre_point, k=num_neurons)[1]
        pattern_neurons = list(set(all_neurons).intersection(
            set(self.neuronsE)))
        self.__strengthen_pattern_connections(pattern_neurons)
        if track:
            self.__track_pattern(pattern_neurons)
        logging.debug(
            "Number of patterns stored: {}".format(
                self.pattern_count))

    def setup_pattern_for_recall(self, pattern_number):
        """
        Set up a pattern for recall.

        Creates a new poisson generator and connects it to a recall subset of
        this pattern - the poisson stimulus will run for the set recall_duration
        from the invocation of this method.
        """
        # set up external stimulus
        pattern_neurons = self.patterns[pattern_number - 1]
        recall_neurons = []
        num_recall_neurons = int(math.ceil(len(pattern_neurons) *
                                           self.recall_percent))
        recall_neurons = pattern_neurons[-num_recall_neurons:]

        stim_time = nest.GetKernelStatus()['time']
        neuronDictStim = {'rate': 200.,
                          'origin': stim_time,
                          'start': 0., 'stop': self.recall_duration}
        stim = nest.Create('poisson_generator', 1,
                           neuronDictStim)

        nest.Connect(stim, recall_neurons,
                     conn_spec=self.connDictStim)

        logging.debug("ANKUR>> Number of recall neurons for pattern"
                      "{}: {}".format(pattern_number, len(recall_neurons)))
        self.recall_neurons.append(recall_neurons)

    def recall_last_pattern(self, time):
        """
        Only setup the last pattern.

        An extra helper method, since we'll be doing this most.
        """
        logging.info("SIMULATION: RECALLING LAST PATTERN")
        self.recall_pattern(time, self.pattern_count)

    def recall_pattern(self, time, pattern_number):
        """Recall a pattern."""
        self.setup_pattern_for_recall(pattern_number)
        self.run_simulation(time)

    def __dump_neuron_set(self, file_name, neurons):
        """Dump a set of neuronIDs to a text file."""
        with open(file_name, 'w') as file_handle:
            for neuron in neurons:
                print(neuron, file=file_handle)

    def __dump_synaptic_weights(self):
        """Dump synaptic weights."""
        current_simtime = (str(nest.GetKernelStatus()['time']))

        conns = nest.GetConnections(target=self.neuronsE,
                                    source=self.neuronsI)
        weightsIE = nest.GetStatus(conns, "weight")
        print("{}, {}".format(
            current_simtime,
            str(weightsIE).strip('[]').strip('()')),
            file=self.weights_file_handle_IE)
        if len(weightsIE) > self.num_synapses_IE:
            self.num_synapses_IE = len(weightsIE)

        conns = nest.GetConnections(target=self.neuronsI,
                                    source=self.neuronsI)
        weightsII = nest.GetStatus(conns, "weight")
        print("{}, {}".format(
            current_simtime,
            str(weightsII).strip('[]').strip('()')),
            file=self.weights_file_handle_II)
        if len(weightsII) > self.num_synapses_II:
            self.num_synapses_II = len(weightsII)

        conns = nest.GetConnections(target=self.neuronsI,
                                    source=self.neuronsE)
        weightsEI = nest.GetStatus(conns, "weight")
        print("{}, {}".format(
            current_simtime,
            str(weightsEI).strip('[]').strip('()')),
            file=self.weights_file_handle_EI)
        if len(weightsEI) > self.num_synapses_EI:
            self.num_synapses_EI = len(weightsEI)

        conns = nest.GetConnections(target=self.neuronsE,
                                    source=self.neuronsE)
        weightsEE = nest.GetStatus(conns, "weight")
        print("{}, {}".format(
            current_simtime,
            str(weightsEE).strip('[]').strip('()')),
            file=self.weights_file_handle_EE)
        if len(weightsEE) > self.num_synapses_EE:
            self.num_synapses_EE = len(weightsEE)

    def dump_data(self):
        """Master datadump function."""
        logging.info("Rank {}: Printing data to files".format(self.rank))
        self.__dump_synaptic_weights()

    def close_files(self):
        """Close all files when the simulation is finished."""
        logging.info("Rank {}: Closing open files".format(self.rank))
        # Comma printed so that pandas can read it as a dataframe point
        print("{},".format(self.num_synapses_EE),
              file=self.weights_file_handle_EE)
        self.weights_file_handle_EE.close()

        print("{},".format(self.num_synapses_EI),
              file=self.weights_file_handle_EI)
        self.weights_file_handle_EI.close()
        print("{},".format(self.num_synapses_II),
              file=self.weights_file_handle_II)
        self.weights_file_handle_II.close()

        print("{},".format(self.num_synapses_IE),
              file=self.weights_file_handle_IE)
        self.weights_file_handle_IE.close()


if __name__ == "__main__":
    # Set up logging configuration
    logging.basicConfig(
        format='%(funcName)s: %(lineno)d: %(levelname)s: %(message)s',
        level=logging.DEBUG)

    step = False
    simulation = VogelsSprekeler()

    # simulation setup
    # set up neurons, connections, spike detectors, files
    simulation.prerun_setup(
        stabilisation_time=2000.,
        recording_interval=250.)
    # print em up
    simulation.print_simulation_parameters()
    logging.info("Rank {}: SIMULATION SETUP".format(simulation.rank))

    # initial setup
    logging.info("Rank {}: SIMULATION STARTED".format(simulation.rank))
    simulation.stabilise()

    simulation.store_pattern_off_centre([0., 0.], True)
    simulation.store_pattern_with_centre([10000, 2000], 600, True)
    simulation.store_pattern_off_centre([0., 2000.0], True)

    # stabilise network after storing patterns
    simulation.stabilise()

    # recall stored and tracked pattern
    simulation.recall_pattern(50, 1)
    simulation.recall_pattern(50, 2)
    simulation.recall_pattern(50, 3)

    simulation.close_files()
    nest.Cleanup()
    logging.info("Rank {}: SIMULATION FINISHED SUCCESSFULLY".format(
        simulation.rank))
