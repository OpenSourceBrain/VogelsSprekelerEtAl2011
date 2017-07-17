/* 
 * Copyright 2014 Ankur Sinha
 *
 * adapted from sim_isp_orig which is part of the auryn source code
 *
 */
#include <iomanip>
#include <stdlib.h>
#include <string>
#include <iterator>

#include <boost/program_options.hpp>

#include "auryn/auryn_global.h"
#include "auryn/auryn_definitions.h"
#include "auryn/System.h"
#include "auryn/Logger.h"

#include "auryn/NeuronGroup.h"
#include "auryn/TIFGroup.h"
#include "auryn/SparseConnection.h"
#include "auryn/SymmetricSTDPConnection.h"
#include "auryn/StateMonitor.h"
#include "auryn/SpikeMonitor.h"
#include "auryn/RateChecker.h"
#include "auryn/RateMonitor.h"
#include "auryn/PatternMonitor.h"
#include "auryn/PoissonGroup.h"
#include "auryn/PopulationRateMonitor.h"
#include "auryn/WeightSumMonitor.h"

/*  Number of excitatory neurons */
#define NE 8000
/*  Number of inhibitory neurons */
#define NI 2000
/*  Number of poisson neurons for stimulation */
#define NP 1000
/*  Unused - not sure what it's for */
#define NSTIM 20

using namespace std;
namespace po = boost::program_options;

int main(int ac, char* av[]) 
{
    /* ***** synaptic conductances **** */
    /*  Basic weight unit - 3nS apparently */
    double w = 0.3 ;
    /*  w_ext_e */
    double w_ext = 25. * w ;
    /*  Multiplier - scaling factor for inhibitory weights */
    double gamma = 10. ;
    /*  w_ii */
    double w_ii = gamma * w;
    /*  Maximum allowed weight for STDP connections */
    double wmax = 10. * w_ii;
    /* inhibitory weight multiplier  */
    double winh = -1.;
    /*  weight e->i multiplier */
    double k_w_ei = 1.;
    /*  w_ee */
    double w_ee = w;
    /*  w_ei */
    double w_ei = w;

    /*  Connection probability */
    double sparseness = 0.02 ;

    /* learning rate */
    double eta = 1e-4 ;
    /*  target rate for STDP connections - kappafudge in code, alpha in paper */
    double kappa = 3;
    /*  Decay constant of pre and post synaptic trace for STDP connections */
    double tau_stdp = 20e-3 ;
    /*  Enable stdp */
    bool stdp_active = true;
    /*  Enable stimulus from poisson neurons */
    bool poisson_stim = true;
    /*  current multiplier */
    double chi = 10.;
    /* background current constant value */
    double bg_current = 2e-2;

    /*  Mean firing rate of all poisson neurons in the group */
    double poisson_rate = 100.;
    /*  Sparsensess for afferent connections ext -> E */
    double sparseness_afferents = 0.05;

    /*  Unused - probably to control verbosity of logs */
    bool quiet = false;
    /*  Total simulation time */
    double simtime = 3600. ;
    /*  Since the EI connections are static, it doesn't matter how long the
     *  stimlus is enabled for - it'll have the same effect on the E and
     *  therefore the I neurons 
     *
     *  This variable only comes in use if you want to switch off the poisson
     *  input after a while, which isn't supposed to be done in the simulation
     */
    double stage1_time = 60;
    double stage2_time = 600;
    double stage3_time = 5;
    double stage4_time = 600;
    double stage5_time = 2;
    double stage6_time = 300;

    /*  sampling interval for rate monitor */
    double rate_mon_sampling_interval = 0.05;


    /*  Which neuron we're recording from when not the entire population */
    NeuronID record_neuron = 30;
    // handle command line options

    string infilename = "";
    string outputfile = "out";
    string stimfile = "";
    vector<string> patfilenames;
    string strbuf ;
    string msgbuf;
    ostringstream converter;

    int errcode = 0;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("quiet", "quiet mode")
            ("load", po::value<string>(), "input weight matrix")
            ("out", po::value<string>(), "output filename")
            ("stimfile", po::value<string>(), "stimulus file")
            ("pats", po::value<vector<string> >()-> multitoken(), "pattern file(s)")
            ("eta", po::value<double>(), "learning rate")
            ("kappa", po::value<double>(), "target rate")
            ("simtime", po::value<double>(), "simulation time")
            ("active", po::value<bool>(), "toggle learning")
            ("poisson", po::value<bool>(), "toggle poisson stimulus")
            ("winh", po::value<double>(), "inhibitory weight multiplier")
            ("k_w_ei", po::value<double>(), "ei weight multiplier")
            ("chi", po::value<double>(), "chi current multiplier")
            ;

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        if (vm.count("quiet")) {
            quiet = true;
        } 

        if (vm.count("load")) {
            cout << "input weight matrix " 
                << vm["load"].as<string>() << ".\n";
            infilename = vm["load"].as<string>();
        } 

        if (vm.count("pats")) {
            patfilenames = vm["pats"].as<vector<string> >();
            for (vector<string>::iterator it = patfilenames.begin(); it != patfilenames.end(); ++it){
                cout << "patfile filename(s) " 
                    << *it << ".\n";
            }
        } 

        if (vm.count("out")) {
            cout << "output filename " 
                << vm["out"].as<string>() << ".\n";
            outputfile = vm["out"].as<string>();
        } 

        if (vm.count("stimfile")) {
            cout << "stimfile filename " 
                << vm["stimfile"].as<string>() << ".\n";
            stimfile = vm["stimfile"].as<string>();
        } 

        if (vm.count("eta")) {
            cout << "eta set to " 
                << vm["eta"].as<double>() << ".\n";
            eta = vm["eta"].as<double>();
        } 

        if (vm.count("kappa")) {
            cout << "kappa set to " 
                << vm["kappa"].as<double>() << ".\n";
            kappa = vm["kappa"].as<double>();
        } 

        if (vm.count("simtime")) {
            cout << "simtime set to " 
                << vm["simtime"].as<double>() << ".\n";
            simtime = vm["simtime"].as<double>();
        } 

        if (vm.count("active")) {
            cout << "stdp active : " 
                << vm["active"].as<bool>() << ".\n";
            stdp_active = vm["active"].as<bool>();
        } 

        if (vm.count("poisson")) {
            cout << "poisson active : " 
                << vm["poisson"].as<bool>() << ".\n";
            poisson_stim = vm["poisson"].as<bool>();
        } 


        if (vm.count("winh")) {
            cout << "inhib weight multiplier : " 
                << vm["winh"].as<double>() << ".\n";
            winh = vm["winh"].as<double>();
        } 

        if (vm.count("k_w_ei")) {
            cout << "ei weight multiplier : " 
                << vm["k_w_ei"].as<double>() << ".\n";
            k_w_ei = vm["k_w_ei"].as<double>();
        } 

        if (vm.count("chi")) {
            cout << "chi multiplier : " 
                << vm["chi"].as<double>() << ".\n";
            chi = vm["chi"].as<double>();
        } 

    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
    }

    // BEGIN Global definitions
    mpi::environment env(ac, av);
    mpi::communicator world;
    communicator = &world;

    /*  Set up log file */
    stringstream oss;
    oss << outputfile << "." << world.rank();
    outputfile = oss.str();
    oss << ".log";
    string logfile = oss.str();
    logger = new Logger(logfile,world.rank());

    /*  Set up the system */
    sys = new System(&world);
    // END Global definitions



    /*  Sets up the inhibitory and excitatory neuron groups */
    logger->msg("Setting up neuron groups ...",PROGRESS,true);
    TIFGroup * neurons_e = new TIFGroup(NE);
    TIFGroup * neurons_i = new TIFGroup(NI);
    PoissonGroup *poisson = new PoissonGroup(NP, poisson_rate);

    /*  Initialise to random membrane potentials */
/*     neurons_e->random_mem(-60e-3,5e-3);
 *     neurons_i->random_mem(-60e-3,5e-3);
 */


    /*  Set up synaptic connections */
    logger->msg("Setting up connections ...",PROGRESS,true);
    /*  Static: From E->I */
    SparseConnection * con_ei = new SparseConnection(neurons_e,neurons_i,k_w_ei*w_ei,sparseness,GLUT);

    /*  Static: From I->I - recurrent */
    SparseConnection * con_ii = new SparseConnection(neurons_i,neurons_i,w_ii,sparseness,GABA);

    /*  Static: From external stimulus (poisson neurons) -> excitatory */
    SparseConnection * con_exte = new SparseConnection(poisson,neurons_e,0,sparseness_afferents,GLUT);

    /*  Static: From E->E - recurrent */
    SparseConnection * con_ee = new SparseConnection(neurons_e,neurons_e,k_w_ei*w_ee,sparseness,GLUT);

    /*  STDP: From I->E */
    SymmetricSTDPConnection * con_ie = new SymmetricSTDPConnection(neurons_i,neurons_e,
            gamma*w,sparseness,
            gamma*eta,kappa,tau_stdp,wmax,
            GABA);


    logger->msg("STDP deactivated to start with ...",PROGRESS,true);
    con_ie->stdp_active = false;



    if (!infilename.empty()) {
        sys->load_network_state(infilename);
    }

    /*  winh is -1 so this loop is ignored */
    if (winh>=0)
        con_ie->set_all(winh);

    logger->msg("Setting up monitors ...",PROGRESS,true);
    strbuf = outputfile;
    strbuf += "_e.ras";
    SpikeMonitor * smon_e = new SpikeMonitor( neurons_e , strbuf.c_str() );

    strbuf = outputfile;
    strbuf += "_i.ras";
    SpikeMonitor * smon_i = new SpikeMonitor( neurons_i, strbuf.c_str() );


    /* 	strbuf = outputfile;
     * 	strbuf += ".volt";
     * 	StateMonitor * vmon = new StateMonitor( neurons_e, record_neuron, "mem", strbuf.c_str() );
     * 
     * 	strbuf = outputfile;
     * 	strbuf += ".ampa";
     * 	StateMonitor * amon = new StateMonitor( neurons_e, record_neuron, "g_ampa", strbuf.c_str() );
     * 
     * 	strbuf = outputfile;
     * 	strbuf += ".gaba";
     * 	StateMonitor * gmon = new StateMonitor( neurons_e, record_neuron, "g_gaba", strbuf.c_str() );
     * 
     * 	strbuf = outputfile;
     * 	strbuf += "_e.prate";
     * 	PopulationRateMonitor * pmon_e = new PopulationRateMonitor( neurons_e , strbuf.c_str() );
     * 
     * 	strbuf = outputfile;
     * 	strbuf += "_i.prate";
     * 	PopulationRateMonitor * pmon_i = new PopulationRateMonitor( neurons_i , strbuf.c_str() );
     */

    /*  Firing rate monitors */
    strbuf = outputfile;
    strbuf += "_e.rate";
    RateMonitor * rmon_e = new RateMonitor( neurons_e , strbuf.c_str(), rate_mon_sampling_interval );

    strbuf = outputfile;
    strbuf += "_i.rate";
    RateMonitor * rmon_i = new RateMonitor( neurons_i , strbuf.c_str(), rate_mon_sampling_interval);

    /*  Weight monitors */
    strbuf = outputfile;
    strbuf += "_ie_stdp.weightinfo";
    WeightSumMonitor * weightmon_ie = new WeightSumMonitor (con_ie, strbuf.c_str(), rate_mon_sampling_interval );

    strbuf = outputfile;
    strbuf += "_ii_static.weightinfo";
    WeightSumMonitor * weightmon_ii = new WeightSumMonitor (con_ii, strbuf.c_str(), rate_mon_sampling_interval );

    strbuf = outputfile;
    strbuf += "_ee_static.weightinfo";
    WeightSumMonitor * weightmon_ee = new WeightSumMonitor (con_ee, strbuf.c_str(), rate_mon_sampling_interval );

    strbuf = outputfile;
    strbuf += "_ei_static.weightinfo";
    WeightSumMonitor * weightmon_ei = new WeightSumMonitor (con_ei, strbuf.c_str(), rate_mon_sampling_interval );

    strbuf = outputfile;
    strbuf += "_exte_static.weightinfo";
    WeightSumMonitor * weightmon_exte = new WeightSumMonitor (con_exte, strbuf.c_str(), rate_mon_sampling_interval );

    /*     if (!patfilename.empty()) {
     *         logger->msg("Enabling pattern monitors ...",PROGRESS,true);
     *         strbuf = outputfile;
     *         strbuf += ".e.patact";
     *         PatternMonitor * patmon_e = new PatternMonitor(neurons_e, strbuf.c_str(),patfilename.c_str());
     * 
     *         strbuf = outputfile;
     *         strbuf += ".i.patact";
     *         PatternMonitor * patmon_i = new PatternMonitor(neurons_i, strbuf.c_str(),patfilename.c_str());
     * 
     *         strbuf = outputfile;
     *         strbuf += ".stim.pact";
     *         PatternMonitor * patmon_poisson = new PatternMonitor( poisson, strbuf.c_str(),patfilename.c_str());
     *     }
     */

    RateChecker * chk = new RateChecker( neurons_e , 0.1 , 1000. , 100e-3);

    /*  Fixed input currents to all inputs */
    for ( int j = 0; j < NE ; j++ ) {
        neurons_e->set_bg_current(j,bg_current);

    }

    for ( int j = 0; j < NI ; j++ ) {
        neurons_i->set_bg_current(j,bg_current);

    }

    if (stage1_time > 0)
    {
        converter << stage1_time;
        msgbuf = "Stage 1a - Stimulating for " + converter.str() + " seconds ...";
        logger->msg(msgbuf,PROGRESS,true);
        sys->run(stage1_time);
    }

    // stimulus
    logger->msg("Simulating ... with non-poisson stimulus",PROGRESS,true);
    for ( int j = 0; j < NE ; j++ ) {
        neurons_e->set_bg_current(j,chi*bg_current);
    }
    for ( int j = 0; j < NI ; j++ ) {
        neurons_i->set_bg_current(j,chi*bg_current);
    }

    /* 	logger->msg("Stage 1b - Switched off poisson stimulus",PROGRESS,true);
     *     NeuronID counter = 0;
     *     while (counter < NE)
     *     {
     *         for (int i = 0 ; i < NP ; ++i)
     *             con_exte->set(i,counter,0);
     * 
     *         counter++;
     *     }
     */

    if (stage2_time > 0)
    {
        logger->msg("Switched on stdp and set all stdp synapses to 0",PROGRESS,true);
        con_ie->stdp_active = true;
        con_ie->set_all(0);

        converter.str("");
        converter.clear();
        converter << stage2_time;
        msgbuf = "Stage 2 - Stimulating for " + converter.str() + " seconds ...";
        logger->msg(msgbuf,PROGRESS,true);
        sys->run(stage2_time);
    }

    if(!patfilenames.empty() && stage3_time > 0)
    {
        /* Stage 3 - strengthening excitatory synapses of the two memory patterns
         * Doing it in three blocks */
        logger->msg("Stage 3 - strengthening synapses of loaded memory patterns ...",PROGRESS,true);
        for (vector<string>::iterator it = patfilenames.begin(); it != patfilenames.end(); ++it)
        {
            con_ee->load_patterns(*it,w_ee*15., false, false);
        }
        sys->run(stage3_time);
    }

    if (stage4_time > 0)
    {
        converter.str("");
        converter.clear();
        converter << stage4_time;
        msgbuf = "Stage 4 - stabilizing for " + converter.str() + " seconds ...";
        logger->msg(msgbuf,PROGRESS,true);
        sys->run(stage4_time);
    }


    converter.str("");
    converter.clear();
    converter << stage5_time;
    msgbuf = "Stage 5 - poisson stimulus to a quarter of the lower pattern for " + converter.str() + " seconds ...";
    logger->msg(msgbuf,PROGRESS,true);
    /*  
     *  if we have a stimulus file
     *
     *  Go over all excitatory neurons, and if the stimulus file says 1:
     *      if poisson stimulus is enabled, set all the synapses from poisson
     *      neurons to the excitatory neurons as w_ext
     *      otherwise, set the current of the E neurons to bg current
     **/
    if (!stimfile.empty()) 
    {
        bool run_while = true;
        logger->msg("Simulating ... with stimulus pattern",PROGRESS,true);
        string::size_type sz;
        ifstream fin(stimfile.c_str());
        if (!fin.is_open())
        {
            logger->msg("Something went wrong opening the file. Please abort immediately!", PROGRESS, true);
            run_while = false;
        }


        while (run_while) 
        {
            string line;
            NeuronID neuron;

            getline(fin, line);
            if (line.empty ())
                break;

            /*  stoi requires c++11  */
            neuron = stoi(line, &sz);
            if (poisson_stim==true) 
            {
                for (int i = 0 ; i < NP ; ++i)
                    con_exte->set(i,neuron,w_ext);
            } 
            else 
            {
                neurons_e->set_bg_current(neuron,chi*bg_current);
            }
            converter.str("");
            converter.clear();
            converter << neuron;
            msgbuf = "Modified neuron" + converter.str();
            logger->msg(msgbuf,PROGRESS,true);
        }
        fin.close();
        sys->run(stage5_time);

        /*  remove stimulus */

        /*  I know I can rewind. I'll do that later. */
        ifstream fin1(stimfile.c_str());
        if (!fin1.is_open())
        {
            logger->msg("Something went wrong opening the file. Please abort immediately!", PROGRESS, true);
            run_while = false;
        }
        while (run_while) 
        {
            string line;
            NeuronID neuron;

            getline(fin1, line);
            if (line.empty ())
                break;

            /*  stoi requires c++11  */
            neuron = stoi(line, &sz);
            if (poisson_stim==true) 
            {
                for (int i = 0 ; i < NP ; ++i)
                    con_exte->set(i,neuron,0);
            } 
            else 
            {
                neurons_e->set_bg_current(neuron,bg_current);
            }
            converter.str("");
            converter.clear();
            converter << neuron;
            msgbuf = "Modified neuron" + converter.str();
            logger->msg(msgbuf,PROGRESS,true);
        }
        fin1.close();
    }

    if (stage6_time > 0)
    {
        converter.str("");
        converter.clear();
        converter << stage6_time;
        msgbuf = "Stage 6 - checking recall for " + converter.str() + " seconds ...";
        logger->msg(msgbuf,PROGRESS,true);
        sys->run(stage6_time);
    }
    logger->msg("Saving network state ...",PROGRESS,true);
    sys->save_network_state(outputfile);

    logger->msg("Freeing ...",PROGRESS,true);
    delete sys;

    logger->msg("Exiting ...",PROGRESS,true);
    return errcode;
}
