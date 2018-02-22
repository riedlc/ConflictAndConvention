### Readme for simulation code in "Conflict and Convention in Dynamic Networks"
Paper by Foley, Forber, Smead, Riedl
Royal Society Interface (citation to come)

This README describes how to run code used in above paper.  The code simulates a hawk-## dove game with reinforcement learning on both strategies and adjacency links.  Agents ## thus learn what strategy to play and who to play against.  Agents interact in two     ## contexts: as visitor and as host.  The strategies associated with these contexts are  ## indepedent within a single agent.

#### This project includes this README, three python files that are used to set up simulation, inputs, a C++ simulation file, and “ESD_Seeds_All_Ordered.csv” which C++ reads a list of seeds from.

###########################################

#### Parameters of this model are the following:

payoffs, pop_list_in, init_strategy_str_list_in, net_discount_list_in, strat_discount_list_in, net_speed_list_in, strat_speed_list_in,net_tremble_list_in, strat_tremble_list_in, init_cond_hawk_p1_list_in, init_cond_hawk_p2_list_in

- Population [pop, integer > 1]: Number of agents (default 20)

- Payoffs [p1_payoffs,p2_payoffs, array w/ values in [0,1]: Interaction payoffs given by visitor and host strategy played in a single game. See table in Figure 1 of paper for constraints.  

- Initial Strategy [init_strategy_str, string]: initial strategy configuration, either “random” or “uniform” (default “uniform”)

- Network Discount [net_discount, float in [0,1]]: memory factor in link updating.  All current outgoing network weights are multiplied by net_discount just before network learning occurs. As it is a multiplication factor, 1 implies no discounting, 0 implies maximum discounting.  Default value 0.99.

- Strategy Discount [strat_discount, float in [0,1]]: memory factor in strategy updating. Both (hawk, dove) strategy weights are multiplied by strat_discount just before strategy learning occurs.  As it is a multiplication factor, 1 implies no discounting, 0 implies maximum discounting.  Default value 0.99.

- Network Learning Speed [net_learning_speed, float >= 0]: multiplier for network payoff additions (larger value corresponds to faster network learning).  0 implies no learning.

- Strategy Learning Speed [strat_learning_speed, float >= 0]: multiplier for strategy payoff additions (larger value corresponds to faster strategy learning)

- Network Tremble [net_tremble, float in [0,1]]: network error rate between 0 and 1 (default 0.01).  When agents make a network error, they choose a random agent in the network to play against regardless of adjacency weights.  0 implies no errors, 1 means that agents make a random partner choice every round.

- Strategy Tremble [strat_tremble, float in [0,1]]: strategy error rate between 0 and 1 (default 0.01).  When agents make a strategy error they flip a coin to decide which strategy to play, rather than relying on strategy weights.

- Initial Visitor Hawk Condition [init_cond_hawk_p1, float in [0,100]]: initial visitor hawk strategy percentage.  (i.e., if 75, all agents will start simulation with 75% chance to play hawk when visiting).

- Initial Host Hawk Condition [init_cond_hawk_p2, float in [0,100]]: initial host hawk strategy percentage.  (i.e., if 75, all agents will start simulation with 75% chance to play hawk when hosting).

###########################################

#### First, C++ simulation code must be compiled. 
Compile Requirements: On Mac OS 10.12.6, I have downloaded boost-1.55.0 and gsl

To download gsl, see here: https://www.gnu.org/software/gsl/

To download boost, see here: http://www.boost.org/users/history/ (code should work with newest version of boost, but I can only guarantee that it does work with 1.55.0)

I have compiled with both g++ and clang.  For g++, see here: http://www-scf.usc.edu/~csci104/20142/installation/gccmac.html

#### Once you have all those, compile code is:

g++ -g -O3 Conflict_Dynamic_Networks_SimCode.cpp -I /usr/local/boost-1.55.0/include -L /usr/local/boost-1.55.0/lib -std=c++11 -lgsl -o DynNet_HawkDoveSim

#### OR, this code supports multi-threading.  To compile with multi-threading, use:

g++ -g -O3 -fopenmp Conflict_Dynamic_Networks_SimCode.cpp -I /usr/local/boost-1.55.0/include -L /usr/local/boost-1.55.0/lib -std=c++11 -lgsl -o DynNet_HawkDoveSim

#### To setup the Simulation:

Next, open driver.py and choose the model parameters that you would like to simulate.  As long as the datatypes are compatible (outlined above) and each input list has at least one element, it will work.

BEWARE choosing model parameters.  “setup_simulation.py” will create an input parameter set for every possible combination of lists that you enter.  Choosing 5 populations, 4 payoff points, 3 trembles, 2 initial visitor hawk conditions, and 2 learning speeds will result in 5*4*3*2*2 = 240 total simulation parameter sets (times however many seeds are specified).

Once input parameters are chosen, set run_now = 1 if you want python to call the simulation code directly (e.g. if you’re running the simulation locally, right now).  Otherwise set run_now = 0.  If you set run_now = 1, be sure to set num_seeds to the number of random seeds you want to run for each parameter configuration.

#### Then to run:

python driver.py

If you set run_now = 0, you will need to know 3 things:  Number of seeds to run (num_seeds below), the name of the input folder (input_folder below) where the input files are stored (inside HD_Input after you run python driver.py), and the number of threads to run in parallel.  If you compiled with no “-fopenmp” flag, the number of threads must be 1.

#### To run the 1st configuration file in input_folder:

./DynNet_HawkDoveSim 1 HD input_folder 0 num_seeds 0 0 0 0 0 0

#### To run the 5th configuration file:

./DynNet_HawkDoveSim 1 HD input_folder 4 num_seeds 0 0 0 0 0 0


Can easily wrap this in a bash script to run all configuration files.  By default, python will run all configuration files created when run_now = 1.


#### Notes:

Populations larger than 100 are quite slow and should be run on a dedicated server if possible.  Small populations are very fast and low memory usage, so splitting runs among many threads is preferable.  At about population 500 the opposite holds, higher RAM is more important than more threads/cores.

There are more parameters available than the ones specified here.  See the C++ code and setup_sim.py for details.  You can change the max simulation time, the symmetry of the ## strategy or network, etc.  Not all other parameters in setup_sim are supported by C++ ## code as of right now.
