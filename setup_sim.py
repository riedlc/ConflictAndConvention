""" This code corresponds to simulations run in "Conflict and Convention in Dynamic Networks" by Foley, Forber, Smead, Riedl
    Royal Society Interface
    Referred to hereafter as CCDN
    IF ATTEMPTING TO REPLICATE CCDN RESULTS, DO NOT CHANGE VARIABLES COMMENTED WITH "FIXED IN CCDN"
    Otherwise, feel free to play around with any variables
    
    README available in main folder
    
   """

import os, sys
from itertools import product
import numpy as np
import random
import string
import pandas as pd
import re
from itertools import combinations
from subprocess import Popen

# Helper function to flatten list of lists
flatten = lambda l: [item for sublist in l for item in sublist] 

# Header of input files (these are all parameters that C++ simulation code uses)
full_input_list = ['Base','Num_Strats_P1','Num_Strats_P2','Pop','TMax','Net_in','InitStratStr','InitNetStr','InitNetFill','DeathRate','NetDiscount','StratDiscount','NetLearningSpeed','StratLearningSpeed','NetSymmetric','StratSymmetric','NetTremble','StratTremble','Game','OutFolder','Key']

# Total input variables
num_input_vars = len(full_input_list)
bracket_string = "{} " * (num_input_vars-1) + "{}\n"

def setup_simulation(payoffs,pop_list_in = [20],init_strategy_str_list_in = ["uniform"],net_discount_list_in = [0.99],strat_discount_list_in = [0.99],net_speed_list_in = [1],
                   strat_speed_list_in = [1],net_tremble_list_in = [0.01], strat_tremble_list_in = [0.01],init_cond_hawk_p1_list_in = [50],init_cond_hawk_p2_list_in = [50],payoff_type = "User",run_now=1,num_seeds = 1):
    """ 
    THE FOLLOWING SETS ALL INPUT PARAMETERS FOR MODEL
    
    payoffs = list of payoffs determined by get_payoffs
    pop_list_in = list of simulation populations (integer)
    init_strategy_str_list_in = list of strings that determine initial strategies ("random" or "uniform")
    net_discount_list_in = list of network discount values (float range [0,1])
    strat_discount_list_in = list of strategy discount values (float range[0,1])
    net_speed_list_in = list of network learning speeds (float >= 0, where 0 implies no learning)
    strat_speed_list_in = list of strategy learning speeds (float >= 0, where 0 implies no learning)
    net_tremble_list_in = list of network trembles (called errors in paper) (float range[0,1])
    strat_tremble_list_in = list of strategy trembles (float range[0,1])
    init_cond_hawk_p1_list_in = list of initial hawk strategy percentages for visitor
    init_cond_hawk_p2_list_in = list of initial hawk strategy percentages for host
    payoff_type = see get_payoffs for payoff_type options
    run_now = if 1, run simulation directly from this function, if 0, this function creates all input files, see readme on how to run simulation on cluster, etc.  
    
    Beware: if run_now == 1, some simulations take a very long time.  Keep population < 100 and seeds < 100 to run local.
    """

    # Each element in these lists corresponds to one parameter point.  For payoff symmetry, set dd1s = dd2s, dh1s == dh2s. (Can do manually or just add an if statement).  Must have equal number of elements in all 4 lists!

    # THE TWO STRATEGIES ARE HAWK AND DOVE FOR BOTH HOST AND VISITOR
    # FIXED IN CCDN
    num_strats_p1 = 2
    num_strats_p2 = 2

    # Game Type (Hawk Dove)
    # FIXED IN CCDN, THIS IS JUST FOR FILE NAMING
    game = "HD"

    # FOLDER WHERE INPUT FILES ARE STORED
    input_folder = game + "_Input/"

    # Fixed base at 0.0001 in hawk-dove
    # This number is added to all payoffs (to avoid computational issues with floats below minimum representable float)
    ## FIXED IN CCDN
    base_in = 0.0001

    # Fully connected initial network, with agent outweights uniformly distributed among neighbors
    # Changes to this not currently supported (TO-DO)
    # FIXED IN CCDN
    network_list_in = ["random-100"]

    # How long to run simulation (max time)
    # One million time steps typically enough for convergence, however edge cases exist
    # FIXED IN CCDN
    tmax_in = 1000000

    # Initial network string (if not uniform, outgoing weights are drawn randomly and renormalized)
    # FIXED IN CCDN
    init_net_str_list_in = ["uniform"]

    # No death rate
    # FIXED IN CCDN
    death_rate_list_in = [0]

    # Network and strategy symmetry (0 for asymmetric, 1 for symmetric)
    # FIXED IN CCDN
    net_sym_list_in = [0]
    strat_sym_list_in = [0]

    # Parameters to string - leave this
    nd_str = "-".join(map(str, net_discount_list_in))
    sd_str = "-".join(map(str, strat_discount_list_in))
    nls_str = "-".join(map(str, net_speed_list_in))
    sls_str = "-".join(map(str, strat_speed_list_in))
    net_tremble_str = "-".join(map(str, net_tremble_list_in))
    strat_tremble_str = "-".join(map(str, strat_tremble_list_in))
    n_sym_str = "-".join(map(str, net_sym_list_in))
    s_sym_str = "-".join(map(str, strat_sym_list_in))
    pop_str = "-".join(map(str, pop_list_in))
    init_cond_hawk_p1 = "-".join(map(str, init_cond_hawk_p1_list_in))
    init_cond_hawk_p2 = "-".join(map(str, init_cond_hawk_p2_list_in))
    
    if max(pop_list_in) < 100:
        file_lines = 6
    elif max(pop_list_in) < 500:
        file_lines = 2
    else:
        file_lines = 1


    strats_p1_str = str(num_strats_p1)
    strats_p2_str = str(num_strats_p2)

    time_str = str(tmax_in/1000000)

    # Simulation description
    description_in = "Pop-"+ pop_str + "_Discount-" + sd_str + "_NLS-" + nls_str + "_SLS-" + sls_str + "_SymN-"+ n_sym_str + "_SymS-" + s_sym_str + "_Tremble-" + strat_tremble_str + "_T-" + time_str + "M_Payoffs-" + payoff_type + "_" + str(init_cond_hawk_p1) + "-" + str(init_cond_hawk_p2)

    data_description = game + "_" + description_in

    if not os.path.exists(game + "_Output_Data"):
        os.makedirs(game + "_Output_Data")

    # Get all combinations of parameter lists above
    param_combos = list(product(payoffs,pop_list_in,network_list_in,init_strategy_str_list_in,init_net_str_list_in,init_cond_hawk_p1_list_in,init_cond_hawk_p2_list_in,death_rate_list_in,net_discount_list_in,strat_discount_list_in,net_speed_list_in,strat_speed_list_in,net_sym_list_in,strat_sym_list_in, net_tremble_list_in,strat_tremble_list_in))
    
    total_files = 0
    
    input_params_list = []
    for ind,i in enumerate(param_combos):
        
        # CONVERT PERCENTAGES TO INITIAL WEIGHTS (TOTAL 2)
        hawk_p1_weight = np.round(2 * i[5]/100.,1)
        hawk_p2_weight = np.round(2 * i[6]/100.,1)
        
        # Set local parameter variables
        p1_payoffs_in = i[0][0]
        p2_payoffs_in = i[0][1]
        pop_in = i[1]
        network_in = i[2]
        init_strategy_str_in = i[3]
        init_net_str_in = i[4]
        init_p1strategy_fill_in = np.array(np.ones(shape=(pop_in,2)) * np.array([hawk_p1_weight,2-hawk_p1_weight])).T
        init_p2strategy_fill_in = np.array(np.ones(shape=(pop_in,2)) * np.array([hawk_p2_weight,2-hawk_p2_weight])).T
        init_net_fill_in = np.round(19*(1.0/(pop_in-1)),5)
        death_rate_in = i[7]
        net_discount_in = i[8]
        strat_discount_in = i[9]
        net_speed_in = i[10]
        strat_speed_in = i[11]
        net_sym_in = i[12]
        strat_sym_in = i[13]
        net_tremble = i[14]
        strat_tremble = i[15]
        
        
        # Keep discount and tremble equal for parsimony
        if net_tremble == strat_tremble and net_discount_in == strat_discount_in:
        
            key = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(8))
            
            print(key)
            print(data_description)
            
            payoff_folder = input_folder + data_description + "/Payoffs/"
            initw_folder = input_folder + data_description + "/InitWeights/"
            if not os.path.exists(payoff_folder):
                os.makedirs(payoff_folder)
                os.makedirs(initw_folder)
            
            pd.DataFrame(p1_payoffs_in).to_csv(payoff_folder + "P1_Payoffs_" + key + ".csv",header=None,index=False)
            pd.DataFrame(p2_payoffs_in).to_csv(payoff_folder + "P2_Payoffs_" + key + ".csv",header=None,index=False)
            
            pd.DataFrame(init_p1strategy_fill_in).to_csv(initw_folder + "P1_InitWeights_" + key + ".csv",header=None,index=False)
            pd.DataFrame(init_p2strategy_fill_in).to_csv(initw_folder + "P2_InitWeights_" + key + ".csv",header=None,index=False)
            
            input_params = bracket_string.format(base_in,num_strats_p1,num_strats_p2,pop_in,tmax_in,network_in,init_strategy_str_in,init_net_str_in,init_net_fill_in,death_rate_in,net_discount_in,strat_discount_in,net_speed_in,strat_speed_in,net_sym_in,strat_sym_in,net_tremble,strat_tremble,game,data_description,key)
            
            if input_params not in input_params_list:
                input_params_list.append(input_params)

    if not os.path.exists(input_folder + data_description):
        os.makedirs(input_folder + data_description)

    for ind,row in enumerate(input_params_list):
        if not os.path.exists(input_folder + data_description + "/Input_" + data_description + "_" + str(int(ind/file_lines)) + ".conf"):
            with open(input_folder + data_description + "/Input_" + data_description + "_" + str(int(ind/file_lines)) + ".conf",'a') as a:
                if ind % file_lines == 0:
                    a.write(bracket_string.format(*full_input_list))
                    a.write(row)
                else:
                    a.write(row)

    total_files = int(ind/file_lines) + 1
    if run_now:
        for run_ind in range(total_files):
            test_in2 = ["./DynNet_HawkDoveSim","1","HD", data_description, str(run_ind), str(num_seeds), "0", "0", "0", "0", "0", "0"]
            Popen(test_in2,shell=False)
