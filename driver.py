"""This is the driver code for the model described in "Conflict and Convention in Dynamic Networks", by Foley, Forber, Smead, Riedl, Royal Society Interface (citation to come)
    
    After compiling "Conflict_Dynamic_Networks_SimCode.cpp",
    Choose parameter lists of interest below and then run this file (i.e. python driver.py)
    All arguments are configured in this file, and this file directly calls the C++ simulation code
    
    """

from setup_sim import setup_simulation
from get_payoffs import get_payoffs

"""
    Set Input Parameters
    
    """

# Run_now (run locally, immediately when calling setup_simulation
run_now = 1

num_seeds = 10 # Change this as you see fit

# If run_now == 0, see readme for details

# Population (size of network)
# CHOOSE LIST OF POPULATIONS TO SIMULATE (INTEGER ONLY)
pop_list_in = [20]

# Initial strategy string - Options "uniform" or "random"
# "uniform" means they're all equal to 1, "random" is random draw, which is later renormalized
init_strategy_str_list_in = ["uniform"]

# Network and strategy discounts
# Discounting determines memory of agents
# Acceptable values are in [0,1] range inclusive
# IN CCDN, STRATEGY AND NETWORK DISCOUNTS ARE EQUAL
net_discount_list_in = [0.99]
strat_discount_list_in = net_discount_list_in

# Network and strategy learning speed 
# CCDN COVERS [0.1,1,10]
# 1 is default
net_speed_list_in = [1]
strat_speed_list_in = [1]

# Network and strategy tremble (0.01 is default)
# CCDN COVERS [0,0.001,0.01]
# STRATEGY AND NETWORK TREMBLE EQUAL IN CCDN
net_tremble_list_in = [0.01]
strat_tremble_list_in = net_tremble_list_in

# INITIAL CONDITIONS FOR VISITOR AND HOST STRATEGY
# SET AT INITIAL PERCENTAGE TO PLAY HAWK (P1 IS VISITOR, P2 IS HOST)
# DEFAULT 50, 50 (EQUAL CHANCE TO PLAY HAWK OR DOVE FOR BOTH VISITOR AND HOST AT SIMULATION START)
init_cond_hawk_p1_list_in = [50]
init_cond_hawk_p2_list_in = [50]

## User input payoffs (see get_payoffs.py for full details)
payoff_type = "User"
if payoff_type == "User":
    user_p1_payoff = [[0,1],[0.2,0.6]]
    user_p2_payoff = [[0,0.2],[1,0.6]]

# Get Payoffs 
# See "get_payoffs.py" to change arguments from defaults
payoffs = get_payoffs(payoff_type,user_p1_payoff = user_p1_payoff, user_p2_payoff = user_p2_payoff)

"""
    Run simulation with given parameters
    """

setup_simulation(payoffs, pop_list_in, init_strategy_str_list_in, net_discount_list_in, strat_discount_list_in, net_speed_list_in, strat_speed_list_in,net_tremble_list_in, strat_tremble_list_in, init_cond_hawk_p1_list_in, init_cond_hawk_p2_list_in, payoff_type,run_now,num_seeds)
