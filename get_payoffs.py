import numpy as np

"""
    ######   CCDN SIMULATION PAYOFFS
    ##
    ## Called from driver.py (change all inputs there)
    ##
    ## payoff_type = choose from predetermined list of payoffs (possible arguments: "AlphaV","AlphaH", "SymSpace", "ASymSpace", "User")
    ## dh_spacing = determines spacing interval for range of dh_payoffs
    ##
    ## For Alpha payoffs:
    ## Fix either host or visitor payoffs, while varying the other across some range
    ## Difference between dove-dove and dove-hawk payoffs is equal (to alpha) for both host and visitor
    ## alpha = y_2 - x_2 = y_1 - x_1 (from Conflict in Dynamic Networks paper, Foley et al.)
    ## 
    ## To Run:  Choose payoff_type (either "AlphaH" to vary host payoffs, or "AlphaV" to vary visitor payoffs)
    ## alpha_list = list of differences between dove-dove and dove-hawk
    ## 
    
    ## For Symmetric Payoffs (Spanning payoff space):
    ## Set payoff_type = "SymSpace"
    
    ## For Asymmetric payoffs (Spanning payoff space):
    ## Set payoff_type = "ASymSpace
    
    ## For User input payoffs
    ## Set payoff_type = "User"
    ## Requires user_p1_payoff (list of lists, set up as [[0,1],[x_1, y_1]] with y_1 >= x_1)
    ## Requires user_p2_payoff (list of lists, set up as [[0,x_1],[1, y_1]] with y_1 >= x_1)
    """

def get_payoffs(payoff_type,dh_spacing=0.05,alpha_list = [0.1],user_p1_payoff=[[0,1],[0.2,0.6]],user_p2_payoff=[[0,0.2],[1,0.6]]):
    # Initialize list of payoffs for p1 (visitor) and p2 (host)
    # P1 PAYOFFS IN CCDN ARE [[0,1],[x_1, y_1]] with y_1 >= x_1
    # P1 PAYOFFS IN CCDN ARE [[0,x_2],[1, y_2]] with y_2 >= x_2
    p1_payoffs_list_in = []
    p2_payoffs_list_in = []

    if payoff_type.startswith("Alpha"):
        dh_payoffs = np.arange(0.1,0.81,dh_spacing)
        for i in dh_payoffs:
            for alpha in alpha_list:
                if payoff_type == "AlphaV":
                    this_payoff_p1 = np.array([[0,1],[np.round(i,2),np.round(i + alpha,2)]])
                    this_payoff_p2 = np.array([[0,0.5 - alpha],[1,0.5]])
                if payoff_type == "AlphaH":
                    this_payoff_p1 = np.array([[0,1],[0.5 - alpha,0.5]])
                    this_payoff_p2 = np.array([[0,np.round(i,2)],[1,np.round(i + alpha,2)]])
                p1_payoffs_list_in.append(this_payoff_p1)
                p2_payoffs_list_in.append(this_payoff_p2)
    elif payoff_type == "SymSpace": # Symmetric payoff space
        payoff_dd = np.linspace(0.1,0.9,40)
        payoff_dh = np.linspace(0.1,0.9,40)
        for dh in payoff_dh:
            for dd in payoff_dd:
                if dh < dd and dh >= 0.1 and dd <= 0.9:
                    this_payoff_p1 = np.array([[0,1],[np.round(dh,2),np.round(dd,2)]])
                    this_payoff_p2 = np.array([[0,np.round(dh,2)],[1,np.round(dd,2)]])
                    p1_payoffs_list_in.append(this_payoff_p1)
                    p2_payoffs_list_in.append(this_payoff_p2)
    elif payoff_type == "ASymSpace": # Asymmetric payoff space
        p_list = []
        dd2_payoffs = np.arange(1,9.1,0.25)
        dd1_payoffs = np.arange(1,9.1,0.25)
        for dd2_ratio in dd2_payoffs:
            for dd1_ratio in dd1_payoffs:
                dh2 = 1/(1+dd2_ratio)
                dh1 = 1/(1+dd1_ratio)
                dd2 = np.round(1 - dh2,2)
                dd1 = np.round(1 - dh1,2)
                this_payoff_list = [0,1,np.round(dh1,2),np.round(dd1,2),0,np.round(dh2,2),1,np.round(dd2,2)]
                if dd1 >= dh1 and dd2 >= dh2 and this_payoff_list not in p_list:
                    this_payoff_p1 = np.array([[0,1],[np.round(dh1,2),np.round(dd1,2)]])
                    this_payoff_p2 = np.array([[0,np.round(dh2,2)],[1,np.round(dd2,2)]])
                    p1_payoffs_list_in.append(this_payoff_p1)
                    p2_payoffs_list_in.append(this_payoff_p2)
                    p_list.append(this_payoff_list)
    elif payoff_type == "User":
        this_payoff_p1 = np.array(user_p1_payoff)
        this_payoff_p2 = np.array(user_p2_payoff)
        
        p1_payoffs_list_in.append(this_payoff_p1)
        p2_payoffs_list_in.append(this_payoff_p2)

    return zip(p1_payoffs_list_in,p2_payoffs_list_in)
