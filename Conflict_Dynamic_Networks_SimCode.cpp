// If compiled with -fopenmp, include omp (for multithreading)
#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_get_max_threads() 1
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <stdint.h>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <array>
#include <numeric>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <dirent.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>

typedef boost::mt19937 Engine;
typedef boost::uniform_real<double> UDistribution;
typedef boost::variate_generator< Engine &, UDistribution > UGenerator;

// Global trackers in case we want to store more data (more time steps)
int time_tracker1;
int time_tracker2;
int time_tracker3;
int burn_in_time;

struct MersenneRNG {
    MersenneRNG() : dist(0.0, 1.0), rng(eng, dist) {}
    
    Engine eng;
    UDistribution dist;
    UGenerator rng;
};

time_t starttime;

void run_evolution(int num_strats_p1,int num_strats_p2,std::vector<double>& adjacency_weights,std::vector<double>& adjacency_weights_new, UGenerator rng, int seed_id, int pop, float p1_payoff[],float p2_payoff[],std::vector<double>& p1_strategy,std::vector<double>& p2_strategy,std::vector<double>& p1_strategy_t,std::vector<double>& p2_strategy_t,std::vector<double>& adjacency_weights_t,std::vector<int>& player1_s1_t,std::vector<int>& player2_s1_t, float base, float death_rate, int max_time, float net_discount, float strat_discount, float net_learning_speed, float strat_learning_speed, int net_symmetric, int strat_symmetric,  float net_tremble_prob, float strat_tremble_prob, std::vector<double>& strat_corr_t, std::vector<double>& perc_inters_t, std::vector<double>& p1_strat_var_t, std::vector<double>& p2_strat_var_t,std::vector<double>& p1_strat_mean_t, std::vector<double>& p2_strat_mean_t,std::vector<double>& strength_var_t, std::vector<int>& stats_time, std::vector<double>& payoffs_p1_t,std::vector<double>& payoffs_p2_t); // One randomly seeded simulation run
void one_time_step(int num_strats_p1, int num_strats_p2,std::vector<double>& adjacency_weights, std::vector<double>& adjacency_weights_new, int t, UGenerator rng, int pop, float p1_payoff[], float p2_payoff[], std::vector<double>& p1_strategy, std::vector<double>& p1_strategy_new, std::vector<double>& p2_strategy, std::vector<double>& p2_strategy_new, long int& total_p1, long int& total_p2, float base, float net_discount, float strat_discount,float net_learning_speed, float strat_learning_speed, int net_symmetric, int strat_symmetric, float net_tremble_prob, float strat_tremble_prob, std::vector<int>& agent_seq, std::vector<double>& past_payoffs_p1,std::vector<double>& past_payoffs_p2); // Run simulation over all agents for one time step
int * intdup(int const * src, size_t len);
long * longdup(long const * src, size_t len);
float * floatdup(float const * src, size_t len);
void print_matrix(long int matrix[], int pop); // Helper function to print out matrix
void print_as_matrix(float matrix[],int row_len, int col_len);
void print_dubarray(float array[],int size); // Helper to print out array
void print_intarray(long int array[],int size);
//void print_vector(igraph_vector_t *v);

int main(int argc, char *argv[])
/* Read in or build network
 Read in list of seeds (used only to read in spaces)
 Run barebones simulation 1000 times
 */
{
    
    //argv[1] is number of threads for openmp
    //argv[2] should be "HD" for Hawk Dove
    //argv[2] is subfolder containing input parameter file
    //argv[3] is number of input file (starting at 0)
    //argv[4] is number of random seeds (100 is typically sufficient), fewer for large populations
    //argv[5] is index of starting seed (default 0)
    //argv[6] is index of initial input line within input file
    //Next 3 argv are time trackers (1 is for EvoStats files, between 2 and 3 is recording full network/strategy data)
    //argv[11] is burn in time (no imitation takes place then, just network updating)
    
    starttime = time(NULL);
    
    int thread_ct = atoi(argv[1]);
    int input_filenum = atoi(argv[4]);
    int num_seeds = atoi(argv[5]);
    int base_seed = atoi(argv[6]);
    int start_key = atoi(argv[7]);
    time_tracker1 = atoi(argv[8]);
    time_tracker2 = atoi(argv[9]);
    time_tracker3 = atoi(argv[10]);
    burn_in_time = atoi(argv[11]);
    
    /*
     Read seeds for repeatable random draws
     */
    ////////////////////////////////////////////
    
    FILE *seedFile;
    char seed_filename[128];
    strcpy(seed_filename,"ESD_Seeds_All_Ordered.csv");
    int seeds[num_seeds];
    int throwaway;
    
    seedFile = fopen(seed_filename, "r");
    
    if(seedFile == NULL)
    {
        printf("\n seed file opening failed ");
    }
    
    for(int line = 0;line < base_seed+num_seeds; line++) // Read seed file line by line
    {
        if(line < base_seed){
            fscanf(seedFile, "%d", &throwaway);
        }
        else{
            fscanf(seedFile, "%d", (seeds + line - base_seed)); // float check how efficient this is
        }
    }
    
    fclose(seedFile);
    
    
    //////////////////////////////////////////////
    
    /*
     Read input parameters into array
     */
    int max_line_len = 500;
    int lines_allocated = 128;
    
    /* Allocate lines of text */
    char **words = (char **)malloc(sizeof(char*)*lines_allocated);
    
    /* Allocate lines of text */
    if (words==NULL)
    {
        fprintf(stderr,"Out of memory (1).\n");
        exit(1);
    }
    
    char inputFolder[300];
    
    snprintf(inputFolder,sizeof(inputFolder),"%s_Input/%s",argv[2],argv[3]);
    
    char inputFilename[600];
    
    snprintf(inputFilename,sizeof(inputFilename),"%s/Input_%s_%d.conf",inputFolder,argv[3],input_filenum);
    
    
    FILE *inputFile = fopen(inputFilename, "r");
        
    if (inputFile == NULL)
    {
        fprintf(stderr,"Error opening file.\n");
        exit(2);
    }
    
    char buffer[500];
    
    fgets(buffer, 500, inputFile);
    
    int k;
    for (k=0; 1;k++)
    {
        int j;
        
        /* Have we gone over our line allocation? */
        if (k >= lines_allocated)
        {
            int new_size;
            
            /* float our allocation and re-allocate */
            new_size = lines_allocated*2;
            words = (char **)realloc(words,sizeof(char*)*new_size);
            
            if (words == NULL)
            {
                fprintf(stderr,"Out of memory.\n");
                exit(3);
            }
            
            lines_allocated = new_size;
        }
        
        
        /* Allocate space for the next line */
        words[k] = (char *)malloc(sizeof(char *) * max_line_len);
        if (words[k]==NULL)
        {
            fprintf(stderr,"Out of memory (3).\n");
            exit(4);
        }
        if (fgets(words[k],max_line_len-1,inputFile)==NULL)
        break;
        
        /* Get rid of CR or LF at end of line */
        for (j=strlen(words[k])-1;j>=0 && (words[k][j]=='\n' || words[k][j]=='\r');j--)
        ;
        words[k][j+1]='\0';
    }
    /* Close file */
    fclose(inputFile);
    
    //////////////////////////////////////////////////////////////
    
    //igraph_i_set_attribute_table(&igraph_cattribute_table);
    
    MersenneRNG mrng; // Set random number generator
    
    //How to seed it
    long seed = (long) 0; // For repeatable results, otherwise can use computer clock, etc.
    mrng.rng.engine().seed(seed);
    mrng.rng.distribution().reset();
    
	#ifdef _OPENMP
    {
		#pragma omp parallel for num_threads(thread_ct) //start thread_ct parallel for loops (each is one simulation)
	#endif
        for(int seeded_run = num_seeds*start_key; seeded_run < k * num_seeds; seeded_run++)
        {
            
            int run_num = seeded_run/num_seeds; // Index of current line (defining input paramaters) in input file
            int seed_ind = seeded_run%num_seeds; // Index of curent seed
            
            //printf("%d\n",run_num);
            //printf("%d\n",seed_ind);
            
            
            /* Read run_num(th) line from input file and parse parameters here
             */
            
            // All input parameters defined in README
            
            char* network; // Network
            char* init_strategy_str;
            char* init_net_str;
            char* game;
            char* key;
            char* sim_description;
            
            game = (char*) malloc(16*sizeof(char));
            key = (char*) malloc(16*sizeof(char));
            network = (char*) malloc(16 * sizeof(char));
            init_strategy_str = (char*) malloc(16 * sizeof(char));
            init_net_str = (char*) malloc(16 * sizeof(char));
            sim_description = (char*) malloc(200*sizeof(char));
            
            int pop;
            int max_time;
            int num_strats_p1;
            int num_strats_p2;
            float base; // Add this to all payoffs
            float death_rate;
            float net_discount;
            float strat_discount;
            float init_net_fill;
            float net_learning_speed;
            float strat_learning_speed;
            int net_symmetric;
            int strat_symmetric;
            float net_tremble_prob;
            float strat_tremble_prob;
            
            //p1_payoff[0] = S1[0] * S2[0], p1_payoff[1] = S1[0] * S2[1], p1_payoff[2] = S1[1] * S2[0], p1_payoff[3] = S1[1] * S2[1]
            //p2_payoff is the same
            
            //printf("%s\n",words[run_num]);
            
            // Read in one line
            sscanf(words[run_num],"%f %d %d %d %d %s %s %s %f %f %f %f %f %f %d %d %f %f %s %s %s", &base,&num_strats_p1,&num_strats_p2,&pop,&max_time,network,init_strategy_str,init_net_str,&init_net_fill,&death_rate, &net_discount, &strat_discount, &net_learning_speed, &strat_learning_speed, &net_symmetric, &strat_symmetric, &net_tremble_prob, &strat_tremble_prob, game, sim_description, key);
            
            //printf("%f\n",net_discount);
            //printf("%f\n",net_tremble_prob);
            //printf("%s\n",game);
            
            // Initialize payoff matrices for visitor (p1) and host (p2)
            float p1_payoff[num_strats_p1*num_strats_p2];
            float p2_payoff[num_strats_p1*num_strats_p2];
            
            // Initialize payoff filenames
            char p1payoff_filename[400];
            char p2payoff_filename[400];
            
            // Initialize output folder
            char out_folder[325];
            
            snprintf(out_folder,sizeof(out_folder),"Output_%s",sim_description);
            snprintf(p1payoff_filename,sizeof(p1payoff_filename),"%s/Payoffs/P1_Payoffs_%s.csv",inputFolder,key);
            snprintf(p2payoff_filename,sizeof(p2payoff_filename),"%s/Payoffs/P2_Payoffs_%s.csv",inputFolder,key);
            
            // Read in visitor payoff matrix
            FILE *p1PayoffMat = fopen(p1payoff_filename, "r");
            
            if (p1PayoffMat == NULL)
            {
                fprintf(stderr,"Error opening file.\n");
                exit(2);
            }
            
            for(int row_c = 0; row_c < num_strats_p1; row_c++)
            {
                for(int col_c = 0; col_c < num_strats_p2; col_c++)
                {
                    fscanf(p1PayoffMat,"%f,",&p1_payoff[row_c * num_strats_p2 + col_c]);
                }
                
            }
            fclose(p1PayoffMat);
            
            // Read in host payoff matrix
            FILE *p2PayoffMat = fopen(p2payoff_filename, "r");
            
            if (p2PayoffMat == NULL)
            {
                fprintf(stderr,"Error opening file.\n");
                exit(2);
            }
            
            for(int row_c = 0; row_c < num_strats_p1; row_c++)
            {
                for(int col_c = 0; col_c < num_strats_p2; col_c++)
                {
                    fscanf(p2PayoffMat,"%f,",&p2_payoff[row_c * num_strats_p2 + col_c]);
                    
                }
                
            }
            fclose(p2PayoffMat);
            
            // Initialize starting strategy weights
            float p1_init_weights[num_strats_p1 * pop];
            float p2_init_weights[num_strats_p2 * pop];
            
            char p1initweight_filename[400];
            char p2initweight_filename[400];
            
            snprintf(p1initweight_filename,sizeof(p1initweight_filename),"%s/InitWeights/P1_InitWeights_%s.csv",inputFolder,key);
            snprintf(p2initweight_filename,sizeof(p2initweight_filename),"%s/InitWeights/P2_InitWeights_%s.csv",inputFolder,key);
            
            // Read in starting visitor strategy weights from file
            FILE *p1IWMat = fopen(p1initweight_filename, "r");
            
            if (p1IWMat == NULL)
            {
                fprintf(stderr,"Error opening file.\n");
                exit(2);
            }
            
            for(int w = 0; w < num_strats_p1; w++)
            {
                for(int p_ind = 0; p_ind < pop; p_ind ++){
                    fscanf(p1IWMat,"%f,",&p1_init_weights[w * pop + p_ind]);
                }
            }
            fclose(p1IWMat);
            
            // Read in starting host strategy weights from file
            FILE *p2IWMat = fopen(p2initweight_filename, "r");
            
            if (p2IWMat == NULL)
            {
                fprintf(stderr,"Error opening file.\n");
                exit(2);
            }
            
            for(int w = 0; w < num_strats_p2; w++)
            {
                for(int p_ind = 0; p_ind < pop; p_ind ++){
                    fscanf(p2IWMat,"%f,",&p2_init_weights[w * pop + p_ind]);
                }
            }
            fclose(p2IWMat);
            
            
            ////////////////////////////////////////////////////////////////////
            
            struct stat st = {0};
            
            //printf("%s\n",key);
            
            // Initialize path to output folder
            char out_folder_complete_path[350];
            
            snprintf(out_folder_complete_path,sizeof(out_folder_complete_path),"%s_Output_Data/%s",argv[2],out_folder);
            
            // If output directory doesn't exist, create it
            if(stat(out_folder_complete_path, &st) == -1){
                mkdir(out_folder_complete_path, 0700);
            }
            
            // Initialize output filenames
            char out_file_adj[400];
            char out_file_p1t[400];
            char out_file_p2t[400];
            char out_file_p1_s1_t[400];
            char out_file_p2_s1_t[400];
            char out_file_stats[400];
            char out_file_payoffs_p1_t[400];
            char out_file_payoffs_p2_t[400];
            
            
            snprintf(out_file_adj,sizeof(out_file_adj),"%s/%s_Weights_%s_%d.csv",out_folder_complete_path,game,key, seeds[seed_ind]);
            snprintf(out_file_p1t,sizeof(out_file_p1t),"%s/%s_StrategyP1t_%s_%d.csv",out_folder_complete_path,game,key,seeds[seed_ind]);
            snprintf(out_file_p2t,sizeof(out_file_p2t),"%s/%s_StrategyP2t_%s_%d.csv",out_folder_complete_path,game,key,seeds[seed_ind]);
            snprintf(out_file_p1_s1_t,sizeof(out_file_p1_s1_t),"%s/%s_ActionsP1t_%s_%d.csv",out_folder_complete_path,game,key,seeds[seed_ind]);
            snprintf(out_file_p2_s1_t,sizeof(out_file_p2_s1_t),"%s/%s_ActionsP2t_%s_%d.csv",out_folder_complete_path,game,key,seeds[seed_ind]);
            snprintf(out_file_stats,sizeof(out_file_stats),"%s/%s_EvoStats_%s_%d.csv",out_folder_complete_path,game,key,seeds[seed_ind]);
            snprintf(out_file_payoffs_p1_t,sizeof(out_file_payoffs_p1_t),"%s/%s_PayoffsP1t_%s_%d.csv",out_folder_complete_path,game,key,seeds[seed_ind]);
            snprintf(out_file_payoffs_p2_t,sizeof(out_file_payoffs_p2_t),"%s/%s_PayoffsP2t_%s_%d.csv",out_folder_complete_path,game,key,seeds[seed_ind]);
            
            // If no network learning, net_symmetric is irrelevant
            if(net_learning_speed == 0){
                net_discount = 1;
                net_symmetric = 0;
            }
            
            // Check if output files already exist, if they do, skip this simulation
            if( (access(out_file_adj, F_OK ) != -1) && (access(out_file_p1t, F_OK ) != -1) && (access(out_file_p2t, F_OK ) != -1) && (access(out_file_p1_s1_t, F_OK ) != -1) && (access(out_file_p2_s1_t, F_OK ) != -1) && (access(out_file_stats, F_OK ) != -1))
            {
                continue;
            } else // If output files don't yet exist, run this simulation
            {
                // Set up random engine, uniform distribution, corresponding rng
                Engine eng(seeds[seed_ind]);
                UDistribution udst(0.0, 1.0);
                UGenerator rng(eng, udst);
                
                // Initialize state variables for loop
                
                std::vector<double> p1_strategy;
                std::vector<double> p2_strategy;
                
                double total_init_strat;
                double p1_strat_draw;
                
                // If random initial strategies, create initial strategy weights here
                if(strcmp("random", init_strategy_str) == 0){
                    for(int pop_ind = 0; pop_ind < pop; pop_ind++){
                        for(int p1_fill_ind = 0; p1_fill_ind < num_strats_p1; p1_fill_ind++){
                            total_init_strat = p1_init_weights[p1_fill_ind * pop + pop_ind] + p2_init_weights[p1_fill_ind * pop + pop_ind];
                            p1_strat_draw = rng() * total_init_strat;
                            p1_strategy.push_back(p1_strat_draw);
                            p2_strategy.push_back(total_init_strat - p1_strat_draw);

                        }
                    }
                // If uniform initial strategies, create initial strategy weights here
                }else if(strcmp("uniform", init_strategy_str) == 0){
                    for(int pop_ind = 0; pop_ind < pop; pop_ind++){
                        for(int p1_fill_ind = 0; p1_fill_ind < num_strats_p1; p1_fill_ind++){
                            p1_strategy.push_back(p1_init_weights[p1_fill_ind * pop + pop_ind]);
                        }
                        for(int p2_fill_ind = 0; p2_fill_ind < num_strats_p2; p2_fill_ind++){
                            p2_strategy.push_back(p2_init_weights[p2_fill_ind * pop + pop_ind]);
                        }
                    }
                }
                // Initialize tracking vectors

                std::vector<double> adjacency_weights; // Old Network
                std::vector<double> adjacency_weights_new; // New Network
                                
                std::vector<double> *p1_strategy_t = new std::vector<double>; // Track visitor strategies for output (all agents)
                std::vector<double> *p2_strategy_t = new std::vector<double>; // Track host strategies for output (all agents)
                std::vector<double> *adjacency_weights_t = new std::vector<double>; // Track network for output
                std::vector<double> *payoffs_p1_t = new std::vector<double>; // Track visitor hawk actions for output
                std::vector<double> *payoffs_p2_t = new std::vector<double>; // Track host hawk actions for output
                
                std::vector<double> *strat_corr_t = new std::vector<double>; // Strategy correlation between visitor and host (0 if all different, 1 if all equal) over time
                std::vector<double> *perc_inters_t = new std::vector<double>; // Track percentage of interactions for output over time
                std::vector<double> *p1_strat_var_t = new std::vector<double>; // Track aggregate variance of visitor strategy over time
                std::vector<double> *p2_strat_var_t = new std::vector<double>; // Track aggregate variance of host strategy over time
                std::vector<double> *p1_strat_mean_t = new std::vector<double>; // Track aggregate mean of visitor strategy over time
                std::vector<double> *p2_strat_mean_t = new std::vector<double>; // Track aggregate mean of host strategy over time
                std::vector<double> *strength_var_t = new std::vector<double>; // Track node in strength variance over time
                std::vector<int> *stats_time = new std::vector<int>; // Track more statistics over time
                
                
                
                
                std::vector<int> player1_s1_t;
                std::vector<int> player2_s1_t;
                
                // If uniform network initial weights
                if(strcmp("uniform", init_net_str) == 0)
                {
                    for(int ag_ind = 0; ag_ind < pop;ag_ind++)
                    {
                        for(int nid = 0; nid < pop; nid++)
                        {
                            if(ag_ind == nid)
                            {
                                //adjacency_weights[ag_ind*pop + nid] = 0;
                                //adjacency_weights.push_back(0);
                                adjacency_weights.push_back(0);
                            }
                            else{
                                //adjacency_weights[ag_ind*pop + nid] = 1;
                                //adjacency_weights.push_back(1);
                                adjacency_weights.push_back(init_net_fill);
                            }
                        }
                        
                    }
                // If random network initial weights
                }else if(strcmp("random", init_net_str) == 0)
                {
                    for(int ag_ind = 0; ag_ind < pop;ag_ind++)
                    {
                        for(int nid = 0; nid < pop; nid++)
                        {
                            if(ag_ind == nid)
                            {
                                //adjacency_weights[ag_ind*pop + nid] = 0;
                                //adjacency_weights.push_back(0);
                                adjacency_weights.push_back(0);
                            }
                            else{
                                //adjacency_weights[ag_ind*pop + nid] = 1;
                                //adjacency_weights.push_back(1);
                                adjacency_weights.push_back(init_net_fill * rng());
                            }
                        }
                        
                    }
                }
                
                // Set new network equal to current network (for first time step)
                adjacency_weights_new = adjacency_weights;
                
                
                // Run simulation with given parameters, trackers, etc.
                run_evolution(num_strats_p1,num_strats_p2, adjacency_weights, adjacency_weights_new, rng, seed_ind, pop, p1_payoff,p2_payoff,p1_strategy,p2_strategy,*p1_strategy_t,*p2_strategy_t,*adjacency_weights_t,player1_s1_t,player2_s1_t,base,death_rate, max_time, net_discount, strat_discount, net_learning_speed, strat_learning_speed, net_symmetric, strat_symmetric, net_tremble_prob, strat_tremble_prob, *strat_corr_t, *perc_inters_t, *p1_strat_var_t, *p2_strat_var_t,*p1_strat_mean_t, *p2_strat_mean_t,*strength_var_t, *stats_time, *payoffs_p1_t,*payoffs_p2_t);
                
                
                // Variables for output
                int time;
                int index;
                float sum_row;
                float sum_strat_row;
                
                // Number of timesteps to track across different stats
                int measured_timesteps = player1_s1_t.size();
                int measured_time_adj = adjacency_weights_t->size()/(pop*pop);
                int measured_time_stats = stats_time->size();
                int actions_timesteps = measured_timesteps - 1;
                
                //printf("%lf\n", p1_strategy[10]
                
                
                //time_t simtime;
                //simtime = time(NULL);
                
                //printf("%lu\n",player1_s1_t.size());
                //printf("%d\n",measured_time_adj);
                
                
                // Output weighted adjacency matrix
                FILE *out_f_adj = fopen(out_file_adj, "wb");
                
                if(out_f_adj == NULL){
                    perror("Error");
                }
                
                for(int j=0; j<pop; j++)
                {
                    for(int i=0; i<pop*measured_time_adj; i++)
                    {
                        time = i/pop;
                        index = i%pop;
                        //printf("%f\n",std::accumulate((*adjacency_weights_t).begin()+time*pop*pop + j*pop, (*adjacency_weights_t).begin()+time*pop*pop + j*pop+20, 0.0));
                        sum_row = std::accumulate((*adjacency_weights_t).begin()+time*pop*pop + j*pop, (*adjacency_weights_t).begin()+time*pop*pop + j*pop+pop, 0.0);
                        
                        if(i < pop * measured_time_adj - 1)
                        {
                            fprintf(out_f_adj,"%2.4f,", adjacency_weights_t->at(time*pop*pop + j*pop + index)/sum_row);
                        }
                        else
                        {
                            fprintf(out_f_adj,"%2.4f",adjacency_weights_t->at(time*pop*pop + j*pop + index)/sum_row);
                        }
                    }
                    fprintf(out_f_adj,"\n");
                    
                }
                fclose(out_f_adj);
                
                // Output visitor strategies over time
                FILE *out_f_p1t = fopen(out_file_p1t, "wb");
                
                if(out_f_p1t == NULL){
                    perror("Error");
                }
                
                for(int j=0; j<pop; j++)
                {
                    for(int i = 0; i < num_strats_p1 * measured_timesteps; i++)
                    {
                        time = i/num_strats_p1;
                        index = i%num_strats_p1;
                        
                        sum_strat_row = std::accumulate((*p1_strategy_t).begin()+time*num_strats_p1*pop + j*num_strats_p1, (*p1_strategy_t).begin()+time*num_strats_p1*pop + j*num_strats_p1+num_strats_p1, 0.0);
                        
                        
                        if(i < num_strats_p1*measured_timesteps - 1)
                        {
                            fprintf(out_f_p1t,"%2.4f,", (p1_strategy_t->at(time*num_strats_p1*pop + j*num_strats_p1 + index))/((sum_strat_row)));
                        }
                        else
                        {
                            fprintf(out_f_p1t,"%2.4f",(p1_strategy_t->at(time*num_strats_p1*pop + j*num_strats_p1 + index))/(sum_strat_row));
                        }
                    }
                    fprintf(out_f_p1t,"\n");
                    
                }
                fclose(out_f_p1t);
                
                // Output host strategies over time
                FILE *out_f_p2t = fopen(out_file_p2t, "wb");
                
                if(out_f_p2t == NULL){
                    perror("Error");
                }
                
                for(int j=0; j<pop; j++)
                {
                    for(int i = 0; i < num_strats_p2 * measured_timesteps; i++)
                    {
                        time = i/num_strats_p2;
                        index = i%num_strats_p2;
                        
                        sum_strat_row = std::accumulate((*p2_strategy_t).begin()+time*num_strats_p2*pop + j*num_strats_p2, (*p2_strategy_t).begin()+time*num_strats_p2*pop + j*num_strats_p2+num_strats_p2, 0.0);
                        
                        if(i < num_strats_p2*measured_timesteps - 1)
                        {
                            fprintf(out_f_p2t,"%2.4f,", (p2_strategy_t->at(time*num_strats_p2*pop + j*num_strats_p2 + index))/(sum_strat_row));
                        }
                        else
                        {
                            fprintf(out_f_p2t,"%2.4f",(p2_strategy_t->at(time*num_strats_p2*pop + j*num_strats_p2 + index))/(sum_strat_row));
                        }
                    }
                    fprintf(out_f_p2t,"\n");
                    
                }
                fclose(out_f_p2t);
                
                // Output visitor hawk actions
                FILE *out_p1_s1_t = fopen(out_file_p1_s1_t, "wb");
                
                if(out_p1_s1_t == NULL){
                    perror("Error");
                }
                
                
                for(int j=0; j<measured_timesteps; j++)
                {
                    fprintf(out_p1_s1_t,"%d\n",player1_s1_t[j]);
                }
                fclose(out_p1_s1_t);
                
                // Output host hawk actions
                FILE *out_p2_s1_t = fopen(out_file_p2_s1_t, "wb");
                
                if(out_p2_s1_t == NULL){
                    perror("Error");
                }
                
                
                for(int j=0; j<measured_timesteps; j++)
                {
                    fprintf(out_p2_s1_t,"%d\n",player2_s1_t[j]);
                }
                fclose(out_p2_s1_t);
                
                // Output evolutionary stats (node strength variance, average strategy across all agents, proportion of interactions)
                FILE *out_stats = fopen(out_file_stats,"wb");
                
                if(out_stats == NULL){
                    perror("Error");
                }
                
                int base_ind;
                int base_ind_2;
                int base_ind_perc;
                
                char evostat_header[2000];
                char single_head_p1[60];
                char single_head_p2[60];
                char single_head_inter[7];
                
                strcpy(evostat_header, "Time,Node-Strength-Variance,");
                
                for(int j = 0; j < num_strats_p1; j++){
                    snprintf(single_head_p1,sizeof(single_head_p1),"P1_Strategy%d-Mean,P1_Strategy%d-Variance,",j+1,j+1);
                    strcat(evostat_header, single_head_p1);
                }
                
                for(int j = 0; j < num_strats_p2; j++){
                    snprintf(single_head_p2,sizeof(single_head_p2),"P2_Strategy%d-Mean,P2_Strategy%d-Variance,",j+1,j+1);
                    strcat(evostat_header, single_head_p2);
                }
                for(int j = 0; j < num_strats_p1; j++){
                    for(int k = 0; k < num_strats_p2; k++){
                        if(j == num_strats_p1 - 1 && k == num_strats_p2 - 1){
                            snprintf(single_head_inter,sizeof(single_head_inter),"%d-%d\n",j,k);
                        }else{
                            snprintf(single_head_inter,sizeof(single_head_inter),"%d-%d,",j,k);
                        }
                        strcat(evostat_header,single_head_inter);
                    }
                    
                }
                
                for(int j = 0; j < measured_time_stats; j++){
                    base_ind = j * num_strats_p1;
                    base_ind_2 = j * num_strats_p2;
                    base_ind_perc = j * num_strats_p1 * num_strats_p2;
                    if(j == 0){
                        //fprintf(out_stats,"Time,Node-Strength_Variance,Strategy_Pearsonr-Corr,Strategy1-P1_Mean,Strategy1-P2_Mean,Strategy1-P1_Variance,Strategy1-P2_Variance\n");
                        fprintf(out_stats,"%s",evostat_header);
                        
                    }
                    fprintf(out_stats,"%d,%2.3f,",stats_time->at(j),strength_var_t->at(j));
                    for(int n = 0; n < num_strats_p1; n++){
                        fprintf(out_stats,"%2.3f,%2.3f,",p1_strat_mean_t->at(base_ind+n),p1_strat_var_t->at(base_ind+n));
                    }
                    for(int n = 0; n < num_strats_p2; n++){
                        fprintf(out_stats,"%2.3f,%2.3f,",p2_strat_mean_t->at(base_ind_2+n),p2_strat_var_t->at(base_ind_2+n));
                    }
                    for(int n = 0; n < num_strats_p2*num_strats_p1; n++){
                        if(n < num_strats_p1*num_strats_p2-1){
                            fprintf(out_stats,"%2.3f,",perc_inters_t->at(base_ind_perc+n));
                        }else{
                            fprintf(out_stats,"%2.3f\n",perc_inters_t->at(base_ind_perc+n));
                        }
                    }
                }
                fclose(out_stats);
                
                
                // Output visitor payoffs over time
                FILE *out_payoffs_p1_t = fopen(out_file_payoffs_p1_t, "wb");
                
                if(out_payoffs_p1_t == NULL){
                    perror("Error");
                }
                
                
                for(int j=0; j<pop; j++)
                {
                    for(int i = 0; i < actions_timesteps; i++)
                    {
                        if(i < actions_timesteps - 1)
                        {
                            fprintf(out_payoffs_p1_t,"%2.4f,", (payoffs_p1_t->at(i*pop + j)));
                        }
                        else
                        {
                            fprintf(out_payoffs_p1_t,"%2.4f",(payoffs_p1_t->at(i*pop + j)));
                        }
                    }
                    fprintf(out_payoffs_p1_t,"\n");
                    
                }
                
                //Output host payoffs over time
                fclose(out_payoffs_p1_t);
                
                FILE *out_payoffs_p2_t = fopen(out_file_payoffs_p2_t, "wb");
                
                if(out_payoffs_p2_t == NULL){
                    perror("Error");
                }
                
                
                for(int j=0; j<pop*2; j=j+2)
                {
                    for(int i = 0; i < actions_timesteps; i++)
                    {
                        if(i < actions_timesteps - 1)
                        {
                            fprintf(out_payoffs_p2_t,"%2.4f,", (payoffs_p2_t->at(i*pop*2 + j)));
                            fprintf(out_payoffs_p2_t,"%2.4f,", (payoffs_p2_t->at(i*pop*2 + j + 1)));
                        }
                        else
                        {
                            fprintf(out_payoffs_p2_t,"%2.4f,",(payoffs_p2_t->at(i*pop*2 + j)));
                            fprintf(out_payoffs_p2_t,"%2.4f",(payoffs_p2_t->at(i*pop*2 + j + 1)));
                            
                        }
                    }
                    fprintf(out_payoffs_p2_t,"\n");
                    
                }
                
                fclose(out_payoffs_p2_t);
                
                
                
                
                delete adjacency_weights_t;
                //delete[] adjacency_weights;
                //delete[] adjacency_weights_new;
                
                //delete adjacency_weights_new;
                
                delete p1_strategy_t;
                delete p2_strategy_t;
                
                delete strength_var_t;
                delete p1_strat_var_t;
                delete p2_strat_var_t;
                delete p1_strat_mean_t;
                delete p2_strat_mean_t;
                delete strat_corr_t;
                
                
                //igraph_destroy(&agent_net);
                //igraph_vector_destroy(&degree_seq);
                //igraph_matrix_destroy(&agent_net2);
                
                //free(p1_strategy);
                //free(p2_strategy);
                
                //p1_strategy = NULL;
                //p2_strategy = NULL;
                
                
            }
            
            //free(network);
            free(init_strategy_str);
            free(init_net_str);
            free(game);
            free(key);
            free(sim_description);
            
            network = NULL;
            init_strategy_str = NULL;
            init_net_str = NULL;
            game = NULL;
            key = NULL;
            sim_description = NULL;
            
        }
#ifdef _OPENMP
    }
#endif
    
    
    /* Good practice to free memory */
    for(k; k >= 0; k--)
    {
        free(words[k]);
    }
    free(words);
    words = NULL;
    
    time_t endtime;
    endtime = time(NULL);
    
    float seconds = difftime(endtime,starttime);
    //float seconds_sim = difftime(simtime,starttime);
    
    printf ("%.f seconds total.\n", seconds);
    //printf ("%.f seconds to sim.\n", seconds_sim);
    
    
    return 0;
}

void run_evolution(int num_strats_p1,int num_strats_p2, std::vector<double>& adjacency_weights, std::vector<double>& adjacency_weights_new, UGenerator rng, int seed_id, int pop, float p1_payoff[], float p2_payoff[], std::vector<double>& p1_strategy, std::vector<double>& p2_strategy,std::vector<double>& p1_strategy_t, std::vector<double>& p2_strategy_t,std::vector<double>& adjacency_weights_t,std::vector<int>& player1_s1_t,std::vector<int>& player2_s1_t, float base, float death_rate, int max_time, float net_discount, float strat_discount, float net_learning_speed, float strat_learning_speed, int net_symmetric, int strat_symmetric, float net_tremble_prob, float strat_tremble_prob, std::vector<double>& strat_corr_t,std::vector<double>& perc_inters_t, std::vector<double>& p1_strat_var_t, std::vector<double>& p2_strat_var_t,std::vector<double>& p1_strat_mean_t, std::vector<double>& p2_strat_mean_t,std::vector<double>& strength_var_t, std::vector<int>& stats_time, std::vector<double>& payoffs_p1_t,std::vector<double>& payoffs_p2_t)
/* Single simulation code for evolutionary game theory model
 */
{
    std::vector<double> past_payoffs_p1(pop,0.0); // Track visitor payoffs
    std::vector<double> past_payoffs_p2(pop*2,0.0); // Track host payoffs and how many times a host got visited
    
    // Another way to do past payoffs.  Would need to keep track of what timesteps earned which payoffs though.  Vector (population) of vector (time) of vectors (payoffs?
    /*vector< vector<int> > vec;
     
     for (int i = 0; i < 10; i++) {
     vector<int> row; // Create an empty row
     for (int j = 0; j < 20; j++) {
     row.push_back(i * j); // Add an element (column) to the row
     }
     vec.push_back(row); // Add the row to the main vector
     }
     
     */
    
    // Initialize node strength
    std::vector<double> node_strength(pop, 0.0);;
    
    
    long int total_p1 = 0;
    long int total_p2 = 0;
    
    for(int i = 0; i < pop; i++)
    {
        for(int k = 0; k < num_strats_p1; k++){
            p1_strategy_t.push_back(p1_strategy.at(i*num_strats_p1 + k));
        }
        for(int k = 0; k < num_strats_p2; k++){
            p2_strategy_t.push_back(p2_strategy.at(i*num_strats_p2 + k));
        }
    }
    
    player1_s1_t.push_back(total_p1);
    player2_s1_t.push_back(total_p2);
    
    
    adjacency_weights_t.insert(adjacency_weights_t.end(), adjacency_weights.begin(), adjacency_weights.end());
    
    //std::vector<float> adjacency_weights_new = adjacency_weights;
    
    std::vector<double> p1_strategy_new = p1_strategy;
    std::vector<double> p2_strategy_new = p2_strategy;
    
    std::vector<int> agent_seq;
    for(int iter = 0; iter < pop; iter++)
    {
        agent_seq.push_back(iter);
    }
    
    // Run simulation for max_time timesteps
    for (int t = 1; t < max_time+2 && (std::accumulate(p1_strategy.begin(),p1_strategy.end(),0.0) > 0); t++)
    {
        // Update agent location at eac, h time step
        std::fill(past_payoffs_p1.begin(),past_payoffs_p1.end(),0.0);
        std::fill(past_payoffs_p2.begin(),past_payoffs_p2.end(),0.0);
        
        // Run simulation for one time step, loop through all agents once
        one_time_step(num_strats_p1,num_strats_p2, adjacency_weights, adjacency_weights_new, t, rng, pop, p1_payoff, p2_payoff, p1_strategy, p1_strategy_new, p2_strategy, p2_strategy_new, total_p1, total_p2, base, net_discount, strat_discount, net_learning_speed, strat_learning_speed, net_symmetric, strat_symmetric, net_tremble_prob, strat_tremble_prob, agent_seq, past_payoffs_p1,past_payoffs_p2);
        
        //if(t < time_tracker1 || (t > time_tracker2 && t < time_tracker3)){
        if(t <= 100 || (t <= 1000 && t%50 == 0) || t == 5000 || t == 10000 || t == 50000 || t == 100000 || t == 1000000 || t == 2000000 || t == 5000000 || t == 10000000 || t == 20000000 || t == 100000000 || t == 500000000 || t == 1000000000 || (t >= time_tracker2 && t <= time_tracker3)){
            p1_strategy_t.insert(p1_strategy_t.end(), p1_strategy.begin(), p1_strategy.end());
            p2_strategy_t.insert(p2_strategy_t.end(), p2_strategy.begin(), p2_strategy.end());
            
            //for(int i = 0; i < pop*num_strats; i++)
            //{
            //    p1_strategy_t.push_back(p1_strategy.at(i));
            //    p2_strategy_t.push_back(p2_strategy.at(i));
            //}
            
            player1_s1_t.push_back(total_p1);
            player2_s1_t.push_back(total_p2);
            
            //if(t == 25 || t == 50 || t == 100 || t == 1000 || t == 10000 || t == 100000 || t == 1000000 || t == 2000000 || t == 5000000 || t == 10000000 || t == 100000000 || t == 500000000 || t == 1000000000)
            //{
            adjacency_weights_t.insert(adjacency_weights_t.end(), adjacency_weights.begin(), adjacency_weights.end());
            
            payoffs_p1_t.insert(payoffs_p1_t.end(),past_payoffs_p1.begin(),past_payoffs_p1.end());
            payoffs_p2_t.insert(payoffs_p2_t.end(),past_payoffs_p2.begin(),past_payoffs_p2.end());
            
            //}
            
            
            //    for(int i = 0; i < pop*pop; i++)
            //    {
            //        adjacency_weights_t.push_back(adjacency_weights.at(i));
            //    }
            
            
        }
        
        // Record keeping
        if((max_time <= 5000000 && t % 100 == 0) || (max_time > 5000000 && t % 10000 == 0) || t < time_tracker1 || (t > time_tracker2 && t < time_tracker2 + 1000)){
            
            std::vector<double> assort(num_strats_p1*num_strats_p2,0.0);
            
            //gsl_vector_const_view gsl_p1 = gsl_vector_const_view_array( &p1_strategy[0], (size_t) p1_strategy.size() );
            //gsl_vector_const_view gsl_p2 = gsl_vector_const_view_array( &p2_strategy[0], (size_t) p2_strategy.size() );
            
            //double strat_corr = gsl_stats_correlation( (double*) gsl_p1.vector.data, 1,
            //                                       (double*) gsl_p2.vector.data, 1,
            //p1_strategy.size());
            
            std::vector<double> strat_tracker(pop);
            std::vector<double> p1_strat_sums(pop);
            std::vector<double> p2_strat_sums(pop);
            
            
            // Get mean and variance of visitor and host strategies
            double p1_strat_var;
            double p1_strat_mean;
            double p2_strat_mean;
            double p2_strat_var;
            
            for(int p1s_ind = 0; p1s_ind < num_strats_p1;p1s_ind++){
                for(int pop_ind = 0; pop_ind < pop; pop_ind++){
                    p1_strat_sums.at(pop_ind) = std::accumulate(p1_strategy.begin() + pop_ind*num_strats_p1, p1_strategy.begin() + pop_ind*num_strats_p1 + num_strats_p1,0.0);
                    strat_tracker.at(pop_ind) = p1_strategy.at(pop_ind * num_strats_p1 + p1s_ind)/p1_strat_sums.at(pop_ind);
                }
                p1_strat_var = gsl_stats_variance(&strat_tracker[0], 1, (size_t) strat_tracker.size());
                p1_strat_mean  = gsl_stats_mean(&strat_tracker[0], 1, (size_t) strat_tracker.size());
                
                p1_strat_var_t.push_back(p1_strat_var);
                p1_strat_mean_t.push_back(p1_strat_mean);
            }
            
            for(int p2s_ind = 0; p2s_ind < num_strats_p2;p2s_ind++){
                for(int pop_ind = 0; pop_ind < pop; pop_ind++){
                    p2_strat_sums.at(pop_ind) = std::accumulate(p2_strategy.begin() + pop_ind*num_strats_p2, p2_strategy.begin() + pop_ind*num_strats_p2 + num_strats_p2,0.0);
                    strat_tracker.at(pop_ind) = p2_strategy.at(pop_ind * num_strats_p2 + p2s_ind)/p2_strat_sums.at(pop_ind);
                }
                p2_strat_var = gsl_stats_variance(&strat_tracker[0], 1, (size_t) strat_tracker.size());
                p2_strat_mean  = gsl_stats_mean(&strat_tracker[0], 1, (size_t) strat_tracker.size());
                
                p2_strat_var_t.push_back(p2_strat_var);
                p2_strat_mean_t.push_back(p2_strat_mean);
            }
            
            // Calculate proportion of interactions between all strategy pairs
            std::fill(node_strength.begin(),node_strength.end(),0.0);
            
            double sum_row;
            int strat_ind_p1;
            int strat_ind_p2;
            for(int mat_row = 0; mat_row < pop; mat_row++){
                sum_row = std::accumulate(adjacency_weights.begin()+ mat_row*pop, adjacency_weights.begin()+mat_row*pop + pop, 0.0);
                for(int mat_col = 0; mat_col < pop; mat_col++){
                    node_strength.at(mat_col) = node_strength.at(mat_col) + adjacency_weights.at(mat_row * pop + mat_col)/sum_row;
                    
                    for(int strat_ind = 0; strat_ind < num_strats_p1 *num_strats_p2; strat_ind++){
                        strat_ind_p1 = strat_ind/num_strats_p2;
                        strat_ind_p2 = strat_ind%num_strats_p2;
                        
                        assort.at(strat_ind) += (adjacency_weights.at(mat_row * pop + mat_col)/(sum_row) * p1_strategy.at(num_strats_p1*mat_row+strat_ind_p1)/p1_strat_sums.at(mat_row) * p2_strategy.at(num_strats_p2*mat_col+strat_ind_p2)/p2_strat_sums.at(mat_col))/pop;
                    }
                }
            }
            
            gsl_vector_const_view gsl_ns = gsl_vector_const_view_array( &node_strength[0], node_strength.size() );
            double strength_var = gsl_stats_variance ((double*) gsl_ns.vector.data, 1, (size_t) node_strength.size());
            
            //strat_corr_t.push_back(strat_corr);
            strength_var_t.push_back(strength_var); 
            perc_inters_t.insert(perc_inters_t.end(), assort.begin(), assort.end());
            stats_time.push_back(t);
        }
        
        
    }
    
    
}

void one_time_step(int num_strats_p1,int num_strats_p2, std::vector<double>& adjacency_weights, std::vector<double>& adjacency_weights_new, int t, UGenerator rng, int pop, float p1_payoff[], float p2_payoff[], std::vector<double>& p1_strategy, std::vector<double>& p1_strategy_new, std::vector<double>& p2_strategy, std::vector<double>& p2_strategy_new, long int& total_p1, long int& total_p2, float base, float net_discount, float strat_discount, float net_learning_speed, float strat_learning_speed, int net_symmetric, int strat_symmetric, float net_tremble_prob, float strat_tremble_prob, std::vector<int>& agent_seq, std::vector<double>& past_payoffs_p1,std::vector<double>& past_payoffs_p2)
/* Run simulation for one time step.  Loops through all agents.  Agents interact randomly with neighbors according to
 the link weight between them and each neighbor.  Once an interaction occurs, the links are updated according to some rule.
 Further, agent types are updated according to a given payoff matrix (UL, UR, BL, BR)
 
 Inputs agent adjacency matrix (network), nk space, agent initial location
 Outputs updated agent locations
 */
{
    
    total_p1 = 0;
    total_p2 = 0;
    
    
    // Shuffle agents in random order (updating is synchronous anyways so this only serves as another layer of randomness)
    
    std::random_shuffle(agent_seq.begin(), agent_seq.end());
    
    //////////////////////////////////////////
    
    int agent;
    
    // Loop through agents
    for(int agent_num = 0; agent_num < pop; agent_num++)
    {
        
        //printf("%d\n",agent_num);
        
        agent = agent_seq.at(agent_num); // Current agent (shuffled order)
        
        std::vector<int> temp_agent_seq(agent_seq);
        
        temp_agent_seq.erase(std::remove(temp_agent_seq.begin(), temp_agent_seq.end(), agent), temp_agent_seq.end());
        
        //printf("%d\n",agent);
        //for (auto const& c : temp_agent_seq)
        //    std::cout << c << ' ';
        
        //printf("\n");
        
        
        /////////////////////////////////
        
        /*
         
         Loop through all agents, choose interaction partner according to network weights
         
         */
        
        /////////////////////////////////
        
        int friend_ind = -1; // Flag (goes >= 0 as the index) for when the neighbor is picked
        int nid;
        float rand_tremble = rng();
        
        // If agent doesn't make an error
        if(rand_tremble > net_tremble_prob){
            std::vector<double> sum_vec(pop);
            //float sum_vec[pop];
            
            std::partial_sum(adjacency_weights.begin()+agent*pop, adjacency_weights.begin()+agent*pop+pop, sum_vec.begin());
            
            //printf("%f\n",adjacency_weights.at(agent*pop));
            
            //printf("%f\n",sum_vec[pop-1]);
            
            double interaction_random_draw = rng() * sum_vec[pop-1];
            
            //auto up = std::upper_bound(sum_vec.begin(), sum_vec.end(), interaction_random_draw);
            
            //int friend_ind = (up - sum_vec.begin());
            
            
            //Iterate over neighbors
            
            for(nid = 0; nid < pop; nid++)
            {
                // Add viable neighbor
                // If it's time to copy
                
                if(friend_ind == -1 && interaction_random_draw <= sum_vec.at(nid) && nid != agent){
                    friend_ind = nid; // This is agents partner
                }
                // Discount neighbors
                adjacency_weights_new.at(agent*pop + nid) = (adjacency_weights_new.at(agent*pop + nid)) * net_discount;
            }
        }else{ // If agent makes an error
            for(nid = 0; nid < pop; nid++)
            {
                // Discount all neighbors
                adjacency_weights_new.at(agent*pop + nid) = (adjacency_weights_new.at(agent*pop + nid)) * net_discount;
            }
            // Choose random neighbor
            int temp_ind = (int) (rng() * (pop - 1));
            friend_ind = temp_agent_seq.at(temp_ind);
            
        }
        
        adjacency_weights_new.at(agent*pop + agent) = 0;
        
        // Initialize payoffs for visitor (utility[0]) and host (utility[1])
        float utility[2];
                
        // Initialize cumulative sum of strategy weights
        std::vector<double> stratp1_sum_vec(num_strats_p1);
        std::vector<double> stratp2_sum_vec(num_strats_p2);
        
        std::partial_sum(p1_strategy.begin()+agent*num_strats_p1, p1_strategy.begin()+agent*num_strats_p1+num_strats_p1, stratp1_sum_vec.begin());
        std::partial_sum(p2_strategy.begin()+friend_ind*num_strats_p2, p2_strategy.begin()+friend_ind*num_strats_p2+num_strats_p2, stratp2_sum_vec.begin());
        
        //printf("%f\n",adjacency_weights.at(agent*pop));
        
        //printf("%f\n",sum_vec[pop-1]);
        
        //Draw random numbers to determine strategy for host and visitor
        double stratp1_draw = rng() * stratp1_sum_vec[num_strats_p1-1];
        double stratp2_draw = rng() * stratp2_sum_vec[num_strats_p2-1];
        
        //auto up = std::upper_bound(sum_vec.begin(), sum_vec.end(), interaction_random_draw);
        
        //int friend_ind = (up - sum_vec.begin());
        
        
        //Iterate over neighbors, apply discount to current weights and choose partner to play against
        int p1_strat_ind = -1;
        int p2_strat_ind = -1;
        
        // This all only applies when number of host and visitor strategies are different (not in hawk dove)
        int max_strats;
        
        if(num_strats_p2 >= num_strats_p1){
            max_strats = num_strats_p2;
        }else{
            max_strats = num_strats_p1;
        }
        
        // Discount strategies of agent and friend
        for(int strat_ind = 0; strat_ind < max_strats; strat_ind++)
        {
            // Add viable neighbor
            // If it's time to copy
            
            if(p1_strat_ind == -1 && stratp1_draw <= stratp1_sum_vec.at(strat_ind)){
                p1_strat_ind = strat_ind;
            }
            if(p2_strat_ind == -1 && stratp2_draw <= stratp2_sum_vec.at(strat_ind)){
                p2_strat_ind = strat_ind;
            }
            if(strat_ind < num_strats_p1){
                p1_strategy_new.at(agent*num_strats_p1 + strat_ind) = (p1_strategy_new.at(agent*num_strats_p1 + strat_ind)) * strat_discount;
            }
            if(strat_ind < num_strats_p2){
                p2_strategy_new.at(friend_ind*num_strats_p2 + strat_ind) = (p2_strategy_new.at(friend_ind*num_strats_p2 + strat_ind)) * strat_discount;
            }
        }
        
        float p1_rand_strat = rng();
        float p2_rand_strat = rng();
        
        if(p1_rand_strat < strat_tremble_prob){
            p1_strat_ind = (int) (rng() * (num_strats_p1 - 1));
        }
        if(p2_rand_strat < strat_tremble_prob){
            p2_strat_ind = (int) (rng() * (num_strats_p2 - 1));
        }
        
        //printf("Player 1 Draw: %lf, Player 2 Draw: %lf\n",p1_strategy_draw,p2_strategy_draw);
        //printf("Player 1 Strategy: %d, Player 2 Strategy: %d\n",p1_strat_ind,p2_strat_ind);
        
        // See above for payoff matrix explanation: 0x0 is index 0, 0x1 is index 1, 1x0 index 2, 1x1 index 3
        int payoff_index = p1_strat_ind * num_strats_p2 + (p2_strat_ind);
        
        if(p1_strat_ind == 0)
        {
            total_p1 += 1;
        }
        if(p2_strat_ind == 0)
        {
            total_p2 += 1;
        }
        //printf("Payoff index: %d\n", payoff_index);
        
        /*
         Interact
         */
        
        //printf("%f\n%f\n",p1_payoff[payoff_index] + base,p2_payoff[payoff_index] + base);
        
        //Set payouts for each player
        utility[0] = p1_payoff[payoff_index] + base;
        utility[1] = p2_payoff[payoff_index] + base;
        
        past_payoffs_p1.at(agent) += utility[0]; // Track payoff for visitor
        past_payoffs_p2.at(friend_ind*2) += utility[1]; // Track payoff for host
        past_payoffs_p2.at(friend_ind*2 + 1) += 1; // Add one host interaction
        
        /*
         Update network weights
         */
        
        adjacency_weights_new.at(agent*pop + friend_ind) = adjacency_weights_new.at(agent*pop + friend_ind) + net_learning_speed * utility[0];
        
        if(net_symmetric == 1) // This mean that agents partner updates their network weights as well
        {
            int friend_id;
            for(friend_id = 0; friend_id < pop; friend_id++){
                adjacency_weights_new.at(friend_ind*pop + friend_id) = (adjacency_weights_new.at(friend_ind*pop + friend_id)) * net_discount;
            }
            adjacency_weights_new.at(friend_ind*pop + agent) = (adjacency_weights_new.at(friend_ind*pop + agent)) + net_learning_speed * utility[1];
            adjacency_weights_new.at(friend_ind*pop + friend_ind) = 0;
        }
        //print_as_matrix(p1_strategy_new,pop,2);
        
        /*
         Update Strategy of visitor and host
         */        
        
        p1_strategy_new.at(agent*num_strats_p1 + p1_strat_ind) = (p1_strategy_new.at(agent*num_strats_p1 + p1_strat_ind)) + strat_learning_speed * utility[0]; // Update player 1 strategy played
        p2_strategy_new.at(friend_ind*num_strats_p2 + p2_strat_ind) = (p2_strategy_new.at(friend_ind*num_strats_p2 + p2_strat_ind)) + strat_learning_speed * utility[1]; // Update strategy played (player 2)
        
        if(strat_symmetric == 1){  // If strategy is symmetric, agents are forced to have the same host and visitor strategies
            for(int strat_ind = 0; strat_ind < num_strats_p1; strat_ind++){
                p1_strategy_new.at(friend_ind*num_strats_p1 + strat_ind) = (p1_strategy_new.at(friend_ind*num_strats_p1 + strat_ind)) * strat_discount;
            }
            p1_strategy_new.at(friend_ind*num_strats_p1 + p1_strat_ind) = p1_strategy_new.at(friend_ind*num_strats_p1 + p1_strat_ind) + strat_learning_speed * utility[1];
            
        }
        //printf("\n");
        //printf("Utility P1: %lf, Utility P2: %lf\n",utility[0],utility[1]);
        //printf("\n");
        
        
    }
    
    // Update current adjacency weights and strategy weights to use for next time step
    adjacency_weights = adjacency_weights_new;
    p1_strategy = p1_strategy_new;
    p2_strategy = p2_strategy_new;
}

int * intdup(int const * src, size_t len)
{
    int * p = (int*) malloc(len * sizeof(int));
    memcpy(p, src, len * sizeof(int));
    return p;
}

long * longdup(long const * src, size_t len)
{
    long * p = (long*) malloc(len * sizeof(long));
    memcpy(p, src, len * sizeof(long));
    return p;
}

float * floatdup(float const * src, size_t len)
{
    float * p = (float*) malloc(len * sizeof(float));
    memcpy(p, src, len * sizeof(float));
    return p;
}

void print_matrix(long int matrix[],int pop)
{
    for (int row=0; row<pop; row++)
    {
        for(int columns=0; columns<pop; columns++)
        printf("%ld ", matrix[row * pop + columns]);
        printf("\n");
    }
    
    printf("\n\n");
}

void print_as_matrix(float matrix[],int row_len, int col_len)
{
    for (int row=0; row<row_len; row++)
    {
        for(int columns=0; columns<col_len; columns++)
        printf("%lf ", matrix[row * col_len + columns]);
        printf("\n");
    }
    
    printf("\n\n");
}

void print_dubarray(float array[],int size)
{
    int i;
    for(i = 0; i < size; i++)
    {
        printf("%lf\n",array[i]);
    }
    printf("\n");
}

void print_intarray(long int array[],int size)
{
    int i;
    for(i = 0; i < size; i++)
    {
        printf("%ld\n",array[i]);
    }
    printf("\n");
}



