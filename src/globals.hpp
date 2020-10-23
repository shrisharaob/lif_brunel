#ifndef _GLOBAL_VARS
#define _GLOBAL_VARS

//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
//
#include <iostream> // required for libconfig++
#include <iomanip>
#include <fstream>
#include <libconfig.h++>
#include <vector>
#include <random>
#include <chrono>
//


/* #include "libconfig.h" */

// network size and connectivity
unsigned int  NE, NI, N, K, K_I;
double sqrt_K;

// // connectivity
// unsigned int IS_RANDOM=1;
// double alpha = 0.0;

//
unsigned long long int n_connections = 0;
// , n_stdp_connections=0; // stdp only in e-to-e synapses

// simulation params
double dt, t_stop, cur_t;
size_t n_steps = 0;

// lif state vectors
double *Vm, *V_threshold;

// lif params
double V_rest, V_threshold_initial, V_reset;
double refractory_period=0.0;
// double d_threshold; // if a neuron spikes, increase its threshold by this value
double *last_spike_time; // time of the latest spike of a neuron

double tau_membrane; // , tau_threshold;
double *g_e, *g_i;

// synaptic strengths
double g_brunel = 0;
double J, Je0, Ji0, Jee, Jei, Jie, Jii;

// transmission delay
double delay_syn;
int n_delay_bins=1;
int **syn_delay_buffer=NULL;

// external input 
double v_ext;
std::vector<double> I_of_t;

std::default_random_engine *poiss_generator_e, *poiss_generator_i;

//
unsigned int *nPostNeurons = NULL, *sparseConVec = NULL, *idxVec = NULL;
unsigned int *n_pre_neurons = NULL, *pre_sparseVec = NULL, *pre_idxVec = NULL;

//
unsigned int n_spikes[] = {0, 0};
double pop_rate_e = 0, pop_rate_i = 0;
double discard_time=0.0;

//
FILE *spk_fp;

// stp
// int STP_ON = 0;
// double stp_U, stp_A, stp_tau_d, stp_tau_f;
// /* double *u_plus, *u_minus, *u, *x, *x_minus; */
// double *stp_u, *stp_x_old, *stp_x;

// // STDP
// int test_stdp = 0;
// int STDP_ON = 0;
// double *stdp_pre_trace, *stdp_post_trace;
// // stdp weights are scaled as 1 / K
// // and they are indexed by connection number
// double *stdp_weights; // *stdp_pre_post_spk_tracking;
// unsigned int *IS_STDP_CON; // 1 if STDP CON, used to implement sqrt(K) stdp connections 

// double stdp_initial_weights=.01, stdp_tau_pre=100.0, stdp_tau_post=200.0;
// double stdp_lr_pre=0.05, stdp_lr_post=0.05, stdp_max_weight=1.0;

// unsigned int temp_stdp_con_idx=0;

#endif
