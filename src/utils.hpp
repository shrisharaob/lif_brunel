#ifndef _UTILS
#define _UTILS
#include <vector>
#include "globals.hpp"
//

using namespace libconfig;
void print_matrix(int **mat, int row, int clmn) {
  std::cout << "printing matrix "  << "\n";
  for(int i =0; i < row; ++i) {
    for(int j = 0; j < clmn; ++j) {
      std::cout << mat[i][j] << " ";
    }
    std::cout << "\n";
  }
}


// ----------------------------------------- //
unsigned int* int_vector(unsigned int n, unsigned int init_val) {
  unsigned int *x = (unsigned int *) malloc(n * sizeof(unsigned int));
  // init to zero
  for(unsigned int i=0; i < n; ++i) {
    x[i] = init_val;
  }
  return x;
}

double* double_vector(unsigned int n, double init_val) {
  double *x = (double *) malloc(n * sizeof(double));
  for(unsigned int i=0; i < n; ++i) {
    x[i] = init_val;
  }
  return x;
}

double* double_vector(unsigned long long n, double init_val) {
  double *x = (double *) malloc(n * sizeof(double));
  for(unsigned int i=0; i < n; ++i) {
    x[i] = init_val;
  }
  return x;
}

void vector_divide(std::vector<double> &a, double z) { 
  for(unsigned int i = 0; i < N; ++i) {
    a[i] /= z;
  }
}

void vector_sum(std::vector<double> &a, std::vector<double> &b) { 
  for(unsigned int i = 0; i < N; ++i) {
    a[i] += b[i];
  }
}

void vector_init_to_zero(double *a) {
  for(unsigned int i = 0; i < N; ++i) {
    a[i] = 0;
  }
}


void shift_matrix(int **mat, int rows, int clmns) {
  // shift matrix to the left by one and set the last element to zero
  int row = 0;
  while (row < rows) {
    for(int clmn = 1; clmn < clmns; ++clmn) {
      mat[row][clmn - 1] = mat[row][clmn];
    }
    mat[row][clmns-1] = 0;
    row += 1; 
  }
}

int** create_2d_matrix(int rows, int clmns, int init_val) {
  int **arr = new int* [rows];
  // arr = new int* [rows];
  for(int i = 0; i < rows; ++i) {
    arr[i] = new int [clmns];
  }
  // initialize
  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < clmns; ++j) {
      arr[i][j] = init_val;
    }
  }
  std::cout << "allocation done" << "\n";
  return arr;
}

void clear_2d_matrix(int **arr, int rows, int clmns) {
  for(int i = 0; i < rows; ++i) {
    delete [] arr[i];
  }
  delete [] arr;
}


void get_ff_input(const char *filename) {
  double x;
  std::ifstream inFile;
  inFile.open(filename);
  std::cout << "reading time dependent ff input"  << "\n";
  // inFile.open("test.txt");
  if (!inFile) {
    std::cout << "Unable to open file \n";
    std::cout << "setting time varying ff input to zero \n";
    for(size_t i = 0; i < n_steps; ++i){
      I_of_t.push_back(0.0);
    }
  }
  while (inFile >> x) {
    I_of_t.push_back(x);
  }
  inFile.close();
}

void GenSparseMat(unsigned int *conVec, unsigned int rows, unsigned int clms,
		  unsigned int* sparseVec, unsigned int* idxVec,
		  unsigned int* nPostNeurons ) {
  /* generate sparse representation
     conVec       : input vector / flattened matrix 
     sparseVec    : sparse vector
     idxVec       : every element is the starting index in sparseVec
                    for ith row in matrix conVec
     nPostNeurons : number of non-zero elements in ith row 
  */
  // printf("\n MAX Idx of conmat allowed = %u \n", rows * clms);
  unsigned long long int i, j, counter = 0, nPost;
  for(i = 0; i < rows; ++i) {
    nPost = 0;
    for(j = 0; j < clms; ++j) {
      // printf("%llu %llu %llu %llu\n", i, j, i + clms * j, i + rows * j);
      if(conVec[i + rows * j]) { /* i --> j  */
        sparseVec[counter] = j;
        counter += 1;
        nPost += 1;
      }
    }
    nPostNeurons[i] = nPost;
  }
  
  idxVec[0] = 0;
  for(i = 1; i < rows; ++i) {
    idxVec[i] = idxVec[i-1] + nPostNeurons[i-1];
  }
}

void gen_conmat() {
  // connection matrix (i, j) = i + j * n,
  // i is pre,  j is post
  // e.g, 
  // 1 3 6
  // 2 4 7
  // 3 5 8
  
  const gsl_rng_type * T;
  gsl_rng *r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  double u_rand; // uniform random number
  double con_prob = 0; // (double) K / NE;
  double con_prob_E=0, con_prob_I=0;
  con_prob_E = (double) K / NE;
  con_prob_I = (double) K_I / NI;
  //
  n_connections = 0;
  unsigned int *conmat = int_vector(N * N, 0);;
  unsigned int n_ee_cons = 0, n_ei_cons = 0;
  for(size_t i=0; i < N; ++i) {
    if (i < NE) {
      con_prob = con_prob_E;
    }
    else {
      con_prob = con_prob_I;
    }
    for(size_t j=0; j < N; ++j) {
      u_rand = gsl_rng_uniform(r);
      if(u_rand <= con_prob){
	conmat[i + N * j] = 1;
	n_connections += 1;
	if(i < NE && j < NE) {
	  n_ee_cons += 1;
	}

	if(i >= NE && j >= NE) {
	  n_ei_cons += 1;
	}
	
      }
    }
  }

  // remove self connections
  for(size_t i = 0; i < N; ++i) {
    conmat[i + N * i] = 0;
  }

  std::cout << " connections done!  "  << "\n"; 
  // std::cout << "n cons = " << (double)n_connections  << "\n";
  // std::cout << "c prob = " << con_prob  << "\n";
  // std::cout << "n ee cons = " << n_ee_cons << "\n";
  // std::cout << "n ee cons per cell = " << (double)n_ee_cons / NE << "\n";
  // std::cout << "n ii cons per cell = " << (double)n_ei_cons / NI << "\n";      
  sparseConVec = int_vector(n_connections, 0);
  //
  GenSparseMat(conmat, N, N, sparseConVec, idxVec, nPostNeurons);
  //
  free(conmat);
  gsl_rng_free(r);
}


int read_params() {
  Config cfg;

  // Read the file. If there is an error, report it and exit.
  try {
    cfg.readFile("params.cfg");
  }
  catch(const FileIOException &fioex)
    {
      std::cerr << "I/O error while reading file." << std::endl;
      return(EXIT_FAILURE);
    }
  catch(const ParseException &pex)
    {
      std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
		<< " - " << pex.getError() << std::endl;
      return(EXIT_FAILURE);
    }

  const Setting& root = cfg.getRoot();
  // set params
  // network size
  NE = root["NE"];
  NI = root["NI"];
  N = NE + NI;
  K = root["K"];
  K_I = root["K_I"];

  tau_membrane = root["tau_mem"];  
  refractory_period = root["refractory_period"];
  // Brunel parameters
  delay_syn = root["delay_syn"];
  g_brunel = root["g_brunel"];
  J = root["J"];
  V_threshold_initial = root["V_threshold_initial"];  
  v_ext = root["v_ext"];  // input rates
  // recurrent weights
  Jee = J;  
  Jei = -1.0 * g_brunel * J;
  Jie = J;
  Jii = -1.0 * g_brunel * J;
  // feedforward weights
  Je0 = J;
  Ji0 = J;
  //
  // d_threshold = root["d_threshold"];
  V_reset = root["V_reset"];
  // simulation time
  t_stop = root["t_stop"];
  dt = root["dt"];
  discard_time = root["discard_time"];
  t_stop += discard_time;

  // display params 
  std::cout << "-- -- -- --- Network --- -- -- --" << "\n";
  std::cout << "NE = " << NE << "\n";
  std::cout << "NI = " << NI << "\n";
  std::cout << "CE = " << K << "\n";
  std::cout << "CI = " << K_I << "\n";  
  std::cout << "V_reset = " << V_reset << "mV" << "\n";
  std::cout << "-- -- --- brunel params --- -- --" << "\n";
  std::cout << "transmission delay D  = " << delay_syn << "ms" << "\n";
  std::cout << "J = " << J << "mV" << "\n";
  std::cout << "g = " << g_brunel << "\n";
  std::cout << "v_ext = " << v_ext * 1e3 << "Hz" << "\n";    
  std::cout << "firing threshold theta = " <<
    V_threshold_initial  << "mV" << "\n";
  double v_thresh_brunel = V_threshold_initial / (Je0 * K * tau_membrane);
  std::cout <<  "v_ext / v_thresh = " << v_ext / v_thresh_brunel << "\n";
  std::cout << "-- -- --- ~~~~~~ ~~~~~~ --- -- --" << "\n";

  // //
  // std::cout << "-- --- recurrent interactions --- --" << "\n";
  // std::cout << "  " << "\n";
  // std::cout << Jee << " " << Jei << "\n";
  // std::cout << Jie << " " << Jii << "\n";
  // std::cout << " " << "\n";
  // std::cout << "-- -- - --- FF strength --- -- -- --" << "\n";  
  // std::cout << Je0 << "\n" << Ji0 << "\n";
  // std::cout << "-- -- ---- ~~~~~ ~~~~~ ~~~~~~ --- -- --" << "\n";
  

  
  return 0;
}

FILE* push_spike_init() {
  FILE *spk_fp = fopen("./data/spikes.txt", "w");
  return spk_fp;
}


void setup_poiss_generators() {
  // each neuron has a E and I external poisson generator
  poiss_generator_e = new std::default_random_engine[N];
  // poiss_generator_i = new std::default_random_engine[N];  
  for(size_t i = 0; i < N; ++i) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    // std::default_random_engine tmp (seed);
    // poiss_generator_i[i] = tmp;
    //
    std::default_random_engine tmpe (seed);
    poiss_generator_e[i] = tmpe;
    
  }
}

void init_state_vectors() {
  // this function must be called after calling read_params
  Vm = double_vector(N, 0.0);
  V_threshold = double_vector(N, V_threshold_initial);  
  g_e = double_vector(N, 0); 
  g_i = double_vector(N, 0);
  nPostNeurons = int_vector(N, 0);
  idxVec = int_vector(N, 0);
  spk_fp = push_spike_init();
  // assume that all neurons fired at t0 - refractory_period
  last_spike_time = double_vector(N, -1.0 * refractory_period);
  // buffer to store previous spikes upto (t - D)
  n_delay_bins = (int) (delay_syn / dt);
  std::cout << "# delay bins " << n_delay_bins << "\n";
  syn_delay_buffer = create_2d_matrix(N, n_delay_bins, 0);
  setup_poiss_generators();
}

void delete_state_vectors() {
  // call from main
  free(Vm);
  free(V_threshold);
  free(g_i);
  free(g_e);
  free(nPostNeurons);
  free(idxVec);
  free(sparseConVec);
  free(last_spike_time);
  //
  delete [] poiss_generator_i;
  delete [] poiss_generator_e;  
  //
  clear_2d_matrix(syn_delay_buffer, N, n_delay_bins);
  //
  fflush(spk_fp);
  fclose(spk_fp);
}


void push_spike(double spike_time, unsigned int neuron_idx) {
  // save one excitatory spikes 
  if(neuron_idx < NE) {
    fprintf(spk_fp, "%f %u\n", spike_time, neuron_idx);
  }
}

void propagate_spikes_brunel(unsigned int pre_neuron_idx) {
  // update all the post synaptic neurons to presynaptic neuron
  unsigned int tmpIdx, cntr, post_neuron_idx;
  cntr = 0;
  tmpIdx = idxVec[pre_neuron_idx];
  if (pre_neuron_idx < NE){
    n_spikes[0] += 1;    
  }
  else {
    n_spikes[1] += 1;
  }
  while(cntr < nPostNeurons[pre_neuron_idx]) {
    post_neuron_idx = sparseConVec[tmpIdx + cntr];
    cntr += 1;
    // update post synaptic neuron only after the refractory period
    if(cur_t >= last_spike_time[post_neuron_idx] + refractory_period)  { 
      if(pre_neuron_idx < NE) {
	/* --    E-to-E    -- */      
	if(post_neuron_idx < NE) {
	  g_e[post_neuron_idx] += Jee; 
	}
	/* --    E-to-I    -- */      
	else {
	  g_e[post_neuron_idx] += Jie; 
	}
      }
      else {
	/* --    I-to-E    -- */      
	if(post_neuron_idx < NE) {
	  g_i[post_neuron_idx] += Jei; 
	}
	/* --    I-to-I    -- */      
	else {
	  g_i[post_neuron_idx] += Jii; 
	}
      }
    }
  }
}

void get_pop_rates(double time_interval) {
  /* if(cur_t > discard_time) { */
  pop_rate_e = (double)n_spikes[0] / (time_interval * NE);
  pop_rate_i = (double)n_spikes[1] / (time_interval * NI);
  /* } */
  n_spikes[0] = 0;
  n_spikes[1] = 0;
}


void detect_spikes_brunel(double t) {
  // detect if Vm > V_threshold and add 1 to g_x vector
  for (unsigned int neuron_idx = 0; neuron_idx < N; ++neuron_idx) {
    //
    if(syn_delay_buffer[neuron_idx][0]) {
      // propagate spikes after delay period
      propagate_spikes_brunel(neuron_idx);
    }
    // if(Vm[neuron_idx] >= V_threshold_[neuron_idx]) {
    if(Vm[neuron_idx] >= V_threshold_initial) {
      Vm[neuron_idx] = V_reset;
      // required to implement ref
      last_spike_time[neuron_idx] = t; 
      // update threshold
      // V_threshold[neuron_idx] += d_threshold;
      syn_delay_buffer[neuron_idx][n_delay_bins - 1] += 1;
      if (t > discard_time) {
	push_spike((t - discard_time) * 1e-3, neuron_idx);
      }
    }
  }
}


void ProgressBar(float progress, float me, float mi) {
  int barWidth = 31;
  std::cout << "Progress: [";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)  std::cout << "\u25a0"; //std::cout << "=";
    else std::cout << " ";
  }
  std::cout << std::fixed;    
  std::cout << "] " << int(progress * 100.0) << "% done | << t = " <<
    std::setprecision(2) << cur_t * 1e-3 << " mE = " << std::setprecision(2) <<
    me << " mI = " << std::setprecision(2) << mi << "\r";
  std::cout.flush();
  if(progress == 1.) {
    //      std::cout << std::endl;
    std::cout << std::fixed;    
    std::cout << "] " << int(progress * 100.0) << "% done | << t = " <<
      std::setprecision(2) << cur_t * 1e-3 << " mE = " <<
      std::setprecision(2) << me << " mI = " << std::setprecision(2) << mi <<
      std::endl;
    std::cout.flush();
  }
}


bool AreSame(float a, float b) {
  return fabs(a - b) < dt;
}



void integrate_brunel() {

  n_steps = (size_t) (t_stop / dt); // t_stop is set in read_params func to tstop + discard_time
  size_t n_discard_steps = (size_t) discard_time / dt;
  get_ff_input("./data/ff_input.txt");
  double dt_over_tau; //,  dt_over_tau_th;
  dt_over_tau = dt / tau_membrane;

  std::cout << " -- -- --- ~~~~~ --- -- -- "  << "\n";  
  std::cout << "Euler Integration"  << "\n";
  std::cout << "dt = " << dt << "\n";
  std::cout << "n steps = " << n_steps  << "\n";  
  //
  FILE *fpvm = fopen("./data/vm.txt", "w");
  FILE *fpip = fopen("./data/input.txt", "w");
  FILE *fp_rates_i = fopen("./data/pop_rates_i.txt", "w");
  FILE *fp_rates_e = fopen("./data/pop_rates_e.txt", "w");  
  // double cur_t;
  double time_interval = 1e-3 * dt * n_steps / 100;  // in seconds
  std::cout << "tim win = " << time_interval << "s" << "\n";
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - */
  double input_ext= 0.0, net_input=0.0;
  //
  // all neurons receive external inputs from K independent Poisson generators
  // with a rate of v_ext, which is the same as a input from a Poisson
  // gen of rate ext_rate = (K * v_ext)
  double ext_rate = K * v_ext;
  std::poisson_distribution<int> gen_poiss_counts(ext_rate);
  // std::uniform_real_distribution<int> uniform_rand(0., 1.);
  // time loop
  for(size_t t_i = 0; t_i < n_steps; ++t_i) {
    cur_t = dt * t_i;    
    for(size_t neuron_idx=0; neuron_idx < N; ++neuron_idx) {
      // adaptive threshold 
      // V_threshold[neuron_idx] += dt_over_tau_th *
      // (V_threshold_initial - V_threshold[neuron_idx]);
      // update membrane voltage only if the neuron is not in the ref. period
      if(cur_t >= last_spike_time[neuron_idx] + refractory_period)  {
	input_ext = Je0 * gen_poiss_counts(poiss_generator_e[neuron_idx]);
	// net input is the sum of external, recurrent exe and inh inputs
	net_input = input_ext + g_e[neuron_idx] + g_i[neuron_idx];
	Vm[neuron_idx] = (1.0 - dt_over_tau) * Vm[neuron_idx] + net_input;
      }
    }
    /* - - - */
    // variable g_e and g_i store the total input from recurent spikes
    // at a time step, delta synapses
    vector_init_to_zero(g_e);
    vector_init_to_zero(g_i);
    //
    detect_spikes_brunel(cur_t);
    //
    shift_matrix(syn_delay_buffer, N, n_delay_bins);

    // ---------- testing ----
    fprintf(fpvm, "%f %f %f %f %f\n", cur_t, Vm[0], input_ext, g_e[0], g_i[0]);
    // input to the last neuron
    fprintf(fpip, "%f\n", input_ext);
    /* - - - - - */
    if((t_i >= discard_time && t_i % (unsigned int)((unsigned int)n_steps / 100)
	== 0) || t_i == n_steps-1) {
      get_pop_rates(time_interval);
      ProgressBar((float)t_i / (float)n_steps, pop_rate_e, pop_rate_i);
      if(t_i >= n_discard_steps) {
	fprintf(fp_rates_e, "%f %f\n", cur_t - discard_time, pop_rate_e);
	fprintf(fp_rates_i, "%f\n", pop_rate_i);
      }
    }
  }
  /* - - - */
  fflush(fpvm);
  fclose(fpvm);
  fflush(fpip);
  fclose(fpip);
  fflush(fp_rates_e);
  fclose(fp_rates_e);
  fflush(fp_rates_i);
  fclose(fp_rates_i);
  // gsl_rng_free(r);
}

#endif
