Brunel LIF network
-------

## Setup

**Download**

`git clone git@gitlab.com:rbm-hippo/lif_stp.git <path_to_dir>`

**Requirements**

`libconfig`: http://macappstore.org/libconfig/

python: see requirements


**Build**

`cd src`

`CC="g++ -lconfig++ -lgsl -stdlib=libc++" python setup.py build --force`

Tested on Mac OS 10.14.6

**Configuration**

Edit the file `config.toml` and set the path to `rbm_analysis` folder

`rbm_analysis` can be downloaded as:

`git@gitlab.com:rbm-hippo/rbm_analysis.git <path_to_dir>`

## Usage

Example usage in ./notebooks


Default simulation parameters are in the file src/params_bkp.cfg

First import module **simulate**

To change parameters pass them as keyword argument pairs:

e.g The following will run the simulation with 100 excitatory neurons

`import simulate`

`simulate.runsim(NE=100)`

To pass a parameter file, use function the `runsim_paramfile`: 

`simulate.runsim_paramfile(<param_file_name>)`

### TOFIX: 

- set parameter types, currently floating point parameters such as time constants must be passed with a decimal point
- move test code to a different folder


## Model

The model consists $`N_E`$ excitatory and $`N_I`$ inhibitory LIF neurons.


Neuron $`i`$ of population $`A`$ i.e. $`(i, A)`$ receives a connection from neuron $`(j, B)`$ with a probability given by:

$`P(C_{AB}^{ij} = 1) = \frac{K}{N_B}`$

The membrane voltage evolves as follows, 

$` \frac{d}{dt} V_{A}^{i} = \frac{-1}{\tau} V_A^i + I_{rec}^i + I^i_{FF}`$


Reccurent input $`I_{rec}^i`$ is depends on all presynatic spikes,

$`I_{rec}^i = \tau \,  \sum_{B = (E, I)} J_{AB} \sum_j \, C_{AB}^{ij} \sum_k \delta(t - t^j_k - D)`$

D is the transmission delay





Recurrents weights are set by parameter $`J`$ and $`g`$ as follows, 

$` J_{EE} = J`$,   $` J_{EI} = -g \, J`$

$` J_{IE} = J`$,  $` J_{II} = -g \, J`$


The network receives feedforward input from Poission spike generator whose rate is set to $` K \, v_{ext} `$.
The feedforward weights $`J_{E, ext}`$ $`J_{I, ext}`$ are set to $`J`$






