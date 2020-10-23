"jupytext_formats: ipynb,py"

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ipdb  # noqa
import toml
from configobj import ConfigObj
from scipy.signal import square as square_wave
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn
import glob

# set src paths
# home_dir = Path.home()
script_path = str(Path(__file__).resolve().parent)
# pyconf = toml.load(script_path + '/config.toml')
# rbm_analysis_scr = home_dir / pyconf['rbm_analysis']['rbm_analysis_src']
# # sys.path.append(str(rbm_analysis_scr))
# sys.path.append(str(rbm_analysis_scr / 'helpers'))
# # import analyse_dataset as ad  # noqa

import plot_utils as pu  # noqa
seaborn.set()
# import C++ simulation lib
cpp_shared_lib_path = glob.glob(script_path + '/src/build/lib*')[0]
# sys.path.append(script_path + '/src/build/lib.macosx-10.7-x86_64-3.7')
sys.path.append(cpp_shared_lib_path)
import brunel  # noqa


def runsim(**params):
    # load default params
    conf = ConfigObj('params_bkp.cfg')
    conf.filename = 'params.cfg'
    conf.write()
    # make changes in parameters if provided
    update_sim_param(**params)
    new_conf = ConfigObj('params.cfg')
    # generate_ff_input_file(new_conf)
    # run the simulation
    brunel.simu()
    try:
        # display the simulation results
        display_sim_results(new_conf)
        plt.ion()
        plt.show()
    except Exception as err:
        print(err)
        pass
    restore_default_params()


def runsim_paramfile(paramfile):
    # load default params
    conf = ConfigObj(paramfile)
    conf.filename = 'params.cfg'
    conf.write()
    #
    brunel.simu()
    try:
        display_sim_results(conf)
    except Exception as err:
        print(err)
        pass
    restore_default_params()


def restore_default_params():
    conf = ConfigObj('src/params_bkp.cfg')
    conf.filename = 'params.cfg'
    conf.write()


def update_sim_param(filename='params.cfg', **params):
    # overwrites the file
    conf = ConfigObj(filename)
    for key, value in params.items():
        conf[key] = value
    conf.write()


def generate_ff_input_file(conf, duty_val=1.0):
    dt = float(conf['dt'])
    t_stop = float(conf['t_stop'])
    discard_time = float(conf['discard_time'])
    print(t_stop)
    n_steps = int((t_stop) / dt)
    n_discard_steps = int((discard_time) / dt)
    ff_input_zeros = np.zeros((n_discard_steps, ))
    t = np.linspace(0, t_stop, n_steps)
    duty_cycle = np.zeros_like(t)
    idx = np.random.randint(10, duty_cycle.size, 10)
    idx_start = idx - 100
    for i in range(idx.size):
        duty_cycle[idx_start[i]:idx[i]] = duty_val
    #
    ff_input = 4.0 * np.heaviside(square_wave(t, duty_cycle), 0.0)
    ff_input = np.hstack((ff_input_zeros, ff_input))
    #
    fig = plt.figure()
    gs = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(gs[:, :-1])
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[1, 2])
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    ax2.set_ylabel(' ')
    ax1.set_ylabel(' ')
    fig.set_size_inches(20, 4.8)
    ax0.plot(t * 1e-3, ff_input[n_discard_steps:])
    np.savetxt('./data/ff_input.txt', ff_input, delimiter="\n")


def print_rates():
    re = np.loadtxt('data/pop_rates_e.txt')
    ri = np.loadtxt('data/pop_rates_i.txt')
    print(r'r_E = ' + f'{re[:, 1].mean():.4f}Hz ')
    print(r'r_I = ' + f'{ri.mean():.4f}Hz ')


def plot_avg_rates(axobj=None, label_tag='', inset=False):
    re = np.loadtxt('data/pop_rates_e.txt')
    ri = np.loadtxt('data/pop_rates_i.txt')
    acfunc = auto_corr
    ac_re = acfunc(re[:, 1])
    ac_ri = acfunc(ri)
    if axobj is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 4.8)
    else:
        if inset:
            # ax = inset_axes(axobj, width="25%", height="25%", loc=1)
            ax = inset_axes(axobj,
                            width="100%",
                            height="100%",
                            bbox_to_anchor=(1.05, .6, .5, .4),
                            bbox_transform=axobj.transAxes,
                            loc=2,
                            borderpad=0)
        else:
            ax1 = axobj[0]
            ax2 = axobj[1]
    t = re[:, 0] * 1e-3
    ax1.plot(t,
             re[:, 1],
             label=r'$r_E$' + f'({re[:, 1].mean():.2f}Hz) ' + label_tag)
    ax1.plot(t, ri, label=r'$r_I$' + f'({ri.mean():.2f})Hz ' + label_tag)
    ax1.set_ylabel('Avg pop rates')
    ax1.set_xlabel('Time (s)')
    ax1.legend(frameon=False, loc=0)  # , prop={'size': 16})

    print(r'r_E = ' + f'{re[:, 1].mean():.4f}Hz ')
    print(r'r_I = ' + f'{ri.mean():.4f}Hz ')

    # plt.legend(frameon=False,
    #            prop={'size': 16},
    #            bbox_to_anchor=(1.05, 1),
    #            loc='upper left')
    max_lag = ac_re.size
    ax2.plot(t[:max_lag], ac_re[:max_lag])
    ax2.plot(t[:max_lag], ac_ri[:max_lag])
    ax2.set_ylabel('Auto-Corr')
    ax2.set_xlabel('lag (s)')
    return t


def auto_corr(y):
    yunbiased = y - np.mean(y)
    ynorm = np.sum(yunbiased**2)
    acor = np.correlate(yunbiased, yunbiased, "same") / ynorm
    # use only second half
    acor = acor[len(acor) // 2:]
    return acor


def plot_random_slection(spk_times, dt, t_min=0, t_max=100, n=50):
    # st is passed in seconds
    # t_min, t_max is in ms!! fix this

    # to reproduce fig.8 brunel
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(4, 1)
    ax0 = fig.add_subplot(gs[:-1, :])
    ax1 = fig.add_subplot(gs[-1, :], sharex=ax0)
    #
    st = np.copy(spk_times)
    st[:, 0] *= 1e3
    idx = np.logical_and(st[:, 0] > t_min, st[:, 0] <= t_max)
    st = st[idx, :]
    bins = np.arange(t_min, t_max + dt, dt)
    ax1.hist(st[:, 0], bins, color='k', edgecolor='k')
    # raster for n randomly chosen exc. neurons
    neuron_idx = np.random.randint(0, np.max(st[:, 1]), 2 * n)
    st_ = []
    valid_cnt = 0  # number of neurons with at least 1 spike
    for i, idx in enumerate(neuron_idx):
        tmp = st[st[:, 1] == i, 0]
        if len(tmp) > 0:
            tmp2 = [(k, i) for k in tmp]
            st_.append(tmp2)
            valid_cnt += 1
            if (valid_cnt == n):
                break
    st_ = np.vstack(st_)
    pu.plot_raster(st_, ax=ax0)

    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.set_ylabel('Randomly Selected Neurons')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Spike Count')


def display_sim_results(params):
    #
    print("loading spikes")
    st = np.loadtxt('data/spikes.txt')
    print("done")

    print_rates()

    plot_random_slection(st, float(params['dt']))

    n_neurons = int(params["NE"]) + int(params["NI"])
    coeff_of_var = np.nanmean(cv(st, n_neurons))
    print(f"CV = {coeff_of_var}")
    # fig = plt.figure()
    # gs = fig.add_gridspec(2, 3)
    # ax0 = fig.add_subplot(gs[:, :-1])
    # ax1 = fig.add_subplot(gs[0, 2])
    # ax2 = fig.add_subplot(gs[1, 2])
    # pu.plot_raster(st, ax=ax0)
    # fig.set_size_inches(20, 4.8)
    # plt.ylabel('Neuron #')
    # plt.xlabel('Time (s)')
    # #
    # _ = plot_avg_rates(axobj=[ax1, ax2])


def cv_aux(isi):
    mean_isi = isi.mean()
    if mean_isi <= 0:
        return np.nan
    return isi.std() / mean_isi


def cv(st, n_neurons, min_spikes=3):
    """  USAGE: cv(st)

Args
    ----
      st: list or array with 2 clmns: (spike times, cell_idx)

    Returns
    -------
      out: cv of each neuron

    """
    neuron_idx = np.unique(np.array(st[:, 1], dtype=int))
    # n_neurons = np.unique(neuron_idx)
    cv = np.empty((n_neurons, ))
    cv[:] = np.nan
    for i in neuron_idx:
        idx = st[:, 1] == i
        st_i = st[idx, 0]
        if len(st_i) > min_spikes:
            isi = np.diff(st_i)
            cv[i] = cv_aux(isi)
    return cv
