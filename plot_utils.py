import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.transforms import Bbox


def set_color_palette(palette='muted'):
    current_palette = sns.color_palette('muted')
    sns.set_palette(current_palette)


def gen_n_colors(n, palette='muted'):
    '''  returns rgb
    '''
    current_palette = sns.color_palette(palette, n)
    return np.array(current_palette)


def map_color_to_data(labels, palette='muted'):
    """  USAGE: map_color_to_data(labels)

    returns colors for each data point labeled by
    unique color for each label category

    Args
    ----
      labels: int categories  1 x n_points

    Returns
    -------
      out: rgb mat 3 x n_points

    """
    labels = np.array(labels, dtype=int)  # in case labels is a bool array
    n_unique_categories = len(np.unique(labels))
    n_colors = gen_n_colors(n_unique_categories, palette)
    colors = n_colors[labels]
    return colors


def scatter_plot_mat(data, alpha=0.5, markersize=2, ax=None, colors=None):
    """  USAGE: scatter_plot_mat(data, alpha=0.5, markersize=2)

    plot data[:, 0] vs data[:, 1]

    Args
    ----
      data: 2 x n_points
      alpha: float, optional  (default=0.5)
      markersize: int, optional  (default=2)

    Returns
    -------
      out: fig, ax

    """
    rows, columns = data.shape
    if rows == 2:
        x = data[0, :]
        y = data[1, :]
    elif columns == 2:
        x = data[:, 0]
        y = data[:, 1]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    if colors is None:
        colors = 'k'
    else:
        current_palette = sns.color_palette('muted')
        sns.set_palette(current_palette)
        map_var = np.random.rand(x.size) < .5
        colors = map_color_to_data(map_var)

    ax.scatter(x, y, marker='.', c=colors, s=markersize)
    ax.set_aspect('equal')

    # plt.show()

    return fig, ax


def fixed_heatmap(x,
                  x_ticks_step=6,
                  y_ticks_steps=8,
                  cmap='GnBu',
                  xticklabels=('0', r'$\frac{\pi}{2}$', r'$\pi$'),
                  ax=None,
                  show_cbar=True):
    """  USAGE: fixed_heatmap(x, n_x_ticks=5, n_y_ticks=5, cmap='GnBu')

    fixes the ugly dense ticks produces in seaborn
    Args
    ----
      x: argtype
      n_x_ticks: int, optional  (default=5)
      n_y_ticks: int, optional  (default=5)
      cmap: string, optional  (default='GnBu')
      ax: (default=None)
      show_cbar: bool (default=True)

    Returns
    -------
      out:

    """
    if ax is None:
        fg, ax = plt.subplots()
    else:
        fg = plt.gcf()

    mask = np.isnan(x)
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    sns.heatmap(x,
                ax=ax,
                cmap=cmap,
                mask=mask,
                cbar=show_cbar,
                cbar_kws={
                    'ticks': [x_min, 0.5 * (x_min + x_max), x_max],
                    'format': '%.2f',
                    'pad': 0.5
                })
    xticks = ax.get_xticks()
    x_tick_min = xticks[0]
    x_tick_max = xticks[-1]
    x_tick_middle = int(0.5 * (x_tick_max + x_tick_min))
    ax.set_xticks((x_tick_min, x_tick_middle, x_tick_max))
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    else:
        pass
        # ax.set_xticklabels(xticklabels)
    yticks = ax.get_yticks()
    y_tick_min = yticks.min()
    y_tick_max = yticks.max()

    ax.set_yticks((y_tick_min, y_tick_max))
    yticklabels = [f'{y_tick_min}', f'{y_tick_max}']
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    else:
        pass
    # y_tick_labels = ax.get_yticklabels()
    # [label.set_visible(False) for label in ax.get_yticklabels()]
    # for label in y_tick_labels[::y_ticks_steps]:
    #     label.set_visible(True)

    plt.draw()
    return fg, ax


def create_variable_clm_subplots(rows,
                                 columns_list,
                                 hide_ticks=True,
                                 fig_aspect=1):
    """  USAGE: create_variable_clm_subplots(rows, columns_list)

    Create subplot with #rows and variable columns
    Args
    ----
      rows: int
      columns_list: int array of length = rows
      hide_ticks: bool, (default=Ture)
      fig_aspect: float height/width (default=1)

    Returns
    -------
      out: fig, ax

    """
    n_columns = np.max(columns_list)
    fig, ax = plt.subplots(rows,
                           n_columns,
                           figsize=plt.figaspect(fig_aspect),
                           constrained_layout=True)
    for i in range(rows):
        for k in np.arange(columns_list[i], n_columns):
            ax[i, k].set_axis_off()
    if hide_ticks:
        for i in range(rows):
            for k in range(n_columns):
                ax[i, k].set_xticks([])
                ax[i, k].set_yticks([])
    # plt.draw()
    return fig, ax


def distribute_subplot(n, max_columns=5, hide_ticks=True, fig_aspect=1):
    columns = 4
    rows = int(np.ceil(n / 4))
    # _, _, rows, columns = tile_subplots(n)
    n_columns = np.min([max_columns, columns])
    fig, ax = plt.subplots(rows,
                           n_columns,
                           figsize=plt.figaspect(fig_aspect),
                           constrained_layout=True)
    for i in range(rows):
        for k in range(columns):
            if i * columns + k >= n:
                ax[i, k].set_axis_off()
    if hide_ticks:
        for i in range(rows):
            for k in range(n_columns):
                ax[i, k].set_xticks([])
                ax[i, k].set_yticks([])

    return fig, ax


def tile_subplots(n):
    height = 3
    if n > 3:
        height = int(np.ceil(n / 3.))

    n_cloumns = 3  # int(np.ceil(n/3))
    width = 3
    n_rows = int(np.floor(n / n_cloumns))
    if n_cloumns * n_rows < n:
        n_rows += 1
    if n_rows == 0:
        n_rows = 1
    if n_cloumns > 1:
        return width * n_cloumns, height * n_rows, n_rows, n_cloumns
    if n_rows > 1:
        return (width * n_cloumns, height * n_rows * n_cloumns, n_rows,
                n_cloumns)


def plot_raster(data, color='k', neuron_idx_offset=0, NE=None, ax=None):
    """  USAGE: PlotRaster()

    data: spike_times x neuron_idx
    Args
    ----

    Returns
    -------
      out:

    """
    if ax is None:
        _, ax = plt.subplots()

    if NE is None:
        ax.plot(data[:, 0], data[:, 1] + neuron_idx_offset, '|', color=color)
    else:
        ne_indices = data[:, 1] < NE
        data_ne = data[ne_indices, :]
        data_ni = data[np.logical_not(ne_indices), :]
        ax.plot(data_ne[:, 0],
                data_ne[:, 1] + neuron_idx_offset,
                '|',
                color='b')
        ax.plot(data_ni[:, 0],
                data_ni[:, 1] + neuron_idx_offset,
                '|',
                color='r')
        ax.ylabel('Neuron #')
        ax.xlabel('Time (s)')


def PlotSeq(data, cell_order=None, ax=None, colors=None):
    '''
    data : a boolean matrix of shape n_samples x n_neurons

    '''
    if cell_order is not None:
        n_neurons = len(np.squeeze(cell_order))
        seq = np.squeeze(data[:, cell_order])
    else:
        n_neurons = data.shape[1]
        seq = data
        cell_order = np.arange(n_neurons)
    event_times = []
    [event_times.append(np.where(seq[:, i])) for i in range(n_neurons)]
    if colors is None:
        colors = sns.color_palette(None, n_neurons)
    else:
        colors = ['k' for ii in range(n_neurons)]
    color_idx = {}
    increasing_order = np.sort(cell_order)
    fig = None
    for k, n in enumerate(increasing_order):
        color_idx[n] = k
    if ax is None:
        fig, ax = plt.subplots()
    for i in range(n_neurons):
        if len(event_times[i][0]) > 0:
            y = np.ones(event_times[i][0].size) * i + 0.8
            ax.plot(event_times[i][0],
                    y,
                    '|',
                    color=colors[color_idx[cell_order[i]]],
                    alpha=0.4)
    plt.xlabel('Frame #')
    return fig, ax


def plot_colorbar(cmap_name="GnBu",
                  vmin=0,
                  vmax=np.log10(2),
                  label='',
                  ax=None,
                  orientation='vertical',
                  shrink=1):
    # label=r'$\log_{10}(r(\alpha))$'):
    cmap = mpl.cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    if ax is not None:
        cbar = plt.colorbar(sm,
                            ticks=np.linspace(vmin, vmax, 3),
                            format='%.2f',
                            ax=ax,
                            orientation=orientation,
                            shrink=shrink)
    else:
        cbar = plt.colorbar(sm,
                            ticks=np.linspace(vmin, vmax, 3),
                            format='%.2f',
                            orientation=orientation,
                            shrink=shrink)
    cbar.set_label(label)
    return cbar


def summary_figure_layout(pad=0.25):

    # fig.set_size_inches(6.4, 4.8)
    fig_size = (6.4 * 1.25 + 1, 4.8 * 1.25 + 1)
    fig = plt.figure(figsize=fig_size)
    sup_title = fig.suptitle('....SUP_TITLE...', fontsize='x-large')
    sup_title.set_y(0.99)
    # fig.set_size_inches(*fig_size)

    ax = []

    ax.append(fig.add_subplot(451))  # umap rbm
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('RBM-UMAP')

    ax.append(fig.add_subplot(452))  # umap roi with fit
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('ROI fit')

    ax.append(fig.add_subplot(453))  # tuning curves
    ax[2].set_title(r'$\log_{10}(1 + r(\alpha))$')
    plt.setp(ax[2].get_xticklabels()[1], visible=False)
    insetaxes = {}
    insetaxes['si_hist'] = inset_axes(
        ax[2],
        width="25%",  # width = 30% of parent_bbox
        height="25%",  # height : 1 inch
        loc=1)

    insetaxes['si_hist'].spines['top'].set_visible(False)
    insetaxes['si_hist'].spines['right'].set_visible(False)
    insetaxes['si_hist'].spines['left'].set_visible(False)
    plt.setp(insetaxes['si_hist'].get_yticklabels(), visible=False)
    insetaxes['si_hist'].set_yticks([])
    plt.setp(insetaxes['si_hist'].get_xticklabels(), fontsize=6)
    insetaxes['si_hist'].set_xlabel('SI', fontsize=6)
    # insetaxes['si_hist'].spines['bottom'].set_visible(False)
    # insetaxes['si_hist'].spines['left'].set_linewidth(0.5)

    ax.append(fig.add_subplot(454))  # clustered sce covmat
    plt.setp(ax[3].get_xticklabels(), visible=False)
    plt.setp(ax[3].get_yticklabels(), visible=False)
    ax[3].set_ylabel('SCE #')
    ax[3].set_xlabel('SCE #')
    ax[3].set_title('Cov mat')

    insetaxes['clu_hist'] = inset_axes(
        ax[3],
        width="25%",  # width = 30% of parent_bbox
        height="25%",  # height : 1 inch
        loc=1)
    plt.setp(insetaxes['clu_hist'].get_yticklabels(), fontsize=8)
    plt.setp(insetaxes['clu_hist'].get_xticklabels(), fontsize=8)
    insetaxes['clu_hist'].spines['top'].set_visible(False)
    insetaxes['clu_hist'].spines['right'].set_visible(False)
    insetaxes['clu_hist'].spines['left'].set_visible(False)
    plt.setp(insetaxes['clu_hist'].get_yticklabels(), visible=False)
    insetaxes['clu_hist'].set_yticks([])

    plt.setp(insetaxes['clu_hist'].get_xticklabels(), fontsize=6)
    insetaxes['clu_hist'].set_xlabel('clusters', fontsize=6)

    ax.append(fig.add_subplot(455))  # neuron # vs sce #
    plt.setp(ax[4].get_xticklabels(), visible=False)
    ax[4].set_xlabel('Sorted SCEs')
    ax[4].set_title('Assemblies')

    # alphas
    ax.append(plt.subplot2grid(shape=(4, 5), loc=(1, 0), colspan=5))
    ax[5].set_ylabel(r'$\alpha$')
    ax[5].set_ylim([0, np.pi])
    ax[5].set_yticks([0, np.pi / 2, np.pi])
    ax[5].set_yticklabels([f'0', r'$\frac{\pi}{2}$', r'$\pi$'])
    # turn off xticklabels since shared with the next axis
    plt.setp(ax[5].get_xticklabels(), visible=False)

    # insetaxes['alpha_autocorr'] = inset_axes(
    #     ax[5],
    #     width="50%",  # width = 30% of parent_bbox
    #     height="50%",  # height : 1 inch
    #     loc=1)
    insetaxes['alpha_autocorr'] = ax[5].inset_axes((.885, .4, .09, .5))
    # insetaxes['alpha_autocorr'].set_aspect('equal')

    # raster
    # ax.append(plt.subplot2grid(shape=(4, 5), loc=(2, 0), colspan=5))
    ax.append(
        plt.subplot2grid(shape=(4, 5), loc=(2, 0), colspan=5, sharex=ax[5]))
    ax[6].set_xlabel('Frame #')
    ax[6].set_ylabel('Sorted cell #')

    # sce raster
    ax.append(plt.subplot2grid(shape=(4, 5), loc=(3, 0), colspan=5))
    ax[7].set_ylabel('Sorted Cell #')
    ax[7].set_xlabel('SCE #')

    plt.tight_layout(pad=pad)
    fig.subplots_adjust(top=0.9)

    ax5_pos = ax[5].get_position()
    ax6_pos = ax[6].get_position()
    y_shift = (ax5_pos.y0 - ax6_pos.y1) * .37
    # move down
    ax5_pos.y0 -= y_shift
    ax5_pos.y1 -= y_shift
    # move up
    ax6_pos.y0 += y_shift
    ax6_pos.y1 += y_shift

    ax[5].set_position(ax5_pos)
    ax[6].set_position(ax6_pos)
    plt.draw()

    return fig, ax, sup_title, insetaxes
