import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

from concise.preprocessing.sequence import DNA, RNA, AMINO_ACIDS
from concise.utils.letters import all_letters
from collections import OrderedDict

from matplotlib import pyplot
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from descartes.patch import Polygon, PolygonPath, PolygonPatch
from shapely.wkt import loads as load_wkt

from shapely import affinity
import re


def heatmap(w, vmin=None, vmax=None, diverge_color=False,
            ncol=1,
            plot_name=None, vocab=["A", "C", "G", "T"], figsize=(6, 2)):
    """Plot a heatmap from weight matrix w

    vmin, vmax = z axis range
    diverge_color = Should we use diverging colors?
    plot_name = plot_title
    vocab = vocabulary (corresponds to the first axis)
    """
    # Generate y and x values from the dimension lengths
    assert len(vocab) == w.shape[0]
    plt_y = np.arange(w.shape[0] + 1) + 0.5
    plt_x = np.arange(w.shape[1] + 1) - 0.5
    z_min = w.min()
    z_max = w.max()

    if vmin is None:
        vmin = z_min
    if vmax is None:
        vmax = z_max

    if diverge_color:
        color_map = plt.cm.RdBu
    else:
        color_map = plt.cm.Blues

    fig = plt.figure(figsize=figsize)
    # multiple axis
    if len(w.shape) == 3:
        #
        n_plots = w.shape[2]
        nrow = math.ceil(n_plots / ncol)
    else:
        n_plots = 1
        nrow = 1
        ncol = 1

    for i in range(n_plots):
        if len(w.shape) == 3:
            w_cur = w[:, :, i]
        else:
            w_cur = w
        ax = plt.subplot(nrow, ncol, i + 1)
        plt.tight_layout()
        im = ax.pcolormesh(plt_x, plt_y, w_cur, cmap=color_map,
                           vmin=vmin, vmax=vmax, edgecolors="white")
        ax.grid(False)
        ax.set_yticklabels([""] + vocab, minor=False)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(np.arange(w_cur.shape[1] + 1))
        ax.set_xlim(plt_x.min(), plt_x.max())
        ax.set_ylim(plt_y.min(), plt_y.max())

        # nice scale location:
        # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        if plot_name is not None:
            if n_plots > 0:
                pln = plot_name + " {0}".format(i)
            else:
                pln = plot_name
            ax.set_title(pln)
        ax.set_aspect('equal')
    return fig


# -----------------------------------------------------------------------
#
#
# Code adoped from https://github.com/kundajelab/dragonn, (c) 2016 Kundaje Lab


def standardize_polygons_str(data_str):
    """Given a POLYGON string, standardize the coordinates to a 1x1 grid.
    Input : data_str (taken from above)
    Output: tuple of polygon objects
    """
    # find all of the polygons in the letter (for instance an A
    # needs to be constructed from 2 polygons)
    path_strs = re.findall("\(\(([^\)]+?)\)\)", data_str.strip())

    # convert the data into a numpy array
    polygons_data = []
    for path_str in path_strs:
        data = np.array([
            tuple(map(float, x.split())) for x in path_str.strip().split(",")])
        polygons_data.append(data)

    # standardize the coordinates
    min_coords = np.vstack([data.min(0) for data in polygons_data]).min(0)
    max_coords = np.vstack([data.max(0) for data in polygons_data]).max(0)
    for data in polygons_data:
        data[:, ] -= min_coords
        data[:, ] /= (max_coords - min_coords)

    polygons = []
    for data in polygons_data:
        polygons.append(load_wkt(
            "POLYGON((%s))" % ",".join(" ".join(map(str, x)) for x in data)))

    return tuple(polygons)


# ----------------------
letter_polygons = {k: standardize_polygons_str(v) for k, v in all_letters.items()}

VOCABS = {"DNA": OrderedDict([("A", "green"),
                              ("C", "blue"),
                              ("G", "orange"),
                              ("T", "red")]),
          "RNA": OrderedDict([("A", "green"),
                              ("C", "blue"),
                              ("G", "orange"),
                              ("U", "red")]),
          "AA": OrderedDict([('A', '#CCFF00'),
                             ('B', "orange"),
                             ('C', '#FFFF00'),
                             ('D', '#FF0000'),
                             ('E', '#FF0066'),
                             ('F', '#00FF66'),
                             ('G', '#FF9900'),
                             ('H', '#0066FF'),
                             ('I', '#66FF00'),
                             ('K', '#6600FF'),
                             ('L', '#33FF00'),
                             ('M', '#00FF00'),
                             ('N', '#CC00FF'),
                             ('P', '#FFCC00'),
                             ('Q', '#FF00CC'),
                             ('R', '#0000FF'),
                             ('S', '#FF3300'),
                             ('T', '#FF6600'),
                             ('V', '#99FF00'),
                             ('W', '#00CCFF'),
                             ('Y', '#00FFCC'),
                             ('Z', 'blue')]),
          "RNAStruct": OrderedDict([("P", "red"),
                                    ("H", "green"),
                                    ("I", "blue"),
                                    ("M", "orange"),
                                    ("E", "violet")]),
          }
# make sure things are in order
VOCABS["AA"] = OrderedDict((k, VOCABS["AA"][k]) for k in AMINO_ACIDS)
VOCABS["DNA"] = OrderedDict((k, VOCABS["DNA"][k]) for k in DNA)
VOCABS["RNA"] = OrderedDict((k, VOCABS["RNA"][k]) for k in RNA)


def add_letter_to_axis(ax, let, col, x, y, height):
    """Add 'let' with position x,y and height height to matplotlib axis 'ax'.
    """
    if len(let) == 2:
        colors = [col, "white"]
    elif len(let) == 1:
        colors = [col]
    else:
        raise ValueError("3 or more Polygons are not supported")

    for polygon, color in zip(let, colors):
        new_polygon = affinity.scale(
            polygon, yfact=height, origin=(0, 0, 0))
        new_polygon = affinity.translate(
            new_polygon, xoff=x, yoff=y)
        patch = PolygonPatch(
            new_polygon, edgecolor=color, facecolor=color)
        ax.add_patch(patch)
    return


# TODO - add figsize???
def seqlogo(letter_heights, vocab="DNA", ax=None):
    """Make a logo plot

    # Arguments
        letter_heights: "motif length" x "vocabulary size" numpy array
    Can also contain negative values.
        vocab: str, Vocabulary name. Can be: DNA, RNA, AA, RNAStruct.
        ax: matplotlib axis
    """
    ax = ax or plt.gca()

    assert letter_heights.shape[1] == len(VOCABS[vocab])
    x_range = [1, letter_heights.shape[0]]
    pos_heights = np.copy(letter_heights)
    pos_heights[letter_heights < 0] = 0
    neg_heights = np.copy(letter_heights)
    neg_heights[letter_heights > 0] = 0

    for x_pos, heights in enumerate(letter_heights):
        letters_and_heights = sorted(zip(heights, list(VOCABS[vocab].keys())))
        y_pos_pos = 0.0
        y_neg_pos = 0.0
        for height, letter in letters_and_heights:
            color = VOCABS[vocab][letter]
            polygons = letter_polygons[letter]
            if height > 0:
                add_letter_to_axis(ax, polygons, color, 0.5 + x_pos, y_pos_pos, height)
                y_pos_pos += height
            else:
                add_letter_to_axis(ax, polygons, color, 0.5 + x_pos, y_neg_pos, height)
                y_neg_pos += height

    # if add_hline:
    #     ax.axhline(color="black", linewidth=1)
    ax.set_xlim(x_range[0] - 1, x_range[1] + 1)
    ax.grid(False)
    ax.set_xticks(list(range(*x_range)) + [x_range[-1]])
    ax.set_aspect(aspect='auto', adjustable='box')
    ax.autoscale_view()


def seqlogo_fig(letter_heights, vocab="DNA", figsize=(10, 2), ncol=1, plot_name=None):
    """

    # Arguments
        plot_name: Title of the plot. Can be a list of names
    """
    fig = plt.figure(figsize=figsize)

    if len(letter_heights.shape) == 3:
        #
        n_plots = letter_heights.shape[2]
        nrow = math.ceil(n_plots / ncol)
        if isinstance(plot_name, list):
            assert len(plot_name) == n_plots
    else:
        n_plots = 1
        nrow = 1
        ncol = 1

    for i in range(n_plots):
        if len(letter_heights.shape) == 3:
            w_cur = letter_heights[:, :, i]
        else:
            w_cur = letter_heights
        ax = plt.subplot(nrow, ncol, i + 1)
        plt.tight_layout()

        # plot the motif
        seqlogo(w_cur, vocab, ax)

        # add the title
        if plot_name is not None:
            if n_plots > 0:
                if isinstance(plot_name, list):
                    pln = plot_name[i]
                else:
                    pln = plot_name + " {0}".format(i)
            else:
                pln = plot_name
            ax.set_title(pln)
    return fig
