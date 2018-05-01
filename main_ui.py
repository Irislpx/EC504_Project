import cv2
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from ek import Graph
from sklearn.mixture import GaussianMixture


def build_bayes_graph(img, prob_fg, prob_bg, sigma=1e2, kappa=2):
    """ Build a graph from 4-neighborhood of pixels.
        Foreground and background is determined from
        labels (1 for foreground, 0 for background)
        and is modeled with naive Bayes classifiers."""
    m, n = img.shape[:2]
    # RGB vector version (one pixel per row)
    vim = img.reshape((-1, 3))
    # create graph with m*n+2 nodes
    gr = Graph()
    gr.add_node(range(m * n + 2))
    source = m * n  # second to last is source
    sink = m * n + 1  # last node is sink
    # normalize
    for i in range(vim.shape[0]):
        vim[i] = vim[i] / np.linalg.norm(vim[i])
    # go through all nodes and add edges
    for i in range(m * n):
        print(i)
        # add edge from source
        gr.add_edge((source, i), (prob_fg[i] / (prob_fg[i] + prob_bg[i])))
        # add edge to sink
        gr.add_edge((i, sink), (prob_bg[i] / (prob_fg[i] + prob_bg[i])))
        # add edges to neighbors
        if i % n != 0:  # left exists
            edge_wt = kappa * \
                np.exp(-1.0 * sum((vim[i] - vim[i - 1])**2) / sigma)
            gr.add_edge((i, i - 1), edge_wt)
        if (i + 1) % n != 0:  # right exists
            edge_wt = kappa * \
                np.exp(-1.0 * sum((vim[i] - vim[i + 1])**2) / sigma)
            gr.add_edge((i, i + 1), edge_wt)
        if i // n != 0:  # up exists
            edge_wt = kappa * \
                np.exp(-1.0 * sum((vim[i] - vim[i - n])**2) / sigma)
            gr.add_edge((i, i - n), edge_wt)
        if i // n != m - 1:  # down exists
            edge_wt = kappa * \
                np.exp(-1.0 * sum((vim[i] - vim[i + n])**2) / sigma)
            gr.add_edge((i, i + n), edge_wt)
    gr.build_flow(source, sink)
    return gr


def cut_graph(gr, imsize):
    h, w = imsize
    flows = gr.edmonds_karp()
    cuts = gr.find_cut()
    # convert graph to image with labels
    res = np.zeros(h * w)
    for pos in list(cuts.keys())[0:-2]:  # don't add source/sink
        res[pos - 1] = cuts[pos]
    return res.reshape((h, w))


def graph_cuts(img, scale, x, y):
    img_down = imresize(img, scale, interp='bilinear')  # downsample
    size = img_down.shape[:2]
    img_flat = np.concatenate((img_down[:, :, 0].flatten().reshape(-1, 1),
                               img_down[:, :, 1].flatten().reshape(-1, 1),
                               img_down[:, :, 2].flatten().reshape(-1, 1)), axis=1)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        max_iter=500,
        n_init=5).fit(img_flat)
    prob = gmm.predict_proba(img_flat)
    labels = np.argmax(gmm.predict_proba(img_flat), axis=1)
    prob_fg, prob_bg = np.array([i[0] for i in prob.tolist()]), np.array([
        i[1] for i in prob.tolist()])
    # create graph
    g = build_bayes_graph(img_down, prob_fg, prob_bg, sigma=1e20, kappa=1)
    # cut the graph
    mask = cut_graph(g, size)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))  # upsample
    mask = np.array(mask, dtype=np.uint8)
    cuts = cv2.bitwise_and(img, img, mask=mask)
    sy = 300 / img.shape[0]
    sx = 450 / img.shape[1]
    print(size, int(y / sy), int(x / sx), sy, sx)
    if np.all(cuts[int(y / sy), int(x / sx)] == 0):
        cuts = img - cuts
    plt.figure()
    plt.imshow(cv2.cvtColor(cuts, cv2.COLOR_BGR2RGB))
    return mask, cuts


def main_gmm(path, scale, x, y):
    img = cv2.imread(path)
    start_time = time.time()
    mask, cuts = graph_cuts(img, scale, x, y)
    end_time = time.time()
    total = end_time - start_time
    print('Running time: {:2}s'.format(total))
    plt.show()
    return mask, cuts
