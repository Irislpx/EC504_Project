from scipy.misc import imresize
import cv2
import numpy as np
import time
import sys
from ek import Graph
from bayes import BayesClassifier
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def build_bayes_graph(img, labels, sigma=1e2, kappa=2):
    """ Build a graph from 4-neighborhood of pixels.
        Foreground and background is determined from
        labels (1 for foreground, -1 for background, 0 otherwise)
        and is modeled with naive Bayes classifiers."""
    m, n = img.shape[:2]
    # RGB vector version (one pixel per row)
    vim = img.reshape((-1, 3))
    # RGB for foreground and background
    foreground = img[labels == 1].reshape((-1, 3))
    background = img[labels == 0].reshape((-1, 3))
    train_data = [foreground, background]
    # train naive Bayes classifier
    bc = BayesClassifier()
    bc.train(train_data)
    # get probabilities for all pixels
    bc_lables, prob = bc.classify(vim)
    prob_fg, prob_bg = prob[0], prob[1]
    print(np.amax(prob_fg), np.max(prob_bg))
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
    for pos in list(cuts.keys())[1:-1]:  # don't add source/sink
        res[pos - 1] = cuts[pos]
    return res.reshape((h, w))


def graph_cuts(img, scale):
    img_down = imresize(img, scale, interp='bilinear')  # downsample
    size = img_down.shape[:2]
    vim = img_down.reshape((-1, 3)).astype('float32')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, center = cv2.kmeans(
        vim, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape((size))
    print(labels)
    # create graph
    g = build_bayes_graph(img_down, labels, sigma=1e20, kappa=1)
    # cut the graph
    mask = cut_graph(g, size)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))  # upsample
    mask = np.array(mask, dtype=np.uint8)
    back = cv2.bitwise_and(img, img, mask=mask)
    fore = img - back
    return mask, fore, back


if __name__ == '__main__':
    scale = 0.25
    img_name = sys.argv[1]
    img = cv2.imread(img_name)
    start_time = time.time()
    mask, fore, back = graph_cuts(img, scale)
    end_time = time.time()
    total = end_time - start_time
    print('Running time: {:2}s'.format(total))
    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(mask)
    plt.figure()
    plt.imshow(fore)
    plt.figure()
    plt.imshow(back)
    plt.show()
