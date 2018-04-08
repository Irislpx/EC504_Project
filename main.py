import cv2
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from ek import Graph
from bayes import BayesClassifier


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
    background = img[labels == -1].reshape((-1, 3))
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


def create_msr_labels(anno):
    """ Create label matrix for training from
        user annotations. """
    labels = np.zeros(anno.shape[:2])

    anno_hsv = cv2.cvtColor(anno, cv2.COLOR_BGR2HSV)

    # lower red mask (0-10)
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(anno_hsv, lower_red, upper_red)

    # upper red mask (170-180)
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(anno_hsv, lower_red, upper_red)

    # join red masks
    red_mask = mask0 + mask1

    # blue mask
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    blue_mask = cv2.inRange(anno_hsv, lower_blue, upper_blue)

    # background
    labels[np.where(red_mask > 0)] = -1
    # foreground
    labels[np.where(blue_mask > 0)] = 1

    return labels


def graph_cuts(img, anno, scale):
    img_down = imresize(img, scale, interp='bilinear')  # downsample
    anno_down = imresize(anno, scale, interp='nearest')  # downsample
    size = img_down.shape[:2]
    # create label
    labels = create_msr_labels(anno_down)
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
    img_name, anno_name = sys.argv[1:]
    img = cv2.imread(img_name)
    anno = cv2.imread(anno_name)
    start_time = time.time()
    mask, fore, back = graph_cuts(img, anno, scale)
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
