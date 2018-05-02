from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSlot
#from main_ui import *
import cv2
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
#from ek import Graph
from sklearn.mixture import GaussianMixture
#---------ek------
class Graph:

    def __init__(self):
        self.node = []
        self.edge = {}
        self.neighbors = {}
        self.graph = []
        self.residual = []  # residual graph
        self.row = None
        self.source = None
        self.sink = None
        self.sset = None
        self.tset = None

    # add nodes
    def add_node(self, node=[]):
        self.node = node

    # add edges
    def add_edge(self, node=(), capacity=None):
        self.edge.setdefault(node, capacity)

    # build the network flow
    def build_flow(self, source=None, sink=None):
        for i in range(len(self.node)):
            self.graph.append([])
            self.graph[i] = [0 for j in range(len(self.node))]
            self.neighbors.setdefault(i, [])
        for i, j in self.edge.keys():
            self.graph[i][j] = self.edge[(i, j)]
            self.neighbors[i].append(j)
            if i not in self.neighbors[j]:
                self.neighbors[j].append(i)
        self.residual = [i[:] for i in self.graph]
        self.row = len(self.graph)
        self.source = source
        self.sink = sink

    def edmonds_karp(self):
        flow = 0
        length = len(self.graph)
        flows = [[0 for i in range(length)] for j in range(length)]
        while True:
            max, parent = self.bfs(flows)
            print(max)
            if max == 0:
                self.sset = [self.source] + \
                    [i for i, v in enumerate(parent) if v >= 0]
                self.tset = [x for x in self.node if x not in self.sset]
                print(self.sset, self.tset)
                break
            flow = flow + max
            v = self.sink
            while v != self.source:
                u = parent[v]
                flows[u][v] = flows[u][v] + max
                self.residual[u][v] -= max
                flows[v][u] = flows[v][u] - max
                self.residual[v][u] += max
                v = u
        return flow, flows

    def bfs(self, flows):
        length = self.row
        parents = [-1 for i in range(length)]  # parent table
        parents[self.source] = -2  # make sure source is not rediscovered
        M = [0 for i in range(length)]  # Capacity of path to vertex i
        M[self.source] = float('Inf')  # this is necessary!

        queue = []
        queue.append(self.source)
        while queue:
            u = queue.pop(0)
            for v in self.neighbors[u]:
                # if there is available capacity and v is is not seen before in
                # search
                if self.graph[u][v] - flows[u][v] > 0 and parents[v] == -1:
                    parents[v] = u
                    # it will work because at the beginning M[u] is Infinity
                    M[v] = min(M[u], self.graph[u][v] - flows[u]
                               [v])  # try to get smallest
                    if v != self.sink:
                        queue.append(v)
                    else:
                        return M[self.sink], parents
        return 0, parents

    def find_cut(self):
        cut = {}
        for i in self.sset:
            cut[i] = 0
        for i in self.tset:
            cut[i] = 1
        return cut
#-------end ek-------

#-------gmm--------
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

#-------end gmm------



class UI(QWidget):
    # global imagepath

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Segmentation")
        self.setGeometry(100, 100, 500, 550)

        self.inputImagePath = None
        self.x = 0
        self.y = 0

        self.scalevalue = 1

        self.imageselected = False

        # self.coordinateIsButton = False

        self.openButton = QPushButton('Select Image', self)
        self.openButton.setToolTip('format = jpg/png')
        self.openButton.clicked.connect(self.openButtonOnClick)
        self.openButton.setShortcut("Ctrl+O")

        # self.saveButton = QPushButton('Save Image', self)
        # self.saveButton.clicked.connect(self.saveImage)
        # self.saveButton.setShortcut("Ctrl+S")

        self.closeButton = QPushButton('Close', self)
        self.closeButton.clicked.connect(self.closeApp)
        self.closeButton.setShortcut("Ctrl+E")

        self.inputImageView = QLabel("input image")


        # self.outputImageView = QLabel("output image")
        # self.foreground = QLabel("foreground")
        # self.background = QLabel("background")

        self.startButton = QPushButton('Start', self)
        self.startButton.setShortcut("Ctrl+S")
        self.startButton.clicked.connect(self.startButtonOnClick)

        self.scaleLabel = QLabel(self)
        self.scaleLabel.setText('Scale value(0-1):')
        self.line = QLineEdit(self)

        self.okbutton = QPushButton('OK')
        self.okbutton.clicked.connect(self.okButtonOnClick)


        self. coordinatesLabel = QLabel(self)
        self.coordinatesLabel.setText("coordinates = (" + str(self.x) + " , " + str(self.y) + ")")
        self.coordinatesLabel.setFixedHeight(15)

        # self.startCoorBtn = QPushButton('Start from Selected Coordinates')
        # self.startCoorBtn.setShortcut("Ctrl+C")
        # self.startCoorBtn.clicked.connect(self.startCoorOnClick)

        self.horizontal = QHBoxLayout()
        self.horizontal.addWidget(self.scaleLabel)
        self.horizontal.addWidget(self.line)
        self.horizontal.addWidget(self.okbutton)

        # self.horizontalImages2 = QHBoxLayout()
        # self.horizontalImages2.addWidget(self.foreground)
        # self.horizontalImages2.addWidget(self.background)

        self.GroupBox1 = QGroupBox()
        self.GroupBox1.setLayout(self.horizontal)
        self.GroupBox1.setFixedHeight(50)

        # self.imageGroupBox2 = QGroupBox("Segmentation")
        # self.imageGroupBox2.setLayout(self.horizontalImages2)

        self.vlayout = QVBoxLayout()
        # self.vlayout.addWidget(self.imageGroupBox1)
        # self.vlayout.addWidget(self.imageGroupBox2)
        self.vlayout.addWidget(self.inputImageView)
        self.vlayout.addWidget(self.coordinatesLabel)
        self.vlayout.addWidget(self.GroupBox1)
        self.vlayout.addWidget(self.openButton)
        self.vlayout.addWidget(self.startButton)
        # self.vlayout.addWidget(self.line)
        # self.vlayout.addWidget(self.startCoorBtn)
        # self.vlayout.addWidget(self.saveButton)
        self.vlayout.addWidget(self.closeButton)
        # self.vlayout.addWidget(self.horizontal)
        self.setLayout(self.vlayout)

        self.show()

    def okButtonOnClick(self):
        try:
            tmp = float(self.line.text())
            if tmp > 1 or tmp < 0:
                notInRangeWarning = QMessageBox.information(self, "Warning", "Please input number in range(0, 1)!")
            # print("input is not in range 0-1")
            else:
                self.scalevalue = float(self.line.text())
                self.coordinatesLabel.setText("coordinates = (" + str(self.x) + " , " + str(self.y) + ")")
                # print(self.scalevalue)
        except BaseException:
            notNumberWarning = QMessageBox.information(self, "Warning", "Please input valid number!")
            # print("input is not number")

    def closeApp(self):
        reply = QMessageBox.question(
            self,
            "Close Message",
            "Are you sure to exit?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()

    def openFileNameDialog(self):
        self.inputImagePath, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpeg *.jpg *.bmp)")
        if self.inputImagePath:
            # print(self.inputImagePath)
            self.showInputImageView()

    def showInputImageView(self):
        pixmap = QPixmap(self.inputImagePath).scaled(450, 300)
        self.inputImageView.setPixmap(pixmap)
        self.imageselected = True

    def mousePressEvent(self, QMouseEvent):
        tmpx = QMouseEvent.x() - 20
        tmpy = QMouseEvent.y() - 20
        if tmpx < 450 and tmpx > 0 and tmpx > 0 and tmpy < 300:
            self.x = QMouseEvent.x() - 20
            self.y = QMouseEvent.y() - 20
            self.coordinatesLabel.setText("coordinates = (" + str(self.x) + " , " + str(self.y) + ")")
            # print("self.x = ", self.x, "self.y = ", self.y)
        # print("click x = ", QMouseEvent.x(), "click y = ", QMouseEvent.y())

    # def checkInputFormatMsgBox(self):
    #     checkbox = QMessageBox.about(self, "Image Format Incorrect", "Please select jpg/png format only")

    # @pyqtSlot()
    def openButtonOnClick(self):
        self.openFileNameDialog()

    def startButtonOnClick(self):
        if self.imageselected:
            imagepath = self.inputImagePath
            # print(imagepath)
            # print("self.scalevalue = ", self.scalevalue)
            mask, cuts = main_gmm(imagepath, self.scalevalue, self.x, self.y)
        else:
            msg = QMessageBox.information(self, "Warning", "No image selected")
            # print("no image selected")

    # def startCoorOnClick(self):
    #     print(self.x - 20, self.y - 20)


#if __name__ == '__main__':
def main():
    app = QApplication(sys.argv)
    firstUI = UI()
    sys.exit(app.exec_())

