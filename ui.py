from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSlot
from main_ui import *


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
        self.coordinatesLabel.setText(
            "coordinates = (" + str(self.x) + " , " + str(self.y) + ")")
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
                notInRangeWarning = QMessageBox.information(
                    self, "Warning", "Please input number in range(0, 1)!")
            # print("input is not in range 0-1")
            else:
                self.scalevalue = float(self.line.text())
                self.coordinatesLabel.setText(
                    "coordinates = (" + str(self.x) + " , " + str(self.y) + ")")
                # print(self.scalevalue)
        except BaseException:
            notNumberWarning = QMessageBox.information(
                self, "Warning", "Please input valid number!")
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
            self.coordinatesLabel.setText(
                "coordinates = (" + str(self.x) + " , " + str(self.y) + ")")
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    firstUI = UI()
    sys.exit(app.exec_())
