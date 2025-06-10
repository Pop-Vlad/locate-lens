import io
import threading
from functools import partial

from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QBuffer
from PyQt5.QtGui import QIcon, QImage, QPainter
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QAction, QLabel, QPushButton, QFileDialog, QMessageBox, \
    QHBoxLayout, QSpacerItem, QSizePolicy, QStatusBar, QMenuBar, QMenu

import utils
from ui.Adapter import Adapter
from ui.map import QGoogleMap

icons_dir = "ui/icons/"
default_model = utils.ModelType.ViT


class InputCanvas(QWidget):

    # The canvas where the user can upload an image
    def __init__(self):
        super().__init__()
        # Create a new image with fixed size and white background
        self.image = QImage(self.width(), self.height(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.setFixedHeight(512)
        self.setFixedWidth(512)
        self.update()

    # Override the paint event to draw the image
    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())


class UiMainWindow(object):

    # The main window of the application
    def setupUi(self, mainWindow):
        self.adapter = Adapter(default_model)

        mainWindow.setObjectName("GeoLocate")
        mainWindow.resize(1150, 597)
        self.window = mainWindow
        self.centralwidget = QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 1130, 542))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        # input area
        self.inputArea = InputCanvas()
        self.verticalLayout_3.addWidget(self.inputArea)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)

        # run button
        self.runButton = QPushButton(self.horizontalLayoutWidget)
        self.runButton.setMinimumSize(QtCore.QSize(0, 0))
        self.runButton.setObjectName("runButton")
        self.runButton.clicked.connect(self.startLocalsatioinThread)
        self.horizontalLayout_2.addWidget(self.runButton)

        spacerItem1 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        spacerItem2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")

        # output area
        self.outputMap = QGoogleMap()
        self.outputMap.centerAt(0, 0)
        self.outputMap.setZoom(2)
        self.outputMap.update()
        self.verticalLayout_4.addWidget(self.outputMap)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)

        # coordinates label
        self.coordinatesLabel = QLabel(self.horizontalLayoutWidget)
        self.coordinatesLabel.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.coordinatesLabel)

        spacerItem4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem4)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        mainWindow.setCentralWidget(self.centralwidget)

        self.statusbar = QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        # menu bar
        self.menubar = QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1170, 25))
        self.menubar.setObjectName("menubar")
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuModelType = QMenu(self.menubar)
        self.menuModelType.setObjectName("menuModelType")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        mainWindow.setMenuBar(self.menubar)
        self.actionOpen = QAction(QIcon(icons_dir + "open.png"), "Open Image")
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.triggered.connect(self.open)
        self.actionOpen.setShortcut("Ctrl+O")
        self.actionSave = QAction(QIcon(icons_dir + "save.png"), "Save result")
        self.actionSave.setObjectName("actionSave")
        self.actionSave.triggered.connect(self.saveOutput)
        self.actionSave.setShortcut("Ctrl+Shift+S")
        self.actionClear = QAction(QIcon(icons_dir + "clear.png"), "Clear canvas")
        self.actionClear.setObjectName("actionClear")
        self.actionClear.triggered.connect(self.clear)
        self.actionClear.setShortcut("Ctrl+Del")
        self.actionExit = QAction(QIcon(icons_dir + "exit.png"), "Exit")
        self.actionExit.setObjectName("actionExit")
        self.actionExit.triggered.connect(self.exitProgram)
        self.actionExit.setShortcut("Ctrl+Q")
        self.actionAbout = QAction(QIcon(icons_dir + "about.png"), "About")
        self.actionAbout.setObjectName("actionAbout")
        self.actionAbout.setShortcut("Ctrl+I")
        self.actionAbout.triggered.connect(self.displayAbout)

        # Model type selection
        self.changeModelTypeActions = []
        for model_type in utils.ModelType:
            model_type_name = str(model_type.name.split(".")[-1])
            change_model_action = QAction(model_type_name)
            change_model_action.setCheckable(True)
            if model_type == default_model:
                change_model_action.setChecked(True)
            change_model_action.triggered.connect(partial(self.changeModelType, model_type))
            self.changeModelTypeActions.append(change_model_action)
            self.menuModelType.addAction(self.changeModelTypeActions[-1])

        # add menu items
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionClear)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuModelType.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.runButton.setText(_translate("MainWindow", "Run"))
        self.coordinatesLabel.setText(_translate("MainWindow", "Coordinates"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuModelType.setTitle(_translate("MainWindow", "Model type"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionOpen.setText(_translate("MainWindow", "Open Image"))
        self.actionSave.setText(_translate("MainWindow", "Save Result"))
        self.actionClear.setText(_translate("MainWindow", "Clear"))
        self.actionExit.setText(_translate("MainWindow", "Quit"))
        self.actionAbout.setText(_translate("MainWindow", "About"))

    def startLocalsatioinThread(self):
        # run localization inference in separate thread
        t = threading.Thread(target=self.locateImage, args=())
        t.start()

    def locateImage(self):
        # retrieve image from canvas and convert it to PIL image
        image = self.inputArea.image
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        image.save(buffer, "JPG")
        pil_im = Image.open(io.BytesIO(buffer.data()))

        # run inference
        lat, lng = self.adapter.runForImage(pil_im)
        print("Located image at lat: {}, lng: {}".format(lat, lng))

        # update coordinates and map
        self.coordinatesLabel.setText("Coordinates: " + str(lat) + ", " + str(lng))
        self.outputMap.centerAt(lat, lng)
        self.outputMap.addMarker("MyDragableMark", lat, lng, **dict(
            icon="http://maps.gstatic.com/mapfiles/ridefinder-images/mm_20_red.png",
            draggable=True,
            title="marker"
        ))

    def changeModelType(self, model_type):
        # change model type
        self.adapter.changeModel(model_type)
        # update menu items
        for action in self.changeModelTypeActions:
            action.setChecked(False)
        self.changeModelTypeActions[model_type.value - 1].setChecked(True)
        print("Model type changed to " + str(model_type.name.split(".")[-1]))

    def saveOutput(self):
        # save map as an image
        filePath, _ = QFileDialog.getSaveFileName(self.menuFile, "Save Image", "",
                                                  "PNG(*.png);;JPG(*.jpg *.jpeg);;All Files (*.*)")
        if filePath == "":
            return
        self.outputMap.grab().save(filePath)
        print("Saved map to " + filePath)

    def open(self):
        # open image from file and display it on canvas
        filePath, _ = QFileDialog.getOpenFileName(self.menuFile, "Open Image", "",
                                                  "JPG(*.jpg *.jpeg);;PNG(*.png);;All Files (*.*)")
        if filePath == "":
            return
        with open(filePath, 'rb') as f:
            content = f.read()

        # Load the data from the file to the image.
        self.inputArea.image.loadFromData(content)
        # Scales it and updates the drawing area.
        self.inputArea.image = self.inputArea.image.scaled(self.inputArea.width(), self.inputArea.height(),
                                                           QtCore.Qt.IgnoreAspectRatio)
        self.inputArea.update()
        print("Loaded image from " + filePath)

    def clear(self):
        # clear canvas and coordinates
        # fill input with white
        self.inputArea.image.fill(Qt.white)
        self.inputArea.update()
        # remove output coordinates
        self.coordinatesLabel.setText("Coordinates")
        print("Cleared input and output")

    def exitProgram(self):
        QtCore.QCoreApplication.quit()

    def displayAbout(self):
        text = "<p>This is an image geo-localisatioin application that uses a Transformer-based approach to find the coordinates of a given iamge." \
               "You can open a ground-level view image into the left panel. Then press \"Run\" in order to run the geo-localisation algorithm and see the location of the image in the right panel." \
               "You can save the output map as an images. PNG and JPG image formats are supported.</p>"
        QMessageBox.about(self.window, "About", text)
        print("Displayed about message")
