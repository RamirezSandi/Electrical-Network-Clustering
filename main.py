
"""
Created on Mon Aug 17 20:02:32 2020

@author: sramirez
"""
import sys
import os

#Dependencias para el UI
from PyQt5.QtCore import *
from PyQt5 import QtCore 
from PyQt5.QtGui import QDesktopServices
from PyQt5 import QtGui #Paquetes requeridos para crear ventanas de diálogo e interfaz gráfica.
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox, QPushButton, QDialogButtonBox, QRadioButton, QGroupBox
import traceback

#Dependencias para leer Datasets
import pandas as pd
import geopandas as gpd
import numpy as np
import csv
from sklearn import preprocessing
from statistics import mode, mean

#Dependencias para los 3 algoritmos de clustering
import math
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn import metrics
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn import datasets
import matplotlib.ticker as ticker
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids

"""
Aqui empieza el codigo
"""

class ClusteringExe(QMainWindow):
    def __init__(self):
        super(ClusteringExe,self).__init__()
        uic.loadUi('Clustering_UI.ui',self)
        
        # self.progressbar = self.findChild(QGroupBox, "Progreso")
        # self.progressbar.hide()
        # self.setup = self.findChild(QGroupBox, "Setup")
        
        #Select QGIS or Tabla de Datos
        self.rbtn1 = self.findChild(QRadioButton, "BotonTDD")
        self.rbtn2 = self.findChild(QRadioButton, "BotonQGIS")
        

        
        #If select TablaDeDatos
        self.tD = self.findChild(QLineEdit, "TextTDD")
        self.findTD = self.findChild(QPushButton, "BotonFindTDD")
        self.findTD.clicked.connect(self.selectTD)
    
        
        #if select QGIS
        self.selectQGIS()        
        
        
        #select Output Folder
        self.output = self.findChild(QLineEdit, "CarpetaDestino")
        self.findOutput = self.findChild(QPushButton, "FindCarpetaDestino")
        self.findOutput.clicked.connect(self.selectOutput)
        
        self.accept = self.findChild(QDialogButtonBox, "ButtonBox")
        self.accept.accepted.connect(self.run)
        
        
        self.show()
        
    def selectTD(self):
        tablaDD, _ = QFileDialog.getOpenFileName(self, "Seleccione el archivo con la tabla de datos","", "*.csv" )
        self.tD.setText(tablaDD)
        
    def selectOutput(self):
        foldername = QFileDialog.getExistingDirectory(self, "Seleccione la carpeta donde se guardarán los resultados", "", )
        self.output.setText(foldername)

        
    def selectQGIS(self):
        #lineas de MEDIA TENSION
        self.MT1 = self.findChild(QLineEdit, "TextMT1")
        self.MT2 = self.findChild(QLineEdit, "TextMT2")
        self.MT3 = self.findChild(QLineEdit, "TextMT3")
        
        self.findMT1 = self.findChild(QPushButton, "BotonFindMT1")
        self.findMT2 = self.findChild(QPushButton, "BotonFindMT2")
        self.findMT3 = self.findChild(QPushButton, "BotonFindMT3")
        
        self.findMT1.clicked.connect(self.selectMT1)
        self.findMT2.clicked.connect(self.selectMT2)
        self.findMT3.clicked.connect(self.selectMT3)

        #lineas de Baja Tension
        self.BT1 = self.findChild(QLineEdit, "TextBT1")
        self.BT2 = self.findChild(QLineEdit, "TextBT2")
        self.BT3 = self.findChild(QLineEdit, "TextBT3")
        
        self.findBT1 = self.findChild(QPushButton, "BotonFindBT1")
        self.findBT2 = self.findChild(QPushButton, "BotonFindBT2")
        self.findBT3 = self.findChild(QPushButton, "BotonFindBT3")
        
        self.findBT1.clicked.connect(self.selectBT1)
        self.findBT2.clicked.connect(self.selectBT2)
        self.findBT3.clicked.connect(self.selectBT3)
        
        #trafos
        self.Trafo1 = self.findChild(QLineEdit, "TextTrafo1")
        self.Trafo2 = self.findChild(QLineEdit, "TextTrafo2")
        self.Trafo3 = self.findChild(QLineEdit, "TextTrafo3")
        
        self.findTrafo1 = self.findChild(QPushButton, "BotonFindTrafo1")
        self.findTrafo2 = self.findChild(QPushButton, "BotonFindTrafo2")
        self.findTrafo3 = self.findChild(QPushButton, "BotonFindTrafo3")
        
        self.findTrafo1.clicked.connect(self.selectTrafo1)
        self.findTrafo2.clicked.connect(self.selectTrafo2)
        self.findTrafo3.clicked.connect(self.selectTrafo3)

        #cargas
        self.Cargas1 = self.findChild(QLineEdit, "TextCarga")
        self.Cargas2 = self.findChild(QLineEdit, "TextCarga_2")
        self.Cargas3 = self.findChild(QLineEdit, "TextCarga_3")
        self.Cargas4 = self.findChild(QLineEdit, "TextCarga_4")
        self.Cargas5 = self.findChild(QLineEdit, "TextCarga_5")
        self.Cargas6 = self.findChild(QLineEdit, "TextCarga_6")
        
        self.findCargas1 = self.findChild(QPushButton, "BotonFindCarga")
        self.findCargas2 = self.findChild(QPushButton, "BotonFindCarga_2")
        self.findCargas3 = self.findChild(QPushButton, "BotonFindCarga_3")
        self.findCargas4 = self.findChild(QPushButton, "BotonFindCarga_4")
        self.findCargas5 = self.findChild(QPushButton, "BotonFindCarga_5")
        self.findCargas6 = self.findChild(QPushButton, "BotonFindCarga_6")
        
        self.findCargas1.clicked.connect(self.selectCargas1)
        self.findCargas2.clicked.connect(self.selectCargas2)
        self.findCargas3.clicked.connect(self.selectCargas3)
        self.findCargas4.clicked.connect(self.selectCargas4)
        self.findCargas5.clicked.connect(self.selectCargas5)
        self.findCargas6.clicked.connect(self.selectCargas6)
        
        #Archivo con redes de distribucion
        self.redDistribucion = self.findChild(QLineEdit, "TextRD")
        self.findRD = self.findChild(QPushButton, "BotonFindRD")
        self.findRD.clicked.connect(self.selectRD)
        
    def selectMT1(self):
        lineaMT1, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Media Tension","","*.shp" )
        self.MT1.setText(lineaMT1)
    def selectMT2(self):
        lineaMT2, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Media Tension","","*.shp" )
        self.MT2.setText(lineaMT2)        
    def selectMT3(self):
        lineaMT3, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Media Tension","","*.shp" )
        self.MT3.setText(lineaMT3)    

    def selectBT1(self):
        lineaBT1, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Baja Tension","","*.shp" )
        self.BT1.setText(lineaBT1)
    def selectBT2(self):
        lineaBT2, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Baja Tension","","*.shp" )
        self.BT2.setText(lineaBT2)        
    def selectBT3(self):
        lineaBT3, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Baja Tension","","*.shp" )
        self.BT3.setText(lineaBT3)  
            
    def selectTrafo1(self):
        trafo1, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Transformadores","","*.shp" )
        self.Trafo1.setText(trafo1)
    def selectTrafo2(self):
        trafo2, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Transformadores","","*.shp" )
        self.Trafo2.setText(trafo2)        
    def selectTrafo3(self):
        trafo3, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Transformadores","","*.shp" )
        self.Trafo3.setText(trafo3)  
            
    def selectCargas1(self):
        carga1, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Cargas","","*.shp" )
        self.Cargas1.setText(carga1)
    def selectCargas2(self):
        carga2, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Cargas","","*.shp" )
        self.Cargas2.setText(carga2)
    def selectCargas3(self):
        carga3, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Cargas","","*.shp" )
        self.Cargas3.setText(carga3)
    def selectCargas4(self):
        carga4, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Cargas","","*.shp" )
        self.Cargas4.setText(carga4)
    def selectCargas5(self):
        carga5, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Cargas","","*.shp" )
        self.Cargas5.setText(carga5)
    def selectCargas6(self):
        carga6, _ = QFileDialog.getOpenFileName(self, "Seleccione la capa de Cargas","","*.shp" )
        self.Cargas6.setText(carga6)            
    
    def selectRD(self):
        redD, _ = QFileDialog.getOpenFileName(self, "Seleccione el archivo de redes de distribucion","","*.txt" )
        self.redDistribucion.setText(redD)
    
    
    
    def run(self):
        """
        Primero revisamos si existe el output folder donde vamos a guardar los resultados
        """
        # self.setup.hide()
        # self.progressbar.show()
        
        
        if os.path.exists(self.output.text()):
            print("Output Folder exists")
        else:
            mensaje = str( "Seleccionar el directorio donde se guardaran los resultados")
            QMessageBox.critical(None,"Error: Output Files", mensaje)       
        
        #archivo de redes de distribucion
        if (os.path.isfile(self.redDistribucion.text())):
            print("found the file")
            red = open(self.redDistribucion.text(), 'r')
            reader = csv.reader(red, delimiter = '\t')
            red = [row for row in reader]
            redDistribucion = []
            for i in range(len(red)):
                redDistribucion.append(red[i])
                
        else:
            mensaje = str( "Seleccionar un archivo con las redes de distribucion")
            QMessageBox.critical(None,"Error: Cargas", mensaje) 
        
        """
        Luego revisamos si se trabaja con una tabla de datos o con capas de QGIS
        """

        """
        En caso de ser una tabla de datos, se va a guardar la tabla.
        """
        if self.rbtn1.isChecked():
            print("Tabla de datos")
            
            if (os.path.isfile(self.tD.text())):
                print("found the file")

                
                tD = pd.read_csv(self.tD.text())
                tDcolumns = tD.columns.tolist()
                print(tD)
                min_max_scaler = preprocessing.MinMaxScaler()
                tDnorm = min_max_scaler.fit_transform(tD)
                tablaDeDatos = pd.DataFrame(tDnorm, columns = tDcolumns)
                tD = tablaDeDatos.values.tolist()
                
            else:
                mensaje = str( "Seleccionar el archivo donde se encuentra la tabla de datos")
                QMessageBox.critical(None,"Error: Tabla de Datos", mensaje)
        
        
        # en caso de ser capas de QGIS, se van a revisar cada capa y luego se forma la tabla de datos
        
        elif self.rbtn2.isChecked():
            print("Capas de Qgis")
            redes = []
            # ------------------------------------------
            #iniciamos con las capas de media tension
            if (os.path.isfile(self.MT1.text())):
                print("found the file")
                MT1 = gpd.read_file(self.MT1.text())
                MT1columns = MT1.columns.tolist()
                MT1values =  MT1.values.tolist()
                redes.append([MT1, "MT1"])
                
                MT2 = ""
                MT3 = ""
                if self.MT2.text() != "" and os.path.isfile(self.MT2.text()):
                    MT2 = gpd.read_file(self.MT2.text())
                    MT2columns = MT2.columns.tolist()
                    MT2values =  MT2.values.tolist()
                    redes.append([MT2, "MT2"])
                    
                if self.MT3.text() != "" and os.path.isfile(self.MT3.text()):
                    MT3 = gpd.read_file(self.MT3.text())
                    MT3columns = MT3.columns.tolist()
                    MT3values =  MT3.values.tolist()
                    redes.append([MT3, "MT3"])
                    
            else:
                mensaje = str( "Seleccionar minimo un archivo de capas de media tension")
                QMessageBox.critical(None,"Error: LineaMediaTension", mensaje)
                
            #------------------------------------------
            
            #------------------------------------------
            #Capas de Baja tension
            if (os.path.isfile(self.BT1.text())):
                print("found the file")
                BT1 = gpd.read_file(self.MT1.text())
                BT1columns = BT1.columns.tolist()
                BT1values =  BT1.values.tolist()
                redes.append([BT1, "BT1"])
                
                BT2 = ""
                BT3 = ""
                if self.BT2.text() != "" and os.path.isfile(self.BT2.text()):
                    BT2 = gpd.read_file(self.BT2.text())
                    BT2columns = BT2.columns.tolist()
                    BT2values =  BT2.values.tolist()
                    redes.append([BT2, "BT2"])
                    
                if self.BT3.text() != "" and os.path.isfile(self.BT3.text()):
                    BT3 = gpd.read_file(self.BT3.text()) 
                    BT3columns = BT3.columns.tolist()
                    BT3values =  BT3.values.tolist()
                    redes.append([BT3, "BT3"])
                
            else:
                mensaje = str( "Seleccionar minimo un archivo de capas de baja tension")
                QMessageBox.critical(None,"Error: LineaBajaTension", mensaje)
                
            #------------------------------------------
            
            #------------------------------------------
            #Capas de Transformadores

            if (os.path.isfile(self.Trafo1.text())):
                print("found the file")
                TR1 = gpd.read_file(self.Trafo1.text())
                TR1columns = TR1.columns.tolist()
                TR1values = TR1.values.tolist()
                redes.append([TR1, "TR1"])
                
                TR2 = ""
                TR3 = ""
                if self.Trafo2.text() != "" and os.path.isfile(self.Trafo2.text()):
                    TR2 = gpd.read_file(self.Trafo2.text())
                    TR2columns = TR2.columns.tolist()
                    TR2values = TR2.values.tolist()
                    redes.append([TR2, "TR2"])
                    
                if self.Trafo3.text() != "" and os.path.isfile(self.Trafo3.text()):
                    TR3 = gpd.read_file(self.Trafo3.text())
                    TR3columns = TR3.columns.tolist()
                    TR3values = TR3.values.tolist()
                    redes.append([TR3, "TR3"])
                    
            else:
                mensaje = str( "Seleccionar minimo un archivo de capas de transformadores")
                QMessageBox.critical(None,"Error: Transformadores", mensaje)
            
            #------------------------------------------
            
            #------------------------------------------
            #Capas de cargas
            
            if (os.path.isfile(self.Cargas1.text())):
                print("found the file")
                Cargas1 = gpd.read_file(self.Cargas1.text())
                Cargas1columns = Cargas1.columns.tolist()
                Cargas1values = Cargas1.values.tolist()
                redes.append([Cargas1, "Cargas1"])

                Cargas2 = ""
                Cargas3 = ""
                Cargas4 = ""
                Cargas5 = ""
                Cargas6 = ""
                if self.Cargas2.text() != "" and os.path.isfile(self.Cargas2.text()):
                    Cargas2 = gpd.read_file(self.Cargas2.text())
                    Cargas2columns = Cargas2.columns.tolist()
                    Cargas2values = Cargas2.values.tolist()
                    redes.append([Cargas2, "Cargas2"])
                    
                if self.Cargas3.text() != "" and os.path.isfile(self.Cargas3.text()):
                    Cargas3 = gpd.read_file(self.Cargas3.text())
                    Cargas3columns = Cargas3.columns.tolist()
                    Cargas3values = Cargas3.values.tolist()
                    redes.append([Cargas3, "Cargas3"])
                    
                if self.Cargas4.text() != "" and os.path.isfile(self.Cargas4.text()):
                    Cargas4 = gpd.read_file(self.Cargas4.text())
                    Cargas4columns = Cargas4.columns.tolist()
                    Cargas4values = Cargas4.values.tolist()
                    redes.append([Cargas4, "Cargas4"])
                    
                if self.Cargas5.text() != "" and os.path.isfile(self.Cargas5.text()):
                    Cargas5 = gpd.read_file(self.Cargas5.text())
                    Cargas5columns = Cargas5.columns.tolist()
                    Cargas5values = Cargas5.values.tolist()
                    redes.append([Cargas5, "Cargas5"])
                    
                if self.Cargas6.text() != "" and os.path.isfile(self.Cargas6.text()):
                    Cargas6 = gpd.read_file(self.Cargas6.text())
                    Cargas6columns = Cargas6.columns.tolist()
                    Cargas6values = Cargas6.values.tolist()
                    redes.append([Cargas6, "Cargas6"])
                                    
            else:
                mensaje = str( "Seleccionar minimo un archivo de capas de cargas")
                QMessageBox.critical(None,"Error: Cargas", mensaje)
            #------------------------------------------
            
            #------------------------------------------              
                
            #------------------------------------------
            #archivo con valores de resistencia

            res =[]
            res.append(['calibre', 'cobre', 'aluminio'])
            res.append(['6', '1.61', '2.66'])
            res.append(['4', '1.02', '1.67'])
            res.append(['2', '0.66', '1.05'])
            res.append(['1/0', '0.43', '0.69'])
            res.append(['2/0', '0.33', '0.52'])
            res.append(['3/0', '0.269', '0.43'])
            res.append(['4/0', '0.220', '0.36'])
            res.append(['250', '0.187', '0.295'])
            res.append(['300', '0.161', '0.249'])
            res.append(['350', '0.141', '0.217'])
            res.append(['400', '0.125', '0.194'])
            res.append(['500', '0.105', '0.157'])
            res.append(['600', '0.092', '0.135'])
            res.append(['750', '0.079', '0.112'])
            res.append(['1000', '0.062', '0.089'])
            #set up la tabla de datos
            tD = []
            
            for i in range(len(redDistribucion)):
                """
                aqui se obtienen las longitudes de los circuitos de la base de datos
                """
                lMT1 = self.longitud(MT1columns, MT1values, redDistribucion[i][1])
                lMT2=0
                lMT3=0
                if isinstance(MT2, pd.DataFrame):
                    lMT2 = self.longitud(MT2columns, MT2values, redDistribucion[i][1])
                if isinstance(MT3, pd.DataFrame):
                    lMT3 = self.longitud(MT3columns, MT3values, redDistribucion[i][1])
                
                longitudTotal = lMT1 +lMT2 +lMT3
                
                """
                aqui se obtiene la tension electrica y la configuracion de las lineas
                """
                volt1, conf1 = self.voltage(MT1columns, MT1values, redDistribucion[i][1])
                volt2 = conf2 = 0
                volt3 = conf3 = 0
                
                if isinstance(MT2, pd.DataFrame):
                    volt2, conf2 = self.voltage(MT2columns, MT2values, redDistribucion[i][1])
                if isinstance(MT3, pd.DataFrame):
                    volt3, conf3 = self.voltage(MT3columns, MT3values, redDistribucion[i][1])
                
                v = [volt1, volt2, volt3]
                c = [conf1, conf2, conf3]
                v.sort(reverse = True)
                c.sort(reverse = True)
                
                volt = v[0]
                config = c[0]
                
                """
                aqui se obtiene si la red es aerea o subt
                """
                tipo1 = self.subt(MT1columns,MT1values,redDistribucion[i][1])
                tipo2 = tipo3 = None
                
                
                if isinstance(MT2, pd.DataFrame):
                    tipo2 = self.subt(MT2columns,MT2values,redDistribucion[i][1])
                if isinstance(MT3, pd.DataFrame):
                    tipo3 = self.subt(MT3columns,MT3values,redDistribucion[i][1])
                    print(tipo3)
                if tipo1 == None:
                    if tipo2!= None:
                        tipo=tipo2
                    if tipo3!= None:
                        tipo=tipo3
                else:
                    if tipo2 == None and tipo3 == None:
                        tipo = tipo1
                    if tipo2 == None and tipo3 != None:
                        tipo = (tipo1+tipo3)/2
                    if tipo3 == None and tipo2 != None:
                        tipo = (tipo1+tipo2)/2
                
                """
                aqui se obtiene la impedancia de los circuitos de la base de datos
                """
                iMT1 = self.impe(MT1columns, MT1values, redDistribucion[i][1], res)
                iMT2=0
                iMT3=0
                if isinstance(MT2, pd.DataFrame):
                    iMT2 = self.impe(MT2columns, MT2values, redDistribucion[i][1], res)
                if isinstance(MT3, pd.DataFrame):
                    iMT3 = self.impe(MT3columns, MT3values, redDistribucion[i][1], res)
                
                impedanciaTotal = iMT1 +iMT2 +iMT3
                
                """
                aqui se obtiene la cantidad de transformadores de la base de datos
                """
                monoT1, triT1 = self.transf(TR1columns, TR1values, redDistribucion[i][1])
                monoT2=triT2=0
                monoT3=triT3=0
                if isinstance(TR2, pd.DataFrame):
                    monoT2, triT2 = self.impe(TR2columns, TR2values, redDistribucion[i][1])
                if isinstance(TR3, pd.DataFrame):
                    monoT3, triT3 = self.impe(TR3columns, TR3values, redDistribucion[i][1])
                
                monoT = monoT1 + monoT2 + monoT3
                triT = triT1 + triT2 +triT3
            
                """
                aqui se obtiene el numero de cargas de la base de datos
                """
                cTotal1, cResidencial1, cComercial1, cIndustrial1, kwhT1 = self.loads(Cargas1columns, Cargas1values, redDistribucion[i][1])
                cTotal2 = cResidencial2 = cComercial2 = cIndustrial2 = kwhT2 = 0
                cTotal3 = cResidencial3 = cComercial3 = cIndustrial3 = kwhT3 = 0
                cTotal4 = cResidencial4 = cComercial4 = cIndustrial4 = kwhT4 = 0
                cTotal5 = cResidencial5 = cComercial5 = cIndustrial5 = kwhT5 = 0
                cTotal6 = cResidencial6 = cComercial6 = cIndustrial6 = kwhT6 = 0
                if isinstance(Cargas2, pd.DataFrame):
                    cTotal2, cResidencial2, cComercial2, cIndustrial2, kwhT2 = self.loads(Cargas2columns, Cargas2values, redDistribucion[i][1])
                if isinstance(Cargas3, pd.DataFrame):
                    cTotal3, cResidencial3, cComercial3, cIndustrial3, kwhT3 = self.loads(Cargas3columns, Cargas3values, redDistribucion[i][1])  
                if isinstance(Cargas4, pd.DataFrame):
                    cTotal4, cResidencial4, cComercial4, cIndustrial4, kwhT4 = self.loads(Cargas4columns, Cargas4values, redDistribucion[i][1])
                if isinstance(Cargas5, pd.DataFrame):
                    cTotal5, cResidencial5, cComercial5, cIndustrial5, kwhT5 = self.loads(Cargas5columns, Cargas5values, redDistribucion[i][1])
                if isinstance(Cargas6, pd.DataFrame):
                    cTotal6, cResidencial6, cComercial6, cIndustrial6, kwhT6 = self.loads(Cargas6columns, Cargas6values, redDistribucion[i][1])
                
                cargasTotal = cTotal1 + cTotal2 +cTotal3 +cTotal4 +cTotal5 +cTotal6
                cargasResidencial = cResidencial1 + cResidencial2 + cResidencial3 + cResidencial4 + cResidencial5 + cResidencial6
                cargasComercial = cComercial1 + cComercial2 +cComercial3 +cComercial4 +cComercial5 +cComercial6
                cargasIndustrial = cIndustrial1 + cIndustrial2 + cIndustrial3 + cIndustrial4 + cIndustrial5 + cIndustrial6
                consumoTotal = kwhT1 + kwhT2 + kwhT3 + kwhT4 + kwhT5 + kwhT6
                
                tD.append([monoT, triT, longitudTotal, impedanciaTotal, volt, cargasResidencial, cargasComercial, cargasIndustrial, tipo])
            tDnorm = []
            tDtranspose = np.array(tD).T
            for i in range(len(tD)):
                normRow = []
                for j in range(len(tD[i])):
                    if (tD[i][j] == min(tDtranspose[j]) and tD[i][j] == max(tDtranspose[j])) :
                        x=1
                    else:
                        x = (tD[i][j] - min(tDtranspose[j])) / (max(tDtranspose[j]) - min(tDtranspose[j]) )
                    normRow.append(x)
                tDnorm.append(normRow)
            tablaDeDatos = pd.DataFrame(tDnorm, columns = ['Transformador Monofasico', 'Transformador Trifasico' , 'Longitud Total [m]', 'Impedancia Total [ohm]', 'Tensión de línea','Cargas Res', 'Cargas Com', 'Cargas Ind', "Tipo"])
            np.savetxt("File.txt", tablaDeDatos.values, fmt='%d')

            
        else: 
            mensaje = str( "Seleccionar el tipo de dato de entrada")
            QMessageBox.critical(None,"Error: Datos de entrada", mensaje)


        """
        Aqui empieza el clustering con la tabla de datos lista
        """
        
        """
        Primero el K-MEANS
        """
        indicadoresKmeans = self.kmeans(tablaDeDatos, tDnorm, self.output.text())
        
        """
        Luego Hierarchy
        """
        indicadoresHierarchy = self.hierarchy(tablaDeDatos, tDnorm, self.output.text())
        
        """
        Finalmente K-Medoids
        """
        indicadoresKmedoids = self.kmedoids(tablaDeDatos, tDnorm, self.output.text())
        
        x = self.compare(indicadoresKmeans, indicadoresHierarchy, indicadoresKmedoids, self.output.text())
        #Recommendation
        for i in range(len(x)):
            print("For %s clusters, recommend using %s" % (x[i][0], x[i][1]) )
            
            
        #Ask for user input
        while True:
            try:
                
                clustermethod = int(input('Which Clustering Method to use (1.KMeans, 2.Hierarchy, 3.KMedoids)? \n'))
                clusternumber = int(input('How many clusters?:  \n'))
            except ValueError:
                print("Enter valid entry for ClusterMethod and Number of Clusters")
                continue
            else:
                break
        
        print("Starting Clustering")

        
        if clustermethod == 1:
            centroid, preds = self.kmeansFinal(int(clusternumber), tablaDeDatos, tD, tDnorm, redDistribucion, self.output.text())
        elif clustermethod ==2:
            centroid, preds = self.hierarchyFinal(int(clusternumber), tablaDeDatos, tD, tDnorm, redDistribucion, self.output.text())
        elif clustermethod == 3:
            centroid, preds = self.kmedoidsFinal(int(clusternumber), tablaDeDatos, tD, tDnorm, redDistribucion, self.output.text())
        
        
        print("Saving Final Clusters")
        
        if self.rbtn1.isChecked():
            self.finalC(int(clusternumber), centroid, tD, tDnorm, preds, redDistribucion, self.output.text(), tablaDeDatos)
        elif self.rbtn2.isChecked():
            self.finalC(int(clusternumber), centroid, tD, tDnorm, preds, redDistribucion, self.output.text(), tablaDeDatos, redes)
        
        print("Done")
        
    """
    Analisis de Redes en capas Qgis
    """    
    def longitud(self, capaC, capaV, red):
    # obtener TOTAL CONDUCTOR LENGHT

        longTotal = 0
        
        for i in range(len(capaC)):
            if capaC[i] == 'FEEDERID':
                feederA = i
            if capaC[i] == 'LENGTH':
                lengthA = i

        
        for i in range(len(capaV)):
            if (capaV[i][feederA] == red):
                longTotal = longTotal + capaV[i][lengthA]

        return longTotal/1000
    
    def impe(self, capaC, capaV, red, res):
        #obtener impedancia total
    
        for i in range(len(capaC)):
            if capaC[i] == 'FEEDERID':
                feeder = i
            if capaC[i] == 'NEUTMAT':
                neutro = i
            if capaC[i] == 'NEUTSIZ':
                calibreneu = i
            if capaC[i] == 'LENGTH':
                length = i
    
        impedancia = 0
        for i in range(len(capaV)):
            imp = 0
            if (capaV[i][feeder] == red):
                if (capaV[i][neutro] == 'CU'):
                    for j in range(len(res)):
                        if capaV[i][calibreneu] == res[j][0]:
                            imp = capaV[i][length] * float(res[j][1])/1000
                elif (capaV[i][neutro] == 'AAC' or capaV[i][neutro] == 'AAAC' or capaV[i][neutro] == None):
                    for j in range(len(res)):
                        if capaV[i][calibreneu] == res[j][0]:
                            imp = capaV[i][length] * float(res[j][2])/1000
                
            impedancia = impedancia + imp
        
        return impedancia
    
    def transf(self, capaC, capaV, red): #transfValues
        #OBTENER # DE TRAFOS EN RED DE DISTRIBUCION
        tfmono = 0
        tftri = 0
        for i in range(len(capaC)):
            if capaC[i] == 'FEEDERID':
                feeder = i
            if capaC[i] == 'CONEXBANCO':
                conexion = i 
            
        
        for i in range(len(capaV)):
            if (capaV[i][feeder] == red):
                
                if (capaV[i][conexion] == 'MONOFASICO'):
                    tfmono = tfmono +1
                elif (capaV[i][conexion] == 'TRIFASICO' or capaV[i][conexion] == 'BOOSTER' or capaV[i][conexion] == 'DELTAABIERTO' or capaV[i][conexion] == 'PARALELO'):
                    tftri = tftri +1
                else:
                    tfmono = tfmono
                    tftri =tftri

        return tfmono, tftri
    
    def loads(self, capaC, capaV, red):
        clientesTotal = 0
        clientesRes = 0
        clientesCom = 0
        clientesInd = 0
        kwhT = 0
        
        for i in range(len(capaC)):
            if capaC[i] == 'FEEDERID':
                feeder= i
            if capaC[i] == 'SECTOR':
                sector = i
            if capaC[i] == 'KWH':
                kwh = i
            
        
        for i in range(len(capaV)):
            if (capaV[i][feeder] == red):
                
                clientesTotal = clientesTotal +1
                
                if (str(capaV[i][sector]) == '1.0'):
                    clientesRes = clientesRes+1
                elif (str(capaV[i][sector]) == '2.0'):
                    clientesCom = clientesCom+1
                elif (str(capaV[i][sector]) == '3.0'):
                    clientesInd = clientesInd+1
                    
                kwhT = kwhT + float(capaV[i][kwh])
        
        return clientesTotal, clientesRes, clientesCom,clientesInd, kwhT
    
    
    def voltage(self, capaC, capaV, red):
         
        code = []
        code.append([120,150,160,210,230,260,270,340,380])
        code.append([4.16,7.20,7.20,12.5,13.2,13.8,13.8,24.9,34.5])
        code.append([1,2,1,1,1,2,1,1,1])
        
        for i in range(len(capaC)):
            if capaC[i] == 'FEEDERID':
                feederA = i
            if capaC[i] == 'NOMVOLT':
                voltageA = i
            
        
        v=[]
        for i in range(len(capaV)):
            if (capaV[i][feederA] == red):
                v.append(capaV[i][voltageA])
        
        
        if not v:
            return 0 , 0
        
        voltage=mode(v)
        for i in range(len(code[0])):
            if voltage == code[0][i]:
                volt = code[1][i]
                conf = code[2][i]

        return volt, conf
    
    def subt(self, capaC, capaV, red):
        for i in range(len(capaC)):
            if capaC[i] == 'FEEDERID':
                feederA = i
        
        for i in range(len(capaV)):
            if (capaV[i][feederA] == red):
                if ("SUB") in capaV[i][feederA]:
                    return 0
                else: 
                    return 1
        
    
    """
    Algoritmos de agrupamiento
    """
    def kmeans(self, tablaDeDatos, tDnorm, destination):
        
        directory = destination + "\Indicadores"
        if os.path.exists(directory) == False:
            os.mkdir(directory)
            
        range_n_clusters = list (range(2,20))
        
        see=[]
        indicadores = []
        sse = {}
        for n_clusters in range_n_clusters:
            
            # Silhouette Coefficient
            clusterer = KMeans(n_clusters=n_clusters , random_state=0)
            preds = clusterer.fit_predict(tablaDeDatos)
            centers = clusterer.cluster_centers_
            sse[n_clusters] = clusterer.inertia_
        
        
            unique, cnt = np.unique(preds, return_counts=True)
            counts = dict(zip(unique, cnt))
            # print(counts)
            
            score = silhouette_score(tablaDeDatos, preds)
            print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
            sample_silhouette_values = silhouette_samples(tablaDeDatos, preds)
                    
        
            #Variance Ratio Criterion
            labels = clusterer.labels_
        
            VRC =  (metrics.calinski_harabasz_score(tablaDeDatos, labels))
            # print('VRC = ' + str(VRC))
        
            #Similarity Matrix Indicator
            #Obtengo la matriz de distancia entre los centros
            Matrix = []
            for i in range(len(centers)):
                point1 = (centers[i][0], centers[i][1], centers[i][2], centers[i][3], centers[i][4], centers[i][5])
                Row = []
                for j in range(len(centers)):
                    point2 = (centers[j][0], centers[j][1], centers[j][2], centers[j][3], centers[j][4], centers[j][5])
                    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
                    Row.append(distance)
                Matrix.append(Row)
        
                  
        
            #obtengo el indicador a partir de esta distancia
            SMI = 0.0
            smirow = []
            for i in range(len(Matrix)):
                for j in range(len(Matrix)):
                    if i > j:
                        smirow.append(  (1- (1/ (math.log(Matrix[i][j]) ) ) )**(-1) )
                    else:
                        break        
            SMI = max(smirow)**(-1)
            # print('SMI = ' + str(SMI))
            
            SSE=0
            for j in range(n_clusters):
                for k in range(len(preds)):
                    point1 = centers[j]
                    if preds[k]==j:
                        point2 = tDnorm[k]
                        SSE = SSE + math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
                    
            # print('SSE = '+ str(SSE))
            see.append(SSE)
            
            indicadores.append([n_clusters, score, sum(sample_silhouette_values)/len(sample_silhouette_values), VRC, SMI, SSE])
            
        # print (indicadores)
        with open(directory+"\Indicadores de Cluster, K-means++, RedesAereas.txt", "w") as output:
            for row in indicadores:
                output.write(str(row) + '\n')
                
        return indicadores
    
    def kmeansFinal (self, n_clusters, tablaDeDatos, tD, tDnorm, redDistribucion, destination):
        
        directory = destination + '\Images'
        if os.path.exists(directory) == False:
            os.mkdir(directory)
            
        kmeans = KMeans(n_clusters, max_iter=3000, init='k-means++', random_state=0)
        preds = kmeans.fit_predict(tablaDeDatos)
        
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        iteration = kmeans.n_iter_
        
        unique, cnt = np.unique(preds, return_counts=True)
        counts = dict(zip(unique, cnt))
        
        score = silhouette_score(tablaDeDatos, preds)
        # print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
        sample_silhouette_values = silhouette_samples(tablaDeDatos, preds)
        
        sse = kmeans.inertia_
        
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, tablaDeDatos)
        fig, ax1 = plt.subplots()
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[preds==i]
        
            ith_cluster_silhouette_values.sort()
        
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
        
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor="black", alpha=0.7)
        
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))
        
            # Compute the new y_lower for next plot
            y_lower = y_upper +5# 10 for the 0 samples
            
        ax1.axvline(x=score, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.set_title("The silhouette plot for the various clusters." + str(n_clusters))
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        plt.savefig(directory + '\FinalResultGS' + str(n_clusters) + '.pdf')
        plt.show()
        
            #Variance Ratio Criterion
        
        VRC =  (metrics.calinski_harabasz_score(tablaDeDatos, labels))
        # print('VRC = ' + str(VRC))
        
        alpha = []
        for i in range(n_clusters):
            alpha.append(str(i+1))
        
        #Similarity Matrix Indicator
        #Obtengo la matriz de distancia entre los centros
        Matrix = []
        for i in range(len(centers)):
            point1 = (centers[i][0], centers[i][1], centers[i][2], centers[i][3], centers[i][4], centers[i][5], centers[i][6], centers[i][7])
            Row = []
            for j in range(len(centers)):
                point2 = (centers[j][0], centers[j][1], centers[j][2], centers[j][3], centers[j][4], centers[j][5], centers[j][6], centers[j][7])
                distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
                Row.append(distance)
            Matrix.append(Row)
        
        fig, ax2 = plt.subplots(1, 1)    
        cax = ax2.matshow(Matrix, cmap='jet',  vmin=0, vmax=1)
        ax2.set_title("Colormap K-means for N= " + str(n_clusters))
        fig.colorbar(cax)
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax2.set_xticklabels(['']+alpha)  
        ax2.set_yticklabels(['']+alpha)
        plt.xticks(rotation=90)
        ax2.set_title("Distancia entre grupos")
        fig.colorbar(cax)
        
        for (i, j), z in np.ndenumerate(Matrix):
            ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
            
        plt.savefig(directory +"\FullColormap.pdf")
        
        
            #obtengo el indicador a partir de esta distancia
        SMI = 0.0
        smirow = []
        for i in range(len(Matrix)):
            for j in range(len(Matrix)):
                if i > j:
                    smirow.append(  (1- (1/ (math.log(Matrix[i][j]) ) ) )**(-1) )
                else:
                    break        
        SMI = max(smirow)**(-1)
        # print('SMI = ' + str(SMI))
            
        SSE=0
        for j in range(n_clusters):
            for k in range(len(preds)):
                point1 = centers[j]
                if preds[k]==j:
                    point2 = tD[k]
                    SSE = SSE + math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))   
        labels =[]
        for i in range(n_clusters):
            labels.append("C-"+str(i+1))
        fig, ax4 = plt.subplots(1, 1)    
        colours = dict(zip(labels, plt.cm.tab20.colors[:len(labels)]))
        cax = ax4.pie(cnt, labels=labels,colors=[colours[key] for key in labels], wedgeprops={"edgecolor":"k",'linewidth': 1, 'antialiased': True}, rotatelabels =90)
        
        ax4.set_title("Distribución de Redes en los Clústers Resultantes")
        plt.savefig(directory +"\PieChart.pdf")
        plt.show()
        
        for i in range(len(cnt)):
            print("In Cluster " + str(i+1) + " there are " + str(cnt[i]) + " distribution networks. Making up " + str(round(cnt[i]/sum(cnt)*100,2)) + "% of the total count")
        
        centroid = []
        for i in range(len(closest)):
            centroid.append(redDistribucion[closest[i]])
        
        
        #feeder to feeder distance
        # this is the distance from one feeder to the next within the same cluster.
        # a small value indicates that this cluster is tightly grouped together which is good
        
    
    
        return centroid, preds
    
    def hierarchy(self, tablaDeDatos, tDnorm, destination):
        
        directory = destination + "\Indicadores"
        if os.path.exists(directory) == False:
            os.mkdir(directory)
            
        plt.figure(figsize=(10, 7))
        plt.title("Customer Dendograms")
        dend = shc.dendrogram(shc.linkage(tablaDeDatos, method='ward'))
        plt.axhline(y=250, color="red", linestyle="--")
        # plt.savefig('Dendogram.pdf')
        plt.show()
        
        clusterer = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
        clusterer.fit_predict(tablaDeDatos)
        
        #kmeans = KMeans(n_clusters=3).fit(tablaDeDatos)
        #print(centroids)
        
        range_n_clusters = list (range(2,20))
        see=[]
        indicadores = []
        for n_clusters in range_n_clusters:
            
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
            preds = clusterer.fit_predict(tablaDeDatos)
        #    centers = clusterer.cluster_centers_
        
            score = silhouette_score(tablaDeDatos, preds)
            # print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
            sample_silhouette_values = silhouette_samples(tablaDeDatos, preds)
            
            #Variance Ratio Criterion
            labels = clusterer.labels_
        
            VRC =  (metrics.calinski_harabasz_score(tablaDeDatos, labels))
            # print('VRC = ' + str(VRC))
        
            #Similarity Matrix Indicator
            #Obtengo la matriz de distancia entre los centros
            centers = []
            for j in range(n_clusters):
                x=[]
                y=[]
                w=[]
                z=[]
                v=[]
                t=[]
                s=[]
                r=[]
                for i in range(len(preds)):
                    if preds[i] == j:
                        x.append(tDnorm[i][0])
                        y.append(tDnorm[i][1])
                        w.append(tDnorm[i][2])
                        z.append(tDnorm[i][3])
                        v.append(tDnorm[i][4])
                        t.append(tDnorm[i][5])
                        s.append(tDnorm[i][6])
                        r.append(tDnorm[i][7])
                
                centroid = ( sum(x)/len(x) , sum(y)/len(y) , sum(w)/len(w) , sum(z)/len(z) , sum(v)/len(v), sum(t)/len(t), sum(s)/len(s), sum(r)/len(r) )
                centers.append(centroid)        
                        
        
            Matrix = []
            for i in range(len(centers)):
                point1 = (centers[i][0], centers[i][1], centers[i][2], centers[i][3], centers[i][4], centers[i][5], centers[i][6], centers[i][7])
                Row = []
                for j in range(len(centers)):
                    point2 = (centers[j][0], centers[j][1], centers[j][2], centers[j][3], centers[j][4], centers[i][5], centers[j][5], centers[j][5])
                    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
                    Row.append(distance)
                Matrix.append(Row)
        
                   
            #obtengo el indicador a partir de esta distancia
            SMI = 0.0
            smirow = []
            for i in range(len(Matrix)):
                for j in range(len(Matrix)):
                    if i > j:
                        smirow.append(  (1- (1/ (math.log(Matrix[i][j]) ) ) )**(-1) )
                    else:
                        break        
            SMI = max(smirow)**(-1)
            # print('SMI = ' + str(SMI))
            
            #SSE
            SSE=0
            for j in range(n_clusters):
                for k in range(len(preds)):
                    point1 = centers[j]
                    if preds[k]==j:
                        point2 = tDnorm[k]
                        SSE = SSE + math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
                    
            # print('SSE = '+ str(SSE))
            
            see.append(SSE)
            indicadores.append([n_clusters, score, sum(sample_silhouette_values)/len(sample_silhouette_values), VRC, SMI, SSE])
            
        with open(directory+"\Indicadores de Cluster, Hierarchy, RedesAereas.txt", "w") as output:
            for row in indicadores:
                output.write(str(row) + '\n')
        
        return indicadores
    
    def hierarchyFinal(self, n_clusters, tablaDeDatos, tD, tDnorm, redDistribucion, destination):
        

        
        directory = destination + '\Images'
        if os.path.exists(directory) == False:
            os.mkdir(directory)
            
        see=[]
        indicadores = []
        plt.figure(figsize=(10, 7))
        plt.title("Customer Dendograms")
        dend = shc.dendrogram(shc.linkage(tablaDeDatos, method='ward'))
        plt.axhline(y=250, color="red", linestyle="--")
        plt.savefig(directory + '\Dendogram.pdf')
        plt.show()
        
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        preds = clusterer.fit_predict(tablaDeDatos)
    #    centers = clusterer.cluster_centers_
        unique, cnt = np.unique(preds, return_counts=True)
        counts = dict(zip(unique, cnt))
    
    
        score = silhouette_score(tablaDeDatos, preds)
        print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
        sample_silhouette_values = silhouette_samples(tablaDeDatos, preds)

        
        #Variance Ratio Criterion
        labels = clusterer.labels_
    
        VRC =  (metrics.calinski_harabasz_score(tablaDeDatos, labels))
        # print('VRC = ' + str(VRC))
    
        #Similarity Matrix Indicator
        #Obtengo la matriz de distancia entre los centros
        centers = []
        for j in range(n_clusters):
            x=[]
            y=[]
            w=[]
            z=[]
            v=[]
            t=[]
            s=[]
            r=[]
            for i in range(len(preds)):
                if preds[i] == j:
                    x.append(tDnorm[i][0])
                    y.append(tDnorm[i][1])
                    w.append(tDnorm[i][2])
                    z.append(tDnorm[i][3])
                    v.append(tDnorm[i][4])
                    t.append(tDnorm[i][5])
                    s.append(tDnorm[i][6])
                    r.append(tDnorm[i][7])
            
            centroid = ( sum(x)/len(x) , sum(y)/len(y) , sum(w)/len(w) , sum(z)/len(z) , sum(v)/len(v), sum(t)/len(t), sum(s)/len(s), sum(r)/len(r) )
            centers.append(centroid)        
                    
    
        Matrix = []
        for i in range(len(centers)):
            point1 = (centers[i][0], centers[i][1], centers[i][2], centers[i][3], centers[i][4], centers[i][5], centers[i][6], centers[i][7])
            Row = []
            for j in range(len(centers)):
                point2 = (centers[j][0], centers[j][1], centers[j][2], centers[j][3], centers[j][4], centers[i][5], centers[j][5], centers[j][5])
                distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
                Row.append(distance)
            Matrix.append(Row)
        
        closest, _ = pairwise_distances_argmin_min(centers, tablaDeDatos)
    
    
        fig, ax2 = plt.subplots(1, 1)    
        cax = ax2.matshow(Matrix, cmap=plt.cm.YlGn)
        ax2.set_title("Colormap Hierarchy for N= " + str(n_clusters))
        fig.colorbar(cax)
        plt.savefig(directory + '\Colormap, Hierarchy, N = ' + str(n_clusters) + '.pdf')
        plt.show()
    
        #obtengo el indicador a partir de esta distancia
        SMI = 0.0
        smirow = []
        for i in range(len(Matrix)):
            for j in range(len(Matrix)):
                if i > j:
                    smirow.append(  (1- (1/ (math.log(Matrix[i][j]) ) ) )**(-1) )
                else:
                    break        
        SMI = max(smirow)**(-1)
        # print('SMI = ' + str(SMI))
        
        #SSE
        SSE=0
        for j in range(n_clusters):
            for k in range(len(preds)):
                point1 = centers[j]
                if preds[k]==j:
                    point2 = tDnorm[k]
                    SSE = SSE + math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
                
        # print('SSE = '+ str(SSE))
        
        labels =[]
        for i in range(n_clusters):
            labels.append("C-"+str(i+1))
        fig, ax4 = plt.subplots(1, 1)    
        colours = dict(zip(labels, plt.cm.tab20.colors[:len(labels)]))
        cax = ax4.pie(cnt, labels=labels,colors=[colours[key] for key in labels], wedgeprops={"edgecolor":"k",'linewidth': 1, 'antialiased': True}, rotatelabels =90)
        
        ax4.set_title("Distribución de Redes en los Clústers Resultantes")
        plt.savefig(directory +"\PieChart.pdf")
        plt.show()
        
        for i in range(len(cnt)):
            print("In Cluster " + str(i) + " there are " + str(cnt[i]) + " distribution networks. Making up " + str(round(cnt[i]/sum(cnt)*100,2)) + "% of the total count")
        
        centroid = []
        for i in range(len(closest)):
            centroid.append(redDistribucion[closest[i]])
       
        
        
   
        return centroid, preds
    
    def kmedoids(self, tablaDeDatos, tDnorm, destination):

        directory = destination + "\Indicadores"
        if os.path.exists(directory) == False:
            os.mkdir(directory)
    
        range_n_clusters = list (range(2,20))
        
        indicadores = []
        
        for n_clusters in range_n_clusters:
            
        #    fig, ax1 = plt.subplots(1, 1)    
            clusterer = KMedoids(n_clusters=n_clusters, metric='euclidean', init='k-medoids++')
            preds = clusterer.fit_predict(tablaDeDatos)
            centers = clusterer.cluster_centers_
            
            score = silhouette_score(tablaDeDatos, preds)
            # print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
            sample_silhouette_values = silhouette_samples(tablaDeDatos, preds)
        
            
            #Variance Ratio Criterion
            labels = clusterer.labels_
        
            VRC =  (metrics.calinski_harabasz_score(tablaDeDatos, labels))
            # print('VRC = ' + str(VRC))
        
            #Similarity Matrix Indicator
            #Obtengo la matriz de distancia entre los centros
            Matrix = []
            for i in range(len(centers)):
                point1 = (centers[i][0], centers[i][1], centers[i][2], centers[i][3], centers[i][4], centers[i][5])
                Row = []
                for j in range(len(centers)):
                    point2 = (centers[j][0], centers[j][1], centers[j][2], centers[j][3], centers[j][4], centers[i][5])
                    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
                    Row.append(distance)
                Matrix.append(Row)
        
                   
            #obtengo el indicador a partir de esta distancia
            SMI = 0.0
            smirow = []
            for i in range(len(Matrix)):
                for j in range(len(Matrix)):
                    if i > j:
                        smirow.append(  (1- (1/ (math.log(Matrix[i][j]) ) ) )**(-1) )
                    else:
                        break        
            SMI = max(smirow)**(-1)
            # print('SMI = ' + str(SMI))
            
            #SSE
            SSE=0
            for j in range(n_clusters):
                for k in range(len(preds)):
                    point1 = centers[j]
                    if preds[k]==j:
                        point2 = tDnorm[k]
                        SSE = SSE + math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
                    
            # print('SSE = '+ str(SSE))
            
            indicadores.append([n_clusters, score, sum(sample_silhouette_values)/len(sample_silhouette_values), VRC, SMI, SSE])
            
        with open(directory + "\Indicadores de Cluster, K-medoids++, RedesAereas.txt", "w") as output:
            for row in indicadores:
                output.write(str(row) + '\n')
        
        return indicadores

    def kmedoidsFinal(self, n_clusters, tablaDeDatos, tD, tDnorm, redDistribucion, destination):
        
        directory = destination + "\Images"
        if os.path.exists(directory) == False:
            os.mkdir(directory)
        clusterer = KMedoids(n_clusters=n_clusters, metric='euclidean', init='k-medoids++')
        preds = clusterer.fit_predict(tablaDeDatos)
        centers = clusterer.cluster_centers_
        
        score = silhouette_score(tablaDeDatos, preds)
        # print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
        sample_silhouette_values = silhouette_samples(tablaDeDatos, preds)
    
        closest, _ = pairwise_distances_argmin_min(centers, tablaDeDatos)
    
        #Variance Ratio Criterion
        labels = clusterer.labels_
        unique, cnt = np.unique(preds, return_counts=True)
        counts = dict(zip(unique, cnt))
        VRC =  (metrics.calinski_harabasz_score(tablaDeDatos, labels))
        # print('VRC = ' + str(VRC))
    
        #Similarity Matrix Indicator
        #Obtengo la matriz de distancia entre los centros
        Matrix = []
        for i in range(len(centers)):
            point1 = (centers[i][0], centers[i][1], centers[i][2], centers[i][3], centers[i][4], centers[i][5])
            Row = []
            for j in range(len(centers)):
                point2 = (centers[j][0], centers[j][1], centers[j][2], centers[j][3], centers[j][4], centers[i][5])
                distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
                Row.append(distance)
            Matrix.append(Row)
    
        fig, ax2 = plt.subplots(1, 1)    
        cax = ax2.matshow(Matrix, cmap=plt.cm.YlGn)
        ax2.set_title("Colormap K-medoids for N= " + str(n_clusters))
        fig.colorbar(cax)
        plt.savefig(directory + '\Colormap, K-medoids++ N = ' + str(n_clusters) + '.pdf')
        plt.show()
    
        #obtengo el indicador a partir de esta distancia
        SMI = 0.0
        smirow = []
        for i in range(len(Matrix)):
            for j in range(len(Matrix)):
                if i > j:
                    smirow.append(  (1- (1/ (math.log(Matrix[i][j]) ) ) )**(-1) )
                else:
                    break        
        SMI = max(smirow)**(-1)
        # print('SMI = ' + str(SMI))
        
        #SSE
        SSE=0
        for j in range(n_clusters):
            for k in range(len(preds)):
                point1 = centers[j]
                if preds[k]==j:
                    point2 = tDnorm[k]
                    SSE = SSE + math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
                
        # print('SSE = '+ str(SSE))
        
        labels =[]
        for i in range(n_clusters):
            labels.append("C-"+str(i+1))
        fig, ax4 = plt.subplots(1, 1)    
        colours = dict(zip(labels, plt.cm.tab20.colors[:len(labels)]))
        cax = ax4.pie(cnt, labels=labels,colors=[colours[key] for key in labels], wedgeprops={"edgecolor":"k",'linewidth': 1, 'antialiased': True}, rotatelabels =90)
        
        ax4.set_title("Distribución de Redes en los Clústers Resultantes")
        plt.savefig(directory +"\PieChart.pdf")
        plt.show()
        
        for i in range(len(cnt)):
            print("In Cluster " + str(i) + " there are " + str(cnt[i]) + " distribution networks. Making up " + str(round(cnt[i]/sum(cnt)*100,2)) + "% of the total count")
        
        centroid = []
        for i in range(len(closest)):
            centroid.append(redDistribucion[closest[i]])       
        

    
        return centroid, preds   
    
    """
    Medidas de calidad de los clusters
    """
    def compare(self, kmeans, hierarchy, kmedoids, destination):
        
        directory = destination + '\Images' 
        if os.path.exists(directory) == False:
            os.mkdir(directory)
    #indicadores.append(['NumCluster', 'globalSC', 'avgSC', 'VRC', 'SMI'])
        
        kmeans = np.transpose(kmeans)
        hierarchy = np.transpose(hierarchy)
        kmedoids = np.transpose(kmedoids)
        
            
        #Graphing Global Silhouette Score
        plt.figure(1)
        plt.plot(kmeans[0], kmeans[1], label='kmeans', marker='o')
        plt.plot(hierarchy[0], hierarchy[1], label='hierarchy', marker='o')
        plt.plot(kmedoids[0], kmedoids[1], label='kmedoids', marker='o')
        plt.legend()
        plt.xlabel('N° of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid()
        plt.ylim([0.0,0.6])
        plt.xlim([2,20])
        plt.xticks(np.arange(2, 20, step=1))
        plt.savefig(directory + '\Comparing-SilhouetteScore.pdf')
        plt.show
        
        #Graphing VRC
        #valor normalizado VRC segun el estudio de clustering
        
        newx = []
        kmeansNormalizado = []
        hierarchyNormalizado = []
        kmedoidsNormalizado = []
        
        for i in range(len(kmeans[0])-2):
            newx.append(kmeans[0][i+1])
            kmeansNormalizado.append(kmeans[3][i+1]-kmeans[3][i] - (kmeans[3][i+2]-kmeans[3][i+1]))
            hierarchyNormalizado.append(hierarchy[3][i+1]-hierarchy[3][i] - (hierarchy[3][i+2]-hierarchy[3][i+1]))
            kmedoidsNormalizado.append(kmedoids[3][i+1]-kmedoids[3][i] - (kmedoids[3][i+2]-kmedoids[3][i+1]))
            
        
        
        plt.figure(2)
        plt.plot(newx, kmeansNormalizado, label='kmeans', marker='o')
        plt.plot(newx, hierarchyNormalizado, label='hierarchy', marker='o')
        plt.plot(newx, kmedoidsNormalizado, label='kmedoids', marker='o')
        plt.legend()
        plt.grid()
        plt.xlabel('N° of Clusters')
        plt.ylabel('wVRC')
        plt.xticks(np.arange(2, 20, step=1))
        plt.savefig(directory + '\Comparing-VRC.pdf')
        plt.show
        
        #Graphing Silhouette Score
        plt.figure(3)
        plt.plot(kmeans[0], kmeans[4], label='kmeans', marker='o')
        plt.plot(hierarchy[0], hierarchy[4], label='hierarchy', marker='o')
        plt.plot(kmedoids[0], kmedoids[4], label='kmedoids', marker='o')
        plt.legend()
        plt.grid()
        plt.xlabel('N° of Clusters')
        plt.ylabel('SMI')
        plt.xlim([2,20])
        plt.ylim([0,3])
        plt.xticks(np.arange(2, 20, step=1))
        plt.savefig(directory + '\Comparing-SMI.pdf')
        plt.show
        
        plt.figure(4)
        plt.plot(kmeans[0], kmeans[2], label='kmeans', marker='o')
        plt.plot(hierarchy[0], hierarchy[2], label='hierarchy', marker='o')
        plt.plot(kmedoids[0], kmedoids[2], label='kmedoids', marker='o')
        plt.legend()
        plt.grid()
        plt.xlabel('N° of Clusters')
        plt.ylabel('avgSC')
        plt.ylim([0.0,0.7])
        plt.xlim([2,20])
        plt.xticks(np.arange(2, 20, step=1))
        plt.savefig(directory + '\Comparing-AverageSilhouette Score.pdf')
        plt.show
        
        plt.figure(5)
        plt.plot(kmeans[0], kmeans[5], label='kmeans', marker='o')
        plt.plot(hierarchy[0], hierarchy[5], label='hierarchy', marker='o')
        plt.plot(kmedoids[0], kmedoids[5], label='kmedoids', marker='o')
        plt.legend()
        plt.grid()
        plt.xlabel('N° of Clusters')
        plt.ylabel('SSE')
        plt.xlim([2,20])
        plt.xticks(np.arange(2, 20, step=1))
        plt.savefig(directory + '\Comparing-SSE.pdf')
        plt.show
        
        y = []
        for i in range(len(kmeans[0])):
            opt = []
            z=[]
            #SC
            sc = [kmeans[1][i], hierarchy[1][i], kmedoids[1][i]]
            x = sc.index(max(sc))
            opt.append(x)
            
            #VRC
            vrc = [kmeans[3][i], hierarchy[3][i], kmedoids[3][i]]
            x = vrc.index(max(vrc))
            opt.append(x)
            
            #SMI
            smi = [kmeans[4][i], hierarchy[4][i], kmedoids[4][i]]
            x = smi.index(max(smi))
            opt.append(x)
            
            #SSE
            sse = [kmeans[5][i], hierarchy[5][i], kmedoids[5][i]]
            x = sse.index(min(sse))
            opt.append(x)
            
            z.append(kmeans[0][i])
            
            if max(set(opt), key = opt.count) == 0:
                z.append('K-Means') 
                z.append([kmeans[1][i], kmeans[3][i], kmeans[4][i], kmeans[5][i]])
            elif max(set(opt), key = opt.count) == 1:
                z.append('Hierarchy')
                z.append([hierarchy[1][i], hierarchy[3][i], hierarchy[4][i], hierarchy[5][i]])
            elif max(set(opt), key = opt.count) ==2:
                z.append('K-Medoids')
                z.append([kmedoids[1][i], kmedoids[3][i], kmedoids[4][i], kmedoids[5][i]])
            
            y.append(z)
            
        
        return y

   
    def finalC(self, n_clusters, centroid, tD, tDnorm, preds, redDistribucion, destination,tabla, redes=False):
        
        directory = destination
        images = destination + '\Images'
        directory = directory + '\FinalClusters'
        
        if os.path.exists(directory) == False:
            os.mkdir(directory)
    
        for i in range(int(n_clusters)):
            cluster = []
            clusterChar = []
            for j in range(len(preds)):
                if preds[j] == i:
                    cluster.append(redDistribucion[j])
                    clusterChar.append(tD[j])
            with open(directory + "\Cluster" + str(i+1) + '.txt', "w") as output:                    
                output.write("This is Cluster" + str(i+1) + "\n")
                output.write("The centroid is " + str(centroid[i]) + '\n\n')
                output.write('Cluster Members:' + '\n')
                for i in range(len(cluster)):
                    output.write(str(cluster[i]) + '\n')
                    output.write(str(clusterChar[i]) + '\n \n')
    
        unique, cnt = np.unique(preds, return_counts=True)
        counts = dict(zip(unique, cnt))
        
        if redes != False:  
            for i in range(len(centroid)):
                
                for j in range(len(redes)):
                    valuesRed = (redes[j][0]).values.tolist()
                    columnsRed = (redes[j][0]).columns.tolist()
                    for z in range(len(columnsRed)):
                        if columnsRed[z] == 'FEEDERID':
                            feeder = z
                    array = []
                    for k in range(len(valuesRed)):                      
                        if (valuesRed[k][feeder] == centroid[i][1]):
                            array.append(valuesRed[k])

                    new_mapi = pd.DataFrame(array,columns=columnsRed)
                    geometry = new_mapi.geometry
                    new_mapi = new_mapi.drop('geometry', axis=1)
                    new_map = gpd.GeoDataFrame(new_mapi, geometry = geometry)    
                    # fig, ax3 = plt.subplots(1, 1)    
                    # ax3 = new_map.plot()
                    # ax3.set_title("Cluster" + str(j+1))
                    # ax3.set_yticks([])
                    # ax3.set_xticks([])
                    # plt.savefig(directory + '\ClusterNetwork['+str(j+1)+'].pdf')
                    # print(redes[1][j])
                    if not new_map.empty:
                        new_map.to_file(directory + "\Cluster "+str(i+1)+"-"+str(redes[j][1])+".shp")
    
        #feeder to feeder distance
        # this is the distance from one feeder to the next within the same cluster.
        # a small value indicates that this cluster is tightly grouped together which is good
        
        icd = []
        for k in range(n_clusters):
            feeder2centroid=[]  
            matrixFeeders = []
            for i in range(len(tDnorm)):
                if preds[i] == k:
                    point1 = (tDnorm[i][0], tDnorm[i][1], tDnorm[i][2], tDnorm[i][3], tDnorm[i][4], tDnorm[i][5], tDnorm[i][6], tDnorm[i][7])
                    Row = []
                    for j in range(len(tD)):
                        if preds[j] == k:
                            point2 = (tDnorm[j][0], tDnorm[j][1], tDnorm[j][2], tDnorm[j][3], tDnorm[j][4], tDnorm[j][5], tDnorm[j][6], tDnorm[j][7])
                            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
                            Row.append(distance)
                    matrixFeeders.append(Row)
            
            #feeder to centroid
            #calculated as the RMS of the F2F for each member of the cluster
            matrixFeeders = np.array(matrixFeeders)
            for l in range(len(matrixFeeders)):
                f2c = math.sqrt(1/counts[k] *sum( z*z for z in matrixFeeders[l]))
                feeder2centroid.append(f2c)
            # print(feeder2centroid)
            
            #intra class distance
            #calculated by using f2c distance for all values in the cluster
            
            icd.append( math.sqrt(1/(2*counts[k]) * sum(z*z for z in feeder2centroid)) )
            alpha = []
            for i in range(counts[k]):
                alpha.append(str(i+1))
        
            fig, ax5 = plt.subplots(1, 1)    
            cax = ax5.matshow(matrixFeeders, cmap='jet',aspect='auto',vmin=0, vmax=1)
            ax5.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax5.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax5.set_xticklabels(['']+alpha)  
            ax5.set_yticklabels(['']+alpha)
            
            if counts[k] >= 20:
                for (i, j), z in np.ndenumerate(matrixFeeders):
                    ax5.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',fontsize=3)
            else:
                for (i, j), z in np.ndenumerate(matrixFeeders):
                    ax5.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
            
            plt.xticks(rotation=90)
            fig.colorbar(cax)
            plt.savefig(images+ "\Feeder2FeederDistance_cluster_"+str(k)+".pdf")
            plt.show()
        
    #        print(matrixFeeders)
    #    print(icd)
    
        tDT = np.array(tD).T
        for k in range(len(tDT)):
            data = []
            for i in range(n_clusters):
                datax = []
                x=0
                for j in range(len(preds)):
                    if preds[j] == i:
                        datax.append(tDT[k][j])
                data.append(datax)
        
            # Creating plot 
            boxprops = dict(linestyle='-', linewidth=3, color='black')
            flierprops = dict(marker='o', markerfacecolor='green', markersize=10, linestyle='none')
            medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
            
            plt.boxplot(data, widths=0.25, boxprops=boxprops, flierprops=flierprops, medianprops=medianprops)
            plt.title("Boxplot " + tabla.columns[k])
            plt.grid()
            plt.xlabel("Cluster")
            plt.savefig(images+ "\BoxplotFinal" + str(tabla.columns[k])+".pdf")
        
            plt.show()

app = QApplication(sys.argv)
UIWindow = ClusteringExe()
app.exec_()

