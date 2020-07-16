# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5 import uic
Form = uic.loadUiType('gui-matplotlib.ui')[0]  # Load ui
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout

import sys
import numpy as np
from time import sleep

from numba import jit, float64, int32, double, void
import numba as nb
from scipy.fftpack import fft


@jit([void(int32, int32, int32, float64, float64, float64, float64[:,:], 
    float64[:,:], float64[:,:], float64[:,:], int32, int32, int32,
    float64[:,:])], nopython=True)

def step_SIT_SIP (nx, ny, c, dx, dy, dt, ox, oy, px, py, layer, m, kmax, p):
    s = 0
    e = ny - 1
    Ax = np.ones((nx))
    Ay = np.ones((ny))
    Bx = dt / dx * np.ones((nx))
    By = dt / dx * np.ones((ny))

    #calculation of Ax, Ay, Bx, By        
    for i in range (0, nx):
      if i > nx - layer:
          Ax[i] = (1 - dt / 2 * kmax * ((i - nx + layer) / layer)) ** m / (1 + 
            dt / 2 * kmax * ((i - nx + layer) / layer) ** m)
          Ay[i] = (1 - dt / 2 * kmax * ((i - ny + layer) / layer)) ** m / (1 + dt / 2 * kmax * ((i - ny + layer) / layer) ** m)
      elif i < layer:
          Ax[i] = (1 - dt / 2 * kmax * ((layer - i) / layer)) ** m / (1 + dt / 2 * kmax * ((layer - i) / layer) ** m)
          Ay[i] = (1 - dt / 2 * kmax * ((layer - i) / layer)) ** m / (1 + dt / 2 * kmax * ((layer - i) / layer) ** m)
    
    for j in range (0, ny):
      if j > ny - layer:
          Bx[j] = dt / dx / (1 + dt / 2 * kmax * ((j - nx + layer) / layer)) ** m
          By[j] = dt / dx / (1 + dt / 2 * kmax * ((j - nx + layer) / layer)) ** m
      elif j < layer:
          Bx[j] = dt / dx / (1 + dt / 2 * kmax * ((layer - j) / layer)) ** m
          By[j] = dt / dx / (1 + dt / 2 * kmax * ((layer - j) / layer)) ** m

    #calculation of ox, oy, px, py
    #ox
    for j in range (s-1, e):
        for k in range(nx):
            ox[k, j] = Ax[j] * ox[k, j] - Bx[j] * (px[k, j + 1] + py[k, j + 1] - px[k, j] - py[k, j]) # ox is found at t + 1/2

    #oy
    #oy[s, :] = Ay[s] * oy[s, :] - By[s] * (px[s, :] + py[s, :] - py[e, :] - px[e, :])      # % first row
    for i in range (s, e + 1):
        for k in range(ny):
            oy[i, k] = Ay[i] * oy [i, k] - By[i] *  (py[i, k] + px[i, k] - px[i - 1, k] - py[i - 1, k]) # % oy is found at t + 1/2
        
    #px
    #px[:, s] = Ax[s] * px[:, s] - c ** 2 * Bx[s] *  (ox[:, s] - ox[:, e])   # first col
    for j in range(s, e + 1):
        for k in range(ny):
            px[k, j] = Ax[j] * px[k, j] - c ** 2 * Bx[j] * (ox[k, j] - ox[k, j - 1])   # % t + 1
          
    #py
    for i in range(s-1, e):
        for k in range(ny):
            py[i, k] = Ax[i] * py[i, k] - c ** 2 * By[i] * (oy[i + 1, k] - oy[i, k])    # at t + 1  

    for i in range(nx):
        for j in range(ny):
            p[i, j] = px[i, j] + py[i, j]

##############################################################################################
class MyWindow(QMainWindow, Form):
    def __init__(self):
        Form.__init__(self)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.flag = True
        self.start_flag = False
        self.c = 340        #speed of sound
        self.m = 3
        self.kmax = 1000
        self.CFL = 1            #Courant number
        self.nt = 300 / self.CFL #number of time steps in simulation
        self.A = 1
        self.t0 = 2.5E-2
        self.sigma = 5E-5


        #Events
        self.start_pushButton.clicked.connect(self.start)
        self.pushButton_rec.clicked.connect(self.plot_record)
        self.kmax_pushButton.clicked.connect(self.kMax)
        self.stop_pushButton.clicked.connect(self.stop)
        self.file_actionStart.triggered.connect(self.start)
     

    ######class methods######
    def start(self):
        self.start_flag = True

        self.fc = int(self.freq.text())
        self.layer = int(self.Layer.text())


        self.nx = 175 + 2 * self.layer      #number of cells in x direction
        self.ny = 175 + 2 * self.layer     #number of cells in y direction

        
        
        #location of source and receivers in grid
        self.x_source = round(self.nx / 5)
        self.y_source = round(self.ny / 5)

        #pulse properties
        
        self.dx = (self.c / self.fc) / 10.
        self.dy = (self.c / self.fc) / 10.

        self.dt = self.CFL / (self.c * np.sqrt((1 / self.dx ** 2) + (1 / self.dy ** 2)))
 
            
        self.ox = np.zeros((self.nx, self.ny)) 
        self.oy = np.zeros((self.nx, self.ny))
        self.p = np.zeros((self.nx, self.ny))
        self.px = np.zeros((self.nx, self.ny))
        self.py = np.zeros((self.nx, self.ny))


        self.plot_label.setText("در حال کار...")
        
        if self.flag == True : 

            #figure
            self.fig = Figure()
            self.fig.patch.set_color('w')
            self.ax = self.fig.add_subplot(111, frame_on=False)
            self.ax.set_title("My Title")
            self.ax.set_xlim([0, self.nx])
            self.ax.set_ylim([0, self.ny])
            self.canvas = FigureCanvas(self.fig)
            self.y, self.x = np.mgrid[range(self.nx), range(self.ny)]
            self.mesh = self.ax.pcolormesh(self.x, self.y, self.p[:-1, :-1], 
                cmap='RdBu', vmin= -0.01, vmax=0.01)
            #widget
            l = QVBoxLayout(self.matplotlib_widget)
            l.addWidget(self.canvas)



            rect1 = matplotlib.patches.Rectangle((self.nx - self.layer, self.layer),
                width=1, height= self.ny-2*self.layer, alpha=1, facecolor='k')
            self.ax.add_patch(rect1)
            rect2 = matplotlib.patches.Rectangle((self.layer, self.layer), 
                width=1, height= self.ny-2*self.layer, alpha=1, facecolor='k')
            self.ax.add_patch(rect2)
            rect3 = matplotlib.patches.Rectangle((self.layer, self.nx - self.layer),
                width= self.nx-2*self.layer, height= 1, alpha=1, facecolor='k')
            self.ax.add_patch(rect3)
            rect4 = matplotlib.patches.Rectangle((self.layer,self.layer),
                width=self.nx-2*self.layer, height= 1, alpha=1, facecolor='k')
            self.ax.add_patch(rect4)
            rect5= matplotlib.patches.Rectangle((self.nx / 3, self.layer), 
                width=1, height= self.ny/3-self.layer, alpha=1, facecolor='b')
            self.ax.add_patch(rect5)
            circle= matplotlib.patches.Circle((7 * self.nx / 15, self.nx / 5), 2, facecolor='r')
            self.ax.add_patch(circle)


            dic_wave = {'nt' : self.nt, 'dt' : self.dt, 't0' : self.t0, 'sigma' : self.sigma,
                        'm' : self.m, 'layer' : self.layer, 'x_source' : self.x_source, 
                        'y_source' : self.y_source,'A' : self.A, 'fc' : self.fc, 'nx' : self.nx, 
                        'ny' : self.ny, 'dx' : self.dx, 'dy' : self.dy, 'c' : self.c, 'kmax' : self.kmax}
            dic_array = {'p' :  self.p, 'px' : self.px, 'py' : self.py, 'ox' : self.ox, 'oy' : self.oy, 'ax' : self.ax}
            #thread
            self.thread = PlotThread(dic_wave, dic_array)
            self.thread.update_trigger.connect(self.update_plot)
            self.thread.finished_trigger.connect(self.stop)

            self.thread.exit()
            self.thread.start()
            self.flag = False
        if self.flag == False :
            self.thread.exit()
            self.mesh.set_array(np.zeros((self.nx, self.ny)).ravel())
            self.canvas.draw()
            self.thread.start()
            self.flag = False
            
    def update_plot(self):
        p = self.thread.p
        self.mesh.set_array(p[:-1, :-1].ravel())
        self.canvas.draw()

    def stop(self):
        self.plot_label.setText("متوقف")
        self.thread.stop()
        self.thread.exit()

    def plot_record(self):
        fig1 = plt.figure(1)
        plt.plot(np.abs(self.thread.recorderF))
        plt.show()

    def kMax(self):

        # sumX = 1000000000000
        # sumY = 1000000000000
        # fc_100 = 100
        # dx_100 = (self.c / fc_100) / 10.
        # dy_100 = (self.c / fc_100) / 10.
        # layer_100 = int(self.Layer.text())
        # for k in range(0, 100):
        #     print(k)
        #     self.px = np.zeros((self.nx, self.ny))
        #     self.py = np.zeros((self.nx, self.ny))
        #     self.ox = np.zeros((self.nx, self.ny))
        #     self.oy = np.zeros((self.nx, self.ny))
        #     self.p = np.zeros((self.nx, self.ny))
        #     for it in range(1, 300):
        #         t = (it - 1) * self.dt
                
        #         source = self.A * np.sin(2 * np.pi * fc_100 * (t - self.t0)) * np.exp(-((t - self.t0) ** 2) / (self.sigma)) 
                
        #         self.px[self.x_source - 1 : self.x_source + 1, self.y_source - 1 : self.y_source + 1] = self.px[
        #             self.x_source - 1 : self.x_source + 1, self.y_source - 1 : self.y_source + 1] + source
        #         self.py[self.x_source - 1 : self.x_source + 1, self.y_source - 1 : self.y_source + 1] = self.py[
        #             self.x_source - 1 : self.x_source + 1, self.y_source - 1 : self.y_source + 1] + source  

        #         step_SIT_SIP (self.nx, self.ny, self.c, dx_100, dy_100, self.dt, 
        #             self.ox, self.oy, self.px, self.py, layer_100, self.m, k * 100, self.p)

        #     tmp_sumX = self.vReflection()

        #     if tmp_sumX < sumX:# and tmp_sumY < sumY :
        #         sumX = tmp_sumX
        #         kmax = k
        # self.kmax = kmax * 100

        sumX = 1000000000000
        sumY = 1000000000000
        fc_100 = 100
        layer_100 = int(self.Layer.text())
        self.nx = 175 + 2 * layer_100      #number of cells in x direction
        self.ny = 175 + 2 * layer_100     #number of cells in y direction
        dx_100 = (self.c / fc_100) / 10.
        dy_100 = (self.c / fc_100) / 10.
        dt_100 = self.CFL / (self.c * np.sqrt((1 / dx_100 ** 2) + (1 / dy_100 ** 2)))
        x_source_100 = round(self.nx / 2)
        y_source_100 = round(self.ny / 2)
        for k in range(0, 100):
            print(k)
            self.px = np.zeros((self.nx, self.ny))
            self.py = np.zeros((self.nx, self.ny))
            self.ox = np.zeros((self.nx, self.ny))
            self.oy = np.zeros((self.nx, self.ny))
            self.p = np.zeros((self.nx, self.ny))
            for it in range(1, 300):
                t = (it - 1) * dt_100
                
                source = self.A * np.sin(2 * np.pi * fc_100 * (t - self.t0)) * np.exp(-((t - self.t0) ** 2) / (self.sigma)) 
                
                self.px[x_source_100 - 1 : x_source_100 + 1, y_source_100 - 1 : y_source_100 + 1] = self.px[
                    x_source_100 - 1 : x_source_100 + 1, y_source_100 - 1 : y_source_100 + 1] + source
                self.py[x_source_100 - 1 : x_source_100 + 1, y_source_100 - 1 : y_source_100 + 1] = self.py[
                    x_source_100 - 1 : x_source_100 + 1, y_source_100 - 1 : y_source_100 + 1] + source  

                step_SIT_SIP(self.nx, self.ny, self.c, dx_100, dy_100, dt_100, 
                    self.ox, self.oy, self.px, self.py, layer_100, self.m, k * 100, self.p)

            tmp_sumX = self.vReflection()

            if tmp_sumX < sumX:# and tmp_sumY < sumY :
                sumX = tmp_sumX
                kmax = k
        self.kmax = kmax * 100
        if self.start_flag :
            self.thread.kmax = kmax * 100
        self.plot_label_2.setText(str(self.kmax))

    def vReflection(self):
        sumX = 0
        sumY = 0
        for i in range(self.nx):
            for j in range(self.ny):
                sumX += self.ox[i, j] ** 2
                sumY += self.oy[i, j] ** 2 
        return  sumX+ sumY

    
#############################################################################################################
class PlotThread(QtCore.QThread, MyWindow):
    update_trigger = QtCore.pyqtSignal(np.ndarray)
    finished_trigger = QtCore.pyqtSignal()
    recorderF = np.zeros((300))
    def __init__(self, dic_wave, dic_array):
        QtCore.QThread.__init__(self)
        self.nt = dic_wave['nt']
        self.dt = dic_wave['dt']
        self.t0 = dic_wave['t0']
        self.sigma = dic_wave['sigma']
        self.p = dic_array['p']
        self.px = dic_array['px']
        self.py = dic_array['py']
        self.m = dic_wave['m']
        self.Layer = dic_wave['layer']
        self.x_source = dic_wave['x_source']
        self.y_source = dic_wave['y_source']
        self.A = dic_wave['A']
        self.fc = dic_wave['fc']
        self.nx = dic_wave['nx']
        self.ny = dic_wave['ny']
        self.oy = dic_array['oy']
        self.ox = dic_array['ox']
        self.dx = dic_wave['dx']
        self.dy = dic_wave['dy']
        self.c = dic_wave['c']
        self.ax = dic_array['ax']
        self.kmax = dic_wave['kmax']

        self.flag = False
        self.recorder = np.zeros((self.nt))
        self.recorderF = []
        self.f = []

    def stop(self) :
        self.flag = False
    
    def run(self):
        self.flag = True
        for it in range(1, int(self.nt + 1)):
            if not self.flag :
                return  
            t = (it - 1) * self.dt
            self.px[ : round(self.nx / 3), round(self.ny / 3)] = 0 
            self.py[ : round(self.nx / 3), round(self.ny / 3)] = 0 

            #source updaten at new time
            source = self.A * np.sin(2 * np.pi * self.fc * (t - self.t0)) * np.exp(-((t - self.t0) ** 2) / (self.sigma)) 
            
            #add pressure to source location
            self.px[self.x_source - 1 : self.x_source + 1, self.y_source - 1 : self.y_source + 1] = self.px[
                self.x_source - 1 : self.x_source + 1, self.y_source - 1 : self.y_source + 1] + source
            self.py[self.x_source - 1 : self.x_source + 1, self.y_source - 1 : self.y_source + 1] = self.py[
                self.x_source - 1 : self.x_source + 1, self.y_source - 1 : self.y_source + 1] + source  

            #calculate and update p       
            step_SIT_SIP(self.nx, self.ny, self.c, self.dx, 
                self.dy, self.dt, self.ox, self.oy, self.px, self.py, self.Layer, self.m, self.kmax, self.p)

            self.recorder[it-1] = self.p[7*self.nx/15][self.nx/5]

            if it % 3 == 0:
                self.ax.set_title( str( int(it/3) ) )
                self.update_trigger.emit(self.p)
                sleep(0.1)
        fp = self.fc
        k_max = self.kmax
        m = 3
        N = 3
        c1 = 340
        lp = c1/fp
        md = 1
        d = 1.0
        dx = c1/ fp/ 10
        nd=np.ceil(d/dx)
        ##################################

        if nd % 2 ==1 :#% nd should be even, because nd/2 is needed.
            nd = nd + 1;
        
        if N % 2 ==0 :#% N should be odd
            N = N + 1;
        
        N2 = (N-1) / 2; #% N/2
        np1=lp/dx; #% Does not need to be integer
        nx=7*nd+1; #% Number of cells in x direction
        ny=6*nd+1; #% Number of cells in y direction
        npml= self.Layer; #% Number of cells in PML
        xdim=nx-2+2*npml; #% Total number of columns for course grid
        xdim_f=(nd+1)*N; #% Total number of columns for fine grid
        ydim=ny-2+2*npml; #% Total number of rows
        ydim_f=(nd/2+1)*N;
################################3
        cn = 1/ 1.42
        dt1 = cn* dx /c1 /1.42
        cdtdx=c1*dt1/self.dx
        nt1 =round(2*md/(fp*dt1)+(np.sqrt(xdim**2+ydim**2))/cdtdx)
        fmax = 1/ (2*self.dt)
        NFFT = 2 ** round(np.log2(nt1))
        self.f = fmax * np.linspace(0,1,NFFT/2+1)
        self.recorderF = fft(self.recorder, NFFT)/nt1
        print(self.recorderF)     
        self.finished_trigger.emit()
        QtCore.QThread.__init__(self)


app = QApplication(sys.argv)
window = MyWindow()
window.show()
sys.exit(app.exec_())
