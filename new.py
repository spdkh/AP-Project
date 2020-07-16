import matplotlib
import math
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


def FDTD(SizeX, SizeY, c, dx, dt, Hy,  Hx, Ez, nd, layer ):
    Cdtds = 1.0 / math.sqrt(2.0)  # Courant number  %Courant stability factor
    mu_0 = 4.0 * np.pi * 1.0e-7;  # Permeability of free space
    eps_0 = 8.8542e-12;
    imp0 = np.sqrt(mu_0 / eps_0)  # 377.0
    #c = 3E8  # speed of EM wave
    ############################################
    Cezh = [[Cdtds * imp0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
    Ceze = [[1.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]

    Chxh = [[1.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
    Chxe = [[Cdtds / imp0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]

    Chyh = [[1.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
    Chye = [[Cdtds / imp0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
    print(1)
    ############################################

    # Cezh = Cdtds * imp0
    # Chxe = Cdtds / imp0
    # Chye = Cdtds / imp0

    # print('Chye', Chye)
    # print('Cezh', Cezh)
    # print('Chxe', Chxe)


    ########################################
    # PML CHANGES

    npmls = layer  # Depth of PML region in # of cells
    ip = SizeX - npmls
    jp = SizeY - npmls

    # ***********************************************************************
    # Set up the Berenger's PML material constants
    # ***********************************************************************
    sigmax = -3.0 * eps_0 * c * np.log(1.0e-5) / (2.0 * dx * npmls)
    rhomax = sigmax * (imp0 ** 2)
    sig = [sigmax * ((m - 0.5) / (npmls + 0.5)) ** 2 for m in range(1, npmls + 1)]
    rho = [rhomax * (m / (npmls + 0.5)) ** 2 for m in range(1, npmls + 1)]
    # print(len(rho), 'after sig')

    # ***********************************************************************
    # Set up constants for Berenger's PML
    # ***********************************************************************
    re = [sig[m] * dt / eps_0 for m in range(0, npmls)]
    rm = [rho[m] * dt / mu_0 for m in range(0, npmls)]
    ca = [np.exp(-re[m]) for m in range(0, npmls)]
    cb = [-(np.exp(-re[m]) - 1.0) / sig[m] / dx for m in range(0, npmls)]
    da = [np.exp(-rm[m]) for m in range(0, npmls)]
    db = [-(np.exp(-rm[m]) - 1.0) / rho[m] / dx for m in range(0, npmls)]

    # ***********************************************************************
    #  Initialise all matrices for the Berenger's PML
    # ***********************************************************************
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Ez Fields >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #                 ..... Left and Right PML Regions .....
    for i in range(1, nd + 1):  # =2:nd
        for j in range(3, npmls + 2):  # =2:npmls+1
            m = npmls + 2 - j
            Ceze[i][j] = ca[m]
            Cezh[i][j] = cb[m]
        for j in range(jp, nd - 1):  # =jp+1:nd
            m = j - jp
            Ceze[i][j] = ca[m]
            Cezh[i][j] = cb[m]

    # ..... Front and Back PML Regions .....
    for j in range(1, nd + 1):  # =2:nd
        for i in range(3, npmls + 2):  # =2:npmls+1
            m = npmls + 2 - i
            Ceze[i][j] = ca[m]
            Cezh[i][j] = cb[m]

        for i in range(ip, nd - 1):  # =ip+1:nd
            m = i - ip
            Ceze[i][j] = ca[m]
            Cezh[i][j] = cb[m]

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Hx Fields >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #                 ..... Left and Right PML Regions .....
    for i in range(1, nd + 1):  # =2:nd
        for j in range(2, npmls + 1):  # =1:npmls
            m = npmls + 1 - j
            Chxh[i][j] = da[m]
            Chxe[i][j] = db[m]
        for j in range(jp, nd - 1):  # =jp+1:nd
            m = j - jp
            Chxh[i][j] = da[m]
            Chxe[i][j] = db[m]

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Hy Fields >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #                 ..... Front and Back PML Regions .....
    for j in range(1, nd + 1):  # =2:nd
        for i in range(2, npmls + 1):  # =1:npmls
            m = npmls + 1 - i
            Chyh[i][j] = da[m]
            Chye[i][j] = db[m]
        for i in range(ip, nd - 1):  # =ip+1:nd
            m = i - ip
            Chyh[i][j] = da[m]
            Chye[i][j] = db[m]

    ########################################
    # Initializing the values of Electronic and magnetic wave

    # Update magnetic fieldl
    for mm in range(0, SizeX + 1):
        # Periodic Boundary condition for X axis
        if mm == SizeX:
            for nn in range(0, SizeY + 1):
                Hy[mm][nn] = Chyh[mm][nn] * Hy[mm][nn] + \
                             Chye[mm][nn] * (Ez[0][nn] - Ez[mm][nn])
        else:
            for nn in range(0, SizeY + 1):
                Hy[mm][nn] = Chyh[mm][nn] * Hy[mm][nn] + \
                             Chye[mm][nn] * (Ez[mm + 1][nn] - Ez[mm][nn])

    for mm in range(0, SizeX + 1):
        for nn in range(0, SizeY + 1):
            # Periodic Boundary condition for Y axis
            if nn == SizeY:
                Hx[mm][nn] = Chxh[mm][nn] * Hx[mm][nn] - \
                             Chxe[mm][nn] * (Ez[mm][0] - Ez[mm][nn])
            else:
                Hx[mm][nn] = Chxh[mm][nn] * Hx[mm][nn] - \
                             Chxe[mm][nn] * (Ez[mm][nn + 1] - Ez[mm][nn])

    # Update electrical field
    for mm in range(0, SizeX + 1):
        for nn in range(0, SizeY + 1):
            Ez[mm][nn] = Ceze[mm][nn] * Ez[mm][nn] + \
                         Cezh[mm][nn] * ((Hy[mm][nn] - Hy[mm - 1][nn]) - (Hx[mm][nn] - Hx[mm][nn - 1]))


###############################################################################################
class MyWindow(QMainWindow, Form):
    def __init__(self):
        Form.__init__(self)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.flag = True
        self.start_flag = False
        MaxTime = 1000  # Duration of simulation

        # Required constants and variables
        self.fp = 20.0E9  # Electromagnetic wave Frequency
        self.A = 2.0  # Sine source magnitude
        self.c = 3E8  # speed of EM wave
        self.landa = self.c / self.fp  # EM wavelength
        self.CFL = 1
        self.nt = 300 / self.CFL  # number of time steps in simulation

        #  Events
        self.start_pushButton.clicked.connect(self.start)
        self.pushButton_rec.clicked.connect(self.plot_record)
        #self.kmax_pushButton.clicked.connect(self.kMax)
        self.stop_pushButton.clicked.connect(self.stop)
        self.file_actionStart.triggered.connect(self.start)


    def start(self):
        print('1')
        self.start_flag = True
        self.fc = int(self.freq.text())
        self.layer = int(self.Layer.text())
        landa = self.c / self.fc
        self.d = 1.0  # distance in exercise (in m)
        self.dx = landa / 2  # Spatial discretisation step, at least 20 samples per wavelength
        self.nd = math.ceil(self.d /self.dx)  # number of grids in d

        # Room Dimention
        self.SizeX = self.nd + 1  # X position of the room
        self.SizeY = self.nd + 1  # Y position of the room

        # Source Position
        self.xsrc = math.floor(self.SizeX / 3)
        self.ysrc = math.floor(self.SizeY / 3)
        # remaining useful constants and variables
        self.Cdtds = 1.0 / math.sqrt(2.0)  # Courant number  %Courant stability factor
        self.delta = 1e-3
        self.deltat = self.Cdtds * (self.delta) / self.c

        cn = 1 / math.sqrt(2)
        dt = cn * self.dx / self.c / math.sqrt(2)  # Time

        ######class methods######
        self.plot_label.setText("در حال کار...")
        self.Ez = np.zeros((self.SizeX, self.SizeY))
        self.Hx = np.zeros((self.SizeX, self.SizeY))
        self.Hy = np.zeros((self.SizeX, self.SizeY))

        if self.flag == True:
            # figure
            self.fig = Figure()

            self.fig.patch.set_color('w')
            self.ax = self.fig.add_subplot(111, frame_on=False)

            self.ax.set_title("My Title")
            self.ax.set_xlim([0, self.SizeX])
            self.ax.set_ylim([0, self.SizeY])
            self.canvas = FigureCanvas(self.fig)
            self.y, self.x = np.mgrid[range(self.SizeX), range(self.SizeY)]
            self.x
            self.y
            print(3)
            self.mesh = self.ax.pcolormesh(self.x, self.y, self.Ez[:-1, :-1], cmap='RdBu', vmin=-0.01, vmax=0.01)
            print(3)

            # widget
            l = QVBoxLayout(self.matplotlib_widget)
            l.addWidget(self.canvas)
            print(2)
            rect1 = matplotlib.patches.Rectangle((self.SizeX - self.layer, self.layer),
                                                 width=1, height=self.SizeY - 2 * self.layer, alpha=1, facecolor='k')
            self.ax.add_patch(rect1)
            rect2 = matplotlib.patches.Rectangle((self.layer, self.layer),
                                                 width=1, height=self.SizeY - 2 * self.layer, alpha=1, facecolor='k')
            self.ax.add_patch(rect2)
            rect3 = matplotlib.patches.Rectangle((self.layer, self.SizeX - self.layer),
                                                 width=self.SizeX - 2 * self.layer, height=1, alpha=1, facecolor='k')
            self.ax.add_patch(rect3)
            rect4 = matplotlib.patches.Rectangle((self.layer, self.layer),
                                                 width=self.SizeX - 2 * self.layer, height=1, alpha=1, facecolor='k')
            self.ax.add_patch(rect4)
            rect5 = matplotlib.patches.Rectangle((self.SizeX / 3, self.layer),
                                                 width=1, height=self.SizeY / 3 - self.layer, alpha=1, facecolor='b')
            self.ax.add_patch(rect5)
            circle = matplotlib.patches.Circle((7 * self.SizeX / 15, self.SizeX / 5), 2, facecolor='r')
            self.ax.add_patch(circle)
            print(2)
            dic_wave = {'nt': self.nt,
                         'layer': self.layer, 'xsrc': self.xsrc,
                        'ysrc': self.ysrc, 'A': self.A, 'fc': self.fc, 'SizeX': self.SizeX,
                        'SizeY': self.SizeY, 'dx': self.dx,  'c': self.c}
            print(1)
            dic_array = {'Ez': self.Ez, 'Hx': self.Hx, 'Hy': self.Hy}
            print(4)
            # thread
            self.thread = PlotThread(dic_wave, dic_array)  # dic_wave, dic_array)
            print(5)
            self.thread.update_trigger.connect(self.update_plot)
            self.thread.finished_trigger.connect(self.stop)

            print(3)
            self.thread.exit()
            self.thread.start()
            self.flag = False
            print('3')
        if self.flag == False:
            self.thread.exit()
            self.mesh.set_array(np.zeros((self.SizeX, self.SizeY)).ravel())
            print(10)
            self.canvas.draw()
            print(4)
            dic_wave = {'nt': self.nt,
                        'layer': self.layer, 'xsrc': self.xsrc,
                        'ysrc': self.ysrc, 'A': self.A, 'fc': self.fc, 'SizeX': self.SizeX,
                        'SizeY': self.SizeY, 'dx': self.dx, 'c': self.c}

            dic_array = {'Ez': self.Ez, 'Hx': self.Hx, 'Hy': self.Hy}
            self.thread.set_data(dic_wave, dic_array)
            self.thread.start()
            self.flag = False
            print(10)


    def update_plot(self):
        self.Ez = self.thread.Ez
        self.mesh.set_array(self.Ez[:-1, :-1].ravel())
        self.canvas.draw()

    def stop(self):
        self.plot_label.setText("متوقف")
        self.thread.stop()
        self.thread.exit()

    def plot_record(self):
        fig1 = plt.figure(1)
        plt.plot(np.abs(self.thread.recorderF))
        plt.show()

#############################################################################################################
class PlotThread(QtCore.QThread, MyWindow):
    print(7)
    update_trigger = QtCore.pyqtSignal(np.ndarray)
    finished_trigger = QtCore.pyqtSignal()
    recorderF = np.zeros((300))

    def __init__(self, dic_wave, dic_array):  # , dic_wave, dic_array):
        QtCore.QThread.__init__(self)
        self.nt = dic_wave['nt']
        # self.dt = dic_wave['dt']
        # self.t0 = dic_wave['t0']
        # self.sigma = dic_wave['sigma']
        self.Ez = dic_array['Ez']
        self.Hy = dic_array['Hy']
        self.Hx = dic_array['Hx']
        # self.m = dic_wave['m']
        self.Layer = dic_wave['layer']
        self.xsrc = dic_wave['xsrc']
        self.ysrc = dic_wave['ysrc']
        self.A = dic_wave['A']
        self.fc = dic_wave['fc']
        self.SizeX = dic_wave['SizeX']
        self.SizeY = dic_wave['SizeY']
        self.dx = dic_wave['dx']
        # self.dy = dic_wave['dy']
        self.c = dic_wave['c']
        # self.ax = dic_array['ax']
        print(2)
        self.flag = False
        # fekkon
        # self.recorder = np.zeros((self.nt))
        print(4)
        self.recorderF = []
        print(3)
        self.f = []


    def stop(self):
        self.flag = False

    def run(self):
        self.flag = True
        for it in range(1, int(self.nt + 1)):
            if not self.flag:
                return
            t = (it - 1) * self.dt

            # source updaten at new time
            source = self.A * A*np.sin(2.0*np.pi * self.fp * t)
            # add pressure to source location
            self.Ez[self.xsrc - 1: self.xsrc + 1, self.ysrc - 1: self.ysrc + 1] = self.Ez[
                                                                                        self.xsrc - 1: self.xsrc + 1,
                                                                                        self.ysrc - 1: self.ysrc + 1] + source

            # calculate and update p
            FDTD(self.SizeX, self.SizeY, self.c, self.dx,
                 self.dt, self.Hy, self.Hx, self.Ez, self.nd, self.Layer)

            self.recorder[it - 1] = self.p[7 * self.nx / 15][self.nx / 5]

            if it % 3 == 0:
                self.ax.set_title(str(int(it / 3)))
                self.update_trigger.emit(self.Ez)
                sleep(0.1)
        fp = self.fc
        m = 3
        N = 3
        c1 = 3E8
        lp = c1 / fp
        md = 1
        d = 1.0
        dx = c1 / fp / 10
        nd = np.ceil(d / dx)
        ##################################

        if nd % 2 == 1:  # % nd should be even, because nd/2 is needed.
            nd = nd + 1;

        if N % 2 == 0:  # % N should be odd
            N = N + 1;

        N2 = (N - 1) / 2;  # % N/2
        np1 = lp / dx;  # % Does not need to be integer
        nx = 7 * nd + 1;  # % Number of cells in x direction
        ny = 6 * nd + 1;  # % Number of cells in y direction
        npml = self.Layer;  # % Number of cells in PML
        xdim = nx - 2 + 2 * npml;  # % Total number of columns for course grid
        xdim_f = (nd + 1) * N;  # % Total number of columns for fine grid
        ydim = ny - 2 + 2 * npml;  # % Total number of rows
        ydim_f = (nd / 2 + 1) * N;
        ################################3
        cn = 1 / 1.42
        dt1 = cn * dx / c1 / 1.42
        cdtdx = c1 * dt1 / self.dx
        nt1 = round(2 * md / (fp * dt1) + (np.sqrt(xdim ** 2 + ydim ** 2)) / cdtdx)
        fmax = 1 / (2 * self.dt)
        NFFT = 2 ** round(np.log2(nt1))
        self.f = fmax * np.linspace(0, 1, NFFT / 2 + 1)
        self.recorderF = fft(self.recorder, NFFT) / nt1
        print(self.recorderF)
        self.finished_trigger.emit()
        QtCore.QThread.__init__(self)


app = QApplication(sys.argv)
window = MyWindow()
window.show()
sys.exit(app.exec_())
