"""
" FTDT electromagnetic wave emission simulation
" @version 100%
" @authors :
"           S Parisa Dajkhosh  <9223031>
"           Siminfar Samakoush <9223051>
"           Fatemeh Alimoradi  <9223064>
"""
import matplotlib
import math
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
import sys
import numpy as np
from time import sleep
from numba import jit, float64, int32, double, void
import numba as nb
from scipy.fftpack import fft

global condition

matplotlib.use("Qt5Agg")
Form = uic.loadUiType('gui.ui')[0]  # Load ui

# @jit([void(int32, int32, float64, float64, float64, float64[:, :],
#       float64[:, :], float64[:, :], int32, int32)], nopython=True)


def FDTD(SizeX, SizeY, c, dx, dt, Hy,  Hx, Ez, nd, layer):

    global condition
    Cdtds = 1.0 / math.sqrt(2.0)  # Courant number  %Courant stability factor
    mu_0 = 4.0 * np.pi * 1.0e-7;  # Permeability of free space
    eps_0 = 8.8542e-12;
    imp0 = np.sqrt(mu_0 / eps_0)  # 377.0
    ############################################
    Cezh = [[Cdtds * imp0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
    Ceze = [[1.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]

    Chxh = [[1.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
    Chxe = [[Cdtds / imp0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]

    Chyh = [[1.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
    Chye = [[Cdtds / imp0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]

    ########################################
    if condition == "Reflection":
        ################################################
        for mm in range(1, SizeX-1 ):
            for nn in range(1, SizeY):
                Hy[mm][nn] = Chyh[mm][nn] * Hy[mm][nn] + \
                             Chye[mm][nn] * (Ez[mm + 1][nn] - Ez[mm][nn])

        for mm in range(1, SizeX):
            for nn in range(1, SizeY - 1):
                Hx[mm][nn] = Chxh[mm][nn] * Hx[mm][nn] - \
                             Chxe[mm][nn] * (Ez[mm][nn + 1] - Ez[mm][nn])

        for mm in range(2, SizeX):
            for nn in range(2, SizeY):
                Ez[mm][nn] = Ceze[mm][nn] * Ez[mm][nn] + \
                             Cezh[mm][nn] * ((Hy[mm][nn] - Hy[mm - 1][nn]) - (Hx[mm][nn] - Hx[mm][nn - 1]))

    elif condition == "BC":
        # Update magnetic field
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

    elif condition == "PML":
        #######################################
        # PML CHANGES

        npmls = layer  # Depth of PML region in # of cells
        ip = SizeX - math.ceil(npmls)
        jp = SizeY - math.ceil(npmls)

        """
        " Set up the Berenger's PML material constants
        """

        sigmax = -3.0 * eps_0 * c * np.log(1.0e-5) / (2.0 * dx * math.ceil(npmls))
        rhomax = sigmax * (imp0 ** 2)
        sig = [sigmax * ((m - 0.5) / (math.ceil(npmls) + 0.5)) ** 2 for m in range(1, math.ceil(npmls + 1))]
        rho = [rhomax * (m / (math.ceil(npmls) + 0.5)) ** 2 for m in range(1, math.ceil(npmls + 1))]

        """
        " Set up constants for Berenger's PML
        """
        re = [sig[m] * dt / eps_0 for m in range(0, math.ceil(npmls))]
        rm = [rho[m] * dt / mu_0 for m in range(0, math.ceil(npmls))]
        ca = [np.exp(-re[m]) for m in range(0, math.ceil(npmls))]
        cb = [-(np.exp(-re[m]) - 1.0) / sig[m] / dx for m in range(0, math.ceil(npmls))]
        da = [np.exp(-rm[m]) for m in range(0, math.ceil(npmls))]
        db = [-(np.exp(-rm[m]) - 1.0) / rho[m] / dx for m in range(0, math.ceil(npmls))]

        """
        "  Initialise all matrices for the Berenger's PML
        """
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Ez Fields >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #                 ..... Left and Right PML Regions .....

        for i in range(1, nd + 1):  # =2:nd
            for j in range(3, math.ceil(npmls) + 2):  # =2:npmls+1
                m = math.ceil(npmls) + 2 - j
                Ceze[i][j] = ca[m]
                Cezh[i][j] = cb[m]
            for j in range(jp, nd - 1):  # =jp+1:nd
                m = j - jp
                Ceze[i][j] = ca[m]
                Cezh[i][j] = cb[m]

        #                  ..... Front and Back PML Regions .....
        for j in range(1, nd + 1):  # =2:nd
            for i in range(3, math.ceil(npmls) + 2):  # =2:npmls+1
                m = math.ceil(npmls) + 2 - i
                Ceze[i][j] = ca[m]
                Cezh[i][j] = cb[m]

            for i in range(ip, nd - 1):  # =ip+1:nd
                m = i - ip
                Ceze[i][j] = ca[m]
                Cezh[i][j] = cb[m]

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Hx Fields >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #                 ..... Left and Right PML Regions .....
        for i in range(1, nd + 1):  # =2:nd
            for j in range(2, math.ceil(npmls) + 1):  # =1:npmls
                m = math.ceil(npmls) + 1 - j
                Chxh[i][j] = da[m]
                Chxe[i][j] = db[m]

            for j in range(jp, nd - 1):  # =jp+1:nd
                m = j - jp
                Chxh[i][j] = da[m]
                Chxe[i][j] = db[m]

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Hy Fields >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #                 ..... Front and Back PML Regions .....

        for j in range(1, nd + 1):  # =2:nd
            for i in range(2, math.ceil(npmls) + 1):  # =1:npmls
                m = math.ceil(npmls) + 1 - i
                Chyh[i][j] = da[m]
                Chye[i][j] = db[m]

            for i in range(ip, nd - 1):  # =ip+1:nd
                m = i - ip
                Chyh[i][j] = da[m]
                Chye[i][j] = db[m]

        # Update magnetic field
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
        global condition
        Form.__init__(self)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.setFixedSize(1000, 1000)
        self.flag = True
        self.start_flag = False
        self.flag2 = True

        """
        " Required constants and variables
        """

        self.fMhz = 20000
        self.fp = self.fMhz * 1.0E6  # Electromagnetic wave Frequency
        self.cMms = 300
        self.c = self.cMms * 1.0E6  # speed of EM wave
        self.landa = self.cMms / self.fMhz  # EM wavelength

        self.nt = 300   # number of time steps in simulation
        self.layer = 10
        self.d = 20000/self.fMhz  # distance in exercise (in m)
        self.A = 2  # Sine source magnitude
        self.dx = self.landa / 2  # Spatial discretisation step, at least 20 samples per wavelength
        self.dy = self.dx
        self.nd = math.ceil(self.d / self.dx)  # number of grids in d

        # Room Dimension
        self.SizeX = self.nd + 1  # X position of the room
        self.SizeY = self.nd + 1  # Y position of the room
        # self.layer = self.SizeX/100
        print("SizeX = ", self.SizeX, "SizeY = ", self.SizeY)

        # Source Position
        self.xsrc = math.floor(self.SizeX / 2)
        self.ysrc = math.floor(self.SizeY / 3)
        # remaining useful constants and variables
        self.Cdtds = 1.0 / math.sqrt(2.0)  # Courant number  %Courant stability factor
        self.delta = 1e-9
        self.deltat = self.Cdtds * (self.delta) / self.cMms

        self.cn = 1 / math.sqrt(2)
        self.dt = self.cn * self.dx / self.c / math.sqrt(2)  # Time
        self.Ez = np.zeros((self.SizeX+1, self.SizeY+1))
        self.Hx = np.zeros((self.SizeX+1, self.SizeY+1))
        self.Hy = np.zeros((self.SizeX+1, self.SizeY+1))
        self.barrow = []

        condition = "PML"

        if self.flag == True:
            # figure
            self.fig = Figure()
            self.fig.patch.set_color('w')
            self.ax = self.fig.add_subplot(111, frame_on=False)
            self.ax.set_title("Helloooooooo")
            self.ax.set_xlim([0, self.SizeX])
            self.ax.set_ylim([0, self.SizeY])
            self.canvas = FigureCanvas(self.fig)
            self.y, self.x = np.mgrid[range(self.SizeX+1), range(self.SizeY+1)]
            self.mesh = self.ax.pcolormesh(self.x, self.y, self.Ez,
                                           cmap='PRGn', vmin=-self.A/(self.d * 2), vmax=self.A/(self.d * 2))

            # widget
            l = QVBoxLayout(self.matplotlib_widget)
            l.addWidget(self.canvas)
            rect1 = matplotlib.patches.Rectangle((self.SizeX - self.layer, 0),
                                                 width=self.layer, height=self.SizeY,
                                                 alpha=1, facecolor='cyan', edgecolor = 'cyan')
            self.ax.add_patch(rect1)
            rect2 = matplotlib.patches.Rectangle((0, self.layer),
                                                 width=self.layer, height=self.SizeY,
                                                 alpha=1, facecolor='cyan', edgecolor = 'cyan')
            self.ax.add_patch(rect2)
            rect3 = matplotlib.patches.Rectangle((0, self.SizeY - self.layer),
                                                 width=self.SizeX, height=self.layer,
                                                 alpha=1, facecolor='cyan', edgecolor = 'cyan')
            self.ax.add_patch(rect3)
            rect4 = matplotlib.patches.Rectangle((0, 0),
                                                 width=self.SizeX, height=self.layer,
                                                 alpha=1, facecolor='cyan', edgecolor = 'cyan')
            self.ax.add_patch(rect4)

        # Events
        self.pushButton3.clicked.connect(self.PML)
        self.pushButton2.clicked.connect(self.Reflection)
        self.pushButton1.clicked.connect(self.BC)
        self.pushButton.clicked.connect(self.apply)
        self.start_pushButton.clicked.connect(self.start)
        self.pushButton_rec.clicked.connect(self.plot_record)
        self.stop_pushButton.clicked.connect(self.stop)

    def apply(self):
        self.barrow.append(int(self.BarrowX.text()))
        self.barrow.append(int(self.BarrowY.text()))
        self.barrow.append(int(self.BarrowW.text()))
        self.barrow.append(int(self.BarrowH.text()))
        self.barrow.append(1)

        self.fMhz = int(self.freq.text())
        self.layer = int(self.Layer.text())

        circle = matplotlib.patches.Circle((int(self.recorderY.text()), (int(self.recorderX.text()))),
                                           self.SizeX * 0.01, facecolor='g')
        self.ax.add_patch(circle)

        rect5 = matplotlib.patches.Rectangle((self.barrow[1], self.barrow[0]),
                                             width=self.barrow[3], height=self.barrow[2], facecolor='r')
        self.ax.add_patch(rect5)
        self.xsrc = int(self.srcX.text())
        self.ysrc = int(self.srcY.text())
        self.nt = int(self.tmax.text())   # number of time steps in simulation

    def PML(self):
        global condition
        if self.flag2 == True:
            condition = 'PML'
        print(condition)

    def Reflection(self):
        global condition
        if self.flag2 == True:
            condition = 'Reflection'
        print(condition)

    def BC(self):
        global condition
        if self.flag2 == True:
            condition = 'BC'
        print(condition)

    def start(self):

        self.start_flag = True
        self.flag2 = False

        ######class methods######
        self.plot_label.setText("working ...")

        if self.flag == True:
            self.Ez = np.zeros((self.SizeX + 1, self.SizeY + 1))

            dic_wave = {'nt': self.nt, 'dt': self.dt, 'nd': self.nd,
                         'layer': self.layer, 'xsrc': self.xsrc,
                        'ysrc': self.ysrc, 'A': self.A, 'fp': self.fp, 'SizeX': self.SizeX,
                        'SizeY': self.SizeY, 'dx': self.dx,  'c': self.c, 'deltat': self.deltat}

            dic_array = {'Ez': self.Ez, 'Hx': self.Hx, 'Hy': self.Hy, 'ax': self.ax, 'barrow': self.barrow}

            # thread
            self.thread = PlotThread(dic_wave, dic_array)
            self.thread.update_trigger.connect(self.update_plot)
            self.thread.finished_trigger.connect(self.stop)

            self.thread.exit()
            self.thread.start()
            self.flag = False

        if self.flag == False:
            self.thread.exit()
            self.mesh.set_array(np.zeros((self.SizeX, self.SizeY)).ravel())
            self.canvas.draw()
            self.thread.start()
            self.flag = False

    # print(10)
    # ########################### dorost kon
    def update_plot(self):
        # pass
        self.Ez = self.thread.Ez
        self.mesh.set_array(self.Ez[:-1, :-1].ravel())
        self.canvas.draw()

    def stop(self):
        self.flag2 = True

        self.plot_label.setText("STOP")
        self.thread.stop()
        self.thread.exit()


    def plot_record(self):
        plt.figure(1)
        plt.plot(np.abs(self.thread.recorderF))
        plt.show()

#############################################################################################################


class PlotThread(QtCore.QThread, MyWindow, Form):
    update_trigger = QtCore.pyqtSignal(np.ndarray)
    finished_trigger = QtCore.pyqtSignal()
    recorderF = np.zeros((300))

    def __init__(self, dic_wave, dic_array):
        QtCore.QThread.__init__(self)
        global condition
        self.nt = dic_wave['nt']
        self.nd = dic_wave['nd']
        self.dt = dic_wave['dt']
        self.deltat = dic_wave['deltat']
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
        self.fp = dic_wave['fp']
        self.SizeX = dic_wave['SizeX']
        self.SizeY = dic_wave['SizeY']
        self.dx = dic_wave['dx']
        self.c = dic_wave['c']
        self.ax = dic_array['ax']
        self.last_condition = condition
        self.Barrow = dic_array['barrow']

        self.flag = False
        self.recorder = np.zeros(int(self.nt))
        self.recorderF = []
        self.f = []

    def stop(self):
        self.flag = False

    def run(self):
        self.flag = True

        for it in range(1, int(self.nt + 1)):
            if not self.flag:
                return
            t = (it - 1)

            self.Ez[self.Barrow[0]: self.Barrow[0]+self.Barrow[2],
                    self.Barrow[1]: self.Barrow[1]+self.Barrow[3]] = 0

            if self.last_condition != condition:
                self.Ez = np.zeros((self.SizeX + 1, self.SizeY + 1))
                self.Hx = np.zeros((self.SizeX + 1, self.SizeY + 1))
                self.Hy = np.zeros((self.SizeX + 1, self.SizeY + 1))
                self.last_condition = condition

            if it < 30:
                source = self.A * np.sin(2.0 * np.pi * self.fp * t * self.deltat)
                print(self.deltat)
            else:
                source = 0

            self.Ez[self.xsrc - 1: self.xsrc + 1, self.ysrc - 1: self.ysrc + 1] += source

            print(self.Ez[self.xsrc][self.ysrc])
            # calculate and update Ez
            FDTD(self.SizeX, self.SizeY, self.c, self.dx,
                 self.dt, self.Hy, self.Hx, self.Ez, self.nd, self.Layer)
            self.recorder[it - 1] = self.Ez[math.ceil(7 * self.SizeX / 15)][math.ceil(self.SizeX / 5)]

            self.ax.set_title(str(int(it / 3)))
            self.update_trigger.emit(self.Ez)
            sleep(0.05)
            self.last_condition = condition

        N = 3
        c1 = 3E8
        md = 1
        d = 1.0
        dx = c1 / self.fp / 10
        nd = np.ceil(d / dx)
        ##################################

        if nd % 2 == 1:  # % nd should be even, because nd/2 is needed.
            nd = nd + 1;

        if N % 2 == 0:  # % N should be odd
            N = N + 1;
        nx = 7 * nd + 1;  # % Number of cells in x direction
        ny = 6 * nd + 1;  # % Number of cells in y direction
        npml = self.Layer;  # % Number of cells in PML
        xdim = nx - 2 + 2 * npml;  # % Total number of columns for course grid
        ydim = ny - 2 + 2 * npml;  # % Total number of rows
        ################################3
        cn = 1 / 1.42
        dt1 = cn * dx / c1 / 1.42
        cdtdx = c1 * dt1 / self.dx
        nt1 = round(2 * md / (self.fp * dt1) + (np.sqrt(xdim ** 2 + ydim ** 2)) / cdtdx)
        fmax = 1 / (2 * self.dt)
        NFFT = 2 ** round(np.log2(nt1))

        self.f = fmax * np.linspace(0, 1, NFFT / 2 + 1)
        self.recorderF = fft(self.recorder, NFFT) / nt1
        self.finished_trigger.emit()
        QtCore.QThread.__init__(self)



app = QApplication(sys.argv)
window = MyWindow()
window.show()
sys.exit(app.exec_())
