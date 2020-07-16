#required libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
from PyQt5 import uic
import os
from PyQt5.QtWidgets import QApplication, QMainWindow
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg \
import FigureCanvasQTAgg as FigureCanvas
import psutil as p



Form = uic.loadUiType(os.path.join(os.getcwd(), 'gui.ui'))[0]

# constants:
MaxTime = 100  # Duration of simulation
# Required constants and variables
fp = 20.0E9  # Electromagnetic wave Frequency
A = 2.0  # Sine source magnitude
c = 3E8  # speed of EM wave
landa = c / fp  # EM wavelength
d = 1.0  # distance in exercise (in m)
dx = landa / 2  # Spatial discretisation step, at least 20 samples per wavelength
nd = math.ceil(d / dx)  # number of grids in d

# Room Dimention
SizeX = nd + 1  # X position of the room
SizeY = nd + 1  # Y position of the room
#print(SizeY)

# Source Position
xsrc = math.floor(SizeX / 3)
ysrc = math.floor(SizeY / 3)

# Wall position
wx1 = 50
wx2 = 100
wy1 = 65
wy2 = 70

# remaining useful constants and variables
Cdtds = 1.0 / math.sqrt(2.0)  # Courant number  %Courant stability factor
delta = 1e-3
deltat = Cdtds * delta / c

cn = 1 / math.sqrt(2)
dt = cn * dx / c / math.sqrt(2)  # Time

mu_0 = 4.0 * np.pi * 1.0e-7;  # Permeability of free space
eps_0 = 8.8542e-12;
imp0 = np.sqrt(mu_0 / eps_0)  # 377.0
#print("we are after consts")


class main(FigureCanvas):
    def __init__(self):

        # Initializing the values of Electronic and magnetic wave
        self.Ez = [[0.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
        self.Hx = [[0.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
        self.Hy = [[0.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]

        self.zarib(2)
        self.PML_init()

        # first image setup
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.ax.set_xlim(0, 200)
        self.ax.set_ylim(0, 100)
        # and disable figure-wide autoscale
        self.ax.set_autoscale_on(False)
        # generates first "empty" plots
        self.user = []
        self.l_user, = self.ax.plot([], self.user, label='User %')
        self.ax.legend()
        self.y, self.x = np.mgrid[range(SizeX), range(SizeY)]
        print("sizey",type(self.Ez[:-1, :-1]))
        self.mesh = self.ax.pcolormesh(self.x, self.y, self.Ez[:-1, :-1],
                                       cmap='RdBu', vmin=-0.01, vmax=0.01)
        # self.mesh.set_array(np.zeros((SizeX, SizeY)).ravel())
        self.fig.canvas.draw()
        # initialize the iteration counter
        self.cnt = 0
        # call the update method (to speed-up visualization)
        print("before timer")
        self.timerEvent(None)
        # start timer, trigger event every 200 millisecs
        self.timer = self.startTimer(200)

    def zarib(self, condition):
        if condition == 0:
            pass
        else:
            self.Cezh = [[Cdtds * imp0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
            self.Ceze = [[1.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]

            self.Chxh = [[1.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
            self.Chxe = [[Cdtds / imp0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]

            self.Chyh = [[1.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
            self.Chye = [[Cdtds / imp0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]

    # ezInc Fn Defining the value of a sine wave generator
    def ezInc(self, t):
        return A * np.sin(2.0 * np.pi * fp * t)

    def PML_init(self):
        ########################################
        # PML CHANGES
        npmls = 10  # Depth of PML region in # of cells
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
                self.Ceze[i][j] = ca[m]
                self.Cezh[i][j] = cb[m]
            for j in range(jp, nd - 1):  # =jp+1:nd
                m = j - jp
                self.Ceze[i][j] = ca[m]
                self.Cezh[i][j] = cb[m]

        # ..... Front and Back PML Regions .....
        for j in range(1, nd + 1):  # =2:nd
            for i in range(3, npmls + 2):  # =2:npmls+1
                m = npmls + 2 - i
                self.Ceze[i][j] = ca[m]
                self.Cezh[i][j] = cb[m]

            for i in range(ip, nd - 1):  # =ip+1:nd
                m = i - ip
                self.Ceze[i][j] = ca[m]
                self.Cezh[i][j] = cb[m]

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Hx Fields >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #                 ..... Left and Right PML Regions .....
        for i in range(1, nd + 1):  # =2:nd
            for j in range(2, npmls + 1):  # =1:npmls
                m = npmls + 1 - j
                self.Chxh[i][j] = da[m]
                self.Chxe[i][j] = db[m]
            for j in range(jp, nd - 1):  # =jp+1:nd
                m = j - jp
                self.Chxh[i][j] = da[m]
                self.Chxe[i][j] = db[m]

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Hy Fields >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #                 ..... Front and Back PML Regions .....
        for j in range(1, nd + 1):  # =2:nd
            for i in range(2, npmls + 1):  # =1:npmls
                m = npmls + 1 - i
                self.Chyh[i][j] = da[m]
                self.Chye[i][j] = db[m]
            for i in range(ip, nd - 1):  # =ip+1:nd
                m = i - ip
                self.Chyh[i][j] = da[m]
                self.Chye[i][j] = db[m]

        ########################################

        # Initializing existance of wave in the room (0)
        self.room = [[0.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
        # Initializing existance of wall in the room (0)
        self.wall = [[0.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]


        #print('{:<4}'.format('Size X , Y'), len(self.Hy) - 1, len(self.Hy[0]) - 1)

    def update(self):
        # plt.ion()
        # Update magnetic field
        for mm in range(0, SizeX + 1):
            # Periodic Boundary condition for X axis
            if mm == SizeX:
                for nn in range(0, SizeY + 1):
                    self.Hy[mm][nn] = self.Chyh[mm][nn] * self.Hy[mm][nn] + \
                                      self.Chye[mm][nn] * (self.Ez[0][nn] - self.Ez[mm][nn])
            else:
                for nn in range(0, SizeY + 1):
                    self.Hy[mm][nn] = self.Chyh[mm][nn] * self.Hy[mm][nn] + \
                                      self.Chye[mm][nn] * (self.Ez[mm + 1][nn] - self.Ez[mm][nn])

        for mm in range(0, SizeX + 1):
            for nn in range(0, SizeY + 1):
                # Periodic Boundary condition for Y axis
                if nn == SizeY:
                    self.Hx[mm][nn] = self.Chxh[mm][nn] * self.Hx[mm][nn] - \
                                      self.Chxe[mm][nn] * (self.Ez[mm][0] - self.Ez[mm][nn])
                else:
                    self.Hx[mm][nn] = self.Chxh[mm][nn] * self.Hx[mm][nn] - \
                                      self.Chxe[mm][nn] * (self.Ez[mm][nn + 1] - self.Ez[mm][nn])

        # Update electrical field
        for mm in range(0, SizeX + 1):
            for nn in range(0, SizeY + 1):
                self.Ez[mm][nn] = self.Ceze[mm][nn] * self.Ez[mm][nn] + \
                                  self.Cezh[mm][nn] * (
                                  (self.Hy[mm][nn] - self.Hy[mm - 1][nn]) - (self.Hx[mm][nn] - self.Hx[mm][nn - 1]))

        # Defining wall
        # for mm in range(wx1, wx2):
        #     for nn in range(wy1, wy2):
        #         self.wall[mm][nn] = 2
        #         self.Ez[mm][nn] = 0

        # Finilize the room value = self.wall + self.Ez
        # self.room = [[self.wall[mm][nn] + self.Ez[mm][nn] for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]

        # Plotting the room
        # plt.ion()

        # if Time % 2 == 0:
        #     img = plt.imshow(self.room, vmax=A, vmin=-A)
        #     plt.colorbar(img)
        #     plt.pause(0.0002)
        #     plt.clf()
        return self.Ez

    def timerEvent(self, evt):
        self.Ez[math.ceil(xsrc)][math.ceil(ysrc)] = self.ezInc(deltat * self.cnt)
        result = self.update()
        print(type(self.Ez), self.cnt)

        # # append new data to the data sets
        # self.user.append(result)
        # # update lines data using the lists with new data
        # self.l_user.set_data(range(len(self.user)), self.user)
        # # force a redraw of the Figure
        # self.mesh.set_array(self.Ez[:-1, :-1].ravel())
        self.fig.canvas.draw()
        print("hello")
        if self.cnt == MaxTime:
            # stop the timer
            self.killTimer(self.timer)
        else:
            self.cnt += 1
            print("plus")


class PlotWindow(QMainWindow, Form):
    def __init__(self):
        QMainWindow.__init__(self)
        Form.__init__(self)
        self.setupUi(self)

    def start_clicked(self):
        print("after click")
        widget = main()
        widget.setWindowTitle("start")
        widget.show()


# # create the GUI application
# app = QApplication(sys.argv)
# app.setStyle("Fusion")
# w = PlotWindow()
#
# w.start_Button.clicked.connect(PlotWindow.start_clicked)
#
# w.show()
# sys.exit(app.exec_())


main()
