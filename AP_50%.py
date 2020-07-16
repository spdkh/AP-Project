#required libraries
import math
import numpy as np
import matplotlib.pyplot as plt

MaxTime = 1000  # Duration of simulation

# Required constants and variables
fp = 20.0E9  # Electromagnetic wave Frequency
A = 2.0  # Sine source magnitude
c = 3E8  # speed of EM wave
landa = c / fp  # EM wavelength
d = 1.0  # distance in exercise (in m)
dx = landa / 2  # Spatial discretisation step, at least 20 samples per wavelength
nd = math.ceil(d / dx) # number of grids in d

# Room Dimention
SizeX = nd + 1  # X position of the room
SizeY = nd + 1  # Y position of the room
print(SizeY)

# Source Position
xsrc = math.floor(SizeX/3)
ysrc = math.floor(SizeY/3)

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

mu_0 = 4.0*np.pi*1.0e-7;               # Permeability of free space
eps_0 = 8.8542e-12;
imp0 = np.sqrt(mu_0/eps_0) # 377.0

############################################
Cezh = [[Cdtds * imp0 for mm in range(0, SizeX+1)] for nn in range(0, SizeY+1)]
Ceze = [[1.0 for mm in range(0, SizeX+1)] for nn in range(0, SizeY+1)]

Chxh = [[1.0 for mm in range(0, SizeX+1)] for nn in range(0, SizeY+1)]
Chxe = [[Cdtds / imp0 for mm in range(0, SizeX+1)] for nn in range(0, SizeY+1)]

Chyh = [[1.0 for mm in range(0, SizeX+1)] for nn in range(0, SizeY+1)]
Chye = [[Cdtds / imp0 for mm in range(0, SizeX+1)] for nn in range(0, SizeY+1)]

############################################

# Cezh = Cdtds * imp0
# Chxe = Cdtds / imp0
# Chye = Cdtds / imp0

# print('Chye', Chye)
# print('Cezh', Cezh)
# print('Chxe', Chxe)


########################################
# PML CHANGES
npmls = 10                         # Depth of PML region in # of cells
ip = SizeX - npmls
jp = SizeY - npmls

# ***********************************************************************
# Set up the Berenger's PML material constants
# ***********************************************************************
sigmax = -3.0*eps_0*c*np.log(1.0e-5)/(2.0*dx*npmls)
rhomax = sigmax*(imp0**2)
sig = [sigmax*((m-0.5)/(npmls+0.5))**2 for m in range(1, npmls+1)]
rho = [rhomax*(m/(npmls+0.5))**2 for m in range(1, npmls+1)]
# print(len(rho), 'after sig')

# ***********************************************************************
# Set up constants for Berenger's PML
# ***********************************************************************
re = [sig[m]*dt/eps_0 for m in range(0, npmls)]
rm = [rho[m]*dt/mu_0 for m in range(0, npmls)]
ca = [np.exp(-re[m]) for m in range(0, npmls)]
cb = [-(np.exp(-re[m])-1.0)/sig[m]/dx for m in range(0, npmls)]
da = [np.exp(-rm[m]) for m in range(0, npmls)]
db = [-(np.exp(-rm[m])-1.0)/rho[m]/dx for m in range(0, npmls)]

# ***********************************************************************
#  Initialise all matrices for the Berenger's PML
# ***********************************************************************
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Ez Fields >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                 ..... Left and Right PML Regions .....
for i in range(1, nd+1):  # =2:nd
    for j in range(3, npmls+2):  # =2:npmls+1
        m = npmls+2-j
        Ceze[i][j] = ca[m]
        Cezh[i][j] = cb[m]
    for j in range(jp, nd-1):  # =jp+1:nd
        m = j-jp
        Ceze[i][j] = ca[m]
        Cezh[i][j] = cb[m]

#                 ..... Front and Back PML Regions .....
for j in range (1, nd+1):  # =2:nd
    for i in range(3, npmls+2):  # =2:npmls+1
        m = npmls+2-i
        Ceze[i][j] = ca[m]
        Cezh[i][j] = cb[m]

    for i in range(ip, nd-1):  # =ip+1:nd
        m = i-ip
        Ceze[i][j] = ca[m]
        Cezh[i][j] = cb[m]


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Hx Fields >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                 ..... Left and Right PML Regions .....
for i in range(1, nd+1):  # =2:nd
    for j in range(2, npmls+1):  # =1:npmls
        m = npmls+1-j
        Chxh[i][j] = da[m]
        Chxe[i][j] = db[m]
    for j in range(jp, nd-1):  # =jp+1:nd
        m = j-jp
        Chxh[i][j] = da[m]
        Chxe[i][j] = db[m]

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Hy Fields >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                 ..... Front and Back PML Regions .....
for j in range(1, nd+1):  # =2:nd
    for i in range(2, npmls+1):  # =1:npmls
        m = npmls+1-i
        Chyh[i][j] = da[m]
        Chye[i][j] = db[m]
    for i in range(ip, nd-1):  # =ip+1:nd
        m = i-ip
        Chyh[i][j] = da[m]
        Chye[i][j] = db[m]

########################################

# ezInc Fn Defining the value of a sine wave generator
def ezInc(t):
    return A*np.sin(2.0*np.pi * fp * t)

# Initializing existance of wave in the room (0)
room = [[0.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
# Initializing existance of wall in the room (0)
wall = [[0.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]

# Initializing the values of Electronic and magnetic wave
Ez = [[0.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
Hx = [[0.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
Hy = [[0.0 for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]
print('{:<4}'.format('Size X , Y'), len(Hy) - 1, len(Hy[0]) - 1)

# Let us begin the Simulation
plt.ion()
for Time in range(0, MaxTime):
    # if Time < 100:  # Source wave lifetime
        # Add a source
    Ez[math.ceil(xsrc)][math.ceil(ysrc)] = ezInc(deltat * Time)
    # Update magnetic field
    for mm in range(0, SizeX + 1):
        # Periodic Boundary condition for X axis
        if mm == SizeX :
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

    # Defining wall
    # for mm in range(wx1, wx2):
    #     for nn in range(wy1, wy2):
    #         wall[mm][nn] = 2
    #         Ez[mm][nn] = 0

    # Finilize the room value = wall + Ez
    room = [[wall[mm][nn] + Ez[mm][nn] for mm in range(0, SizeX + 1)] for nn in range(0, SizeY + 1)]

    # Plotting the room
    if Time % 2 == 0:
        img = plt.imshow(room, vmax=A, vmin=-A)
        plt.colorbar(img)
        plt.pause(0.0002)
        plt.clf()
