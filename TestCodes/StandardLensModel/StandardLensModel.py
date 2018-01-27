import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np

def pi_formatter(x, pos):
    m = np.round(x / (np.pi/16))
    n = 16
    for i in range(0, 4):
        if m%2==0: 
            m, n = m/2, n/2
        else:
            break

    if m == 0:
        return "0"
    if m == 1 and n == 1:
        return "$\pi$"
    if n == 1:
        return r"$%d \pi$" % m
    if m == 1:
        return r"$\frac{\pi}{%d}$" % n
    return r"$\frac{%d \pi}{%d}$" % (m,n)


def Equidistant(f, theta):
    return f*theta

def Equisolid(f, theta):
    return 2*f*np.sin(theta * 0.5)

def Orthographic(f, theta):
    return f*np.sin(theta)

def Stereographic(f, theta):
    return 2*f*np.tan(theta * 0.5)

if __name__ == "__main__":
    plt.figure()
    thetaArray = np.arange(0, np.pi * 0.5, 0.005)
    rEdist = []
    rEsolid = []
    rOrtho = []
    rStereo = []
    f = 1
    for i in thetaArray:
        rEdist.append(Equidistant(f, i))
        rEsolid.append(Equisolid(f, i))
        rOrtho.append(Orthographic(f, i))
        rStereo.append(Stereographic(f, i))
    plt.plot(thetaArray, rEdist, 'r-', label  = r"$Equidistant \ \ \ \ \ : r_d = f* \theta$")
    plt.plot(thetaArray, rEsolid, 'g-', label = r"$Equisolid \ \ \ \ \ \ \ \ : r_d = 2f * sin(\frac{\theta}{2})$")
    plt.plot(thetaArray, rOrtho, 'b-', label  = r"$Orthographic \ : r_d = f * sin(\theta$)")
    plt.plot(thetaArray, rStereo, 'y-', label = r"$Stereographic: r_d = 2f * tan(\frac{\theta}{2})$")
    plt.xlim(0, np.pi / 2)
    plt.ylim(0, np.max(rStereo))
    plt.legend()
    plt.grid()
    plt.title("The Classic FishEye Camera Models($f=1$)")
    plt.xlabel(r"$\theta (rad)$")
    plt.ylabel(r"$r_d$")

    ax = plt.gca()

    ax.xaxis.set_major_locator( MultipleLocator(np.pi/16) )
    ax.xaxis.set_major_formatter( FuncFormatter( pi_formatter ) )
    ax.xaxis.set_minor_locator( MultipleLocator(np.pi/80) )

    ax.yaxis.set_minor_locator( MultipleLocator(0.05))

    plt.show()

