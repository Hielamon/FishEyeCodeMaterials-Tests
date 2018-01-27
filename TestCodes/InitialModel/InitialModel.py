import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np

def scaleFigure(fig, scale):
    originSize = fig.get_size_inches()
    originSize *= scale
    fig.set_size_inches(originSize)

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

endTheta = np.pi * (180.0 / 360.0)
baseModel = Equidistant
baseF = 1

def GetInitData(thetaArr, ClassicFishEyeFuc):
    return ClassicFishEyeFuc(baseF, thetaArr)

def PolynomialAngle(k1, k2, theta):
    return k1*theta + k2*np.power(theta, 3)

def PolynomialRadius(a0, a2, theta):
    a = a2*np.sin(theta)
    b = -np.cos(theta)
    c = a0*np.sin(theta)
    b24ac = b*b - 4*a*c
    isArr = isinstance(theta, np.ndarray)

    if (isArr and (b24ac < 0).any()) or (not isArr and b24ac < 0):
       print("Error Parameter in PolynomialRadius")
       return False

    r1 = (-b + np.sqrt(b24ac))/(2*a)
    r2 = (-b - np.sqrt(b24ac))/(2*a)
    rBase = baseModel(baseF, theta)

    result = r1
    
    if not isArr and (np.abs(rBase - r1) > np.abs(rBase -r2)):
        result = r2

    if isArr:
        for i in range(0, len(r1)):
            if np.abs(rBase[i] - r1[i]) > np.abs(rBase[i] -r2[i]):
                result[i] = r2[i]

    return result

def GeyerModel(ksai, m, theta):
    return (ksai + m)*np.sin(theta)/(np.cos(theta) + ksai)
   
def GetLeastSquareData(thetaArr, radii, type):
    thetaArr.resize(len(thetaArr), 1)
    radii.resize(len(radii), 1)
    
    if type == "PolynomialAngle":
        return radii, np.concatenate([thetaArr, np.power(thetaArr, 3)], axis = 1)
    elif type == "PolynomialRadius":
        return radii*np.cos(thetaArr), np.concatenate([np.sin(thetaArr), np.sin(thetaArr) * np.power(radii, 2)], axis = 1)
    elif type == "GeyerModel":
        return radii*np.cos(thetaArr), np.concatenate([np.sin(thetaArr) - radii, np.sin(thetaArr)], axis = 1)

def LeastSquare(A, b):
    A = np.matrix(A)
    b = np.matrix(b)
    At = A.T
    result = ((At * A).I)*At*b
    result = np.array(result)
    result.resize(result.shape[1], result.shape[0])
    return result[0]

if __name__ == "__main__":
    #print(np.arctan2(1.0, 0.0));
    #print(GetInitData(np.pi * 0.5, Orthographic)) 
    fig = plt.figure()
    scaleFigure(fig, 1.28)
    
    thetaArray = np.arange(0.005, endTheta, 0.005)
    
    if baseModel == Equidistant:
        radii = GetInitData(thetaArray, Equidistant)
        plt.plot(thetaArray, radii, 'r-', label  = r"$Equidistant \ \ \ \ \ : r_d = f* \theta$" + r"    $(f = " + str(baseF) + ")$")
    elif baseModel == Equisolid:
        radii = GetInitData(thetaArray, Equisolid)
        plt.plot(thetaArray, radii, 'g-', label = r"$Equisolid \ \ \ \ \ \ \ \ : r_d = 2f * sin(\frac{\theta}{2})$" + r"    $(f = " + str(baseF) + ")$")
    elif baseModel == Orthographic:
        radii = GetInitData(thetaArray, Orthographic)
        plt.plot(thetaArray, radii, 'b-', label  = r"$Orthographic \ : r_d = f * sin(\theta$)" + r"    $(f = " + str(baseF) + ")$")
    elif baseModel == Stereographic:
        radii = GetInitData(thetaArray, Stereographic)
        plt.plot(thetaArray, radii, 'y-', label = r"$Stereographic: r_d = 2f * tan(\frac{\theta}{2})$" + r"    $(f = " + str(baseF) + ")$")

    b, A = GetLeastSquareData(thetaArray, radii, PolynomialAngle.__name__)
    k1, k2 = LeastSquare(A, b)
    radiiPA = PolynomialAngle(k1, k2, thetaArray)
    paraStr = r"    $(k_1 = " + ("%0.2f" % k1) + ",k_2 = " + ("%0.2f" % k2) + ")$"
    plt.plot(thetaArray, radiiPA, 'm-', label = r"$PolynomialAngle: r_d = k_1 * \theta + k_2 * \theta^3$" + paraStr)
    print("k1 = %f, k2 = %f" % (k1, k2))

    b, A = GetLeastSquareData(thetaArray, radii, PolynomialRadius.__name__)
    a0, a2 = LeastSquare(A, b)
    radiiPR = PolynomialRadius(a0, a2, thetaArray)
    paraStr = r"    $(a_1 = " + ("%0.2f" % a0) + ",a_2 = " + ("%0.2f" % a2) + ")$"
    plt.plot(thetaArray, radiiPR, color = "#696969", linestyle = "-", label = r"$PolynomialRadius: \frac{r_d}{a_0+a_2*r_d^2} = \frac{sin(\theta)}{cos(\theta)}$" + paraStr)
    print("a0 = %f, a2 = %f" % (a0, a2))

    b, A = GetLeastSquareData(thetaArray, radii, GeyerModel.__name__)
    ksai, m = LeastSquare(A, b)
    radiiGM = GeyerModel(ksai, m, thetaArray)
    paraStr = r"    $(\xi = " + ("%0.2f" % ksai) + ",m = " + ("%0.2f" % m) + ")$"
    plt.plot(thetaArray, radiiGM, 'c-', label = r"$GeyerModel: r_d = \frac{(\xi + m)*sin(\theta)}{cos(\theta) + \xi}$" + paraStr)
    print("ksai = %f, m = %f" % (ksai, m))

    plt.xlim(0, endTheta)
    plt.ylim(0, 2*baseF*np.tan(endTheta * 0.5))
    plt.legend()
    plt.grid()
    plt.title("Initialization Based on " + baseModel.__name__ + " Model")
    plt.xlabel(r"$\theta (rad)$")
    plt.ylabel(r"$r_d$")

    ax = plt.gca()

    ax.xaxis.set_major_locator( MultipleLocator(np.pi/16) )
    ax.xaxis.set_major_formatter( FuncFormatter( pi_formatter ) )
    ax.xaxis.set_minor_locator( MultipleLocator(np.pi/80) )

    ax.yaxis.set_minor_locator( MultipleLocator(0.05))

    plt.show()

