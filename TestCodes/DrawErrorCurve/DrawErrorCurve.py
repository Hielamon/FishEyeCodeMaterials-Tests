import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np

import sys, getopt

def scaleFigure(fig, scale):
    originSize = fig.get_size_inches()
    originSize *= scale
    fig.set_size_inches(originSize)

def showErrorDetail():
    generalModelName = ["PolynomialAngle","PolynomialRadius","GeyerModel"]
    curveStyle = ["r-", "b-", "g-"]
    errorPath = []
    fig = plt.figure()
    idx = 0
    for name in generalModelName:
        nameTmp = "../OptimizeMetric/" + name + "_error.txt"
        errorPath.append(nameTmp)
        file = open(nameTmp)
        iters = []
        errors = []
        for line in file.readlines():
            line = line.replace("\n", "").split(" ")
            iters.append(int(line[0]))
            errors.append(float(line[1]))
        plt.plot(iters, errors, curveStyle[idx], label = name)

        idx += 1

    plt.legend()
    plt.grid()
    plt.title("The Error Curve of Diff General FishEye Model")
    plt.xlabel(r"iter")
    plt.ylabel(r"error")
    plt.show()

def showErrorWithNoise(typeError, type, title):
    generalModelName = ["PolynomialAngle","PolynomialRadius","GeyerModel"]
    curveStyle = ["r-", "b-", "g-"]
    markerStyle = ["o", "s", "D"]
    errorPath = []
    fig = plt.figure()
    idx = 0
    max_error = 0
    min_error = 10
    for name in generalModelName:
        nameTmp = "../OptimizeMetric/" + name + "_" + typeError + "_" + type + "Errors.txt"
        errorPath.append(nameTmp)
        file = open(nameTmp)
        pairsNums = []
        errors = []
        for line in file.readlines():
            line = line.replace("\n", "").split(" ")
            pairsNums.append(float(line[0]))
            line1 = line[1]
            if(line1 != "-nan(ind)"):
                errTmp = float(line1)
            if errTmp > max_error:
                max_error = errTmp
            if errTmp < min_error:
                min_error = errTmp
            errors.append(errTmp)
        plt.plot(pairsNums, errors, curveStyle[idx], label = name, marker = markerStyle[idx])

        idx += 1

    plt.legend()
    plt.grid()
    plt.title(title)
    #plt.xlim(0.5, 9.5)
    plt.ylim(max(min_error - 0.1, 0), max_error + 0.1)
    plt.xlabel(r"$Pairs Number$")
    plt.ylabel(r"error")
    ax = plt.gca()
    ax.xaxis.set_major_locator( MultipleLocator(pairsNums[1] - pairsNums[0]) )

    plt.show()

def showErrorWithNum(typeError, type, title):
    generalModelName = ["PolynomialAngle","PolynomialRadius","GeyerModel"]
    curveStyle = ["r-", "b-", "g-"]
    markerStyle = ["o", "s", "D"]
    errorPath = []
    fig = plt.figure()
    scaleFigure(fig, 1.5)
    idx = 0
    max_error = 0
    min_error = 10
    for name in generalModelName:
        nameTmp = "../OptimizeMetric/" + name + "_" + typeError + "_" + type + "Errors.txt"
        errorPath.append(nameTmp)
        file = open(nameTmp)
        pairsNums = []
        errors = []
        for line in file.readlines():
            line = line.replace("\n", "").split(" ")
            num = float(line[0])
            pairsNums.append(num)
            line1 = line[1]
            if(line1 != "-nan(ind)"):
                errTmp = float(line1)
                errTmp /= num
            
            if errTmp > max_error:
                max_error = errTmp
            if errTmp < min_error:
                min_error = errTmp
            errors.append(errTmp)
        plt.plot(pairsNums, errors, curveStyle[idx], label = name, marker = markerStyle[idx])

        idx += 1

    plt.legend()
    plt.grid()
    plt.title(title)
    #plt.xlim(0.5, 9.5)
    #plt.ylim(max(min_error - 0.1, 0), max_error + 0.1)
    plt.xlabel(r"$Pairs\ Number$")
    plt.ylabel(r"error")
    ax = plt.gca()
    ax.xaxis.set_major_locator( MultipleLocator(pairsNums[1] - pairsNums[0]) )

    plt.show()

def showCurves(fNames, clabels, cStyles, mStyles, xlabel, xMLocator, ylabel, yMLocator, title, save):
    fig = plt.figure()
    scaleFigure(fig, 1.5)
    idx = 0
    for (fName, clabel, cStyle, mStyle) in zip(fNames, clabels, cStyles, mStyles):
        file = open(fName)
        x = []
        y = []
        lineIdx = 0
        for line in file.readlines():
            line = line.replace("\n", "").split(" ")
            xf = float(line[0])
            x.append(xf)

            line1 = line[1]
            if(line1 != "-nan(ind)"):
                yf = float(line1)
            else:
                print("the yValue is -nan(ind) in %dth line %s file" % lineIdx, fName)

            y.append(yf)

            lineIdx += 1
        plt.plot(x, y, cStyle, label = clabel, marker = mStyle)

    plt.legend()
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(x) >= 1 and xMLocator > 0.0:
        ax = plt.gca()
        ax.xaxis.set_major_locator( MultipleLocator(xMLocator))

    if len(y) >= 1 and yMLocator > 0.0:
        ax = plt.gca()
        ax.yaxis.set_major_locator( MultipleLocator(yMLocator))

    if save:
        fig.savefig(title + ".png")

    plt.show()
 
def parseTest(argv):
    helpStr = 'DrawErrorCurve.py -f <file1 file2 ...> -l <clabel1 clabel2 ...> \
    -s <cStyle1 cStyle2 ...> -m <m1 m2 ...> -x <xlabel> --xM <xMLocator> \
    --yM <yMLocator> -y <ylabel> -t <title> --save <save>'

    fNames = []
    clabels = []
    cStyles = []
    mStyles = []
    xlabel = r'x'
    xMLocator = 0.0
    ylabel = r'y'
    yMLocator = 0.0
    title = 'curve'
    save = False

    try:
        opts, args = getopt.getopt(argv, "hf:l:s:m:x:y:t:", ["xM=", "yM=", "save="])
    except getopt.GetoptError:
        print(helpStr)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(helpStr)
            sys.exit()
        elif opt in ("-f"):
            fNames = arg.split(' ')
        elif opt in ("-l"):
            clabels = arg.split(' ')
        elif opt in ("-s"):
            cStyles = arg.split(' ')
        elif opt in ("-m"):
            mStyles = arg.split(' ')
        elif opt in ("-x"):
            xlabel = arg
        elif opt in ("--xM"):
            xMLocator = float(arg)
        elif opt in ("-y"):
            ylabel = arg
        elif opt in ("--yM"):
            yMLocator = float(arg)
        elif opt in ("-t"):
            title = arg
        elif opt in ("--save"):
            if arg == 'yes':
                save = True

    return fNames, clabels, cStyles, mStyles, xlabel, xMLocator, ylabel, yMLocator, title, save

if __name__ == "__main__":
    fNames, clabels, cStyles, mStyles, xlabel, xMLocator, ylabel, yMLocator, title, save = parseTest(sys.argv[1:])
    showCurves(fNames, clabels, cStyles, mStyles, xlabel, xMLocator, ylabel, yMLocator, title, save)