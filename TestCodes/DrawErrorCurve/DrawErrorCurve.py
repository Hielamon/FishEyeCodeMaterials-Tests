import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np

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

if __name__ == "__main__":
    #showErrorWithNoise("Noise", "mean", "Mean Pixel Error Curves With Diff Noise Level")
    #showErrorWithNoise("Noise", "median", "Median Pixel Error Curves With Diff Noise Level")
    #showErrorWithNoise("NoiseRot", "mean", "Mean Rotation Error Curves With Diff Noise Level")
    #showErrorWithNoise("NoiseRot", "median", "Median Rotation Error Curves With Diff Noise Level")
    #showErrorWithNoise("Translate", "mean", "Mean Pixel Error Curves With Diff Translate Level")
    #showErrorWithNoise("Translate", "median", "Median Pixel Error Curves With Diff Translate Level")
    #showErrorWithNoise("TranslateRot", "mean", "Mean Rotation Error Curves With Diff Translate Level")
    #showErrorWithNoise("TranslateRot", "median", "Median Rotation Error Curves With Diff Translate Level")
    showErrorWithNum("PairsNum", "mean", "Mean Pixel Error Curves With Diff Pairs Number")
    showErrorWithNum("PairsNum", "median", "Median Pixel Error Curves With Diff Pairs Number")
    showErrorWithNum("PairsNumRot", "mean", "Mean Rotation Error Curves With Diff Pairs Number")
    showErrorWithNum("PairsNumRot", "median", "Median Rotation Error Curves With Diff Pairs Number")