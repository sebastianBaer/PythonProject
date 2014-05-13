__author__ = 'Sebastian Baer, s0527674'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============== CONSTANTS ==================
ALPHA = 0.0001
NUM_TRAIN_DATA = 6
NUM_VAL_DATA = 2  # 20 percent
LAMBDA_ = 1.0
VECTOR_DIMENSION = NUM_TRAIN_DATA - 1
NUM_ITERATIONS = 1
NUM_ITERATIONS_UPDATE = 200000

# =============== HYPOTHESIS FUNCTIONS ==================
def callback_mvrHypo(theta):
    hypo = X.dot(theta)
    return hypo


def callback_regHypo(theta):
    hypo = polyXtrain.dot(theta)
    return hypo

# =============== TESTVALUES ==================
# targetfunction:
def t(x):
    return -np.sin(x)


X = np.linspace(0, np.pi, NUM_TRAIN_DATA + NUM_VAL_DATA)

xTrain = []
xTrain.append(X[0])
xTrain.append(X[1])
xTrain.append(X[2])
xTrain.append(X[4])
xTrain.append(X[5])
xTrain.append(X[6])
xTrain = np.array(xTrain)

xVal = []
xVal.append(X[3])
xVal.append(X[7])
xVal = np.array(xVal)

polyXtrain = []
polyXval = []

# create Y with gaussian noise
yTrain = np.array([])
yVal = np.array([])

# Erzeugung des polyXval UND yVal + noise
for x in xVal:
    polyXval.append([])
    for j in range(NUM_TRAIN_DATA):
        polyXval[len(polyXval) - 1].append(np.power(x, j))  #Basis: x; Exponent: j;
    yVal = np.append(yVal, t(x) + np.random.uniform(-.2, .2))

# Erzeugung des polyXtrain UND yTrain + noise
for x in xTrain:
    polyXtrain.append([])
    for j in range(NUM_TRAIN_DATA):
        polyXtrain[len(polyXtrain) - 1].append(np.power(x, j))
    yTrain = np.append(yTrain, t(x) + np.random.uniform(-.2,.2))
#
#for i in range(len(X)):
#    if (i % (NUM_TRAIN_DATA - NUM_VAL_DATA) == 0):
#        polyXval.append([])
#        for j in range(NUM_TRAIN_DATA):
#            polyXval[len(polyXval) - 1].append(np.power(X[i], j))
#        yVal = np.append(yVal, t(X[i]))
#    else:
#        polyXtrain.append([])
#        for j in range(NUM_TRAIN_DATA):
#            polyXtrain[len(polyXtrain) - 1].append(np.power(X[i], j))
#        yTrain = np.append(yTrain, t(X[i]))

polyXtrain = np.array(polyXtrain)
polyXval = np.array(polyXval)

print("polyx", polyXtrain)
print("val:", polyXval)

print("valy", yVal)
print("trainy", yTrain)


#for x in X:
#Y=np.append(Y, t(x)+np.random.uniform(-.5,+.5))
# Y=np.append(Y, t(x))

# =============== multivariate ==================
# calculate the mean of given featurelist
def getMeanOfFeature(features):
    shape = np.shape(features)
    if (shape == (len(features),)):
        means = 0.0
        for i in range(len(features)):
            means += features[i]
        means = means / len(features)
    else:
        means = []
        for i in range(len(features[0])):
            mean = 0.0
            for x in features:
                mean += x[i]
            means.append(mean / len(features))
    return means

# scale features
def scaleFeatures(unscaled, means, stdDeviation):
    shape = np.shape(unscaled)
    if (shape == (len(unscaled),)):
        scaled = []
        for x in unscaled:
            scaled.append((x - means) / stdDeviation)
    else:
        scaled = [[]]
        for x in unscaled:
            tmp = []
            for i in range(len(x)):
                if (i > 0):
                    tmp.append((x[i] - means[i]) / stdDeviation[i])
                else:
                    tmp.append(x[i])
            scaled.append(tmp)
            # remove first empty entry
        scaled.remove([])
    return scaled

# descale features
def descaleFeatures(scaled, means, stdDeviation):
    shape = np.shape(scaled)
    if (shape == (len(scaled),)):
        descaled = []
        for x in scaled:
            descaled.append(x * stdDeviation + means)
    else:
        descaled = [[]]
        for x in scaled:
            tmp = []
            for i in range(len(x)):
                if (i > 0):
                    tmp.append(x[i] * stdDeviation[i] + means[i])
                else:
                    tmp.append(x[i])
            descaled.append(tmp)
            # remove first empty entry
        descaled.remove([])
    return descaled


def update(Y, hypo, theta, lambd):
    m = len(polyXtrain)
    t = []
    for j in range(len(theta)):
        if (j == 0):
            t.append(theta[0] - ALPHA * (1.0 / m) * np.sum(hypo - Y))
        else:
            tj = 0.0
            sum1 = 0.0
            for i in range(1, NUM_TRAIN_DATA):
                sum1 += (hypo[i] - Y[i]) * polyXtrain[i][j]
            tj = theta[j] * (1 - ALPHA * (lambd / m)) - (ALPHA / m) * sum1
            t.append(tj)
    return np.array(t)


def gradientDescentScaled(lambd):
    costs = []
    #    meanX = getMeanOfFeature(polyX)
    #    stdX = np.std(polyX, axis=0)
    #    scldX = scaleFeatures(polyX, meanX, stdX)
    #    scldX = np.array(scldX)
    theta = np.array([1 for x in polyXtrain[0]])
    hypo = polyXtrain.dot(theta)
    #theta = update(yTrain, hypo, theta, lambd)
    #hypo = polyXtrain.dot(theta)
    for i in range(0, NUM_ITERATIONS_UPDATE):
        theta = update(yTrain, hypo, theta, lambd)
        hypo = polyXtrain.dot(theta)
    print("thetas: ", theta)
    return theta

#hier vergleicht man den Abstand zwischen y-Val verrauscht und den y-Werten unseres Modells
#alles mit den Val Daten
def eVal(theta):
    const = 1. / (2. *len(xVal))
    return const * ((polyXval.dot(theta) - yVal)**2).sum()

def printValuesAndModels():
    plt.title("target function, validation data and models")
    plt.plot(xTrain, yTrain, "bo")
    plt.plot(xVal, yVal, "ro")
    x = np.linspace(-0, 3.5, 100)

    poly = []
    for i in range(len(x)):
        poly.append([])
        for j in range(NUM_TRAIN_DATA):
            poly[i].append(np.power(x[i], j))
    poly = np.array(poly)
    #print(poly)
    #    meanX = getMeanOfFeature(poly)
    #    stdX = np.std(poly, axis=0)
    #    scldX = scaleFeatures(poly, meanX, stdX)
    #    scldX = np.array(scldX)

    lambdas = np.array([1.,0.1,0.01,0.001,0.0001,0.])
    y = []
    eOut = []
	
    for l in lambdas:
        theta = gradientDescentScaled(l)
        print "theta: ", theta
        print "lambda: ", l
        eOut.append(eVal(theta))
        y = poly.dot(theta)
        plt.plot(x, y, label="lambda = "+str(l))

    min = eOut[0]
    index = 0
    for i in range(len(eOut)):
        if (eOut[i] < min):
            min = eOut[i]
            index = i
    print "BEST Lambda: ", lambdas[index]
    plt.plot(x, t(x), label="-sin")
    plt.legend()
    plt.show()


# =============== run config ==================
printValuesAndModels()
#printCostPerIterationMulti(callback_mvrHypo, callback_mvrCost)
#ALPHA= 0.001
#print3d(gradientDescentScaled(False, callback_mvrHypo, callback_mvrCost))
