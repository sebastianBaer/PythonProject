__author__ = 'Sebastian Baer, s0527674'

import numpy as np
import random
from numpy import *
from decimal import *
import scipy as sc
from scipy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


###########################################################################
#                                                                         #
#       M A K E   D A T A   A N D   F E A T U R E   S C A L I N G         #
#                                                                         #
###########################################################################

Theta_start = np.array([-3., 1., 1.]).reshape(3, 1)

###########################################################
#                                                         #
#     F E A T U R E   S C A L I N G   F O R   B O T H     #
#                                                         #
###########################################################

def mittelwert(feature, anzTrainData):
    result = 0.
    for x in feature:
        result += x / anzTrainData
    return result

#Standardabweichung mit numpy
def featureScaling(feature, anzTrainData):
    return (feature - mittelwert(feature, anzTrainData)) / feature.std()


########################
#   DATA FOR LINEAR    #
########################
m = 30.
tmpx0 = np.ones((1, m))
x0 = tmpx0[0]
x1 = np.arange(-5., 10., (15. / m))
x2 = np.arange(-10., 20., (30. / m))
#x1 = np.array([1,2,3,4])
#x2 = np.array([3,8,4,7])
x_all = np.array([[x0], [x1], [x2]]).reshape(3, m)
#print "x_all: \n", x_all
#print "x_all_T: \n", x_all.T

#################################
#   REFERENZ THETAS FOR BOTH    #
#################################

Theta_neutral = np.array([2.])
Theta_breite = np.array([4.])
Theta_laenge = np.array([7.])
thetas_all = np.array([[Theta_neutral], [Theta_breite], [Theta_laenge]]).reshape((3, 1))
print "Thetas: \n", thetas_all


#########################
#   DATA FOR LOGISTIC   #
#########################

# covariance matrix and mean
cov0 = np.array([[5., -4.], [-4., 3.]])
mean0 = np.array([2., 3.])
# number of data points
m0 = 20  # 300 fuer Vorfuehrung
# generate m0 gaussian distributed data points with
# mean0 and cov0.
r0 = np.random.multivariate_normal(mean0, cov0, m0)
#print "cov0: \n", cov0
#print "r0 BLAU: \n", r0

# covariance matrix
cov1 = np.array([[5., -3.], [-3., 3.]])
mean1 = np.array([1., 1.])
m1 = 20
r1 = np.random.multivariate_normal(mean1, cov1, m1)
#print "r1 ROT: \n", r1

m_log = m1 * 2.

#############################
#   FEATURES FOR LOGISTIC   #
#############################

tmpfeature_x0 = np.ones((1., m_log))
x0_log = tmpfeature_x0[0]
x1_log = np.append(r0[:, 0], r1[:, 0])
x2_log = np.append(r0[:, 1], r1[:, 1])
x_all_log = np.array([[x0_log], [x1_log], [x2_log]]).reshape(3, m_log)
#x_all_log = np.array([[x0_log], [featureScaling(x1_log, m_log)], [featureScaling(x2_log, m_log)]]).reshape(3, m_log)
#print "x_all_log: ", x_all_log

#plt.scatter(r0[...][..., 0], r0[..., 1], c='b', marker='o')
#plt.scatter(r1[..., 0], r1[..., 1], c='r', marker='x')

test = np.arange(-10., 10., 1.)
def logFunction(z):
    return 1 / (1 + exp(-z))

#plt.plot(test, logFunction(test), c='b')


#####################################################
#                                                   #
#                H Y P O T H E S I S                #
#                                                   #
#####################################################

##############
#   LINEAR   #
##############

def hypotheseMulti(thetas):
    return (x_all * thetas).sum(axis=0)

################
#   LOGISTIC   #
################

def hypotheseLog(thetas):
    #Y-Vektor
    return 1. / (1. + exp(-(thetas * x_all_log).sum(axis=0)))

def hlpHypo(featureVektor, thetaVektor):
    #ein Y-Wert
    return 1. / (1. + exp(-(thetaVektor * featureVektor).sum(axis=1)))
    #return 1. / (1. + exp(-np.dot(featureVektor, thetaVektor)))


###################################################
#                                                 #
#                   M A K E   Y                   #
#                                                 #
###################################################

##############
#   LINEAR   #
##############

Y_Ideal = hypotheseMulti(thetas_all)
Y_Gaus = Y_Ideal + np.random.uniform(-5., 5., m)
#print "Y_Ideal: \n", Y_Ideal
#print "Y_Gaus: \n", Y_Gaus

################
#   LOGISTIC   #
################

Y_log = hypotheseLog(thetas_all)
print "Y_ideal_log: \n", Y_log
#plt.scatter(x1_log, Y_log, c='k', marker='x')
#plt.scatter(x2_log, Y_log, c='g', marker='x')

#plt.show()

#########################################################
#                                                       #
#               C O S T   F U N C T I O N               #
#                                                       #
#########################################################

##############
#   LINEAR   #
##############

def costFunctionMulti(thetas):
    const = (1. / (2. * m))
    tmpCount = 1
    result = 0
    while tmpCount < m:
        result += (((x_all.T[tmpCount] * thetas.T).sum(axis=1) - Y_Gaus[tmpCount]) ** 2)
        tmpCount += 1
    return const * result
#print "kostenfunktion: \n", costFunctionMulti(thetas_all)

###############
#   LOGISTC   #
###############

def costFunctionLog(thetas):
    const = -(1. / m_log)
    c = 1
    result = 0.
    while c < m_log:
        result += ((Y_log[c] * log(hlpHypo(x_all_log.T[c], thetas.T))) + ((1. - Y_log[c]) * log(1. - hlpHypo(x_all_log.T[c], thetas.T))))
        c += 1
    return const * result
#print "kostenfunktionLog: \n", costFunctionLog(thetas_all)

###############################################################
#                                                             #
#               U P D A T E   F U N C T I O N S               #
#                                                             #
###############################################################

##############
#   LINEAR   #
##############

def updateThetas(thetas, alpha):
    thetasOld = thetas
    const = (alpha / m)
    countFeature = 0
    while countFeature < 3:
        result = 0.
        countTraining = 1
        while countTraining < m:
            result += (((x_all.T[countTraining] * thetas.T).sum(axis=1) - Y_Gaus[countTraining]) * x_all[countFeature, countTraining])
            countTraining += 1
        thetasOld[countFeature] = thetasOld[countFeature] - (result * const)
        countFeature += 1
    return thetasOld
#print "updateThetas: \n", updateThetas(thetas_all, 0.1)

################
#   LOGISTIC   #
################

def updateThetasLog(thetas, alpha):
    thetasOld = thetas
    const = (alpha / m_log)
    countFeature = 0
    while countFeature < 3:
        result = 0.
        countTraining = 1
        while countTraining < m_log:
            result += (hlpHypo(x_all_log.T[countTraining], thetas.T) - Y_log[countTraining]) * x_all_log[countFeature, countTraining]
            countTraining += 1
        #print "result: \n", result
        thetasOld[countFeature] = thetasOld[countFeature] - (alpha * result)

        countFeature += 1
    return thetasOld
#print "updateThetasLog: \n", updateThetasLog(thetas_all, 0.001)


###########################################################################
#                                                                         #
#          G R A D I E N T E N A B S T I E G S V E R F A H R E N          #
#                                                                         #
###########################################################################


def gradientenabstieg(thetas, alpha, iterations, switch):
    cost = list()
    if switch == 0:
        cost.append(costFunctionMulti(thetas))
    else:
        cost.append(costFunctionLog(thetas))
    c = 0
    while c < iterations:
        if switch == 0:
            thetas = updateThetas(thetas, alpha)
            print "thetas: ", thetas
            cost.append(costFunctionMulti(thetas))
        else:
            thetas = updateThetasLog(thetas, alpha)
            print "thetas: ", thetas
            cost.append(costFunctionLog(thetas))
        print c
        print "Kosten: ", cost[c]
        c += 1
    return thetas, cost

#############################################################################
#                                                                           #
#            P L O T   C O S T S   O V E R   I T E R A T I O N S            #
#                                                                           #
#############################################################################

fig = plt.figure()
def costOverIterations():
    fig.add_subplot(121)
    plt.title('Kosten der Lernraten\n')
    plt.xlabel('Anzahl der Iterationen')
    plt.ylabel('Kosten')
    ite = 40
    for i in np.arange(0.01, 0.018, 0.001):
        Theta_start = np.array([1., 1., 1.]).reshape(3, 1)
        theta, cost = gradientenabstieg(Theta_start, i, ite, switch)
        plt.plot(range(len(cost)), cost, label="{0:.3f}".format(i))
        plt.legend()

#################################################################
#                                                               #
#                  T R A I N   A   M O D E L                    #
#                                                               #
#################################################################

alpha = 0.001
iterations = 1000
switch = 0  # 0 for linear, 1 for logistic
thetaBest, minKosten = gradientenabstieg(Theta_start, alpha, iterations, switch)
thetaBest.reshape(1, 3)
print "thetaBest: ", thetaBest
costOverIterations()


#############################################################
#                                                           #
#               P L O T   T H E   R E S U L T               #
#                                                           #
#############################################################

def funLinear(x, y):
    return thetas_all[0] + (thetas_all[1] * x) + (thetas_all[2] * y)

def funLog(x, y, thetas):
    return 1 / (1 + exp(-(thetas[0] + np.dot(thetas[1], x) + np.dot(thetas[2], y))))


if switch == 0:
    ax = fig.add_subplot(122, projection='3d')
    x1tmp = x1 + np.random.uniform(-5., 5., m)
    x2tmp = x2 + np.random.uniform(-5., 5., m)
    ax.scatter(x1tmp, x2, Y_Gaus, marker="o",  c="r")
    ax.scatter(x1, x2tmp, Y_Gaus, marker="o",  c="r")
    X, Y = np.meshgrid(x1, x2)
    zs = np.array([funLinear(x1,x2) for x1,x2 in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cstride=5, rstride=5, cmap=plt.cm.hot, alpha=0.4)
    plt.title('Multilinear Regression\n')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
else:
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(x1_log, x2_log, 0.5, marker='x',  c='b')

    ###########
    #  N E W  #
    ###########

    Xs = np.linspace(-10., 10., 40)
    print Xs
    print
    Ys = np.linspace(-10., 10., 40)
    X, Y = np.meshgrid(Xs, Ys)
    #Z = funLog(X,Y)
    zs = np.array([funLog(Xs, Ys, thetaBest) for Xs, Ys in zip(np.ravel(X), np.ravel(Y))])
    #zs = np.array([hlpHypo(x_all_log, thetaBest) for Xs, Ys in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cstride=20, rstride=20, cmap=plt.cm.jet, alpha=0.4)

    plt.title('Logistic Regression\n')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    print "logistic"
