__author__ = 'Sebastian Baer, s0527674'

import numpy as np
import random
from numpy import *
from decimal import *
import matplotlib.pyplot as plt

###############################################################################
#    1:Erstellen von Trainingsdaten:                                          #
#    D.h. Punkte die auf einer Geraden liegen und deren y-Werte mittels eines #
#    gaussverteilten "Rauschen" von idealen Werten abweichen.                 #
#                                                                             #
#    m: Anzahl Trainingsdaten, a: steigung, b: Schnittpunkt y-achse;          #
#    X: Trainingsdaten auf X-Achse; Y: Werte auf y-Achse inklusive Rauschen   #
###############################################################################

weiteInX = 15
schritt = 0.2
m = weiteInX / schritt
X = np.arange(0, weiteInX, schritt)
print "X: ", X

# Choose slope and point of intersection with y-Axis as reference
a = 2.
b = 5.
# reference function
Y_ideal = a * X + b
Y_gaus = Y_ideal + np.random.uniform(-5., 5., weiteInX / schritt)

print "Y_ideal: ", Y_ideal
print "Y_gaus: ", Y_gaus

###############################################################################
#   2:Hypothese fuer das lineare Modell.                                      #
#   Bestimmen der Modellparameter mittels lernen aus der Trainingsmenge       #
###############################################################################

def hypothese(theta1, x, theta0):
    return theta1 * x + theta0

###########################
#   3:Kostenfunktion      #
###########################

def costFunction(theta0, theta1):
    const = (1. / (2. * m))
    tmpCount = 0
    result = 0
    while tmpCount < m:
        result += ((hypothese(theta1, X[tmpCount], theta0) - Y_gaus[tmpCount])**2)
        tmpCount += 1
    return result * const

###############################################################################
#   4:Plotten der Kostenfunktion in der Umgebung des Minimums als             #
#     Contourplot.                                                            #
###############################################################################

def contourPlotCostMin():

    plt.subplot(1, 3, 1)

    tmptheta0 = linspace(-5., 15.)
    tmptheta1 = linspace(-2., 6.)

    # with meshgrid
    [A, B] = meshgrid(tmptheta0, tmptheta1)
    Z = costFunction(A, B)
    plt.contourf(A, B, Z, np.logspace(-2,2,30))#, 1000)

    plt.xlabel(r'$\Theta_0$')
    plt.ylabel(r'$\Theta_1$')
    plt.title("Costfunction: 'Scope Minimum'\n")

    # http://matplotlib.org/examples/pylab_examples/contour_demo.html


###########################################################
#   5:Gradientenabstiegsverfahren unter Benutzung         #
#     der Kostenfunktion und der linearen Hypothese.      #
###########################################################

def ableitungTheta0(theta0, theta1):
    tmpCount = 0
    result = 0
    while tmpCount < m:
        result += (theta0 + (theta1 * X[tmpCount]) - Y_gaus[tmpCount])
        tmpCount += 1
    return result * (1. / m)

def ableitungTheta1(theta0, theta1):
    tmpCount = 0
    result = 0
    while tmpCount < m:
        result += ((theta0 + (theta1 * X[tmpCount]) - Y_gaus[tmpCount]) * X[tmpCount])
        tmpCount += 1
    return result * (1. / m)

def theta0Neu(theta0, theta1, alpha):
    return theta0 - (alpha * ableitungTheta0(theta0, theta1))

def theta1Neu(theta0, theta1, alpha):
    return theta1 - (alpha * ableitungTheta1(theta0, theta1))

def gradientenabstieg(theta0, theta1, alpha, iterations):

    cost = list()
    cost.append(costFunction(theta0, theta1))
    c = 0
    while c < iterations:
        thetaOld = theta0
        theta0 = theta0Neu(theta0, theta1, alpha)
        print "theta0: ", theta0
        theta1 = theta1Neu(thetaOld, theta1, alpha)
        print "theta1: ", theta1
        cost.append(costFunction(theta0, theta1))
        c += 1
        print c
        print "Kosten: ", cost[c]
    return [theta0, theta1], cost


##############################################################
#   6:Trainieren des Modells und plotten der Kosten ueber    #
#     den Iterationen fuer verschiedene Werte der Lernrate.  #
##############################################################

plt.subplot(1, 3, 2)
plt.title('Kosten der Lernraten\n')
plt.xlabel('Anzahl der Iterationen')
plt.ylabel('Kosten')
iterations = 40
for i in np.arange(0.001, 0.009, 0.001):
    theta, cost = gradientenabstieg(1., 1., i, iterations)
    plt.plot(range(len(cost)), cost, label="{0:.3f}".format(i))
    plt.legend()


###################################################################
#   7:Plotten des Modells (Fit-Gerade) zusammen mit den Daten.    #
###################################################################

thetaBest, minKosten = gradientenabstieg(1., 1., 0.001, 10000)
print "thetaBest: ", thetaBest

contourPlotCostMin()
plt.subplot(1,3,3)
#plt.xlim((-10, 40))
#plt.ylim((-10, 40))
plt.xlabel("x-Axis")
plt.ylabel("y-Axis")
plt.title("Trainingsdata & Fit-Line\n")
plt.plot(X, Y_ideal, 'r-')          # Reference Line
plt.plot(X, Y_gaus, 'bo')           # Trainingsdaten
Y_fit = thetaBest[1] * X + thetaBest[0]
plt.plot(X, Y_fit, 'g-')            # Fit-Line
plt.legend(["Reference Line", "Trainingspoints", "Fit-Line"]) #- best $\Theta_0$= " + "{0:.3f}".format(thetaBest[0]) + ", best $\Theta_1$= " + "{0:.3f}".format(thetaBest[1])])

plt.show()