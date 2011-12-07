from numpy import *
from util import *

import pdb

def linear(x,z):
    return dot(x,z)

def poly2(x,z):
    return (1 + dot(x,z)) ** 2

def poly3(x,z):
    return (1 + dot(x,z)) ** 3

def rbf0_2(x,z):
    return exp(-0.2 * dot(x-z, x-z))

def rbf0_5(x,z):
    return exp(-0.5 * dot(x-z, x-z))

def rbf1(x,z):
    return exp(-1 * dot(x-z, x-z))

def rbf2(x,z):
    return exp(-2 * dot(x-z, x-z))

def rbf5(x,z):
    return exp(-5 * dot(x-z, x-z))

def combo(x,z):
    A = rbf0_2(x,z)
    B = rbf0_5(x,z)
    return A * B

def ours(x,z):
    #return 1.0 / (1.0 + exp(-1*((exp(-0.25 * dot(x-z, x-z))-7.5)*100)))
    return 1000.0 / (1.0 + exp(-((sqrt(dot(x, x))-2.75)*1000)) + exp(-((sqrt(dot(z, z))-2.75)*1000)))
