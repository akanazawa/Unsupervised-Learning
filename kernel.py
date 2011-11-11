from numpy import *
from util import *

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

