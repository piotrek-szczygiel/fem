# -*- encoding: utf-8 -*-

import matplotlib.pyplot
import numpy
import scipy.integrate

sin = numpy.sin
cos = numpy.cos
log = numpy.log
exp = numpy.exp


def integrate(f, a, b):
    return scipy.integrate.quad(f, a, b)[0]


def matrix(*dimensions):
    return numpy.zeros(dimensions)


def solve_linear(left, right):
    return numpy.linalg.solve(left, right)


def show_graph(xs, ys):
    matplotlib.pyplot.plot(xs, ys, 'g', linewidth=3)
    matplotlib.pyplot.show()


def generate_xs(a, b, n):
    return numpy.linspace(a, b, n)
