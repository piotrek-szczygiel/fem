# -*- encoding: utf-8 -*-

import matplotlib.pyplot
import numpy
import scipy.integrate

sqrt = numpy.sqrt
cos = numpy.cos
sin = numpy.sin
log = numpy.log
exp = numpy.exp


def integrate(f, l1, l2):
    return scipy.integrate.quad(f, l1, l2)[0]


def matrix(*dimensions):
    return numpy.zeros(dimensions)


def solve_linear(left, right):
    return numpy.linalg.solve(left, right)


def graph(xs, ys):
    matplotlib.pyplot.title('y = u(x)')
    matplotlib.pyplot.plot(xs, ys, 'k', linewidth=3)
    matplotlib.pyplot.show()


def generate_xs(a, b, n):
    return numpy.linspace(a, b, n)
