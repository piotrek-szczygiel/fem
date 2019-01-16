# -*- encoding: utf-8 -*-

import sys
from typing import Callable

from fem import FiniteElementMethod
from utils import *

if __name__ == '__main__':
    a: Callable
    b: Callable
    c: Callable
    f: Callable
    beta: float
    gamma: float
    u1: float
    n: int

    if len(sys.argv) == 2 and sys.argv[1] == 'default':
        a = lambda x: 0
        b = lambda x: 1
        c = lambda x: 0
        f = lambda x: 3 * sin(x * 10) * x
        beta = 0
        gamma = 0
        u1 = 1 / 2
        n = 100
    else:
        exec('def a(x): return ' + input('a(x) = '))
        exec('def b(x): return ' + input('b(x) = '))
        exec('def c(x): return ' + input('c(x) = '))
        exec('def f(x): return ' + input('f(x) = '))
        exec('beta = ' + input('beta = '))
        exec('gamma = ' + input('gamma = '))
        exec('u1 = ' + input('u1 = '))
        exec('n = ' + input('n = '))

    fem = FiniteElementMethod(a, b, c, f, beta, gamma, u1, n)

    u = fem.solve()

    xs = generate_xs(0, 1, 50)
    ys = [u(x) for x in xs]

    show_graph(xs, ys)
