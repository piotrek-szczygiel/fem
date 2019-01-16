# -*- encoding: utf-8 -*-

from fem import FiniteElementMethod
from utils import *

if __name__ == '__main__':
    fem = FiniteElementMethod(
        a=lambda x: 0,
        b=lambda x: 1,
        c=lambda x: 0,
        f=lambda x: 3 * sin(x * 10) * x,
        beta=0,
        gamma=0,
        u1=1 / 2,
        n=100
    )

    u = fem.solve()

    xs = generate_xs(0, 1, 50)
    ys = [u(x) for x in xs]

    show_graph(xs, ys)
