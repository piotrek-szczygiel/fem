# -*- encoding: utf-8 -*-

from utils import *


class FiniteElementMethod:
    """Differential equations solver using Finite Element Method.

    Model conditions:
        (a(x)u'(x))' + b(x)u'(x) + c(x)u(x) = f(x),  x e [0, 1]
        a(0)u'(0) + beta*u(0) = gamma
        u(1) = u1
    """

    def __init__(self, a, b, c, f, beta, gamma, u1, n):
        """Initialize model with provided parameters."""

        self.a = a
        self.b = b
        self.c = c
        self.f = f
        self.beta = beta
        self.gamma = gamma
        self.u1 = u1
        self.n = n

    def e(self, i):
        """Return the basis function."""

        return lambda x: max(0, 1 - abs(x * self.n - i))

    def e_d(self, i):
        """Return the derivative of the basis function."""

        def derivative(x):
            if x < (i - 1) / self.n or x > (i + 1) / self.n:
                return 0
            elif x < i / self.n:
                return self.n
            else:
                return -self.n

        return derivative

    def lhs(self, u, v, u_d, v_d, ia, ib):
        """Return result of the equation's left-hand side - B(u, v)"""

        return (
                -self.beta * u(0) * v(0)
                + integrate(lambda x: self.a(x) * u_d(x) * v_d(x), ia, ib)
                + integrate(lambda x: self.b(x) * u_d(x) * v(x), ia, ib)
                + integrate(lambda x: self.c(x) * u(x) * v(x), ia, ib)
        )

    def lhs_cell(self, i, j):
        """Return result of the equation's lhs for provided matrix indices"""

        a = max(0., (i - 1) / self.n)
        b = min(1., (j + 1) / self.n)

        return self.lhs(
            self.e(i),
            self.e(j),
            self.e_d(i),
            self.e_d(j),
            a,
            b
        )

    def rhs(self, v, ia, ib):
        """Return result of the equation's right-hand side - l(v)"""

        return (
                -self.gamma * v(0)
                + integrate(lambda x: self.f(x) * v(x), ia, ib)
        )

    def u_shift(self, x):
        """Return result of ~u(x) function"""
        return self.u1 * self.e(self.n)(x)

    def u_star(self, x, result):
        """Return result of u*(x) function for provided FEM matrix"""
        return sum(ui * self.e(i)(x) for i, ui in enumerate(result))

    def solve(self):
        """Return approximated u(x) function"""
        left = matrix(self.n, self.n)
        for i in range(self.n):
            for j in range(self.n):
                left[i][j] = self.lhs_cell(j, i)

        right = matrix(self.n)
        for i in range(self.n):
            right[i] = (
                    self.rhs(
                        self.e(i),
                        max(0., (i - 1) / self.n),
                        min(1., (i + 1) / self.n)
                    )
                    -
                    self.lhs(
                        self.u_shift,
                        self.e(i),
                        lambda x: self.u1 * self.e_d(self.n)(x),
                        self.e_d(i),
                        max(0., (i - 1) / self.n),
                        min(1., (i + 1) / self.n)
                    )
            )

        result = solve_linear(left, right)

        return lambda x: self.u_shift(x) + self.u_star(x, result)
