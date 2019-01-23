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

    def solve(self):
        """Return approximated u(x) function"""

        left = matrix(self.n, self.n)
        for j in range(self.n):
            for i in range(self.n):
                left[j][i] = self.lhs_cell(i, j)

        right = matrix(self.n)
        for k in range(self.n):
            right[k] = (
                    self.rhs(
                        self.e(k),
                        self.e_triangle_left(k),
                        self.e_triangle_right(k)
                    )
                    -
                    self.lhs(
                        self.u_shift,
                        self.e(k),
                        lambda x: self.u1 * self.e_d(self.n)(x),
                        self.e_d(k),
                        self.e_triangle_left(k),
                        self.e_triangle_right(k)
                    )
            )

        result = solve_linear(left, right)

        return lambda x: self.u_shift(x) + self.u_star(x, result)

    def e(self, k):
        """Return the basis function."""

        return lambda x: max(0, 1 - abs(self.n * x - k))

    def e_d(self, k):
        """Return the derivative of the basis function."""

        def derivative(x):
            if self.e(k)(x) == 0:
                return 0
            elif x < k / self.n:
                return self.n
            else:
                return -self.n

        return derivative

    def e_triangle_left(self, k):
        """Return left limit of basis function triangle."""

        return max(0., (k - 1) / self.n)

    def e_triangle_right(self, k):
        """Return right limit of basis function triangle."""

        return min(1., (k + 1) / self.n)

    def lhs(self, u, v, u_d, v_d, l1, l2):
        """Return result of the equation's left-hand side - B(u, v)"""

        return (
                - self.beta * v(0) * u(0)
                - integrate(lambda x: self.a(x) * v_d(x) * u_d(x), l1, l2)
                + integrate(lambda x: self.b(x) * v(x) * u_d(x), l1, l2)
                + integrate(lambda x: self.c(x) * v(x) * u(x), l1, l2)
        )

    def lhs_cell(self, i, j):
        """Return result of the equation's lhs for provided matrix indices"""

        l1 = min(self.e_triangle_left(i), self.e_triangle_left(j))
        l2 = max(self.e_triangle_right(i), self.e_triangle_right(j))

        return self.lhs(
            self.e(i),
            self.e(j),
            self.e_d(i),
            self.e_d(j),
            l1,
            l2
        )

    def rhs(self, v, l1, l2):
        """Return result of the equation's rhs - l(v)"""

        return (
                - self.gamma * v(0)
                + integrate(lambda x: v(x) * self.f(x), l1, l2)
        )

    def u_shift(self, x):
        """Return result of u~(x) function"""

        return self.u1 * self.e(self.n)(x)

    def u_star(self, x, result):
        """Return result of u*(x) function for provided FEM matrix"""

        return sum(u_k * self.e(k)(x) for k, u_k in enumerate(result))
