"""The pure-Python linear algebra, and the optimisations layered onto it.

Every routine here was rewritten for speed at some point. Each test states the
textbook form it has to agree with, so a future rewrite that is subtly wrong
fails rather than quietly shifting every standard error in the system.
"""

import random
import unittest

from treatmentrx.estimation import linalg


def _symmetric(size, seed):
    rng = random.Random(seed)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            value = rng.uniform(-1.0, 1.0)
            matrix[i][j] = matrix[j][i] = value
        matrix[i][i] += size  # keep it comfortably invertible
    return matrix


def _textbook_matmul(a, b):
    return [[sum(a[i][t] * b[t][j] for t in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


class MatmulTests(unittest.TestCase):
    def test_row_accumulation_matches_the_textbook_form(self):
        rng = random.Random(1)
        a = [[rng.uniform(-1, 1) for _ in range(9)] for _ in range(7)]
        b = [[rng.uniform(-1, 1) for _ in range(5)] for _ in range(9)]
        self.assertEqual(_close(linalg.matmul(a, b), _textbook_matmul(a, b)), True)

    def test_zero_multipliers_are_skipped_without_changing_the_result(self):
        a = [[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]]
        b = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        self.assertEqual(linalg.matmul(a, b), [[6.0, 8.0], [1.0, 2.0]])


class SandwichProductTests(unittest.TestCase):
    def test_matches_two_full_multiplies(self):
        """A B A computed by halves must equal A B A computed whole."""
        bread, meat = _symmetric(24, seed=2), _symmetric(24, seed=3)
        expected = _textbook_matmul(_textbook_matmul(bread, meat), bread)
        self.assertTrue(_close(linalg.sandwich_product(bread, meat), expected))

    def test_the_result_is_symmetric(self):
        bread, meat = _symmetric(16, seed=4), _symmetric(16, seed=5)
        product = linalg.sandwich_product(bread, meat)
        for i in range(16):
            for j in range(16):
                self.assertAlmostEqual(product[i][j], product[j][i], places=12)


class WeightedLeastSquaresTests(unittest.TestCase):
    def test_skipping_zeros_matches_the_dense_normal_equations(self):
        """The censoring and dWOLS designs are half structural zeros."""
        rng = random.Random(6)
        design, targets, weights = [], [], []
        for _ in range(60):
            row = [rng.uniform(-1, 1) if rng.random() < 0.5 else 0.0 for _ in range(8)]
            row[0] = 1.0  # intercept, always present
            design.append(row)
            targets.append(rng.uniform(0, 1))
            weights.append(rng.uniform(0.5, 1.5))

        p = 8
        xtwx = [[0.0] * p for _ in range(p)]
        xtwy = [0.0] * p
        for row, y, w in zip(design, targets, weights):
            for i in range(p):
                xtwy[i] += w * row[i] * y
                for j in range(p):
                    xtwx[i][j] += w * row[i] * row[j]
        for i in range(p):
            xtwx[i][i] += 1e-4
        expected = linalg.solve(xtwx, xtwy)

        actual = linalg.weighted_least_squares(design, targets, weights, ridge=1e-4)
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=9)


class InverseTests(unittest.TestCase):
    def test_inverse_reproduces_the_identity(self):
        matrix = _symmetric(20, seed=7)
        product = linalg.matmul(matrix, linalg.inverse(matrix))
        for i in range(20):
            for j in range(20):
                self.assertAlmostEqual(product[i][j], 1.0 if i == j else 0.0, places=9)


class QuadraticFormTests(unittest.TestCase):
    def test_sparse_support_matches_the_full_walk(self):
        matrix = _symmetric(12, seed=8)
        vector = [0.0] * 12
        vector[2], vector[7], vector[9] = 1.5, -0.5, 2.0
        expected = sum(
            vector[i] * matrix[i][j] * vector[j] for i in range(12) for j in range(12)
        )
        self.assertAlmostEqual(linalg.quadratic_form(vector, matrix), expected, places=9)

    def test_a_zero_vector_has_zero_variance(self):
        self.assertEqual(linalg.quadratic_form([0.0] * 6, _symmetric(6, seed=9)), 0.0)


def _close(a, b, tolerance=1e-9):
    return all(
        abs(a[i][j] - b[i][j]) < tolerance for i in range(len(a)) for j in range(len(a[0]))
    )


if __name__ == "__main__":
    unittest.main()
