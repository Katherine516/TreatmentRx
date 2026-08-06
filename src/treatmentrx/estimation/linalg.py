"""Minimal pure-Python linear algebra for the Layer 4 estimators.

Kept dependency-free on purpose (the project declares no third-party deps). These
routines are intended for the small designs used here (tens of covariates), not
large-scale problems.
"""

from __future__ import annotations


def solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(matrix)
    # Build an augmented copy.
    a = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            a[pivot][col] += 1e-9  # ridge nudge for near-singular columns
        a[col], a[pivot] = a[pivot], a[col]

        pivot_val = a[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col] / pivot_val
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]

    return [a[i][n] / a[i][i] for i in range(n)]


def weighted_least_squares(
    design: list[list[float]],
    targets: list[float],
    weights: list[float],
    ridge: float = 1e-6,
) -> list[float]:
    """Return beta minimizing sum_i w_i (y_i - x_i . beta)^2 via the normal equations.

    A tiny ridge term stabilizes near-collinear designs.
    """
    if not design:
        return []
    p = len(design[0])
    xtwx = [[0.0 for _ in range(p)] for _ in range(p)]
    xtwy = [0.0 for _ in range(p)]
    for row, y, w in zip(design, targets, weights):
        for i in range(p):
            xtwy[i] += w * row[i] * y
            for j in range(p):
                xtwx[i][j] += w * row[i] * row[j]
    for i in range(p):
        xtwx[i][i] += ridge
    return solve(xtwx, xtwy)


SparseRow = list  # list[tuple[int, float]] — (column index, value) pairs


def sparse_normal_matrix(
    rows: list[SparseRow],
    weights: list[float],
    n_features: int,
    ridge: list[float] | float = 1e-6,
) -> list[list[float]]:
    """Accumulate X'WX (+ ridge) from sparse design rows."""
    penalties = [float(ridge)] * n_features if isinstance(ridge, (int, float)) else list(ridge)
    if len(penalties) != n_features:
        raise ValueError("ridge vector must have one entry per feature")
    xtwx = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    for row, w in zip(rows, weights):
        for i, xi in row:
            wxi = w * xi
            target_row = xtwx[i]
            for j, xj in row:
                target_row[j] += wxi * xj
    for i in range(n_features):
        xtwx[i][i] += penalties[i]
    return xtwx


def sparse_weighted_least_squares(
    rows: list[SparseRow],
    targets: list[float],
    weights: list[float],
    n_features: int,
    ridge: list[float] | float = 1e-6,
) -> list[float]:
    """Weighted least squares over sparse design rows with per-coefficient ridge.

    The Q-learning design is block-sparse: a row activates one stage block and
    one arm block, so ~10 of ~30 columns are non-zero. Accumulating only the
    non-zero pairs keeps the normal equations cheap in pure Python.

    `ridge` may be a scalar or a per-coefficient vector, which is how the
    penalized Q-shared fit shrinks the blip block without touching the
    treatment-free block.
    """
    penalties = [float(ridge)] * n_features if isinstance(ridge, (int, float)) else list(ridge)
    if len(penalties) != n_features:
        raise ValueError("ridge vector must have one entry per feature")

    xtwx = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    xtwy = [0.0 for _ in range(n_features)]
    for row, y, w in zip(rows, targets, weights):
        for i, xi in row:
            wxi = w * xi
            xtwy[i] += wxi * y
            target_row = xtwx[i]
            for j, xj in row:
                target_row[j] += wxi * xj
    for i in range(n_features):
        xtwx[i][i] += penalties[i]
    return solve(xtwx, xtwy)


def inverse(matrix: list[list[float]]) -> list[list[float]]:
    """Invert by Gauss-Jordan elimination on the augmented [A | I].

    Solving against each unit vector separately would be O(n^4); eliminating
    once with all n right-hand sides carried along is O(n^3). Only used on the
    p x p normal-equations matrix, which the sandwich variance formula needs
    explicitly.
    """
    n = len(matrix)
    a = [list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            a[pivot][col] += 1e-9  # ridge nudge for near-singular columns
        a[col], a[pivot] = a[pivot], a[col]

        pivot_value = a[col][col]
        pivot_row = a[col]
        for c in range(col, 2 * n):
            pivot_row[c] /= pivot_value
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            if factor == 0.0:
                continue
            row = a[r]
            for c in range(col, 2 * n):
                row[c] -= factor * pivot_row[c]

    return [row[n:] for row in a]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n, k, m = len(a), len(b), len(b[0])
    return [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(m)] for i in range(n)]


def quadratic_form(vector: list[float], matrix: list[list[float]]) -> float:
    """v' M v — the variance of a linear combination v of the parameters.

    Restricted to the non-zero support of `v`. Every caller here is asking about
    a contrast between two arms, which touches two blip blocks and leaves the
    rest of the parameter vector at zero — 8 non-zeros out of 78 for the
    stage-specific fit. Walking the full matrix would be ~95x the work for the
    same number.
    """
    support = [(i, value) for i, value in enumerate(vector) if value != 0.0]
    return sum(
        vi * sum(matrix[i][j] * vj for j, vj in support) for i, vi in support
    )


def solve_precomputed(
    normal_inverse: list[list[float]],
    rows: list[SparseRow],
    targets: list[float],
    weights: list[float],
    n_features: int,
) -> list[float]:
    """beta = (X'WX)^-1 X'Wy, reusing an already-inverted normal matrix.

    Iterative procedures that only change `y` between passes — Q-learning's
    pseudo-outcome fixed point — would otherwise rebuild and re-solve the same
    matrix every time.
    """
    xtwy = [0.0] * n_features
    for row, y, w in zip(rows, targets, weights):
        wy = w * y
        for i, xi in row:
            xtwy[i] += xi * wy
    return [
        sum(normal_inverse[i][j] * xtwy[j] for j in range(n_features) if xtwy[j] != 0.0)
        for i in range(n_features)
    ]


def dot(vector_a: list[float], vector_b: list[float]) -> float:
    return sum(x * y for x, y in zip(vector_a, vector_b))
