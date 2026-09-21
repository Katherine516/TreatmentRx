"""Minimal pure-Python linear algebra for the Layer 4 estimators.

Kept dependency-free on purpose (the project declares no third-party deps). These
routines are intended for the small designs used here (tens of covariates), not
large-scale problems.
"""

from __future__ import annotations

import math


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

    The dense designs here are half structural zeros — one-hot arm indicators and
    their interactions in the censoring model, the untaken arm's blip block in
    dWOLS — so walking every pair costs several times what the row contains.

    This is the dense front door: it sparsifies the rows and hands them to
    `sparse_weighted_least_squares`, which is the arithmetic. That function
    already existed and had **no callers at all** — a hand-optimised solver
    sitting unused, and therefore unpinned, in the one module invariant 23 says
    must be kept exact. Routing through it means one implementation rather than
    two, and every test of this function now covers it.

    A caller that fits the **same design repeatedly** should sparsify once and
    call the sparse form directly; the propensity model's IRLS does, and that
    removed 204,845 reconstructions of a pattern fixed at construction.
    """
    if not design:
        return []
    return sparse_weighted_least_squares(
        [[(i, value) for i, value in enumerate(row) if value != 0.0] for row in design],
        targets,
        weights,
        len(design[0]),
        ridge,
    )


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


def cholesky_inverse(matrix: list[list[float]]) -> list[list[float]] | None:
    """Invert a symmetric positive-definite matrix via A = L L'.

    Returns `None` when the matrix is not positive definite, which is how
    `inverse` knows to fall back rather than returning a plausible wrong answer.

    Every matrix this package inverts is a normal-equations matrix X'WX plus a
    positive ridge, so it is symmetric positive definite by construction —
    measured, all 23 built during a full fit are. That is what licenses the
    factorisation: Cholesky does not pivot, and for an SPD matrix it does not
    need to.

    Three steps, each about n^3/6, against Gauss-Jordan's ~2n^3 on the augmented
    [A | I]: factor, invert the triangular factor, then A^-1 = (L^-1)' (L^-1),
    whose symmetry means only the upper triangle is computed and mirrored.
    Measured on the real matrices this is 1.5x to 2.4x, least at n=78 because
    that one is 82.5% zeros and the Gauss-Jordan skips zero multipliers.

    **It is not more accurate, and that was worth checking rather than
    assuming.** Residuals of `max |A A^-1 - I|` run 1.8e-15 to 4.9e-14 against
    Gauss-Jordan's 1.8e-15 to 3.8e-14 on the same matrices — better at n=38,
    slightly worse at n=78 and n=98, a wash overall. The case for it is speed.
    """
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row, source = lower[i], matrix[i]
        for j in range(i):
            other, total = lower[j], source[j]
            for k in range(j):
                total -= row[k] * other[k]
            row[j] = total / other[j]
        total = source[i]
        for k in range(i):
            value = row[k]
            total -= value * value
        if total <= 0.0:
            return None
        row[i] = math.sqrt(total)

    # M = L^-1, itself lower triangular.
    inverse_lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row, target = lower[i], inverse_lower[i]
        reciprocal = 1.0 / row[i]
        target[i] = reciprocal
        for j in range(i):
            total = 0.0
            for k in range(j, i):
                total += row[k] * inverse_lower[k][j]
            target[j] = -reciprocal * total

    # A^-1 = M' M. Symmetric, so half the entries are mirrored rather than
    # computed, and M's lower-triangular shape bounds the inner sum below by j.
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            total = 0.0
            for k in range(j, n):
                row = inverse_lower[k]
                total += row[i] * row[j]
            out[i][j] = total
            out[j][i] = total
    return out


def inverse(matrix: list[list[float]]) -> list[list[float]]:
    """Invert, by Cholesky when the matrix admits it and Gauss-Jordan otherwise.

    Every caller here passes a normal-equations matrix, so the Cholesky path is
    the one that runs; the elimination below stays because this function's
    contract is general and a non-positive-definite argument must still get a
    correct answer rather than a silent `None`. `tests/test_linalg.py` asserts
    the two agree and that every matrix a full fit builds takes the fast path.
    """
    factored = cholesky_inverse(matrix)
    if factored is not None:
        return factored
    return gauss_jordan_inverse(matrix)


def gauss_jordan_inverse(matrix: list[list[float]]) -> list[list[float]]:
    """Invert by Gauss-Jordan elimination on the augmented [A | I].

    Solving against each unit vector separately would be O(n^4); eliminating
    once with all n right-hand sides carried along is O(n^3). The general
    fallback for anything Cholesky refuses, and the reference the fast path is
    checked against.
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
    """Row-wise accumulation rather than a generator per output element.

    The textbook `sum(a[i][t] * b[t][j] for t in range(k))` builds one generator
    per entry of the product — n*m of them — and pays the iteration protocol on
    every term. Accumulating a row at a time keeps the inner loop over a plain
    list and skips zero multipliers outright, which matters because the sandwich
    variance multiplies matrices whose blocks are structurally empty.
    """
    n, k, m = len(a), len(b), len(b[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        row_a, row_out = a[i], out[i]
        for t in range(k):
            scale = row_a[t]
            if scale == 0.0:
                continue
            row_b = b[t]
            for j in range(m):
                row_out[j] += scale * row_b[j]
    return out


def sandwich_product(
    bread: list[list[float]], meat: list[list[float]]
) -> list[list[float]]:
    """A B A for symmetric A and B, which is itself symmetric.

    Only the upper triangle of the second multiply is computed and then
    mirrored, because the result cannot be anything else. Three halves of a cubic
    instead of two whole ones.
    """
    n = len(bread)
    left = matmul(bread, meat)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_left, row_out = left[i], out[i]
        for j in range(i, n):
            column = bread[j]  # symmetric, so the row is the column
            total = 0.0
            for t in range(n):
                scale = row_left[t]
                if scale != 0.0:
                    total += scale * column[t]
            row_out[j] = total
            out[j][i] = total
    return out


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
