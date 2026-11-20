"""Gemeinsame Fixtures und Hilfen: die Mini-Instanz (fünf Punkte in der Ebene), an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b), und ein exakter Bruch-Löser als Orakel."""

import itertools
import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hmg_scenario import Instance, fuenf_punkte  # noqa: E402


@pytest.fixture
def mini() -> Instance:
    """Fünf Punkte in der Ebene: (4,3) und (0,0) mit -1, (3,1), (4,1), (2,0) mit +1. Von Hand: genau eine von 21 gemischten Stützmengen ist zulässig,
    alpha = (6/5, 2, 4/5, 0, 0), b = -1, w = (6/5, -8/5), Marge 2/||w|| = 1."""
    return fuenf_punkte()


# --- exakte Hilfen (nur Brüche, kein Optimierer): unabhängig vom Code der Demo ---------------------------------------------------------------------
def solve_frac(A, b):
    """Gauß-Jordan in Brüchen; None bei singulärer Matrix."""
    n = len(A)
    M = [[F(A[i][j]) for j in range(n)] + [F(b[i])] for i in range(n)]
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        if p is None:
            return None
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def gram_frac(X):
    return [[sum(F(a) * F(b) for a, b in zip(x, z)) for z in X] for x in X]


def enum_frac(K, y):
    """Alle KKT-Lösungen des harten Duals in Brüchen: Liste von (Stützmenge, alpha, b) und Zahl geprüfter gemischter Mengen."""
    n = len(y)
    found, tested = [], 0
    for r in range(2, n + 1):
        for S in itertools.combinations(range(n), r):
            if all(y[i] > 0 for i in S) or all(y[i] < 0 for i in S):
                continue
            tested += 1
            A = [[y[i] * y[j] * K[i][j] for j in S] + [y[i]] for i in S] + [[y[j] for j in S] + [0]]
            s = solve_frac(A, [1] * r + [0])
            if s is None:
                continue
            a, b = s[:r], s[r]
            if any(v <= 0 for v in a):
                continue
            al = [F(0)] * n
            for k, i in enumerate(S):
                al[i] = a[k]
            if all(y[i] * (sum(al[j] * y[j] * K[j][i] for j in range(n)) + b) >= 1 for i in range(n)):
                found.append((S, al, b))
    return found, tested


def w2_frac(al, y, K):
    n = len(y)
    return sum(al[i] * y[i] * al[j] * y[j] * K[i][j] for i in range(n) for j in range(n))
