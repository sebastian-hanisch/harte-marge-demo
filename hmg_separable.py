"""Trennbarkeit: Test als lineares Programm (in Gleitkomma mit den Toleranzen von HiGHS) und das Zertifikat der Nicht-Trennbarkeit (Alternativsatz von Gordan als eigene Rechnung).

Trennbar heißt: es gibt (w, b) mit y_i (w.phi(x_i) + b) >= 1 für alle i. Im Eingaberaum (und in jedem expliziten Merkmalsraum) ist das ein lineares Programm.
Genau eins von beiden gilt: das Programm ist zulässig, oder es gibt alpha >= 0, alpha != 0 mit sum alpha_i y_i = 0 und sum alpha_i y_i phi(x_i) = 0 (Zertifikat).
Im Merkmalsraum eines Gauß-Kerns gilt: verschiedene Punkte heißen immer trennbar (die Gram-Matrix ist positiv definit, K c = y löst y_i f(x_i) = 1), ein Punkt mit
beiden Etiketten nie. Ein quadratischer Löser (libsvm, SLSQP, trust-constr) prüft die Voraussetzung „trennbar“ nicht; diese Datei tut es.
"""

import numpy as np
from scipy import optimize


def poly2_features(X) -> np.ndarray:
    """Expliziter Merkmalsraum des Kerns (x.z + 1)^2 in 2D: (x1^2, x2^2, sqrt2 x1 x2, sqrt2 x1, sqrt2 x2, 1)."""
    X = np.asarray(X, float)
    s = np.sqrt(2.0)
    return np.c_[X[:, 0] ** 2, X[:, 1] ** 2, s * X[:, 0] * X[:, 1], s * X[:, 0], s * X[:, 1], np.ones(len(X))]


def lp_separable(Phi, y):
    """Trennbarkeit als lineares Programm y_i (w.phi_i + b) >= 1: gibt (w, b) oder None (unzulässig)."""
    Phi = np.asarray(Phi, float)
    y = np.asarray(y, float)
    n, d = Phi.shape
    A = -(y[:, None] * np.c_[Phi, np.ones(n)])
    res = optimize.linprog(np.zeros(d + 1), A_ub=A, b_ub=-np.ones(n), bounds=[(None, None)] * (d + 1), method="highs")
    if res.status == 0:
        return res.x[:d], float(res.x[d])
    return None


def gordan_certificate(Phi, y):
    """Zertifikat der Nicht-Trennbarkeit: alpha >= 0 mit sum alpha = 1, sum alpha_i y_i = 0 und sum alpha_i y_i phi_i = 0 (None, wenn es keines gibt)."""
    Phi = np.asarray(Phi, float)
    y = np.asarray(y, float)
    n, d = Phi.shape
    A_eq = np.vstack([y[None, :], (y[:, None] * Phi).T, np.ones((1, n))])
    b_eq = np.r_[np.zeros(1 + d), 1.0]
    res = optimize.linprog(np.zeros(n), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n, method="highs")
    if res.status == 0:
        return res.x
    return None


def certificate_residual(Phi, y, alpha) -> float:
    """Größter Betrag von sum alpha_i y_i und sum alpha_i y_i phi_i (soll 0 sein)."""
    Phi = np.asarray(Phi, float)
    y = np.asarray(y, float)
    a = np.asarray(alpha, float)
    return float(max(abs(a @ y), np.abs((a * y) @ Phi).max()))


def duplicate_conflict(X, y, tol: float = 1e-12) -> bool:
    """Gibt es zwei gleiche Punkte mit verschiedenen Etiketten? Dann ist die Instanz in jedem Kern nicht trennbar."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    for i in range(len(y)):
        same = (np.abs(X - X[i]).max(1) <= tol) & (y != y[i])
        if same.any():
            return True
    return False


def distinct_points(X, tol: float = 1e-12) -> bool:
    X = np.asarray(X, float)
    for i in range(len(X) - 1):
        if (np.abs(X[i + 1:] - X[i]).max(1) <= tol).any():
            return False
    return True


def one_class(y) -> bool:
    y = np.asarray(y, float)
    return not ((y > 0).any() and (y < 0).any())


def separability(X, y, kind: str = "linear") -> dict:
    """Test „trennbar?“ für den Kern `kind` (linear, poly2 über den expliziten Merkmalsraum, gauss über Satz und Gegenbeispiel).

    Rückgabe: dict(separable: bool, reason: str, certificate: Array oder None, Phi: Merkmalsmatrix oder None). `reason` nennt den Weg."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if one_class(y):
        return dict(separable=False, reason="eine Klasse fehlt", certificate=None, Phi=None)
    if duplicate_conflict(X, y):
        return dict(separable=False, reason="derselbe Punkt hat beide Etiketten: in jedem Kern nicht trennbar", certificate=None, Phi=None)
    if kind == "gauss":
        ok = distinct_points(X)
        return dict(separable=ok, reason="Gauß-Kern, verschiedene Punkte: die Gram-Matrix ist positiv definit, K c = y löst y f = 1" if ok else "gleiche Punkte mit gleichem Etikett",
                    certificate=None, Phi=None)
    Phi = X if kind == "linear" else poly2_features(X)
    if lp_separable(Phi, y) is not None:
        return dict(separable=True, reason="lineares Programm zulässig", certificate=None, Phi=Phi)
    return dict(separable=False, reason="lineares Programm unzulässig", certificate=gordan_certificate(Phi, y), Phi=Phi)


def violations(yf, tol: float = 1e-6):
    """Zahl der Punkte mit y f < 1 - tol (harte Bedingung verletzt) und Zahl der falsch klassifizierten (y f <= 0)."""
    yf = np.asarray(yf, float)
    return int((yf < 1 - tol).sum()), int((yf <= 0).sum())
