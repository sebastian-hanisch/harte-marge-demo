"""Harte Marge: Gram-Matrix, Aufzählung aller Stützmengen (n <= 8), Abstand der konvexen Hüllen (2D), Richtungsgitter (2D).

Die Marge heißt hier **2/||w||** (Abstand der beiden Randgeraden y f = +-1), nicht 1/||w||. Drei unabhängige Rechenwege für die harte Marge:
(1) Aufzählung: je gemischter Stützmenge S ein lineares KKT-System (kein Optimierer), (2) Abstand der konvexen Hüllen beider Klassen (Geometrie, kein QP),
(3) Richtungsgitter: größte Marge über alle Richtungen u der Wert min über Klasse + von u.x minus max über Klasse - von u.x (kein QP). Das duale QP selbst löst
`hmg_smo.py`.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np

ENUM_MAX_N = 8  # größere Instanzen: 2^n Mengen, nur noch gegen Fremdlöser prüfen


# --- Gram-Matrix ------------------------------------------------------------------------------------------------------------------------------
KERNELS = ("linear", "poly2", "gauss")


def kernel_matrix(X, kind: str = "linear", sigma: float = 1.0) -> np.ndarray:
    """Gram-Matrix K_ij = k(x_i, x_j): linear x.z, Polynom Grad 2 (x.z + 1)^2, Gauß exp(-||x - z||^2 / (2 sigma^2))."""
    X = np.asarray(X, float)
    G = X @ X.T
    if kind == "linear":
        return G
    if kind == "poly2":
        return (G + 1.0) ** 2
    if kind == "gauss":
        d = np.diag(G)
        D = np.maximum(d[:, None] + d[None, :] - 2.0 * G, 0.0)
        return np.exp(-D / (2.0 * sigma * sigma))
    raise ValueError(f"unbekannter Kern: {kind}")


def kernel_cross(Xa, Xb, kind: str = "linear", sigma: float = 1.0) -> np.ndarray:
    """k(a_i, b_j) für zwei Punktmengen (Testpunkte gegen Trainingspunkte)."""
    Xa, Xb = np.asarray(Xa, float), np.asarray(Xb, float)
    G = Xa @ Xb.T
    if kind == "linear":
        return G
    if kind == "poly2":
        return (G + 1.0) ** 2
    if kind == "gauss":
        D = np.maximum((Xa ** 2).sum(1)[:, None] + (Xb ** 2).sum(1)[None, :] - 2.0 * G, 0.0)
        return np.exp(-D / (2.0 * sigma * sigma))
    raise ValueError(f"unbekannter Kern: {kind}")


def margin_from_alpha(K, y, alpha):
    """Marge 2/||w|| im Merkmalsraum mit ||w||^2 = (a y)' K (a y); None, wenn w = 0 (dann ist die Marge nicht definiert)."""
    ya = np.asarray(alpha, float) * np.asarray(y, float)
    w2 = float(ya @ np.asarray(K, float) @ ya)
    if w2 <= 1e-24:
        return None
    return 2.0 / math.sqrt(w2)


def decision_values(K_cross, y, alpha, b):
    """f(x) = sum_j alpha_j y_j k(x, x_j) + b; K_cross hat die Form (Testpunkte, Trainingspunkte)."""
    return np.asarray(K_cross, float) @ (np.asarray(alpha, float) * np.asarray(y, float)) + b


def primal_wb(X, y, alpha, b):
    """Linearer Kern: w = sum alpha_i y_i x_i."""
    return (np.asarray(alpha, float) * np.asarray(y, float)) @ np.asarray(X, float), float(b)


# --- Aufzählung der Stützmengen ---------------------------------------------------------------------------------------------------------------
@dataclass
class SetRow:
    support: tuple  # Indizes der Menge S
    status: str  # "zulässig" | "singulär" | "alpha ≤ 0" | "Randbedingung verletzt"
    alpha: np.ndarray | None  # Länge n (außerhalb von S null), None bei "singulär"
    b: float | None
    detail: str


@dataclass
class Enumeration:
    rows: list
    n: int

    @property
    def feasible(self) -> list:
        return [r for r in self.rows if r.status == "zulässig"]

    @property
    def tested(self) -> int:
        return len(self.rows)


def enumerate_support_sets(K, y, tol_a: float = 1e-10, tol_f: float = 1e-9) -> Enumeration:
    """Alle gemischten Stützmengen S (beide Klassen, mindestens zwei Punkte) mit je einem linearen KKT-System

        sum_{j in S} Q_ij a_j + y_i b = 1 (i in S),   sum_{j in S} y_j a_j = 0,

    zulässig, wenn a_S > 0 und y_i f(x_i) >= 1 für alle i. Kein Optimierer beteiligt. Nur für n <= 8 (2^n Mengen)."""
    K = np.asarray(K, float)
    y = np.asarray(y, float)
    n = len(y)
    if n > ENUM_MAX_N:
        raise ValueError(f"Aufzählung nur bis n = {ENUM_MAX_N} (2^n Mengen), n = {n}")
    Q = (y[:, None] * y[None, :]) * K
    rows = []
    for r in range(2, n + 1):
        for S in itertools.combinations(range(n), r):
            idx = list(S)
            if (y[idx] > 0).all() or (y[idx] < 0).all():
                continue
            A = np.zeros((r + 1, r + 1))
            A[:r, :r] = Q[np.ix_(idx, idx)]
            A[:r, r] = y[idx]
            A[r, :r] = y[idx]
            rhs = np.r_[np.ones(r), 0.0]
            try:
                if np.linalg.cond(A) > 1e12:
                    raise np.linalg.LinAlgError
                sol = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                rows.append(SetRow(S, "singulär", None, None, "das lineare System der Menge ist singulär"))
                continue
            a, b = sol[:r], float(sol[r])
            alpha = np.zeros(n)
            alpha[idx] = a
            if (a <= tol_a).any():
                k = idx[int(np.argmin(a))]
                rows.append(SetRow(S, "alpha ≤ 0", alpha, b, f"alpha_{k + 1} = {a.min():.3g} ≤ 0"))
                continue
            f = K @ (alpha * y) + b
            yf = y * f
            if (yf < 1 - tol_f).any():
                j = int(np.argmin(yf))
                rows.append(SetRow(S, "Randbedingung verletzt", alpha, b, f"y f(x_{j + 1}) = {yf[j]:.3g} < 1"))
                continue
            rows.append(SetRow(S, "zulässig", alpha, b, "alle alpha > 0 und y f ≥ 1"))
    return Enumeration(rows, n)


# --- Abstand der konvexen Hüllen (2D) ---------------------------------------------------------------------------------------------------------
def _point_segments_dist(p, a, b):
    """Kleinster Abstand des Punkts p zu den Strecken a_k b_k (vektorisiert über k)."""
    ab = b - a
    den = (ab ** 2).sum(1)
    t = np.where(den > 0, ((p - a) * ab).sum(1) / np.where(den > 0, den, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    q = a + t[:, None] * ab
    return float(np.sqrt(((p - q) ** 2).sum(1)).min())


def hull_distance(X, y) -> float:
    """Abstand der konvexen Hüllen beider Klassen in 2D: kleinster Abstand eines Punkts der einen Klasse zu einer Strecke zwischen zwei Punkten der anderen
    (die Hüllenränder liegen unter diesen Strecken, die übrigen liegen innen). Kein QP. Gleich der größten Marge, wenn die Klassen trennbar sind."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    P, N = X[y > 0], X[y < 0]
    best = math.inf
    for A, B in ((P, N), (N, P)):
        ii, jj = np.triu_indices(len(B))
        for p in A:
            best = min(best, _point_segments_dist(p, B[ii], B[jj]))
    return best


# --- Richtungsgitter (2D) ---------------------------------------------------------------------------------------------------------------------
def direction_margin(X, y, th: float) -> float:
    """Breite des leeren Streifens in der Richtung u = (cos th, sin th): min über Klasse + von u.x minus max über Klasse - von u.x. Nicht positiv, wenn die
    Richtung nicht trennt (Vorzeichen: Klasse + liegt auf der Seite von u)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X @ np.array([math.cos(th), math.sin(th)])
    return float(p[y > 0].min() - p[y < 0].max())


def margin_curve(X, y, thetas) -> np.ndarray:
    """direction_margin für viele Winkel auf einmal (Bogenmaß)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    th = np.asarray(thetas, float)
    U = np.c_[np.cos(th), np.sin(th)]
    P = X @ U.T
    return P[y > 0].min(0) - P[y < 0].max(0)


def max_margin_by_directions(X, y, grid: int = 3600):
    """Größte Marge über alle Richtungen: Gitter über den Winkel, dann Verfeinerung mit dem Goldenen Schnitt (am Maximum sitzt meist eine Knickstelle, zwei
    Sinuskurven kreuzen sich). Kein QP. Gibt (Marge, Winkel in Bogenmaß) zurück; die Marge ist 0 oder negativ, wenn keine Richtung trennt."""
    ths = np.linspace(0.0, 2.0 * math.pi, grid, endpoint=False)
    vals = margin_curve(X, y, ths)
    i = int(np.argmax(vals))
    d = 2.0 * math.pi / grid
    lo, hi = ths[i] - 2 * d, ths[i] + 2 * d
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c1, c2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = direction_margin(X, y, c1), direction_margin(X, y, c2)
    for _ in range(90):
        if f1 < f2:
            lo, c1, f1 = c1, c2, f2
            c2 = lo + gr * (hi - lo)
            f2 = direction_margin(X, y, c2)
        else:
            hi, c2, f2 = c2, c1, f1
            c1 = hi - gr * (hi - lo)
            f1 = direction_margin(X, y, c1)
    th = 0.5 * (lo + hi)
    best = direction_margin(X, y, th)
    if best < vals[i]:
        best, th = float(vals[i]), float(ths[i])
    return best, th % (2.0 * math.pi)


def wrap_deg(a: float) -> float:
    """Winkel in Grad auf (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def angle_between_deg(u, v) -> float:
    """Winkel zwischen zwei Richtungen (Geraden, also ohne Vorzeichen) in Grad, 0 bis 90."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    c = abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(math.degrees(math.acos(min(1.0, c))))
