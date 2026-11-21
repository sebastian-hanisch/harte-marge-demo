"""Erzeuger der Instanzen (alle ausgedacht, keine Messdaten) und die festen Handbeispiele.

Zufallsquelle überall SplitMix64 (`hmg_rng.py`, plattformfest). Normalwerte für die Instanzen der App kommen aus `normal()` der Vorlage (Box-Muller mit exaktem
Logarithmus und Kosinus, bitgleich auf Windows und Linux); nur die großen Testmengen der Messwerkzeuge (4 000 Punkte) nutzen `fast_normal` (Inverse der
Verteilungsfunktion, nur bis auf die letzte Binärstelle plattformgleich). Erzeuger:

* „Zwei Gebiete mit Puffer“: zwei Gebiete beiderseits einer Grenze mit Pufferstreifen (x1 = y (Lücke + |N(0,1)|), x2 ~ N(0; 1,5)), zufällig gedreht.
* „Gauß-Klassen“: zwei überlappende Klassen N((±Abstand/2, 0), I).
* „Ringe mit Lücke“: zwei Ringe, Radius 1 und 2, Breite 0,1.
* „Liefergebiet (Kreis)“: Kunden gleichverteilt im Quadrat [-1, 1]^2, Klasse +1 innerhalb eines Kreises mit halber Fläche.
* Handbeispiele: „Fünf Punkte in der Ebene“, „XOR (vier Punkte)“, „Rechteck“.
"""

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from hmg_rng import SplitMix64

_ND = NormalDist()


@dataclass
class Instance:
    X: np.ndarray
    y: np.ndarray
    name: str = ""
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)


def fast_normal(rng: SplitMix64) -> float:
    """Standardnormal über die Inverse der Verteilungsfunktion (schnell, für große Testmengen der Messwerkzeuge)."""
    return _ND.inv_cdf(min(max(rng.random(), 1e-12), 1.0 - 1e-12))


def _rot(th: float) -> np.ndarray:
    return np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])


def _labels(n: int) -> np.ndarray:
    return np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


def _gap_of(gap, label: float) -> float:
    """Lücke eines Labels: eine Zahl gilt für beide Klassen, ein Paar (Lücke der Klasse +1, Lücke der Klasse -1) ist unsymmetrisch um die wahre Grenze."""
    return float(gap[0] if label > 0 else gap[1]) if isinstance(gap, tuple) else float(gap)


def puffer(n: int, gap, seed: int, rotate: bool = True) -> Instance:
    """Zwei Gebiete mit Puffer: Lücke `gap` auf jeder Seite der wahren Grenze, Labels abwechselnd, Drehung um einen Zufallswinkel (nur wenn rotate)."""
    rng = SplitMix64(seed)
    th = 2.0 * math.pi * rng.random() if rotate else 0.0
    y = _labels(n)
    pts = []
    for i in range(n):
        z1 = abs(rng.normal())
        z2 = 1.5 * rng.normal()
        pts.append((y[i] * (_gap_of(gap, y[i]) + z1), z2))
    X = np.array(pts) @ _rot(th).T
    return Instance(X, y, "Zwei Gebiete mit Puffer", {"gap": gap, "theta_true": th, "seed": seed})


def puffer_test(m: int, gap, seed: int, theta_true: float) -> Instance:
    """Frische Testpunkte desselben Modells (gleiche wahre Grenze), schnelle Normalwerte; für die Messwerkzeuge."""
    rng = SplitMix64(seed)
    y = _labels(m)
    pts = [(y[i] * (_gap_of(gap, y[i]) + abs(fast_normal(rng))), 1.5 * fast_normal(rng)) for i in range(m)]
    return Instance(np.array(pts) @ _rot(theta_true).T, y, "Testpunkte (Zwei Gebiete mit Puffer)")


def gauss_classes(n: int, sep: float, seed: int) -> Instance:
    """Zwei überlappende Gauß-Klassen N((±sep/2, 0), I), Labels abwechselnd."""
    rng = SplitMix64(seed)
    y = _labels(n)
    X = np.array([(rng.normal(), rng.normal()) for _ in range(n)])
    X[:, 0] += y * sep / 2.0
    return Instance(X, y, "Gauß-Klassen", {"sep": sep, "seed": seed})


def rings(n_per: int, seed: int, r_in: float = 1.0, r_out: float = 2.0, width: float = 0.1) -> Instance:
    """Zwei Ringe mit Lücke: Radius 1 (Klasse -1) und 2 (Klasse +1), Breite ± `width`, gleichverteilter Winkel."""
    rng = SplitMix64(seed)
    pts = []
    for r0 in (r_in, r_out):
        for _ in range(n_per):
            ang = 2.0 * math.pi * rng.random()
            r = r0 + width * (2.0 * rng.random() - 1.0)
            pts.append((r * math.cos(ang), r * math.sin(ang)))
    y = np.r_[-np.ones(n_per), np.ones(n_per)]
    return Instance(np.array(pts), y, "Ringe mit Lücke", {"seed": seed})


def liefergebiet(n: int, seed: int) -> Instance:
    """Liefergebiet (Kreis): Kunden gleichverteilt im Quadrat [-1, 1]^2, Depot in der Mitte, Klasse +1 innerhalb des Kreises mit halber Fläche (Radius sqrt(2/pi))."""
    rng = SplitMix64(seed)
    X = np.array([(2.0 * rng.random() - 1.0, 2.0 * rng.random() - 1.0) for _ in range(n)])
    y = np.where((X ** 2).sum(1) <= 2.0 / math.pi, 1.0, -1.0)
    return Instance(X, y, "Liefergebiet (Kreis)", {"seed": seed})


def xor4() -> Instance:
    """XOR (vier Punkte): (1, 1) und (-1, -1) mit +1, (1, -1) und (-1, 1) mit -1."""
    return Instance(np.array([(1, 1), (-1, -1), (1, -1), (-1, 1)], float), np.array([1.0, 1.0, -1.0, -1.0]), "XOR (vier Punkte)")


def fuenf_punkte() -> Instance:
    """Fünf Punkte in der Ebene (Handbeispiel); die Reihenfolge ist die von alpha = (6/5, 2, 4/5, 0, 0)."""
    return Instance(np.array([(4, 3), (3, 1), (0, 0), (4, 1), (2, 0)], float), np.array([-1.0, 1.0, -1.0, 1.0, 1.0]), "Fünf Punkte in der Ebene")


def rechteck() -> Instance:
    """Rechteck (Entartung): vier Punkte auf den Randgeraden, aber nur zwei Dimensionen."""
    return Instance(np.array([(0, 1), (2, 1), (0, -1), (2, -1)], float), np.array([1.0, 1.0, -1.0, -1.0]), "Rechteck")


def widerspruch(n: int, seed: int) -> Instance:
    """Ein Punkt mit beiden Labels: Gauß-Klassen mit Abstand 4 plus Duplikat des ersten Punkts mit umgekehrtem Label (in jedem Kern nicht trennbar)."""
    base = gauss_classes(max(n - 1, 2), 4.0, seed)
    X = np.vstack([base.X, base.X[:1]])
    y = np.r_[base.y, -base.y[0]]
    return Instance(X, y, "Ein Punkt mit beiden Labels", {"seed": seed})
