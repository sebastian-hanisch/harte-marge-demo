"""Erzeuger der Instanzen: eingefrorene Werte je festem Zufalls-Seed (erste Zahlen und Prüfsumme), Eigenschaften der Modelle als Band, Zufallsquelle SplitMix64.

Die Normalwerte der Instanzen kommen aus `normal()` der Vorlage (exakter Logarithmus und Kosinus, auf Windows und Linux bitgleich); Winkel und Ringe nutzen `math.cos` und
`math.sin` der C-Bibliothek, darum stehen die eingefrorenen Werte mit einer Toleranz von 1e-9 (und die Prüfsumme auf 9 Stellen gerundet)."""

import hashlib
import math

import numpy as np
import pytest

import hmg_scenario as Sc
from hmg_rng import SplitMix64


def _checksum(ins):
    return hashlib.sha256(np.round(np.c_[ins.X, ins.y], 9).tobytes()).hexdigest()[:16]


def test_splitmix64_referenzwerte_und_normalzahlen():
    r = SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [16294208416658607535, 7960286522194355700, 487617019471545679]  # Referenzfolge von SplitMix64 mit Seed 0
    n = SplitMix64(1)
    assert [n.normal() for _ in range(3)] == pytest.approx([-0.028249746095855, -0.227919522867635, 0.10309095168574], abs=1e-12)


def test_eingefrorene_instanzen_je_zufalls_seed():
    p = Sc.puffer(8, 0.25, 7)
    assert np.round(p.X[:3], 6).tolist() == [[-1.028688, 2.787486], [0.499213, -2.347829], [-1.24534, 0.547197]] and p.y[:4].tolist() == [1.0, -1.0, 1.0, -1.0]
    assert p.info["theta_true"] == pytest.approx(2.449372547393552, abs=1e-12) and _checksum(p) == "8fdd3a97ef226e92"
    g = Sc.gauss_classes(8, 2.0, 7)
    assert np.round(g.X[:3], 6).tolist() == [[2.364992, -0.396524], [-0.995501, -0.580613], [-0.712889, 2.062278]] and _checksum(g) == "8dd01fca02cbc58f"
    r = Sc.rings(4, 7)
    assert np.round(r.X[:3], 6).tolist() == [[-0.695433, 0.576566], [0.825282, -0.593597], [-0.907793, 0.279636]] and _checksum(r) == "161a6dd5e2a30129"
    lg = Sc.liefergebiet(10, 7)
    assert np.round(lg.X[:3], 6).tolist() == [[-0.220341, -0.966423], [0.801521, 0.165861], [-0.095116, -0.501137]] and _checksum(lg) == "f43cb73a8bb40af2"
    w = Sc.widerspruch(6, 7)
    assert _checksum(w) == "4c0bafae4564f190" and np.array_equal(w.X[-1], w.X[0]) and w.y[-1] == -w.y[0]


def test_gleicher_zufalls_seed_gleiche_instanz_anderer_seed_andere_instanz():
    assert np.array_equal(Sc.puffer(12, 0.3, 5).X, Sc.puffer(12, 0.3, 5).X) and not np.array_equal(Sc.puffer(12, 0.3, 5).X, Sc.puffer(12, 0.3, 6).X)
    assert np.array_equal(Sc.rings(10, 2).X, Sc.rings(10, 2).X) and not np.array_equal(Sc.liefergebiet(30, 2).X, Sc.liefergebiet(30, 3).X)


def test_puffer_lueckenbedingung_in_der_wahren_achse_und_drehung():
    for seed in range(20):
        ins = Sc.puffer(20, 0.25, seed)
        th = ins.info["theta_true"]
        Z = ins.X @ np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])  # zurückgedreht: x1 hat die Lücke
        assert (ins.y * Z[:, 0] >= 0.25 - 1e-12).all()  # jeder Punkt liegt mindestens die Lücke von der wahren Grenze entfernt
        assert (ins.y > 0).sum() == 10 and (ins.y < 0).sum() == 10
    un = Sc.puffer(40, (0.05, 0.8), 3, rotate=False)
    assert (un.y[un.y > 0] * un.X[un.y > 0, 0] >= 0.05 - 1e-12).all() and (-un.X[un.y < 0, 0] >= 0.8 - 1e-12).all() and un.info["theta_true"] == 0.0


def test_puffer_verteilung_als_band_halbnormal_und_streuung_1_5():
    ins = Sc.puffer(4000, 0.25, 11, rotate=False)
    d = ins.y * ins.X[:, 0] - 0.25  # |N(0,1)|: Mittel sqrt(2/pi) = 0,7979, Standardfehler ~ 0,0076
    assert abs(d.mean() - math.sqrt(2 / math.pi)) < 4.5 * 0.6028 / math.sqrt(4000)
    assert abs(ins.X[:, 1].std() - 1.5) < 4.5 * 1.5 / math.sqrt(2 * 4000)
    assert abs(ins.X[:, 1].mean()) < 4.5 * 1.5 / math.sqrt(4000)


def test_gauss_klassen_mittelwerte_und_streuung_als_band():
    ins = Sc.gauss_classes(4000, 2.0, 3)
    for lab in (1.0, -1.0):
        P = ins.X[ins.y == lab]
        assert abs(P[:, 0].mean() - lab) < 4.5 / math.sqrt(2000) and abs(P[:, 1].mean()) < 4.5 / math.sqrt(2000)
        assert abs(P.std(0) - 1).max() < 4.5 / math.sqrt(2 * 2000)


def test_ringe_radien_und_klassen_und_liefergebiet_kreis_halbe_flaeche():
    r = Sc.rings(300, 9)
    rad = np.linalg.norm(r.X, axis=1)
    assert (np.abs(rad[r.y < 0] - 1.0) <= 0.1 + 1e-12).all() and (np.abs(rad[r.y > 0] - 2.0) <= 0.1 + 1e-12).all()
    lg = Sc.liefergebiet(4000, 4)
    assert (np.abs(lg.X) <= 1).all() and abs((lg.y > 0).mean() - 0.5) < 4.5 * 0.5 / math.sqrt(4000)  # Kreis mit halber Fläche
    assert ((np.linalg.norm(lg.X, axis=1) <= math.sqrt(2 / math.pi)) == (lg.y > 0)).all()


def test_handbeispiele_sind_die_festen_punktmengen():
    assert Sc.fuenf_punkte().X.tolist() == [[4, 3], [3, 1], [0, 0], [4, 1], [2, 0]] and Sc.fuenf_punkte().y.tolist() == [-1, 1, -1, 1, 1]
    assert Sc.xor4().y.tolist() == [1, 1, -1, -1] and Sc.rechteck().X.tolist() == [[0, 1], [2, 1], [0, -1], [2, -1]]


def test_testpunkte_haben_dieselbe_wahre_grenze_und_schnelle_normalwerte_sind_normal():
    ins = Sc.puffer(20, 0.25, 5)
    te = Sc.puffer_test(4000, 0.25, 6, ins.info["theta_true"])
    th = ins.info["theta_true"]
    Z = te.X @ np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    assert (te.y * Z[:, 0] >= 0.25 - 1e-9).all() and te.n == 4000
    rng = SplitMix64(12)
    z = np.array([Sc.fast_normal(rng) for _ in range(20000)])
    assert abs(z.mean()) < 4.5 / math.sqrt(20000) and abs(z.std() - 1) < 4.5 / math.sqrt(2 * 20000)
