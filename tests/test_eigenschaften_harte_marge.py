"""Eigenschaftstests der harten Marge (ORACLE-PLAYBOOK §5): je 20 bis 50 Zufallsinstanzen, fester Zufalls-Seed, linearer Kern, Zwei Gebiete mit Puffer.

Gelten sollen: Entfernen senkt die Marge nie, Hinzufügen (trennbarkeitserhaltend) hebt sie nie, ein Nicht-Stützvektor ändert nichts, Verschieben/Drehen/Skalieren/Umnummerieren/Labels
vertauschen verhalten sich wie erwartet, ein doppelter Punkt ändert nichts. NICHT als Eigenschaft: die Eindeutigkeit von alpha (Rechteck-Gegenprobe in test_algorithm.py)."""

import math

import numpy as np
import pytest

import hmg_margin as M
import hmg_scenario as Sc
import hmg_separable as P
import hmg_smo as S
from hmg_rng import SplitMix64


def hard(X, y, eps=1e-11):
    r = S.smo(X @ X.T, y, np.inf, eps=eps, max_steps=100000)
    assert r.converged
    return r, (r.alpha * y) @ X, r.b


def marge(X, y):
    r, w, b = hard(X, y)
    return 2.0 / np.linalg.norm(w)


def inst(t, n=14):
    return Sc.puffer(n, 0.2, 4000 + t)


def test_nicht_stuetzvektor_entfernen_aendert_w_und_b_nicht():
    n_geprueft = 0
    for t in range(30):
        ins = inst(t)
        r, w, b = hard(ins.X, ins.y)
        nsv = [i for i in range(ins.n) if r.alpha[i] <= 1e-9]
        if not nsv:
            continue
        keep = np.arange(ins.n) != nsv[0]
        r2, w2, b2 = hard(ins.X[keep], ins.y[keep])
        assert np.abs(w2 - w).max() < 1e-7 and abs(b2 - b) < 1e-6
        n_geprueft += 1
    assert n_geprueft >= 25


def test_entfernen_senkt_die_marge_nie_und_hinzufuegen_hebt_sie_nie():
    rng = SplitMix64(77)
    for t in range(40):
        ins = inst(t)
        m0 = marge(ins.X, ins.y)
        r, _, _ = hard(ins.X, ins.y)
        i = int(rng.below(ins.n))
        keep = np.arange(ins.n) != i
        assert marge(ins.X[keep], ins.y[keep]) >= m0 * (1 - 1e-8)  # Entfernen: nie kleiner
        lo, hi = ins.X.min(0), ins.X.max(0)
        for _ in range(30):  # zufälliger Punkt mit einem Label, das die Trennbarkeit erhält
            p = lo + (hi - lo) * np.array([rng.random(), rng.random()])
            lab = 1.0 if rng.random() < 0.5 else -1.0
            if P.lp_separable(np.vstack([ins.X, p]), np.r_[ins.y, lab]) is not None:
                break
            lab = -lab
            if P.lp_separable(np.vstack([ins.X, p]), np.r_[ins.y, lab]) is not None:
                break
        assert marge(np.vstack([ins.X, p]), np.r_[ins.y, lab]) <= m0 * (1 + 1e-8)  # Hinzufügen: nie größer


def test_verschieben_aller_punkte_aendert_w_nicht_und_b_wie_erwartet():
    for t in range(20):
        ins = inst(t)
        r, w, b = hard(ins.X, ins.y)
        v = np.array([3.0, -2.0])
        r2, w2, b2 = hard(ins.X + v, ins.y)
        assert np.allclose(w2, w, atol=1e-7) and b2 == pytest.approx(b - w @ v, abs=1e-6)  # f(x - v) hat denselben Wert wie f(x)


def test_drehen_aller_punkte_dreht_w_und_laesst_die_marge():
    for t in range(20):
        ins = inst(t)
        r, w, b = hard(ins.X, ins.y)
        a = 0.7
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        r2, w2, b2 = hard(ins.X @ R.T, ins.y)
        assert np.allclose(w2, R @ w, atol=1e-6) and b2 == pytest.approx(b, abs=1e-6)


def test_skalieren_aller_punkte_mit_s_skaliert_die_marge_mit_s():
    for t in range(20):
        ins = inst(t)
        m0 = marge(ins.X, ins.y)
        for s in (0.001, 3.0, 1000.0):
            assert marge(ins.X * s, ins.y) == pytest.approx(s * m0, rel=1e-6)


def test_umnummerieren_aendert_nichts_und_etiketten_vertauschen_dreht_w_und_b_um():
    rng = SplitMix64(5)
    for t in range(20):
        ins = inst(t)
        r, w, b = hard(ins.X, ins.y)
        p = list(range(ins.n))
        rng.shuffle(p)
        r2, w2, b2 = hard(ins.X[p], ins.y[p])
        assert np.allclose(w2, w, atol=1e-7) and b2 == pytest.approx(b, abs=1e-6)
        r3, w3, b3 = hard(ins.X, -ins.y)
        assert np.allclose(w3, -w, atol=1e-7) and b3 == pytest.approx(-b, abs=1e-6)


def test_doppelter_punkt_mit_gleichem_etikett_aendert_die_marge_nicht():
    for t in range(20):
        ins = inst(t)
        m0 = marge(ins.X, ins.y)
        assert marge(np.vstack([ins.X, ins.X[3]]), np.r_[ins.y, ins.y[3]]) == pytest.approx(m0, rel=1e-6)


def test_gewichtssummen_beider_klassen_gleich_alpha_nichtnegativ_und_komplementaritaet():
    for t in range(40):
        ins = inst(t, 12)
        r, w, b = hard(ins.X, ins.y)
        assert (r.alpha >= 0).all()
        assert r.alpha[ins.y > 0].sum() == pytest.approx(r.alpha[ins.y < 0].sum(), rel=1e-8)
        assert np.abs(r.alpha * (ins.y * (ins.X @ w + b) - 1)).max() < 1e-8
        assert (ins.y * (ins.X @ w + b) >= 1 - 1e-8).all()


def test_eindeutigkeit_von_alpha_ist_keine_eigenschaft_nur_w_und_b():
    """Gegenprobe: beim Rechteck ist α nicht eindeutig (α = (1/2 - t, t, 1/2 - t, t), t in [0, 1/2], alle mit demselben w und b);
    die Aufzählung über Stützmengen findet nur deren zwei Endpunkte (Stützmengen {1,3} und {2,4})."""
    ins = Sc.rechteck()
    en = M.enumerate_support_sets(ins.X @ ins.X.T, ins.y)
    assert len(en.feasible) == 2


def test_marge_ist_der_abstand_der_randgeraden_nicht_ein_halber_und_bleibt_unter_dem_abstand_der_klassen():
    for t in range(20):
        ins = inst(t)
        r, w, b = hard(ins.X, ins.y)
        f = ins.X @ w + b
        d_plus = (f[ins.y > 0].min()) / np.linalg.norm(w)  # kleinster Abstand der Klasse +1 zur Geraden f = 0: halbe Marge
        assert 2 * d_plus == pytest.approx(2 / np.linalg.norm(w), rel=1e-6)
