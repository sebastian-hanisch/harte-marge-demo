"""Einheiten der Demo gegen die Handbeispiele (exakte Brüche, `handbeispiele_02.py` der Planung) und gegen einen Bruch-Orakel ohne Optimierer.

Je Einheit eine Mini-Instanz, von Hand gerechnet: Gram-Matrix, Aufzählung der Stützmengen, Hüllenabstand, Richtungsgitter, Trennbarkeit und Zertifikat, Perceptron, Kerne."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

import hmg_margin as M
import hmg_perceptron as Pe
import hmg_scenario as Sc
import hmg_separable as P
import hmg_smo as S
from conftest import enum_frac, gram_frac, w2_frac


def _seg_dist2(p, a, b):
    ab = [bb - aa for aa, bb in zip(a, b)]
    ap = [pp - aa for aa, pp in zip(a, p)]
    den = sum(v * v for v in ab)
    t = F(0) if den == 0 else min(F(1), max(F(0), sum(u * v for u, v in zip(ap, ab)) / den))
    q = [aa + t * v for aa, v in zip(a, ab)]
    return sum((pp - qq) ** 2 for pp, qq in zip(p, q))


def _hull_dist2(X, y):
    Pp = [tuple(map(F, x)) for x, yy in zip(X, y) if yy > 0]
    Nn = [tuple(map(F, x)) for x, yy in zip(X, y) if yy < 0]
    best = None
    for A, B in ((Pp, Nn), (Nn, Pp)):
        for p in A:
            for i in range(len(B)):
                for j in range(i, len(B)):
                    d = _seg_dist2(p, B[i], B[j])
                    best = d if best is None else min(best, d)
    return best


def _dir_margin2(X, y, w):
    pos = [sum(F(a) * F(b) for a, b in zip(w, x)) for x, yy in zip(X, y) if yy > 0]
    neg = [sum(F(a) * F(b) for a, b in zip(w, x)) for x, yy in zip(X, y) if yy < 0]
    g = min(pos) - max(neg)
    return None if g <= 0 else g * g / sum(F(a) * F(a) for a in w)


def _solve_float(X, y, kind="linear", sigma=1.0):
    K = M.kernel_matrix(X, kind, sigma)
    en = M.enumerate_support_sets(K, y)
    return K, en


# --- Teil 1: drei Punkte in einer Dimension -----------------------------------------------------------------------------------------------------
def test_drei_punkte_in_1d_eine_von_drei_mengen_alpha_2_2_0():
    X = np.array([[0.0], [1.0], [3.0]])
    y = np.array([-1.0, 1.0, 1.0])
    K, en = _solve_float(X, y)
    assert en.tested == 3 == (2 ** 1 - 1) * (2 ** 2 - 1)  # gemischte Mengen: (2^1 - 1)(2^2 - 1)
    assert len(en.feasible) == 1
    r = en.feasible[0]
    assert r.support == (0, 1) and np.allclose(r.alpha, [2, 2, 0]) and r.b == pytest.approx(-1.0)
    assert M.margin_from_alpha(K, y, r.alpha) == pytest.approx(1.0)
    assert S.dual_value(K, y, r.alpha) == pytest.approx(2.0)  # = ||w||^2 / 2 mit w = 2


# --- Teil 2: fünf Punkte in der Ebene ----------------------------------------------------------------------------------------------------------
def test_fuenf_punkte_aufzaehlung_stimmt_mit_dem_bruch_orakel_ueberein(mini):
    X, y = mini.X, mini.y
    K, en = _solve_float(X, y)
    Xi = [tuple(int(v) for v in r) for r in X]
    yi = [int(v) for v in y]
    sol, tested = enum_frac(gram_frac(Xi), yi)
    assert en.tested == tested == (2 ** 2 - 1) * (2 ** 3 - 1) == 21
    assert len(sol) == 1 and len(en.feasible) == 1
    S0, al, b = sol[0]
    assert S0 == (0, 1, 2) == en.feasible[0].support
    assert al == [F(6, 5), F(2), F(4, 5), F(0), F(0)] and b == -1
    assert np.allclose(en.feasible[0].alpha, [float(a) for a in al], atol=1e-12) and en.feasible[0].b == pytest.approx(-1.0, abs=1e-12)


def test_fuenf_punkte_w_marge_komplementaritaet_dualwert_und_fusspunkte(mini):
    X, y = mini.X, mini.y
    K, en = _solve_float(X, y)
    r = en.feasible[0]
    w, b = M.primal_wb(X, y, r.alpha, r.b)
    assert np.allclose(w, [6 / 5, -8 / 5]) and w @ w == pytest.approx(4.0)  # ||w||^2 = 4, Marge 2/||w|| = 1
    assert M.margin_from_alpha(K, y, r.alpha) == pytest.approx(1.0)
    f = X @ w + b
    assert np.allclose(y * f, [1, 1, 1, 11 / 5, 7 / 5])  # y f = 1 an den Stützvektoren, 11/5 und 7/5 an den anderen
    assert all(abs(a * (yy * ff - 1)) < 1e-12 for a, yy, ff in zip(r.alpha, y, f))  # Komplementarität
    assert abs(r.alpha @ y) < 1e-12 and r.alpha[y > 0].sum() == pytest.approx(2.0) == r.alpha[y < 0].sum()
    assert S.dual_value(K, y, r.alpha) == pytest.approx(2.0)  # Dualwert = ||w||^2/2 = 2/Marge^2
    ap, an = r.alpha * (y > 0), r.alpha * (y < 0)
    fp, fn = (ap @ X) / 2, (an @ X) / 2
    assert np.allclose(fp, [3, 1]) and np.allclose(fn, [12 / 5, 9 / 5]) and np.linalg.norm(fp - fn) == pytest.approx(1.0)


def test_fuenf_punkte_huellenabstand_und_richtungsgitter_geben_marge_eins(mini):
    X, y = mini.X, mini.y
    Xi = [tuple(int(v) for v in r) for r in X]
    yi = [int(v) for v in y]
    assert _hull_dist2(Xi, yi) == 1  # Orakel in Brüchen
    assert M.hull_distance(X, y) == pytest.approx(1.0, abs=1e-12)
    best, th = M.max_margin_by_directions(X, y)
    assert best == pytest.approx(1.0, abs=1e-12)
    u = np.array([math.cos(th), math.sin(th)])
    assert np.allclose(u, [3 / 5, -4 / 5], atol=1e-6)  # Richtung (3, -4)/5, t = -1/2
    assert _dir_margin2(Xi, yi, (3, -4)) == 1
    best_exact = max((m2 for k in range(-400, 401) if (m2 := _dir_margin2(Xi, yi, (1 - F(k, 40) ** 2, 2 * F(k, 40)))) is not None))
    assert best_exact == 1  # über 801 exakt rationale Richtungen ist 1 das Maximum


def test_fuenf_punkte_lp_trennbar_und_smo_gleich_aufzaehlung(mini):
    assert P.lp_separable(mini.X, mini.y) is not None
    s = P.separability(mini.X, mini.y)
    assert s["separable"] and s["certificate"] is None
    r = S.smo(mini.X @ mini.X.T, mini.y, np.inf, eps=1e-11)
    assert np.allclose(r.alpha, [6 / 5, 2, 4 / 5, 0, 0], atol=1e-8) and r.b == pytest.approx(-1.0, abs=1e-8)


# --- Teil 3: Kern tauschen (XOR, Gram-Matrix von Hand) ---------------------------------------------------------------------------------------------
def test_xor_gauss_gram_a_ein_viertel_alpha_16_9_marge_3_4():
    a = F(1, 4)
    Kf = [[F(1), a * a, a, a], [a * a, F(1), a, a], [a, a, F(1), a * a], [a, a, a * a, F(1)]]
    y = [1, 1, -1, -1]
    sol, tested = enum_frac(Kf, y)
    assert tested == 9 and len(sol) == 1 and sol[0][1] == [F(16, 9)] * 4 and sol[0][2] == 0
    assert w2_frac(sol[0][1], y, Kf) == F(64, 9) and F(4) / F(64, 9) == (1 - a) ** 2 == F(9, 16)
    K = np.array([[float(v) for v in r] for r in Kf])
    en = M.enumerate_support_sets(K, np.array(y, float))
    assert len(en.feasible) == 1 and np.allclose(en.feasible[0].alpha, 16 / 9)
    assert M.margin_from_alpha(K, np.array(y, float), en.feasible[0].alpha) == pytest.approx(0.75)
    assert np.linalg.eigvalsh(K).min() == pytest.approx(9 / 16)  # (1 - a)^2


def test_xor_gauss_kern_marge_ist_eins_minus_exp_minus_zwei_durch_sigma_quadrat():
    ins = Sc.xor4()
    for sigma in (0.8, 1.0, 1.6, 3.0):
        K = M.kernel_matrix(ins.X, "gauss", sigma)
        en = M.enumerate_support_sets(K, ins.y)
        assert len(en.feasible) == 1
        assert M.margin_from_alpha(K, ins.y, en.feasible[0].alpha) == pytest.approx(1 - math.exp(-2 / sigma ** 2), rel=1e-10)


# --- Teil 4: nicht trennbar, XOR linear -------------------------------------------------------------------------------------------------------
def test_xor_linear_keine_kkt_loesung_zertifikat_und_lp_unzulaessig():
    ins = Sc.xor4()
    K, en = _solve_float(ins.X, ins.y)
    assert en.tested == 9 and en.feasible == []
    sol, tested = enum_frac(gram_frac([tuple(int(v) for v in r) for r in ins.X]), [1, 1, -1, -1])
    assert tested == 9 and sol == []
    assert P.lp_separable(ins.X, ins.y) is None
    s = P.separability(ins.X, ins.y)
    assert not s["separable"] and s["certificate"] is not None
    assert P.certificate_residual(s["Phi"], ins.y, s["certificate"]) < 1e-9
    assert np.allclose(s["certificate"] / s["certificate"].max(), 1.0)  # alpha = (1, 1, 1, 1) bis auf einen Faktor
    al = np.ones(4)
    assert al @ ins.y == 0 and np.allclose((al * ins.y) @ ins.X, 0) and al @ ((ins.y[:, None] * ins.y[None, :]) * K) @ al == pytest.approx(0.0)  # alpha' Q alpha = 0


def test_xor_linear_weiche_marge_alpha_gleich_c_w_null():
    ins = Sc.xor4()
    r = S.smo(ins.X @ ins.X.T, ins.y, 7.0, eps=1e-9)
    assert r.converged and np.allclose(r.alpha, 7.0) and r.dual == pytest.approx(4 * 7.0)  # Schranke sum alpha = 4 C erreicht
    assert M.margin_from_alpha(ins.X @ ins.X.T, ins.y, r.alpha) is None  # w = 0: die Marge ist nicht definiert


# --- Teil 5: Monotonie, Entfernen, Hinzufügen ------------------------------------------------------------------------------------------------
def _w2(X, y):
    K, en = _solve_float(X, y)
    assert len(en.feasible) == 1
    r = en.feasible[0]
    return float((r.alpha * y) @ K @ (r.alpha * y)), r


def test_entfernen_und_hinzufuegen_in_bruechen(mini):
    X, y = mini.X, mini.y
    w2, r0 = _w2(X, y)
    assert w2 == pytest.approx(4.0)
    keep = [0, 1, 2, 4]  # Nicht-Stützvektor (4,1) entfernen: nichts ändert sich
    w2a, ra = _w2(X[keep], y[keep])
    assert w2a == pytest.approx(4.0) and np.allclose(ra.alpha, [6 / 5, 2, 4 / 5, 0])
    keep = [0, 2, 3, 4]  # Stützvektor (3,1) entfernen: 25/9, Marge 6/5
    w2b, _ = _w2(X[keep], y[keep])
    assert w2b == pytest.approx(25 / 9) and M.hull_distance(X[keep], y[keep]) == pytest.approx(6 / 5)
    keep = [1, 2, 3, 4]  # Stützvektor (4,3) entfernen: 1, Marge 2
    w2c, _ = _w2(X[keep], y[keep])
    assert w2c == pytest.approx(1.0) and M.hull_distance(X[keep], y[keep]) == pytest.approx(2.0)
    p = np.array([3.5, 1.5])  # (7/2, 3/2) mit y = +1: f = 4/5, richtige Seite, in der Marge
    assert 0 < (6 / 5) * p[0] - (8 / 5) * p[1] - 1 < 1
    X2, y2 = np.vstack([X, p]), np.r_[y, 1.0]
    w2d, rd = _w2(X2, y2)
    assert w2d == pytest.approx(400 / 81) and 2 / math.sqrt(w2d) == pytest.approx(9 / 10) and rd.support == (0, 2, 5)
    w_new, _ = M.primal_wb(X2, y2, rd.alpha, rd.b)
    assert M.angle_between_deg(w_new, [6 / 5, -8 / 5]) == pytest.approx(0.0, abs=1e-6)  # w parallel zum alten: keine Drehung


# --- Teil 6: viele trennende Geraden (Perceptron in allen 120 Reihenfolgen) --------------------------------------------------------------------
def test_perceptron_120_reihenfolgen_geben_12_geraden_mit_marge_zwischen_0_263_und_1(mini):
    X, y = mini.X, mini.y
    Xi = [tuple(int(v) for v in r) for r in X]
    yi = [int(v) for v in y]
    lines = {}
    for order in itertools.permutations(range(5)):
        w, b, upd, ep, ok = Pe.perceptron(X, y, list(order))
        assert ok
        lines[(tuple(float(v) for v in w), float(b))] = 1
    assert len(lines) == 12  # 12 verschiedene Geraden (Brüche im Handbeispiel)
    m2 = [_dir_margin2(Xi, yi, w) for (w, b) in lines]
    assert all(v is not None and v <= 1 for v in m2)
    assert float(min(m2)) == pytest.approx(0.0690, abs=1e-4) and float(math.sqrt(min(m2))) == pytest.approx(0.263, abs=1e-3)
    assert sum(1 for v in m2 if v == 1) == 2  # zwei davon treffen die größte Marge


def test_perceptron_kehrt_bei_xor_nicht_zurueck_und_meldet_es():
    ins = Sc.xor4()
    w, b, upd, ep, ok = Pe.perceptron(ins.X, ins.y, [0, 1, 2, 3], max_epochs=50)
    assert not ok and ep == 50 and upd > 0


def test_perceptron_reihenfolgen_sind_je_zufalls_seed_fest_und_verschieden():
    a = Pe.random_orders(10, 4, 3)
    assert a == Pe.random_orders(10, 4, 3) and a != Pe.random_orders(10, 4, 4)
    assert all(sorted(o) == list(range(10)) for o in a) and len({tuple(o) for o in a}) == 4


# --- Teil 7: Entartung ----------------------------------------------------------------------------------------------------------------------
def test_rechteck_zwei_kkt_loesungen_gleiches_w_und_b_verschiedene_alpha():
    ins = Sc.rechteck()
    K, en = _solve_float(ins.X, ins.y)
    sol, _ = enum_frac(gram_frac([tuple(int(v) for v in r) for r in ins.X]), [1, 1, -1, -1])
    assert len(sol) == 2 and len(en.feasible) == 2
    assert {f.support for f in en.feasible} == {(0, 2), (1, 3)}
    ws = {tuple(np.round(M.primal_wb(ins.X, ins.y, f.alpha, f.b)[0], 9)) for f in en.feasible}
    assert ws == {(0.0, 1.0)} and all(f.b == pytest.approx(0.0, abs=1e-12) for f in en.feasible)
    assert not np.allclose(en.feasible[0].alpha, en.feasible[1].alpha)
    mid = 0.5 * (en.feasible[0].alpha + en.feasible[1].alpha)  # das Mittel ist wieder zulässig und gibt dasselbe w
    assert np.allclose((mid * ins.y) @ ins.X, [0, 1]) and abs(mid @ ins.y) < 1e-12


# --- Kerne, Hilfen ----------------------------------------------------------------------------------------------------------------------------
def test_kernmatrizen_gegen_die_formel_von_hand():
    X = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 0.5]])
    assert np.allclose(M.kernel_matrix(X, "linear"), [[5, -2, 4], [-2, 1, -0.5], [4, -0.5, 9.25]])
    assert np.allclose(M.kernel_matrix(X, "poly2"), (X @ X.T + 1) ** 2)
    K = M.kernel_matrix(X, "gauss", 2.0)
    assert np.diag(K) == pytest.approx(1.0) and K[0, 1] == pytest.approx(math.exp(-(1 + 9) / 8)) and np.allclose(K, K.T)
    assert np.allclose(M.kernel_cross(X[:2], X, "gauss", 2.0), K[:2])
    with pytest.raises(ValueError):
        M.kernel_matrix(X, "sigmoid")


def test_poly2_gram_gleich_skalarprodukt_der_expliziten_merkmale():
    X = np.random.default_rng(3).normal(size=(6, 2))
    Phi = P.poly2_features(X)
    assert np.allclose(Phi @ Phi.T, M.kernel_matrix(X, "poly2"), atol=1e-12)


def test_clip_line_und_winkel_hilfen():
    from hmg_visualization import clip_line

    seg = clip_line([1.0, 0.0], 2.0, np.array([0.0, -3.0]), np.array([5.0, 3.0]))  # senkrechte Gerade x = 2
    assert seg is not None and {round(float(seg[0][0]), 9), round(float(seg[1][0]), 9)} == {2.0} and sorted([float(seg[0][1]), float(seg[1][1])]) == [-3.0, 3.0]
    assert clip_line([1.0, 0.0], 9.0, np.array([0.0, 0.0]), np.array([5.0, 3.0])) is None
    assert M.wrap_deg(190.0) == pytest.approx(-170.0) and M.wrap_deg(-180.0) == 180.0
    assert M.angle_between_deg([1, 0], [-1, 0]) == pytest.approx(0.0) and M.angle_between_deg([1, 0], [0, 2]) == pytest.approx(90.0)
