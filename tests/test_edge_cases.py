"""Randfälle der harten Marge (DEMO-PLAYBOOK §12): n = 2, eine Klasse leer, gleiche Punkte, ein Punkt mit beiden Labels, kollineare Punkte, n über 8, Skalen,
Schrittgrenze, Reglerränder."""

import numpy as np
import pytest

import hmg_constants as C
import hmg_evaluation as ev
import hmg_margin as M
import hmg_scenario as Sc
import hmg_separable as P
import hmg_smo as S
from hmg_scenario import Instance


def test_genau_zwei_punkte_marge_ist_der_abstand_alpha_zwei_durch_abstand_zum_quadrat():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    y = np.array([-1.0, 1.0])
    K = X @ X.T
    en = M.enumerate_support_sets(K, y)
    assert en.tested == 1 and len(en.feasible) == 1
    assert np.allclose(en.feasible[0].alpha, 2 / 25) and M.margin_from_alpha(K, y, en.feasible[0].alpha) == pytest.approx(5.0)
    assert M.hull_distance(X, y) == pytest.approx(5.0) and M.max_margin_by_directions(X, y)[0] == pytest.approx(5.0, abs=1e-9)


def test_eine_klasse_leer_meldet_statt_zu_rechnen():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
    y = np.ones(3)
    s = P.separability(X, y)
    assert not s["separable"] and s["reason"] == "eine Klasse fehlt"
    assert M.enumerate_support_sets(X @ X.T, y).tested == 0 and M.enumerate_support_sets(X @ X.T, y).feasible == []
    assert P.one_class(y) and not P.one_class(np.array([1.0, -1.0]))


def test_alle_punkte_gleich_beide_etiketten_sind_nicht_trennbar_in_jedem_kern():
    X = np.zeros((4, 2))
    y = np.array([1.0, -1.0, 1.0, -1.0])
    for kind in ("linear", "poly2", "gauss"):
        s = P.separability(X, y, kind)
        assert not s["separable"] and "beide Labels" in s["reason"]


def test_ein_punkt_mit_beiden_etiketten_ist_in_jedem_kern_nicht_trennbar_und_die_app_meldet_es():
    ins = Sc.widerspruch(11, 5)
    for kern in ("linear", "Polynom Grad 2", "Gauß σ = 1"):
        r = ev.exp4_case(ins, kern, 100, 1e-3, 5000)
        assert not r["sep"]["separable"]
        kind, text = ev.verdict_trap(r)
        assert kind == "error" and "NICHT trennbar" in text and "beide Labels" in text
        assert r["sol"]["violators"] >= 1  # mindestens ein Verletzer: die harte Bedingung kann nicht gelten


def test_kollineare_punkte_auf_einer_geraden_in_2d():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    y = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
    K = X @ X.T
    en = M.enumerate_support_sets(K, y)
    assert len(en.feasible) == 1 and en.feasible[0].support == (1, 2)
    assert M.margin_from_alpha(K, y, en.feasible[0].alpha) == pytest.approx(np.sqrt(2.0))
    assert M.hull_distance(X, y) == pytest.approx(np.sqrt(2.0)) and M.max_margin_by_directions(X, y)[0] == pytest.approx(np.sqrt(2.0), abs=1e-9)


def test_aufzaehlung_ist_ab_neun_punkten_gesperrt_und_sagt_warum():
    ins = Sc.puffer(9, 0.2, 1)
    with pytest.raises(ValueError, match="nur bis n = 8"):
        M.enumerate_support_sets(ins.X @ ins.X.T, ins.y)
    M.enumerate_support_sets(Sc.puffer(8, 0.2, 1).X @ Sc.puffer(8, 0.2, 1).X.T, Sc.puffer(8, 0.2, 1).y)  # genau acht läuft


def test_punkte_weit_auseinander_skalen_1e_minus_6_und_1e6():
    base = Sc.fuenf_punkte()
    for s in (1e-6, 1e6):
        X = base.X * s
        K = X @ X.T
        r = S.smo(K, base.y, np.inf, eps=1e-10)
        assert M.margin_from_alpha(K, base.y, r.alpha) == pytest.approx(s, rel=1e-6)
        assert M.max_margin_by_directions(X, base.y)[0] == pytest.approx(s, rel=1e-9) and M.hull_distance(X, base.y) == pytest.approx(s, rel=1e-9)
        assert P.lp_separable(X / s, base.y) is not None


def test_c_am_reglermaximum_und_schrittgrenze_erreicht_meldet_die_app():
    ins = Sc.gauss_classes(20, 2.0, 13005)
    r = ev.exp4_case(ins, "linear", max(C.E4_C_OPTIONS), 1e-3, 500)  # C = 1e4 und eine zu kleine Grenze
    assert not r["sol"]["converged"] and "Schrittgrenze 500 erreicht" in r["sol"]["message"] and r["sol"]["steps"] == 500
    kind, text = ev.verdict_trap(r)
    assert "Schrittgrenze 500 erreicht" in text and "NICHT trennbar" in text


def test_rechteck_meldet_dass_alpha_nicht_eindeutig_ist_w_und_b_schon():
    r = ev.exp2("Rechteck (α nicht eindeutig)", 1, "linear")
    kind, text = ev.verdict_exp2(r)
    assert len(r["feasible"]) == 2 and "nicht eindeutig" in text and "demselben w und b" in text


def test_xor_linear_meldet_nie_eine_marge_nur_nicht_trennbar_und_w_null():
    r = ev.exp4_xor("linear", 10000, 1e-3, 100000)
    kind, text = ev.verdict_trap(r)
    assert kind == "error" and "NICHT trennbar" in text and "nicht definiert (w = 0)" in text and r["sol"]["margin"] is None
    assert np.allclose(r["sol"]["alpha"], 10000.0)  # alle alpha = C (Handbeispiel Teil 4)
    r2 = ev.exp2("XOR (vier Punkte)", 1, "linear")
    kind2, text2 = ev.verdict_exp2(r2)
    assert kind2 == "warning" and "keine harte Marge" in text2 and r2["feasible"] == []


def test_kernmatrix_mit_gleichen_punkten_gleichen_etiketten_ist_singulaer_aber_die_aufzaehlung_bleibt_stabil():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    y = np.array([-1.0, -1.0, 1.0])
    en = M.enumerate_support_sets(X @ X.T, y)
    assert any(r.status == "singulär" for r in en.rows) or len(en.feasible) >= 1
    assert M.margin_from_alpha(X @ X.T, y, en.feasible[0].alpha) == pytest.approx(2.0)


def test_exp3_randfaelle_punktnummer_ueber_n_wird_begrenzt_und_falsche_seite_ist_nicht_trennbar():
    r = ev.exp3(C.E3_SETS[1], 20, 0.25, 7, 40, 0.8, 3.7)  # Punkt 40 gibt es bei fünf Punkten nicht: auf Punkt 5 begrenzt
    assert r["k"] == 4
    r0 = ev.exp3(C.E3_SETS[1], 20, 0.25, 7, 2, 0.0, 0.0)  # Abstand 0: der neue Punkt liegt auf der Geraden, die Bedingung y f >= 1 ist nur mit kleinerer Marge erfüllbar
    assert r0["add"] is not None and r0["add"]["margin"] < r0["base"]["margin"]


def test_regler_ecken_der_experimente_rechnen_ohne_fehler():
    for n in (C.N1_MIN, C.N1_MAX):
        for pf in (C.PUFFER_OPTIONS[0], C.PUFFER_OPTIONS[-1]):
            r = ev.exp1(n, pf, 1, C.THETA_MIN, C.PERC_MAX)
            assert r["m_max"] > 0 and all(ln["converged"] for ln in r["lines"])
            r = ev.exp1(n, pf, 1, C.THETA_MAX, 0)
            assert r["m_max"] > 0 and r["lines"] == []
    for a in (C.E4_ABSTAND_OPTIONS[0], C.E4_ABSTAND_OPTIONS[-1]):
        for c in (1, C.E4_C_OPTIONS[-1]):
            r = ev.exp4_overlap(C.E4_N_MAX, a, 1, "linear", c, 1e-3, 20000)
            assert r["sol"]["steps"] <= 20000


def test_instanz_hilfen_unveraendert_nach_dem_rechnen():
    ins = Sc.fuenf_punkte()
    X0 = ins.X.copy()
    ev.exp2(C.E2_SETS[0], 1, "linear")
    ev.exp3(C.E3_SETS[1], 20, 0.25, 1, 2, 0.8, 3.7)
    assert np.array_equal(ins.X, X0) and isinstance(ins, Instance)
