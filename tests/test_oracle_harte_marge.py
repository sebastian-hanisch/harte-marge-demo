"""Orakel-Tests der harten Marge: jeder Rechenweg der App gegen einen anderen (ORACLE-PLAYBOOK §2, DEMO-PLAYBOOK §12).

Wege: (1) Aufzählung der Stützmengen (je Menge ein lineares KKT-System), (2) Abstand der konvexen Hüllen, (3) Richtungsgitter, (4) eigenes SMO, (5) SLSQP-Primal und
`SVC` als Fremdreferenz (`importorskip`). Trennbarkeit: lineares Programm gegen das Zertifikat nach Gordan (genau eins von beiden). Die großen Läufe (200 Instanzen)
stehen in `tools/hmg_make_b2.py` und `hmg_b2.json`; hier die kleine, schnelle Fassung."""

import math

import numpy as np
import pytest

import hmg_evaluation as ev
import hmg_margin as M
import hmg_scenario as Sc
import hmg_separable as P
import hmg_smo as S


def _instances(n_inst=40):
    for t in range(n_inst):
        yield Sc.puffer(4 + (t % 5), 0.2, 11000 + t)


def test_vier_wege_der_harten_marge_stimmen_ueberein_40_instanzen():
    for ins in _instances():
        X, y = ins.X, ins.y
        K = X @ X.T
        en = M.enumerate_support_sets(K, y)
        assert len(en.feasible) == 1, ins.info  # in Allgemeinlage genau eine KKT-Lösung
        r = en.feasible[0]
        m0 = M.margin_from_alpha(K, y, r.alpha)
        assert len(r.support) in (2, 3)  # in 2D zwei oder drei Stützvektoren
        assert M.max_margin_by_directions(X, y)[0] == pytest.approx(m0, rel=1e-9)
        assert M.hull_distance(X, y) == pytest.approx(m0, rel=1e-9)
        s = S.smo(K, y, np.inf, eps=1e-10)
        assert M.margin_from_alpha(K, y, s.alpha) == pytest.approx(m0, rel=1e-7)
        assert S.dual_value(K, y, r.alpha) == pytest.approx(2.0 / m0 ** 2, rel=1e-9)  # Dualwert = ||w||^2/2 = 2/Marge^2
        ap, an = r.alpha * (y > 0), r.alpha * (y < 0)
        assert ap.sum() == pytest.approx(an.sum(), rel=1e-9)  # sum alpha_+ = sum alpha_-
        assert np.linalg.norm((ap @ X) / ap.sum() - (an @ X) / an.sum()) == pytest.approx(m0, rel=1e-9)  # Fußpunkte der Hüllen
        w, b = M.primal_wb(X, y, r.alpha, r.b)
        assert (y * (X @ w + b) >= 1 - 1e-9).all() and np.abs(r.alpha * (y * (X @ w + b) - 1)).max() < 1e-9  # zulässig, Komplementarität


def test_marge_gegen_svc_und_slsqp_primal_20_instanzen():
    svm = pytest.importorskip("sklearn.svm")
    from scipy import optimize

    for ins in list(_instances(20)):
        X, y = ins.X, ins.y
        K = X @ X.T
        m0 = M.margin_from_alpha(K, y, M.enumerate_support_sets(K, y).feasible[0].alpha)
        clf = svm.SVC(kernel="linear", C=1e4, tol=1e-10).fit(X, y)
        assert 2 / np.linalg.norm(clf.coef_[0]) == pytest.approx(m0, rel=1e-5)
        z0 = P.lp_separable(X, y)
        d = X.shape[1]
        cons = {"type": "ineq", "fun": lambda z: y * (X @ z[:d] + z[d]) - 1.0, "jac": lambda z: y[:, None] * np.c_[X, np.ones(len(y))]}
        res = optimize.minimize(lambda z: 0.5 * z[:d] @ z[:d], np.r_[z0[0], z0[1]], jac=lambda z: np.r_[z[:d], 0.0], constraints=[cons], method="SLSQP", options={"ftol": 1e-15, "maxiter": 500})
        assert 2 / np.linalg.norm(res.x[:d]) == pytest.approx(m0, rel=1e-6)


def test_aufzaehlung_gegen_svc_mit_gauss_kern_auf_trennbaren_instanzen():
    svm = pytest.importorskip("sklearn.svm")
    for t in range(8):
        ins = Sc.gauss_classes(7, 1.0, 200 + t)  # Gauß-Kern: verschiedene Punkte sind immer trennbar
        K = M.kernel_matrix(ins.X, "gauss", 1.0)
        en = M.enumerate_support_sets(K, ins.y)
        assert len(en.feasible) == 1
        m0 = M.margin_from_alpha(K, ins.y, en.feasible[0].alpha)
        clf = svm.SVC(kernel="precomputed", C=1e6, tol=1e-10).fit(K, ins.y)
        a = np.zeros(7)
        a[clf.support_] = np.abs(clf.dual_coef_[0])
        assert M.margin_from_alpha(K, ins.y, a) == pytest.approx(m0, rel=1e-5)


def test_trennbarkeit_lp_gegen_zertifikat_genau_eins_von_beiden():
    cases = [Sc.xor4(), Sc.rechteck(), Sc.fuenf_punkte()]
    cases += [Sc.gauss_classes(8 + t % 5, 0.5 * (t % 7), 300 + t) for t in range(30)] + [Sc.puffer(8, 0.1, 500 + t) for t in range(10)]
    n_sep = n_not = 0
    for ins in cases:
        lp = P.lp_separable(ins.X, ins.y)
        cert = P.gordan_certificate(ins.X, ins.y)
        assert (lp is None) != (cert is None), ins.name  # genau eins von beiden
        if lp is not None:
            w, b = lp
            assert (ins.y * (ins.X @ w + b) >= 1 - 1e-7).all()
            n_sep += 1
        else:
            assert (cert >= -1e-12).all() and cert.sum() == pytest.approx(1.0) and P.certificate_residual(ins.X, ins.y, cert) < 1e-8
            n_not += 1
    assert n_sep >= 10 and n_not >= 10  # beide Zweige laufen wirklich


def test_gauss_kern_trennbar_wenn_verschieden_nie_bei_widerspruch_und_kc_gleich_y():
    for t in range(15):
        ins = Sc.gauss_classes(12, 0.0, 700 + t)  # vollständig überlappend, trotzdem im Merkmalsraum trennbar
        assert P.separability(ins.X, ins.y, "gauss")["separable"]
        K = M.kernel_matrix(ins.X, "gauss", 1.0)
        c = np.linalg.solve(K, ins.y)
        assert np.abs(K @ c - ins.y).max() < 1e-6  # K c = y löst y_i f(x_i) = 1 für alle i (b = 0)
    w = Sc.widerspruch(11, 3)
    assert P.duplicate_conflict(w.X, w.y) and not P.separability(w.X, w.y, "gauss")["separable"]
    assert not P.separability(w.X, w.y, "linear")["separable"] and not P.separability(w.X, w.y, "poly2")["separable"]


def test_poly2_trennt_die_ringe_und_das_liefergebiet_aber_nicht_das_xor_mit_linearem_kern():
    r = Sc.rings(30, 5)
    assert not P.separability(r.X, r.y, "linear")["separable"] and P.separability(r.X, r.y, "poly2")["separable"]
    lg = Sc.liefergebiet(60, 5)
    assert not P.separability(lg.X, lg.y, "linear")["separable"] and P.separability(lg.X, lg.y, "poly2")["separable"]
    assert P.separability(Sc.xor4().X, Sc.xor4().y, "poly2")["separable"]


def test_exp1_marge_des_smo_gleich_gitter_gleich_huelle_und_kennzahlen_der_anzeige():
    for seed in range(12):
        r = ev.exp1(15, 0.25, seed, 10, 4)
        assert r["m_max"] == pytest.approx(r["grid_max"], rel=1e-8) and r["m_max"] == pytest.approx(r["hull"], rel=1e-8)
        assert r["m_sel"] == pytest.approx(M.direction_margin(r["inst"].X, r["inst"].y, r["th_star"] + math.radians(10)))
        assert r["m_sel"] <= r["m_max"] + 1e-12
        assert 2 <= len(r["sv"]) <= 3
        for ln in r["lines"]:
            assert ln["converged"] and ln["margin"] <= r["m_max"] + 1e-9 and 0 < ln["ratio"] <= 1 + 1e-9  # keine trennende Gerade hat mehr Marge
            assert (r["inst"].y * (r["inst"].X @ ln["w"] + ln["b"]) > 0).all()  # jede Perceptron-Gerade trennt
        # die Kurve: gleicher Wert wie die direkte Rechnung an drei Winkeln
        for k in (30, 150, 300):
            assert r["curve"][k] == pytest.approx(M.direction_margin(r["inst"].X, r["inst"].y, math.radians(r["thetas_deg"][k])))


def test_exp2_und_exp5_kennzahlen_gegen_direkte_rechnung():
    r = ev.exp2("Fünf Punkte in der Ebene (Handbeispiel)", 7, "linear")
    assert len(r["feasible"]) == 1 and r["margin"] == pytest.approx(1.0) and r["dual"] == pytest.approx(2.0) and r["hull"] == pytest.approx(1.0) and r["grid"] == pytest.approx(1.0, abs=1e-9)
    assert r["solution"]["margin"] == pytest.approx(1.0, abs=1e-6) and r["enum"].tested == 21
    r5 = ev.exp5("Ringe mit Lücke", 3, "Gauß σ = 0,8")
    sol = r5["sol"]
    ya = sol["alpha"] * r5["inst"].y
    assert sol["margin"] == pytest.approx(2 / math.sqrt(ya @ sol["K"] @ ya)) and r5["sv_share"] == pytest.approx(len(sol["sv"]) / 60)
    f = sol["K"] @ ya + sol["b"]
    assert r5["yf_min"] == pytest.approx(float((r5["inst"].y * f).min())) and r5["yf_min"] > 1 - 1e-4  # harte Bedingung im Trainingsbereich erreicht
    te = Sc.rings(1000, 3 + 500)
    ft = M.kernel_cross(te.X, r5["inst"].X, "gauss", 0.8) @ ya + sol["b"]
    assert r5["test_error"] == pytest.approx(float((np.sign(ft) != te.y).mean())) and r5["n_test"] == 2000


def test_exp4_kennzahlen_verletzer_defizit_und_schranke_gegen_direkte_rechnung():
    r = ev.exp4_overlap(20, 2.0, 13005, "linear", 100, 1e-3, 100000)
    sol, ins = r["sol"], r["inst"]
    f = ins.X @ ((sol["alpha"] * ins.y) @ ins.X) + sol["b"]
    yf = ins.y * f
    assert sol["violators"] == int((yf < 1 - 1e-6).sum()) and sol["errors"] == int((yf <= 0).sum()) and sol["defizit"] == pytest.approx(max(0.0, 1 - yf.min()))
    assert sol["at_C"] == int((sol["alpha"] >= 100 * (1 - 1e-9)).sum()) and not r["sep"]["separable"]
    assert sol["margin"] == pytest.approx(2 / np.linalg.norm((sol["alpha"] * ins.y) @ ins.X))
    svm = pytest.importorskip("sklearn.svm")
    clf = svm.SVC(kernel="linear", C=100, tol=1e-3).fit(ins.X, ins.y)
    ref = np.abs(clf.dual_coef_[0]).sum() - 0.5 * float(np.linalg.norm(clf.coef_[0])) ** 2
    assert sol["dual"] == pytest.approx(ref, rel=1e-2)  # beide Löser auf der Toleranz 1e-3: gleicher Dualwert bis etwa ein Prozent
