# -*- coding: utf-8 -*-
"""Messwerkzeug: rechnet die eingefrorenen Studien der Demo neu und schreibt/prüft `hmg_b2.json` (nur Werkzeug, scikit-learn nur hier und in den Tests).

Aufruf (mit dem Test-venv, OMP_NUM_THREADS=1):
    python tools/hmg_make_b2.py A B2 E ...        # nur diese Abschnitte rechnen, Teilergebnis nach tools/hmg_parts/<Abschnitt>.json
    python tools/hmg_make_b2.py --merge           # Teilergebnisse zu hmg_b2.json zusammenführen
    python tools/hmg_make_b2.py --check           # eine kleine Teilmenge je Abschnitt neu rechnen und gegen hmg_b2.json halten (Exit 1 bei Abweichung)

Abschnitte (Zähleinheit je Abschnitt im Ergebnis genannt: Instanzen, Zufalls-Seeds, Aktualisierungsschritte):
  A   Orakelkette: Aufzählung = Richtungsgitter = Hüllenabstand = SVC = Primal = Dual = eigenes SMO (n = 4 bis 8, zwei Sätze zu 200 Instanzen)
  B1  XOR linear (vier Punkte): SVC und eigenes SMO bei C = 1, 1e2, 1e4
  B2  überlappende Klassen, nicht trennbar: SVC linear bei C = 1, 1e2, 1e4, 1e6 (drei Sätze zu 100 Instanzen), dazu das eigene SMO (live-Modell)
  B3  dieselben Instanzen mit Gauß-Kern sigma = 1
  B4  derselbe Punkt mit beiden Etiketten
  C   Monotonie und Empfindlichkeit (200 Instanzen, n = 20)
  D   Stützvektoren zählen: Löser und Schwelle (zwei Sätze zu 100 Instanzen, n = 8)
  E   Perceptron gegen größte Marge (drei Sätze zu 200 Instanzen, n = 20, 8 Reihenfolgen, 4 000 Testpunkte)
  G   Gauß-Kern auf zwei Ringen (20 Zufalls-Seeds, 7 Bandbreiten)

Alle Instanzen kommen aus den Erzeugern der Demo (SplitMix64, `hmg_scenario.py`); die Vormessung der Planung lief mit `numpy.random.default_rng`, die Zahlen hier
sind die der App-Ströme (Bänder über Zufalls-Seeds, nicht dieselben Zahlen wie die Vormessung). Keine Sekunden als Befund: die Laufzeit steht nicht im Ergebnis.
"""
import itertools
import json
import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import optimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hmg_margin as M  # noqa: E402
import hmg_perceptron as Pe  # noqa: E402
import hmg_scenario as Sc  # noqa: E402
import hmg_separable as P  # noqa: E402
import hmg_smo as S  # noqa: E402
from hmg_rng import SplitMix64  # noqa: E402

PARTS = ROOT / "tools" / "hmg_parts"
TARGET = ROOT / "hmg_b2.json"


def _svm():
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import SVC

    return SVC, ConvergenceWarning


def svc_lin(X, y, C=1e4, tol=1e-10, max_iter=-1):
    SVC, CW = _svm()
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter("always")
        m = SVC(kernel="linear", C=C, tol=tol, max_iter=max_iter).fit(X, y)
    return m, any(issubclass(w.category, CW) for w in W)


def svc_pre(K, y, C=1e4, tol=1e-10, max_iter=-1):
    SVC, CW = _svm()
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter("always")
        m = SVC(kernel="precomputed", C=C, tol=tol, max_iter=max_iter).fit(K, y)
    a = np.zeros(len(y))
    a[m.support_] = np.abs(m.dual_coef_[0])
    return a, float(m.intercept_[0]), m, any(issubclass(w.category, CW) for w in W)


def alpha_from(m, n):
    a = np.zeros(n)
    a[m.support_] = np.abs(m.dual_coef_[0])
    return a


def Qmat(K, y):
    return (y[:, None] * y[None, :]) * K


def qp_slsqp(Q, y, x0=None, ftol=1e-15, maxiter=1000):
    n = len(y)
    f = lambda a: 0.5 * a @ Q @ a - a.sum()
    g = lambda a: Q @ a - 1.0
    cons = [{"type": "eq", "fun": lambda a: y @ a, "jac": lambda a: y}]
    opts = {"ftol": ftol, "maxiter": maxiter} if ftol is not None else {}
    return optimize.minimize(f, np.zeros(n) if x0 is None else x0, jac=g, bounds=[(0.0, None)] * n, constraints=cons, method="SLSQP", options=opts)


def qp_trust(Q, y, tight=True):
    n = len(y)
    f = lambda a: 0.5 * a @ Q @ a - a.sum()
    g = lambda a: Q @ a - 1.0
    lc = optimize.LinearConstraint(y[None, :], 0.0, 0.0)
    x0 = np.where(y > 0, 0.5 * (y < 0).sum() / max(1, (y > 0).sum()), 0.5)
    opts = {"gtol": 1e-12, "xtol": 1e-14, "maxiter": 3000} if tight else {}
    return optimize.minimize(f, x0, jac=g, hess=lambda a: Q, bounds=optimize.Bounds(np.zeros(n), np.full(n, np.inf)), constraints=[lc], method="trust-constr", options=opts)


def primal_slsqp(X, y):
    n, d = X.shape
    z0 = P.lp_separable(X, y)
    z0 = np.r_[z0[0], z0[1]]
    cons = {"type": "ineq", "fun": lambda z: y * (X @ z[:d] + z[d]) - 1.0, "jac": lambda z: y[:, None] * np.c_[X, np.ones(n)]}
    return optimize.minimize(lambda z: 0.5 * z[:d] @ z[:d], z0, jac=lambda z: np.r_[z[:d], 0.0], constraints=[cons], method="SLSQP", options={"ftol": 1e-15, "maxiter": 500})


def hard_smo(X, y, eps=1e-10, max_steps=200000):
    return S.smo(X @ X.T, y, np.inf, eps=eps, max_steps=max_steps)


def ck(C):
    """Schlüssel für C im Ergebnis: ganze Zahl als Text (1, 100, 10000, 1000000)."""
    return str(int(C))


def med(a):
    return float(np.median(a))


def nonsep_set(seed0, count=100, n=20, sep=2.0):
    """Instanzen aus den Gauß-Klassen, die das lineare Programm als nicht trennbar bestätigt; zählt die gezogenen Zufalls-Seeds."""
    out, seed, tried = [], seed0, 0
    while len(out) < count:
        ins = Sc.gauss_classes(n, sep, seed)
        tried += 1
        seed += 1
        if P.lp_separable(ins.X, ins.y) is None:
            out.append(ins)
    return out, tried


# ================================================================== A
def sec_A(n_inst=200, seeds=(11000, 61000)):
    res = {"einheit": "Instanzen; je Satz 200 trennbare Instanzen, n = 4 bis 8, Erzeuger Zwei Gebiete mit Puffer (Lücke 0,2), linearer Kern", "saetze": {}}
    for label, seed0 in zip("AB", seeds):
        dev = {k: 0.0 for k in ("grid", "hull", "svc", "primal", "dual_slsqp", "smo", "dual_vs_2_over_m2", "fusspunkte")}
        uniq = 0
        nsv = {}
        prim_success = prim_ok = dual_success = 0
        comp = sums = 0.0
        for t in range(n_inst):
            n = 4 + (t % 5)
            ins = Sc.puffer(n, 0.2, seed0 + t)
            X, y = ins.X, ins.y
            K = X @ X.T
            en = M.enumerate_support_sets(K, y)
            if len(en.feasible) != 1:
                continue
            uniq += 1
            r = en.feasible[0]
            alpha, b = r.alpha, r.b
            w = (alpha * y) @ X
            m0 = 2.0 / np.linalg.norm(w)
            nsv[len(r.support)] = nsv.get(len(r.support), 0) + 1
            mg, _ = M.max_margin_by_directions(X, y)
            dev["grid"] = max(dev["grid"], abs(mg - m0) / m0)
            dev["hull"] = max(dev["hull"], abs(M.hull_distance(X, y) - m0) / m0)
            m, _ = svc_lin(X, y, 1e4, 1e-10)
            dev["svc"] = max(dev["svc"], abs(2.0 / np.linalg.norm(m.coef_[0]) - m0) / m0)
            res_p = primal_slsqp(X, y)
            prim_success += int(res_p.success)
            err = abs(2.0 / np.linalg.norm(res_p.x[:2]) - m0) / m0
            dev["primal"] = max(dev["primal"], err)
            prim_ok += int(err < 1e-6)
            Q = Qmat(K, y)
            r2 = qp_slsqp(Q, y)
            dual_success += int(r2.success)
            a2 = np.where(r2.x < 1e-7 * max(1.0, alpha.max()), 0.0, r2.x)
            dev["dual_slsqp"] = max(dev["dual_slsqp"], abs(2.0 / np.linalg.norm((a2 * y) @ X) - m0) / m0)
            s3 = hard_smo(X, y)
            dev["smo"] = max(dev["smo"], abs(2.0 / np.linalg.norm((s3.alpha * y) @ X) - m0) / m0)
            W = S.dual_value(K, y, alpha)
            dev["dual_vs_2_over_m2"] = max(dev["dual_vs_2_over_m2"], abs(W - 2.0 / m0 ** 2) / W)
            ap, an = alpha * (y > 0), alpha * (y < 0)
            dev["fusspunkte"] = max(dev["fusspunkte"], abs(np.linalg.norm((ap @ X) / ap.sum() - (an @ X) / an.sum()) - m0) / m0)
            sums = max(sums, abs(ap.sum() - an.sum()) / max(1.0, ap.sum()))
            comp = max(comp, float(np.abs(alpha * (y * (X @ w + b) - 1.0)).max()))
        res["saetze"][label] = dict(seed0=seed0, instanzen=n_inst, eindeutig=uniq, stuetzvektoren=nsv, abweichung=dev, primal_success=prim_success, primal_richtig=prim_ok,
                                    dual_success=dual_success, gewichtssummen=sums, komplementaritaet=comp)
    return res


# ================================================================== B1
def sec_B1():
    ins = Sc.xor4()
    X, y = ins.X, ins.y
    cert = P.gordan_certificate(X, y)
    res = {"einheit": "XOR linear, vier Punkte", "lp_trennbar": P.lp_separable(X, y) is not None, "zertifikat": [float(v) for v in cert * 4], "svc": {}, "smo": {}}
    for C in (1.0, 1e2, 1e4):
        m, warned = svc_lin(X, y, C, 1e-3, 100000)
        res["svc"][ck(C)] = dict(w=[float(v) for v in m.coef_[0]], b=float(m.intercept_[0]), alpha=[float(v) for v in np.abs(m.dual_coef_[0])], n_iter=int(m.n_iter_[0]), warnung=bool(warned))
        r = S.smo(X @ X.T, y, C, eps=1e-3, max_steps=100000)
        res["smo"][ck(C)] = dict(alpha=[float(v) for v in r.alpha], w=[float(v) for v in (r.alpha * y) @ X], steps=int(r.steps), converged=bool(r.converged))
    return res


# ================================================================== B2 und B3 (dieselben Instanzen)
def sec_B2(count=100, seeds=(13000, 33000, 43000), smo_count=30):
    res = {"einheit": "Instanzen und Aktualisierungsschritte; je Satz 100 nicht trennbare Instanzen (lineares Programm unzulässig), n = 20, Gauß-Klassen mit Abstand 2; SVC linear, tol 1e-3, max_iter 2 000 000",
           "saetze": {}, "paarweise": {}, "tol1e8": {}, "smo": {}}
    sets = {}
    for label, seed0 in zip("ABC", seeds):
        insts, tried = nonsep_set(seed0, count)
        sets[label] = insts
        entry = dict(seed0=seed0, gezogen=tried, nicht_trennbar=len(insts), c={})
        for C in (1.0, 1e2, 1e4, 1e6):
            warned_n = silent = zero_w = 0
            viol, iters, margins, duals = [], [], [], []
            for ins in insts:
                m, warned = svc_lin(ins.X, ins.y, C, 1e-3, 2_000_000)
                w = m.coef_[0]
                f = ins.X @ w + float(m.intercept_[0])
                v = int((ins.y * f < 1 - 1e-6).sum())
                warned_n += int(warned)
                silent += int(v > 0 and not warned)
                viol.append(v)
                nw = float(np.linalg.norm(w))
                margins.append(2.0 / nw if nw > 1e-12 else float("nan"))
                zero_w += int(nw <= 1e-12)
                iters.append(int(m.n_iter_[0]))
                duals.append(float(np.abs(m.dual_coef_[0]).sum() - 0.5 * nw ** 2))
            entry["c"][ck(C)] = dict(warnung=warned_n, ohne_warnung_mit_verletzern=silent, verletzer_median=med(viol), verletzer_min=min(viol), verletzer_max=max(viol),
                                        schritte_median=med(iters), marge_median=float(np.nanmedian(margins)), dual_median=med(duals), w_null=zero_w,
                                        verletzer=viol, schritte=iters, marge=margins)
        res["saetze"][label] = entry
    # paarweiser Vergleich der Marge je Instanz
    for c1, c2 in (("1", "100"), ("100", "10000"), ("10000", "1000000")):
        key = f"{c1}->{c2}"
        for lb in list(sets) + ["alle"]:
            if lb == "alle":
                a = np.concatenate([res["saetze"][k]["c"][c1]["marge"] for k in sets])
                b = np.concatenate([res["saetze"][k]["c"][c2]["marge"] for k in sets])
            else:
                a = np.array(res["saetze"][lb]["c"][c1]["marge"])
                b = np.array(res["saetze"][lb]["c"][c2]["marge"])
            r = (b / a)
            r = r[np.isfinite(r)]
            res["paarweise"].setdefault(key, {})[lb] = dict(median=float(np.median(r)), unter_1_prozent=float(np.mean(np.abs(r - 1) < 0.01)), sinkt_ueber_5_prozent=float(np.mean(r < 0.95)),
                                                           steigt_ueber_5_prozent=float(np.mean(r > 1.05)), instanzen=int(len(r)))
    for lb in ("A", "B"):
        rr = []
        for ins in sets[lb]:
            m1, _ = svc_lin(ins.X, ins.y, 1e2, 1e-8, 5_000_000)
            m2, _ = svc_lin(ins.X, ins.y, 1e4, 1e-8, 5_000_000)
            rr.append(float(np.linalg.norm(m1.coef_[0]) / np.linalg.norm(m2.coef_[0])))
        rr = np.array(rr)
        res["tol1e8"][lb] = dict(median=float(np.median(rr)), unter_1_prozent=float(np.mean(np.abs(rr - 1) < 0.01)), sinkt_ueber_5_prozent=float(np.mean(rr < 0.95)))
    # eigenes SMO (Live-Modell der App): die ersten smo_count Instanzen von Satz A, eps 1e-3, Schrittgrenze 200 000
    for C in (1.0, 1e2, 1e4):
        steps, conv, viol, ratio = [], [], [], []
        for k, ins in enumerate(sets["A"][:smo_count]):
            r = S.smo(ins.X @ ins.X.T, ins.y, C, eps=1e-3, max_steps=200000)
            w = (r.alpha * ins.y) @ ins.X
            f = ins.X @ w + r.b
            steps.append(int(r.steps))
            conv.append(bool(r.converged))
            viol.append(int((ins.y * f < 1 - 1e-6).sum()))
            m_svc = res["saetze"]["A"]["c"][ck(C)]["marge"][k]
            ratio.append(float(2.0 / np.linalg.norm(w) / m_svc) if np.linalg.norm(w) > 1e-12 else float("nan"))
        res["smo"][ck(C)] = dict(instanzen=smo_count, schritte_median=med(steps), schritte_min=min(steps), schritte_max=max(steps), nicht_konvergiert=int(sum(1 for c in conv if not c)),
                                    mit_verletzern=int(sum(1 for v in viol if v > 0)), marge_verhaeltnis_median=float(np.nanmedian(ratio)), marge_verhaeltnis_min=float(np.nanmin(ratio)),
                                    marge_verhaeltnis_max=float(np.nanmax(ratio)), schritte=steps)
    # B3: Gauß-Kern sigma = 1 auf denselben Instanzen
    b3 = {"einheit": "Instanzen; Gauß-Kern sigma = 1, SVC mit vorberechneter Gram-Matrix, C = 1e4, tol 1e-10, max_iter 2 000 000; dieselben 300 Instanzen", "saetze": {}}
    for lb, insts in sets.items():
        viol, deficits, loose, at_bound, lam_min, sv_share, margins = [], [], [], [], [], [], []
        interp_ok = 0
        warned_n = 0
        for ins in insts:
            K = M.kernel_matrix(ins.X, "gauss", 1.0)
            a, b, m, warned = svc_pre(K, ins.y, 1e4, 1e-10, 2_000_000)
            f = K @ (a * ins.y) + b
            viol.append(int((ins.y * f < 1 - 1e-3).sum()))
            loose.append(int((ins.y * f < 1 - 1e-5).sum()))
            deficits.append(float(max(0.0, 1.0 - (ins.y * f).min())))
            at_bound.append(int((a >= 1e4 * (1 - 1e-9)).sum()))
            warned_n += int(warned)
            sv_share.append(float((a > 1e-8 * max(1.0, a.max())).mean()))
            margins.append(2.0 / math.sqrt(float((a * ins.y) @ K @ (a * ins.y))))
            lam_min.append(float(np.linalg.eigvalsh(K).min()))
            c = np.linalg.solve(K, ins.y)
            interp_ok += int(np.abs(K @ c - ins.y).max() < 1e-6)
        nv = sum(1 for v in viol if v > 0)
        b3["saetze"][lb] = dict(erreicht_haerte=100 - nv, verletzung=nv, verletzung_1e5=sum(1 for v in loose if v > 0), warnung=warned_n, positiv_definit=sum(1 for v in lam_min if v > 0),
                                lam_min=min(lam_min), kc_gleich_y=interp_ok, schranke_median=(med([a_ for v, a_ in zip(viol, at_bound) if v > 0]) if nv else 0),
                                defizit_max=max(deficits), anteil_sv_median=med(sv_share), marge_median=med(margins), marge_min=min(margins), marge_max=max(margins))
    res["B3"] = b3
    return res


# ================================================================== B4
def sec_B4(count=30):
    viol, iters, nsat, warned_n = [], [], [], 0
    for t in range(count):
        ins = Sc.widerspruch(11, 14000 + t)
        K = M.kernel_matrix(ins.X, "gauss", 1.0)
        a, b, m, warned = svc_pre(K, ins.y, 1e4, 1e-10, 200_000)
        f = K @ (a * ins.y) + b
        warned_n += int(warned)
        viol.append(int((ins.y * f < 1 - 1e-5).sum()))
        iters.append(int(m.n_iter_[0]))
        nsat.append(int((a >= 1e4 * (1 - 1e-9)).sum()))
    return dict(einheit="Instanzen; Gauß-Kern sigma = 1, 11 Punkte (10 aus Gauß-Klassen mit Abstand 4 plus Duplikat des ersten Punkts mit umgekehrtem Etikett), C = 1e4, tol 1e-10, max_iter 200 000",
                instanzen=count, warnung=warned_n, verletzer_median=med(viol), verletzer_min=min(viol), verletzer_max=max(viol), schritte_median=med(iters), schritte_min=min(iters),
                schritte_max=max(iters), schranke_median=med(nsat))


# ================================================================== C
def sec_C(n_inst=200, seed0=15000):
    rng = SplitMix64(15999)
    c1_dw = c1_dm = 0.0
    c1_n = 0
    c2, c3, c4r, c4a = [], [], [], []
    c4_sv_change = 0
    c4_n = 0
    c2_viol = c3_viol = 0
    for t in range(n_inst):
        ins = Sc.puffer(20, 0.2, seed0 + t)
        X, y = ins.X, ins.y
        m, _ = svc_lin(X, y, 1e4, 1e-10)
        w0, b0 = m.coef_[0].copy(), float(m.intercept_[0])
        m0 = 2.0 / np.linalg.norm(w0)
        a0 = alpha_from(m, 20)
        sv = np.where(a0 > 1e-7 * a0.max())[0]
        nsv = np.setdiff1d(np.arange(20), sv)
        if len(nsv):
            i = int(nsv[rng.below(len(nsv))])
            keep = np.arange(20) != i
            m1, _ = svc_lin(X[keep], y[keep], 1e4, 1e-10)
            c1_dw = max(c1_dw, float(np.abs(m1.coef_[0] - w0).max()))
            c1_dm = max(c1_dm, abs(2.0 / np.linalg.norm(m1.coef_[0]) - m0) / m0)
            c1_n += 1
        i = int(sv[rng.below(len(sv))])
        keep = np.arange(20) != i
        m2, _ = svc_lin(X[keep], y[keep], 1e4, 1e-10)
        r = (2.0 / np.linalg.norm(m2.coef_[0])) / m0
        c2.append(r)
        c2_viol += int(r < 1 - 1e-8)
        lo, hi = X.min(0), X.max(0)
        X3 = y3 = None
        for _try in range(50):
            p = lo + (hi - lo) * np.array([rng.random(), rng.random()])
            labs = [1.0, -1.0] if rng.random() < 0.5 else [-1.0, 1.0]
            for lab in labs:
                Xc, yc = np.vstack([X, p]), np.r_[y, lab]
                if P.lp_separable(Xc, yc) is not None:
                    X3, y3 = Xc, yc
                    break
            if X3 is not None:
                break
        m3, _ = svc_lin(X3, y3, 1e4, 1e-10)
        r3 = (2.0 / np.linalg.norm(m3.coef_[0])) / m0
        c3.append(r3)
        c3_viol += int(r3 > 1 + 1e-8)
        wn = w0 / np.linalg.norm(w0)
        tdir = np.array([-wn[1], wn[0]])
        cpt = -b0 * w0 / (w0 @ w0)
        tpos = -2.0 + 4.0 * rng.random()
        p = cpt + tpos * tdir + 0.25 * (m0 / 2.0) * wn
        X4, y4 = np.vstack([X, p]), np.r_[y, 1.0]
        if P.lp_separable(X4, y4) is not None:
            m4, _ = svc_lin(X4, y4, 1e4, 1e-10)
            c4r.append((2.0 / np.linalg.norm(m4.coef_[0])) / m0)
            c4a.append(M.angle_between_deg(m4.coef_[0], w0))
            a4 = alpha_from(m4, 21)
            sv4 = set(np.where(a4 > 1e-7 * a4.max())[0]) - {20}
            c4_sv_change += int(set(sv) != sv4 or a4[-1] > 1e-7 * a4.max())
            c4_n += 1
    q = lambda a, p_: float(np.percentile(a, p_))
    return dict(einheit="Instanzen; 200 trennbare Instanzen, n = 20, Zwei Gebiete mit Puffer (Lücke 0,2), linearer Kern, SVC C = 1e4 tol 1e-10", seed0=seed0, instanzen=n_inst,
                c1=dict(instanzen=c1_n, w_aenderung_max=c1_dw, marge_aenderung_max=c1_dm),
                c2=dict(median=med(c2), max=max(c2), min=min(c2), verletzungen=c2_viol),
                c3=dict(median=med(c3), min=min(c3), max=max(c3), verletzungen=c3_viol, unter_halb=int(sum(1 for r in c3 if r < 0.5))),
                c4=dict(instanzen=c4_n, median=med(c4r), min=min(c4r), q90=q(c4r, 90), winkel_median=med(c4a), winkel_q90=q(c4a, 90), winkel_max=max(c4a), stuetzmenge_aendert=c4_sv_change))


# ================================================================== D
def sec_D(n_inst=100, seeds=(12000, 52000)):
    thr_list = [0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2]
    methods = ["SLSQP (dual)", "SLSQP (Voreinstellungen)", "trust-constr", "trust-constr (Voreinstellungen)", "SVC tol=1e-10", "SVC tol=1e-3 (Voreinstellung)", "SMO eps=1e-10", "SMO eps=1e-3"]
    res = {"einheit": "Instanzen; 100 trennbare Instanzen je Satz, n = 8, linearer Kern, Zwei Gebiete mit Puffer (Lücke 0,2); exakte Menge aus der Aufzählung", "schwellen": thr_list, "saetze": {}}
    for label, seed0 in zip("AB", seeds):
        equal = {m: {str(t): 0 for t in thr_list} for m in methods}
        merr = {m: [] for m in methods}
        nuniq = 0
        for t in range(n_inst):
            ins = Sc.puffer(8, 0.2, seed0 + t)
            X, y = ins.X, ins.y
            K = X @ X.T
            en = M.enumerate_support_sets(K, y)
            if len(en.feasible) != 1:
                continue
            nuniq += 1
            r = en.feasible[0]
            S0 = set(r.support)
            m0 = 2.0 / np.linalg.norm((r.alpha * y) @ X)
            Q = Qmat(K, y)
            cand = {"SLSQP (dual)": qp_slsqp(Q, y).x, "trust-constr": qp_trust(Q, y).x, "SLSQP (Voreinstellungen)": qp_slsqp(Q, y, np.full(8, 0.1), None).x,
                    "trust-constr (Voreinstellungen)": qp_trust(Q, y, False).x, "SVC tol=1e-10": alpha_from(svc_lin(X, y, 1e4, 1e-10)[0], 8),
                    "SVC tol=1e-3 (Voreinstellung)": alpha_from(svc_lin(X, y, 1e4, 1e-3)[0], 8), "SMO eps=1e-10": S.smo(K, y, 1e4, eps=1e-10).alpha, "SMO eps=1e-3": S.smo(K, y, 1e4, eps=1e-3).alpha}
            for k, a in cand.items():
                a = np.maximum(a, 0.0)
                for thr in thr_list:
                    sv = set(np.where(a > thr * a.max())[0]) if thr > 0 else set(np.where(a > 0)[0])
                    equal[k][str(thr)] += int(sv == S0)
                merr[k].append(abs(2.0 / np.linalg.norm((a * y) @ X) - m0) / m0)
        res["saetze"][label] = dict(seed0=seed0, eindeutig=nuniq, gleich={k: equal[k] for k in methods}, marge_rel_median={k: med(merr[k]) for k in methods}, marge_rel_max={k: float(max(merr[k])) for k in methods})
    return res


# ================================================================== E
def perceptron_fast(X, y, order, max_epochs=20000):
    return Pe.perceptron(X, y, order, max_epochs)


def sec_E(n_inst=200, runs=(("A", 0.25, 16000), ("B", (0.05, 0.8), 16200), ("C", 0.25, 26000))):
    res = {"einheit": "Instanzen; je Satz 200 trennbare Instanzen, n = 20, Zwei Gebiete mit Puffer, 4 000 frische Testpunkte aus demselben Modell, 8 Reihenfolgen je Instanz", "saetze": {}}
    for label, gap, seed0 in runs:
        ratios, angles, updates = [], [], []
        better = worse = equal_ = 0
        runs_b = runs_w = runs_e = 0
        diff, svm_err, perc_err = [], [], []
        conv_fail = 0
        for t in range(n_inst):
            ins = Sc.puffer(20, gap, seed0 + t)
            te = Sc.puffer_test(4000, gap, seed0 + 5000 + t, ins.info["theta_true"])
            r = hard_smo(ins.X, ins.y)
            w, b = (r.alpha * ins.y) @ ins.X, r.b
            e_svm = float((np.sign(te.X @ w + b) != te.y).mean())
            mmax = 2.0 / np.linalg.norm(w)
            errs = []
            for order in Pe.random_orders(20, 8, seed0 + 500 + t):
                wp, bp, upd, ep, ok = perceptron_fast(ins.X, ins.y, order)
                if not ok:
                    conv_fail += 1
                    continue
                updates.append(upd)
                e = float((np.sign(te.X @ wp + bp) != te.y).mean())
                errs.append(e)
                ratios.append(M.direction_margin(ins.X, ins.y, math.atan2(wp[1], wp[0])) / mmax)
                angles.append(M.angle_between_deg(wp, w))
                runs_b += int(e < e_svm)
                runs_w += int(e > e_svm)
                runs_e += int(e == e_svm)
            me = float(np.mean(errs))
            diff.append(me - e_svm)
            better += int(me < e_svm)
            worse += int(me > e_svm)
            equal_ += int(me == e_svm)
            svm_err.append(e_svm)
            perc_err.append(me)
        d = np.array(diff)
        q = lambda a, p_: float(np.percentile(a, p_))
        res["saetze"][label] = dict(luecke=list(gap) if isinstance(gap, tuple) else gap, seed0=seed0, nicht_konvergiert=conv_fail, korrekturen_median=med(updates), korrekturen_min=min(updates),
                                    korrekturen_max=max(updates), verhaeltnis_median=med(ratios), verhaeltnis_q10=q(ratios, 10), verhaeltnis_min=float(min(ratios)), winkel_median=med(angles),
                                    winkel_q90=q(angles, 90), winkel_max=float(max(angles)), fehler_marge=float(np.mean(svm_err)), fehler_marge_se=float(np.std(svm_err, ddof=1) / math.sqrt(n_inst)),
                                    fehler_perceptron=float(np.mean(perc_err)), differenz=float(d.mean()), differenz_se=float(d.std(ddof=1) / math.sqrt(n_inst)), besser=better, gleich=equal_,
                                    schlechter=worse, laeufe_besser=runs_b, laeufe_gleich=runs_e, laeufe_schlechter=runs_w, laeufe=len(ratios))
    return res


# ================================================================== G
def sec_G(n_seeds=20, seed0=18000):
    sigmas = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    rows = {str(s): dict(marge=[], sv=[], test=[], viol=[], loose=[], deficit=[], warn=0) for s in sigmas}
    lin_sep = 0
    for t in range(n_seeds):
        ins = Sc.rings(30, seed0 + t)
        te = Sc.rings(2000, seed0 + 500 + t)
        lin_sep += int(P.lp_separable(ins.X, ins.y) is not None)
        for s in sigmas:
            K = M.kernel_matrix(ins.X, "gauss", s)
            a, b, m, warned = svc_pre(K, ins.y, 1e4, 1e-10, 2_000_000)
            f = K @ (a * ins.y) + b
            ft = M.kernel_cross(te.X, ins.X, "gauss", s) @ (a * ins.y) + b
            r = rows[str(s)]
            r["marge"].append(2.0 / math.sqrt(float((a * ins.y) @ K @ (a * ins.y))))
            r["sv"].append(float((a > 1e-8 * a.max()).mean()))
            r["test"].append(float((np.sign(ft) != te.y).mean()))
            r["viol"].append(int((ins.y * f < 1 - 1e-3).sum()))
            r["loose"].append(int((ins.y * f < 1 - 1e-5).sum()))
            r["deficit"].append(float(max(0.0, 1.0 - (ins.y * f).min())))
            r["warn"] += int(warned)
    out = {"einheit": "Zufalls-Seeds; 20 Zufalls-Seeds ab 18000, Ringe mit Lücke (Radius 1 und 2, Breite 0,1, 2 x 30 Punkte), 2 x 2 000 frische Testpunkte, C = 1e4, tol 1e-10", "n_seeds": n_seeds,
           "linear_trennbar": lin_sep, "sigmas": sigmas, "zeilen": {}}
    for s in sigmas:
        r = rows[str(s)]
        out["zeilen"][str(s)] = dict(marge_median=med(r["marge"]), marge_alle=r["marge"], sv_median=med(r["sv"]), testfehler_mittel=float(np.mean(r["test"])), verfehlt_1e3=int(sum(1 for v in r["viol"] if v > 0)),
                                     verfehlt_1e5=int(sum(1 for v in r["loose"] if v > 0)), defizit_max=max(r["deficit"]), warnung=r["warn"])
    out["grenzwert_klein_sigma"] = 2.0 / math.sqrt(60)
    return out


SECTIONS = {"A": sec_A, "B1": sec_B1, "B2": sec_B2, "B4": sec_B4, "C": sec_C, "D": sec_D, "E": sec_E, "G": sec_G}


def header():
    import platform

    import scipy
    import sklearn

    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, scikit_learn=sklearn.__version__, system=platform.system(), omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                zufallsquelle="SplitMix64 (hmg_rng.py)", hinweis="Zahlen der App-Ströme, Bänder über Zufalls-Seeds; die Vormessung der Planung lief mit numpy default_rng und gibt andere Zahlen")


def run(names):
    PARTS.mkdir(parents=True, exist_ok=True)
    for nm in names:
        out = SECTIONS[nm]()
        (PARTS / f"{nm}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Abschnitt {nm} fertig")


def merge():
    data = json.loads(TARGET.read_text(encoding="utf-8")) if TARGET.exists() else {}
    for nm in SECTIONS:
        f = PARTS / f"{nm}.json"
        if f.exists():
            data[nm] = json.loads(f.read_text(encoding="utf-8"))
    data["kopf"] = header()
    data["B5"] = dict(quelle="Messung Stück 3 (V12 B3, Plan 03 Befund C; nicht neu gemessen, mit numpy-Strömen)",
                      einheit="Zufalls-Seeds; Gauß-Kern sigma = 1, Gauß-Klassen mit 10 % vertauschten Etiketten, n = 60, nicht trennbar, SVC max_iter = 100 000, sechs frische Sätze zu je 30 Zufalls-Seeds",
                      c=["1e2", "1e4", "1e6", "1e8"], warnung_gepoolt=[0, 1, 76, 78], von=180, erste_fassung=[0, 0, 6, 9], von_erste=30)
    TARGET.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("hmg_b2.json geschrieben:", ", ".join(k for k in data if k not in ("kopf", "B5")))


def check():
    """Kleine Teilmengen neu rechnen und mit hmg_b2.json vergleichen (gleiche Plattform: exakt für Zählungen, Band für Gleitkommazahlen)."""
    data = json.loads(TARGET.read_text(encoding="utf-8"))
    bad = []
    # B2: erste 8 Instanzen von Satz A, SVC bei C = 1 und 1e2
    insts, _ = nonsep_set(13000, 8)
    for C in (1.0, 1e2):
        ref = data["B2"]["saetze"]["A"]["c"][ck(C)]
        for k, ins in enumerate(insts):
            m, _w = svc_lin(ins.X, ins.y, C, 1e-3, 2_000_000)
            w = m.coef_[0]
            v = int((ins.y * (ins.X @ w + float(m.intercept_[0])) < 1 - 1e-6).sum())
            if v != ref["verletzer"][k] or abs(2.0 / np.linalg.norm(w) - ref["marge"][k]) > 1e-6 * ref["marge"][k] or abs(int(m.n_iter_[0]) - ref["schritte"][k]) > max(2, 0.05 * ref["schritte"][k]):
                bad.append(f"B2 Satz A, C = {C:g}, Instanz {k}")
    # B1 (exakt)
    b1 = data["B1"]
    r = sec_B1()
    if r["svc"] != b1["svc"] or r["smo"] != b1["smo"]:
        bad.append("B1")
    # E: Satz A, die ersten 6 Instanzen (Verhältnis der Marge, Fehler)
    sub = sec_E(6, (("A", 0.25, 16000),))["saetze"]["A"]
    if sub["nicht_konvergiert"] != 0 or not (0.05 < sub["verhaeltnis_median"] < 1.0):
        bad.append("E Satz A (Teilmenge)")
    # G: Zufalls-Seed 18000 und sigma 0,8
    g = sec_G(1)
    ref = data["G"]["zeilen"]["0.8"]["marge_alle"][0]
    if abs(g["zeilen"]["0.8"]["marge_alle"][0] - ref) > 1e-6 * ref:
        bad.append("G sigma = 0,8, Zufalls-Seed 18000")
    # A: erste 10 Instanzen, genau eine KKT-Lösung
    for t in range(10):
        n = 4 + (t % 5)
        ins = Sc.puffer(n, 0.2, 11000 + t)
        if len(M.enumerate_support_sets(ins.X @ ins.X.T, ins.y).feasible) != 1:
            bad.append(f"A Instanz {t}")
    if bad:
        print("ABWEICHUNG:", "; ".join(bad))
        return 1
    print("hmg_b2.json stimmt mit der Neuberechnung der Teilmengen überein")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--merge"]:
        merge()
    elif args == ["--check"]:
        sys.exit(check())
    else:
        run([a for a in args if a in SECTIONS] or list(SECTIONS))
