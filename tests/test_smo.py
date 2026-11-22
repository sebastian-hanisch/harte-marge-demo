"""hmg_smo.py (SMO, Quelle der Linie für die Stücke 3 und 4): Handbeispiele in Brüchen, Eigenschaften je Schritt, Fremdlöser, Schrittgrenze, Spur, Kopier-Hash.

Jede Prüfung steht für eine Einheit der Datei (Mini-Instanz von Hand gerechnet, DEMO-PLAYBOOK §1b). Die Funktion `smo` ist die Quelle: ihr Hash steht unten,
die Kopien in den Stücken 3 und 4 prüfen dieselbe Bytefolge (Kopier-Test)."""

import hashlib
import inspect

import numpy as np
import pytest

import hmg_smo as S
from hmg_rng import SplitMix64

# --- Daten: Handbeispiele aus handbeispiele_02.py (Stück 2) und handbeispiele_03.py (Stück 3) -------------------------------------------------
X5 = np.array([(4, 3), (3, 1), (0, 0), (4, 1), (2, 0)], float)
Y5 = np.array([-1, 1, -1, 1, 1], float)
XL = np.array([0, 1, 2, 3, 4], float)[:, None]  # fünf Punkte auf der Geraden, einer (x = 3) mit dem falschen Label
YL = np.array([-1, -1, 1, -1, 1], float)

# Hash der Funktion `smo` (Quelltext, Zeilenenden normalisiert): die Kopien in Stück 3 und 4 müssen denselben Wert haben.
SMO_SHA256 = "7ef2a5211d6ae6acff96f78d33bcdbeb74560bef2a6131ed64a23e01cf6a835b"
RNG_SHA256 = "d69b2b2fd67c1234cc0d79249676a662a2189d02803eea7d2edfb04b7a505a3f"


def _overlap(n, seed, sep=2.0):
    """Zwei überlappende Gauß-Klassen mit der Zufallsquelle der Demo (SplitMix64), nur für die Tests dieser Datei."""
    r = SplitMix64(seed)
    y = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    X = np.array([[r.normal(), r.normal()] for _ in range(n)])
    X[:, 0] += y * sep / 2.0
    return X, y


def _w_b(X, y, res):
    return (res.alpha * y) @ X, res.b


# --- Handbeispiele ---------------------------------------------------------------------------------------------------------------------------
def test_fuenf_punkte_in_der_ebene_alpha_in_bruechen():
    r = S.smo(X5 @ X5.T, Y5, np.inf, eps=1e-10)
    assert r.converged and r.stopped_by == "eps" and r.message == ""
    assert np.allclose(r.alpha, [6 / 5, 2, 4 / 5, 0, 0], atol=1e-8)  # 21 gemischte Mengen, eine zulässig: (6/5, 2, 4/5, 0, 0)
    w, b = _w_b(X5, Y5, r)
    assert np.allclose(w, [6 / 5, -8 / 5], atol=1e-8) and b == pytest.approx(-1.0, abs=1e-8)
    assert 2 / np.linalg.norm(w) == pytest.approx(1.0, abs=1e-8)  # Marge 2/||w|| = 1
    assert r.dual == pytest.approx(2.0, abs=1e-8)  # Dualwert = ||w||^2/2 = 2


def test_drei_punkte_in_einer_dimension():
    X = np.array([[0.0], [1.0], [3.0]])
    y = np.array([-1.0, 1.0, 1.0])
    r = S.smo(X @ X.T, y, np.inf, eps=1e-10)
    assert np.allclose(r.alpha, [2, 2, 0], atol=1e-8) and r.b == pytest.approx(-1.0, abs=1e-8)
    assert r.dual == pytest.approx(2.0, abs=1e-8)


def test_xor_mit_der_gram_matrix_des_gauss_kerns_a_gleich_ein_viertel():
    a = 0.25  # a = exp(-2/sigma^2), Gegenüberliegende a^2
    K = np.array([[1, a * a, a, a], [a * a, 1, a, a], [a, a, 1, a * a], [a, a, a * a, 1]])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    r = S.smo(K, y, np.inf, eps=1e-10)
    assert np.allclose(r.alpha, 16 / 9, atol=1e-8) and r.b == pytest.approx(0.0, abs=1e-8)  # alpha = 1/(1-a)^2
    ya = r.alpha * y
    assert 2 / np.sqrt(ya @ K @ ya) == pytest.approx(1 - a, abs=1e-8)  # Marge 3/4
    assert r.dual == pytest.approx(32 / 9, abs=1e-8)


def test_rechteck_w_und_b_eindeutig_alpha_nicht():
    X = np.array([(0, 1), (2, 1), (0, -1), (2, -1)], float)
    y = np.array([1.0, 1.0, -1.0, -1.0])
    for wss in ("second", "first"):
        r = S.smo(X @ X.T, y, np.inf, eps=1e-10, wss=wss)
        w, b = _w_b(X, y, r)
        assert np.allclose(w, [0, 1], atol=1e-8) and b == pytest.approx(0.0, abs=1e-8)
        assert r.alpha.sum() == pytest.approx(1.0, abs=1e-8)  # jede KKT-Lösung hat Summe 1 (Dualwert 1/2 = Summe - 1/2); welche, hängt an der Arbeitsmenge
    r1 = S.smo(X @ X.T, y, np.inf, eps=1e-10, wss="second")
    assert np.count_nonzero(r1.alpha > 1e-9) in (2, 3, 4)


def test_ein_punkt_je_klasse_alpha_ist_zwei_durch_abstand_zum_quadrat():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    y = np.array([-1.0, 1.0])
    r = S.smo(X @ X.T, y, np.inf, eps=1e-12)
    assert np.allclose(r.alpha, 2 / 25, atol=1e-10)  # Abstand 5: alpha = 2/d^2
    assert 2 / np.linalg.norm(_w_b(X, y, r)[0]) == pytest.approx(5.0, abs=1e-9)


@pytest.mark.parametrize("C", [1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 100.0])
def test_weiche_marge_fuenf_punkte_auf_der_geraden_formeln_in_c(C):
    """Handbeispiel Stück 3: Aufzählung in Brüchen gibt für C >= 1/3 alpha = (0, (3C+2)/9, C, C, (3C+2)/9), w = 2/3, Dualwert 2/9 + 8C/3;
    für C <= 1/3 w = 2C und Dualwert 4C - 2C^2."""
    r = S.smo(XL @ XL.T, YL, C, eps=1e-11)
    assert r.converged
    w = float(((r.alpha * YL) @ XL)[0])
    if C >= 1 / 3:
        assert np.allclose(r.alpha, [0, (3 * C + 2) / 9, C, C, (3 * C + 2) / 9], atol=1e-8)
        assert w == pytest.approx(2 / 3, abs=1e-8) and r.b == pytest.approx(-5 / 3, abs=1e-8)
        assert r.dual == pytest.approx(2 / 9 + 8 * C / 3, abs=1e-8)
    else:
        assert w == pytest.approx(2 * C, abs=1e-8)
        assert r.dual == pytest.approx(4 * C - 2 * C * C, abs=1e-8)


# --- Eigenschaften je Schritt ----------------------------------------------------------------------------------------------------------------
def test_nach_jedem_schritt_gilt_summe_y_alpha_null_und_alpha_zwischen_null_und_c_und_der_dualwert_steigt():
    for seed in range(6):
        X, y = _overlap(14, 40 + seed)
        K = X @ X.T
        for C in (0.5, 5.0):
            letzter = 0.0
            for k in range(0, 60):
                r = S.smo(K, y, C, eps=1e-12, max_steps=k)
                assert abs(y @ r.alpha) < 1e-12 and (r.alpha >= 0).all() and (r.alpha <= C).all(), (seed, C, k)
                assert r.dual >= letzter - 1e-12, (seed, C, k)  # der Dualwert steigt in jedem Schritt (nicht streng: Gleichstände)
                assert r.steps == min(k, r.steps)
                letzter = r.dual


def test_dualwert_der_rueckgabe_gleich_neu_gerechnetem_wert_und_kkt_luecke_gleich_der_der_rueckgabe():
    X, y = _overlap(20, 3)
    K = X @ X.T
    r = S.smo(K, y, 2.0, eps=1e-6)
    assert r.dual == pytest.approx(S.dual_value(K, y, r.alpha), rel=1e-12)
    assert S.kkt_gap(K, y, r.alpha, 2.0) == pytest.approx(r.gap, abs=1e-9) and r.gap <= 1e-6


def test_kkt_bedingungen_am_ergebnis_nach_der_definition():
    tol = 1e-6
    for seed in range(8):
        X, y = _overlap(16, 70 + seed)
        K = np.exp(-0.5 * ((X[:, None] - X[None]) ** 2).sum(-1))
        C = 3.0
        r = S.smo(K, y, C, eps=tol)
        f = K @ (r.alpha * y) + r.b
        yf = y * f
        assert (yf[r.alpha < 1e-10] >= 1 - 10 * tol).all()  # alpha = 0: y f >= 1
        assert (np.abs(yf[(r.alpha > 1e-10) & (r.alpha < C - 1e-10)] - 1) <= 10 * tol).all()  # frei: y f = 1
        assert (yf[r.alpha > C - 1e-10] <= 1 + 10 * tol).all()  # alpha = C: y f <= 1


# --- Fremdlöser ------------------------------------------------------------------------------------------------------------------------------
def test_dualwert_gleich_svc_in_hart_und_weich_und_mit_gauss_kern():
    svm = pytest.importorskip("sklearn.svm")
    for seed in range(8):
        X, y = _overlap(24, 200 + seed, sep=1.2)
        for K in (X @ X.T, np.exp(-0.5 * ((X[:, None] - X[None]) ** 2).sum(-1))):
            for C in (0.3, 3.0):
                r = S.smo(K, y, C, eps=1e-9, max_steps=200000)
                m = svm.SVC(kernel="precomputed", C=C, tol=1e-9, max_iter=200000).fit(K, y)
                a = np.zeros(len(y))
                a[m.support_] = np.abs(m.dual_coef_[0])
                ref = a.sum() - 0.5 * (a * y) @ K @ (a * y)
                assert r.dual == pytest.approx(ref, rel=1e-8), (seed, C)


def test_harte_marge_gleich_svc_auf_trennbaren_daten():
    svm = pytest.importorskip("sklearn.svm")
    for seed in range(12):
        r0 = SplitMix64(900 + seed)
        y = np.where(np.arange(12) % 2 == 0, 1.0, -1.0)
        X = np.array([[yy * (0.3 + abs(r0.normal())), 1.5 * r0.normal()] for yy in y])
        r = S.smo(X @ X.T, y, np.inf, eps=1e-10)
        m = svm.SVC(kernel="linear", C=1e6, tol=1e-10).fit(X, y)
        w, _ = _w_b(X, y, r)
        assert 2 / np.linalg.norm(w) == pytest.approx(2 / np.linalg.norm(m.coef_[0]), rel=1e-6)


# --- Arbeitsmengen (jede Wirkung abschaltbar, Zweig-Test) ------------------------------------------------------------------------------------
def test_alle_arbeitsmengen_gleicher_dualwert_aber_verschiedene_schrittzahlen():
    schritte = {"second": [], "first": [], "random": []}
    for seed in range(10):
        X, y = _overlap(30, 500 + seed, sep=1.0)
        K = X @ X.T
        duals = []
        for wss in schritte:
            r = S.smo(K, y, 1.0, eps=1e-9, wss=wss, rng=SplitMix64(seed))
            assert r.converged
            duals.append(r.dual)
            schritte[wss].append(r.steps)
        assert max(duals) - min(duals) < 1e-7 * abs(duals[0])
    assert schritte["second"] != schritte["first"] and schritte["second"] != schritte["random"]  # jede Wahl läuft wirklich
    assert sum(schritte["second"]) < sum(schritte["first"]) < sum(schritte["random"])  # zweite Ordnung braucht in diesen 10 Instanzen am wenigsten Schritte


def test_random_ohne_rng_und_unbekannte_arbeitsmenge_werden_abgelehnt():
    K = np.eye(2)
    y = np.array([1.0, -1.0])
    with pytest.raises(ValueError):
        S.smo(K, y, 1.0, wss="random")
    with pytest.raises(ValueError):
        S.smo(K, y, 1.0, wss="dritte")


def test_random_ist_mit_festem_zufalls_seed_wiederholbar():
    X, y = _overlap(20, 17)
    K = X @ X.T
    a = S.smo(K, y, 1.0, eps=1e-8, wss="random", rng=SplitMix64(5))
    b = S.smo(K, y, 1.0, eps=1e-8, wss="random", rng=SplitMix64(5))
    # andere Zufalls-Seeds laufen anders: nicht für ein einzelnes Paar (auf der Linux-CI hatten die Seeds 5 und 6 gleich viele Schritte), sondern über sechs Seeds
    schritte = {S.smo(K, y, 1.0, eps=1e-8, wss="random", rng=SplitMix64(s)).steps for s in range(5, 11)}
    assert a.steps == b.steps and np.array_equal(a.alpha, b.alpha) and len(schritte) > 1


# --- Schrittgrenze mit Meldung ---------------------------------------------------------------------------------------------------------------
def test_schrittgrenze_meldet_sich_und_stoppt_genau_dort():
    X = np.array([(1, 1), (-1, -1), (1, -1), (-1, 1)], float)  # XOR, linearer Kern: nicht trennbar, das harte Dual ist unbeschränkt
    y = np.array([1.0, 1.0, -1.0, -1.0])
    r = S.smo(X @ X.T, y, np.inf, eps=1e-3, max_steps=200)
    assert not r.converged and r.stopped_by == "max_steps" and r.steps == 200
    assert "Schrittgrenze 200 erreicht" in r.message and "nicht konvergiert" in r.message
    r2 = S.smo(X @ X.T, y, 7.0, eps=1e-3, max_steps=200)  # weiche Marge: alpha = (C, C, C, C), w = 0 (Handbeispiel Teil 4)
    assert r2.converged and np.allclose(r2.alpha, 7.0) and np.allclose((r2.alpha * y) @ X, 0)


def test_schrittgrenze_null_gibt_den_startzustand_zurueck():
    X, y = _overlap(10, 1)
    r = S.smo(X @ X.T, y, 1.0, max_steps=0)
    assert r.steps == 0 and not r.converged and np.all(r.alpha == 0) and r.gap == pytest.approx(2.0)


def test_eine_klasse_gibt_ohne_schritt_no_pair():
    r = S.smo(np.eye(3), np.ones(3), 1.0)
    assert r.stopped_by == "no_pair" and r.converged and r.steps == 0 and np.all(r.alpha == 0)


# --- Spur ------------------------------------------------------------------------------------------------------------------------------------
def test_spur_hat_gleiche_laengen_beginnt_bei_null_und_endet_im_ergebnis():
    X, y = _overlap(20, 9)
    K = X @ X.T
    r = S.smo(K, y, 1.0, eps=1e-6, trace=True)
    t = r.trace
    assert len(t["step"]) == len(t["dual"]) == len(t["gap"]) == r.steps + 1
    assert t["step"][0] == 0 and t["dual"][0] == 0.0 and t["gap"][0] == pytest.approx(2.0)  # Start: a = 0, Lücke 1 - (-1) = 2
    assert t["step"][-1] == r.steps and t["dual"][-1] == pytest.approx(r.dual, rel=1e-12) and t["gap"][-1] == pytest.approx(r.gap, abs=1e-12)
    assert (np.diff(t["dual"]) >= -1e-12).all()  # der Dualwert steigt in jedem Schritt
    assert (np.diff(t["step"]) == 1).all()
    assert S.smo(K, y, 1.0, eps=1e-6).trace is None


def test_spur_mit_ausduennung_behaelt_anfang_und_ende():
    X, y = _overlap(20, 9)
    K = X @ X.T
    voll = S.smo(K, y, 1.0, eps=1e-6, trace=True)
    duenn = S.smo(K, y, 1.0, eps=1e-6, trace=True, trace_every=7)
    assert duenn.steps == voll.steps and duenn.trace["step"][0] == 0 and duenn.trace["step"][-1] == voll.steps
    assert set(duenn.trace["step"][:-1]) <= set(range(0, voll.steps, 7))
    i = int(np.searchsorted(voll.trace["step"], duenn.trace["step"][3]))
    assert duenn.trace["dual"][3] == voll.trace["dual"][i]


def test_spur_bei_schrittgrenze_endet_im_grenzzustand():
    X = np.array([(1, 1), (-1, -1), (1, -1), (-1, 1)], float)
    y = np.array([1.0, 1.0, -1.0, -1.0])
    r = S.smo(X @ X.T, y, np.inf, max_steps=50, trace=True)
    assert r.trace["step"][-1] == 50 and r.trace["gap"][-1] > 1e-3


# --- Randfälle -------------------------------------------------------------------------------------------------------------------------------
def test_kern_ohne_positive_semidefinitheit_rechnet_ohne_absturz():
    r0 = SplitMix64(2)
    X = np.array([[r0.normal(), r0.normal()] for _ in range(10)])
    y = np.where(np.arange(10) % 2 == 0, 1.0, -1.0)
    K = np.tanh(0.5 * X @ X.T - 0.3)  # Sigmoid-Kern: im Allgemeinen nicht positiv semidefinit
    assert np.linalg.eigvalsh(K).min() < 0
    r = S.smo(K, y, 1.0, eps=1e-4, max_steps=20000)
    assert np.isfinite(r.alpha).all() and abs(y @ r.alpha) < 1e-9 and (r.alpha >= 0).all() and (r.alpha <= 1.0).all()


def test_skalen_1e_minus_6_und_1e6_geben_die_um_den_faktor_skalierte_marge():
    for s in (1e-6, 1e6):
        Xs = X5 * s
        r = S.smo(Xs @ Xs.T, Y5, np.inf, eps=1e-10)  # die Lücke hat die Einheit von y f, sie skaliert nicht mit
        w, _ = _w_b(Xs, Y5, r)
        assert 2 / np.linalg.norm(w) == pytest.approx(1.0 * s, rel=1e-6)


def test_gleiche_eingabe_gleiche_ausgabe_und_eingabe_bleibt_unveraendert():
    X, y = _overlap(15, 11)
    K = X @ X.T
    K0, y0 = K.copy(), y.copy()
    a = S.smo(K, y, 1.0, eps=1e-8)
    b = S.smo(K, y, 1.0, eps=1e-8)
    assert np.array_equal(a.alpha, b.alpha) and a.steps == b.steps
    assert np.array_equal(K, K0) and np.array_equal(y, y0)


def test_umnummerieren_und_etiketten_vertauschen():
    X, y = _overlap(14, 21)
    K = X @ X.T
    r = S.smo(K, y, 2.0, eps=1e-10)
    p = np.random.default_rng(0).permutation(14)
    rp = S.smo(K[np.ix_(p, p)], y[p], 2.0, eps=1e-10)
    assert rp.dual == pytest.approx(r.dual, rel=1e-8)
    rn = S.smo(K, -y, 2.0, eps=1e-10)  # Labels vertauschen: gleiche alpha, b wechselt das Vorzeichen
    assert rn.dual == pytest.approx(r.dual, rel=1e-8) and np.allclose(rn.alpha, r.alpha, atol=1e-6) and rn.b == pytest.approx(-r.b, abs=1e-6)


# --- Kopier-Test ---------------------------------------------------------------------------------------------------------------------------------
def test_quelltext_von_smo_ist_unveraendert():
    """Die Stücke 3 und 4 der Linie kopieren hmg_smo.py; ändert sich die Funktion hier, brechen deren Kopier-Tests. Hash der Funktion (nicht der Datei)."""
    src = inspect.getsource(S.smo).replace("\r\n", "\n")
    assert hashlib.sha256(src.encode("utf-8")).hexdigest() == SMO_SHA256


def test_hmg_rng_ist_die_vorlage_bytegleich(tmp_path):
    import pathlib

    mine = pathlib.Path(__file__).resolve().parent.parent / "hmg_rng.py"
    assert hashlib.sha256(mine.read_bytes()).hexdigest() == RNG_SHA256
    vorlage = pathlib.Path(__file__).resolve().parents[2] / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if vorlage.exists():  # im Projects-Ordner vorhanden, in der CI nicht
        assert mine.read_bytes() == vorlage.read_bytes()
