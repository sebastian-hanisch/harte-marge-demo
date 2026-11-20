"""Jede Zahl der README steht hier (DEMO-PLAYBOOK §4, §11a Punkt 13): die Tabellen der README werden gegen den Inhalt von `hmg_b2.json` gehalten (eingefrorene Studie mit den
Strömen der App, SplitMix64), dazu Bänder für die Aussagen („gewinnt“, „hängt nicht“, „meldet nichts“) und die Testzahl. Zahlen, die nur als Abweichung von der Vormessung genannt sind, haben
keinen Test (Zeile „Zahl → Test“ der README)."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import hmg_constants as C
import hmg_texts as T

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
D = json.loads((ROOT / "hmg_b2.json").read_text(encoding="utf-8"))


def f(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def g(x, nd=2):
    return f"{x:.{nd}g}".replace(".", ",").replace("e-0", "e-").replace("e+0", "e")


def sci(x, nd=1):
    m, e = f"{x:.{nd}e}".replace(".", ",").split("e")
    return f"{m}e{int(e)}"


def thou(n):
    return f"{int(n):,}".replace(",", " ")


def test_annahmentabelle_der_app_steht_wortgleich_in_der_readme():
    assert T.assumptions_md().strip() in README


def test_testzahl_der_readme_gleich_der_suite():
    m = re.search(r"(\d+) Tests \(`pytest --collect-only -q tests`\)", README)
    assert m, "Testzahl nicht in der README"
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"], cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    n = int(re.search(r"(\d+) tests? collected", out).group(1))
    assert int(m.group(1)) == n


def test_orakelkette_zahlen_der_readme():
    A = D["A"]["saetze"]
    assert A["A"]["eindeutig"] == A["B"]["eindeutig"] == 200  # genau eine KKT-Lösung in 200 von 200 (zwei Sätze)
    assert f"{A['A']['eindeutig']} und {A['B']['eindeutig']} von 200 genau eine KKT-Lösung" in README
    assert (A["A"]["stuetzvektoren"], A["B"]["stuetzvektoren"]) == ({"2": 57, "3": 143}, {"3": 159, "2": 41}) and sum(A["A"]["stuetzvektoren"].values()) == 200
    assert f"2 in {A['A']['stuetzvektoren']['2']} und {A['B']['stuetzvektoren']['2']}, 3 in {A['A']['stuetzvektoren']['3']} und {A['B']['stuetzvektoren']['3']} von je 200 Instanzen" in README
    for k, limit in (("grid", 1e-12), ("hull", 1e-12), ("primal", 1e-9), ("smo", 1e-8), ("dual_slsqp", 1e-6), ("svc", 1e-5), ("dual_vs_2_over_m2", 1e-12), ("fusspunkte", 1e-12)):
        assert max(A["A"]["abweichung"][k], A["B"]["abweichung"][k]) < limit, k
    mx = lambda k: max(A["A"]["abweichung"][k], A["B"]["abweichung"][k])
    for k in ("grid", "hull", "primal", "smo", "dual_slsqp", "svc"):
        assert sci(mx(k)) in README, k
    assert (A["A"]["primal_success"], A["B"]["primal_success"], A["A"]["dual_success"], A["B"]["dual_success"]) == (196, 199, 191, 192)
    assert f"Primal {A['A']['primal_success']} und {A['B']['primal_success']}, Dual {A['A']['dual_success']} und {A['B']['dual_success']} von je 200" in README
    assert A["A"]["primal_richtig"] == A["B"]["primal_richtig"] == 200  # Erfolgsmeldung sagt über die Richtigkeit nichts: Marge in allen 200 richtig


def test_falle_tabelle_drei_saetze_und_aussage_mit_voraussetzung():
    S = D["B2"]["saetze"]
    for ck, label in (("1", "1"), ("100", "1e2"), ("10000", "1e4"), ("1000000", "1e6")):
        cells = [S[lb]["c"][ck] for lb in "ABC"]
        row = f"| {label}" if ck != "1000000" else "| 1e6"
        line = next(ln for ln in README.splitlines() if ln.startswith(row))
        assert " / ".join(str(c["warnung"]) for c in cells) in line and " / ".join(str(c["ohne_warnung_mit_verletzern"]) for c in cells) in line
        assert " / ".join(thou(c["schritte_median"]) for c in cells) in line and " / ".join(g(c["marge_median"], 4) for c in cells) in line
    # Absolutaussage nur als Zählung mit Voraussetzung: bei C = 1e4 in 98 bis 100 von je 100 Instanzen ohne Warnung (Band), nie „immer“
    silent = [S[lb]["c"]["10000"]["ohne_warnung_mit_verletzern"] for lb in "ABC"]
    assert all(95 <= v <= 100 for v in silent) and silent != [100, 100, 100]
    assert S["A"]["c"]["1"]["warnung"] == S["A"]["c"]["100"]["warnung"] == 0 and all(S[lb]["c"]["1"]["ohne_warnung_mit_verletzern"] == 100 for lb in "ABC")
    assert all(78 <= S[lb]["c"]["1000000"]["warnung"] <= 90 for lb in "ABC")  # bei C = 1e6 hält nur die Grenze max_iter den Lauf an
    assert all(S[lb]["c"][ck]["w_null"] == 0 for lb in "ABC" for ck in ("1", "100", "10000", "1000000"))
    assert [S[lb]["nicht_trennbar"] for lb in "ABC"] == [100, 100, 100] and f"gezogen {S['A']['gezogen']} / {S['B']['gezogen']} / {S['C']['gezogen']} Zufalls-Seeds" in README
    assert S["A"]["c"]["10000"]["verletzer_median"] >= 5 and S["A"]["c"]["10000"]["verletzer_min"] >= 1


def test_marge_wandert_bis_c_1e2_danach_nicht_mehr_paarweise():
    P = D["B2"]["paarweise"]
    p12, p24 = P["1->100"]["alle"], P["100->10000"]["alle"]
    assert p12["instanzen"] == p24["instanzen"] == 300
    assert 0.65 < p12["median"] < 0.8 and p12["sinkt_ueber_5_prozent"] > 0.8 and p12["unter_1_prozent"] < 0.15  # bis C = 1e2 hängt die Marge an C
    assert 0.995 < p24["median"] < 1.005 and p24["unter_1_prozent"] > 0.85 and p24["steigt_ueber_5_prozent"] < 0.02  # danach in den meisten Instanzen nicht mehr
    assert f"im Median das {f(p12['median'], 2)}fache ({f(100 * p12['sinkt_ueber_5_prozent'], 0)} % sinken" in README
    assert f"ändern sich {f(100 * p24['unter_1_prozent'], 0)} % um weniger als 1 %" in README
    t8 = D["B2"]["tol1e8"]
    assert f"{f(100 * t8['A']['unter_1_prozent'], 0)} % und {f(100 * t8['B']['unter_1_prozent'], 0)} % unter 1 %" in README
    assert P["10000->1000000"]["alle"]["steigt_ueber_5_prozent"] > 0.3  # 1e4 -> 1e6 vergleicht nicht konvergierte Läufe, steht so in der README
    assert "sagt deshalb nichts über C" in README


def test_live_modell_gegen_studie_zahlen():
    s4, s2, s1 = (D["B2"]["smo"][k] for k in ("10000", "100", "1"))
    assert s4["instanzen"] == 30 and s4["nicht_konvergiert"] >= 1 and s1["nicht_konvergiert"] == 0 and s2["nicht_konvergiert"] == 0
    assert 0.9 < s4["marge_verhaeltnis_min"] <= s4["marge_verhaeltnis_median"] <= s4["marge_verhaeltnis_max"] < 1.02
    assert f"hält bei C = 1e4 auf {s4['instanzen']} nicht trennbaren Instanzen in {s4['nicht_konvergiert']} an der Schrittgrenze 200 000 an" in README
    assert f"{f(s4['marge_verhaeltnis_min'])} bis {f(s4['marge_verhaeltnis_max'])} (Median {f(s4['marge_verhaeltnis_median'], 4)})" in README


def test_gauss_kern_auf_denselben_instanzen_und_widerspruch():
    B3 = D["B2"]["B3"]["saetze"]
    assert [B3[lb]["positiv_definit"] for lb in "ABC"] == [100, 100, 100] and [B3[lb]["kc_gleich_y"] for lb in "ABC"] == [100, 100, 100]  # alle 300 trennbar
    assert all(90 <= B3[lb]["erreicht_haerte"] <= 100 for lb in "ABC") and all(B3[lb]["warnung"] == 0 for lb in "ABC")
    assert f"in {B3['A']['erreicht_haerte']} / {B3['B']['erreicht_haerte']} / {B3['C']['erreicht_haerte']} von 100" in README
    assert f"{B3['A']['verletzung_1e5']} / {B3['B']['verletzung_1e5']} / {B3['C']['verletzung_1e5']} (Löser-Genauigkeit)" in README
    B4 = D["B4"]
    assert B4["instanzen"] == 30 and B4["warnung"] == 0 and 1 <= B4["verletzer_min"] <= B4["verletzer_max"] <= 3
    assert f"keine Warnung in {B4['instanzen'] - B4['warnung']} von {B4['instanzen']}, {B4['verletzer_min']} bis {B4['verletzer_max']} Verletzer, {B4['schritte_min']} bis {B4['schritte_max']} Aktualisierungsschritte" in README
    B1 = D["B1"]
    assert not B1["lp_trennbar"] and B1["zertifikat"] == [1.0, 1.0, 1.0, 1.0]
    assert [B1["svc"][c]["n_iter"] for c in ("1", "100", "10000")] == [3, 201, 20001] and all(B1["svc"][c]["w"] == [0.0, 0.0] and not B1["svc"][c]["warnung"] for c in B1["svc"])
    assert all(a == float(c) for c in ("1", "100", "10000") for a in B1["svc"][c]["alpha"])  # alle alpha = C
    assert "3 / 201 / 20 001 Aktualisierungsschritte, keine Warnung" in README
    assert D["B5"]["warnung_gepoolt"] == [0, 1, 76, 78] and D["B5"]["von"] == 180 and "0 / 1 / 76 / 78 von 180 Zufalls-Seeds" in README


def test_empfindlichkeit_entfernen_hinzufuegen_punkt_in_der_marge():
    c = D["C"]
    assert c["c2"]["verletzungen"] == 0 and c["c3"]["verletzungen"] == 0 and c["c2"]["min"] >= 1.0 and c["c3"]["max"] <= 1 + 1e-9  # Monotonie: Entfernen senkt nie, Hinzufügen hebt nie
    assert c["c1"]["w_aenderung_max"] < 1e-7 and c["c4"]["stuetzmenge_aendert"] == c["c4"]["instanzen"] == 200
    assert 0.6 < c["c4"]["median"] < 0.75 and 2 < c["c4"]["winkel_median"] < 6
    for s in (f(c["c2"]["median"]), f(c["c2"]["max"]), f(c["c3"]["min"]), f(c["c4"]["median"]), f(c["c4"]["min"]), f(c["c4"]["q90"]), f(c["c4"]["winkel_median"], 1), f(c["c4"]["winkel_q90"], 1), f(c["c4"]["winkel_max"], 1), sci(c["c1"]["w_aenderung_max"])):
        assert s in README, s
    assert f"in {c['c3']['unter_halb']} von 200 unter 0,5" in README


def test_stuetzvektoren_zaehlen_tabelle_und_einstellungsabhaengigkeit():
    S = D["D"]["saetze"]
    names = list(S["A"]["gleich"])
    for nm in names:
        line = next(ln for ln in README.splitlines() if ln.startswith(f"| {nm} |"))
        for t in S["A"]["gleich"][nm]:
            assert f"{S['A']['gleich'][nm][t]} / {S['B']['gleich'][nm][t]}" in line
        assert f"{sci(S['A']['marge_rel_max'][nm])} / {sci(S['B']['marge_rel_max'][nm])}" in line
    for lb in "AB":  # bei alpha > 0: libsvm in 100, trust-constr in 0, SLSQP dazwischen und abhängig von den Einstellungen
        assert S[lb]["gleich"]["SVC tol=1e-10"]["0.0"] == 100 and S[lb]["gleich"]["trust-constr"]["0.0"] == 0 and S[lb]["gleich"]["trust-constr (Voreinstellungen)"]["0.0"] == 0
        assert 0 < S[lb]["gleich"]["SLSQP (Voreinstellungen)"]["0.0"] < S[lb]["gleich"]["SLSQP (dual)"]["0.0"] < 100
        assert S[lb]["gleich"]["SMO eps=1e-10"]["0.0"] >= 99 and S[lb]["gleich"]["SMO eps=1e-3"]["0.0"] >= 98
        assert S[lb]["marge_rel_max"]["SVC tol=1e-3 (Voreinstellung)"] < 1e-3  # die Voreinstellung tol 1e-3 trifft die Marge auf 5e-4 relativ
    assert S["A"]["gleich"]["SLSQP (dual)"]["0.0"] != S["A"]["gleich"]["SLSQP (Voreinstellungen)"]["0.0"]  # die Zahl hängt an den Einstellungen


def test_perceptron_gegen_marge_zaehlung_im_erzeuger_mit_luecke():
    E = D["E"]["saetze"]
    for lb in "ABC":
        e = E[lb]
        assert e["nicht_konvergiert"] == 0 and e["besser"] + e["gleich"] + e["schlechter"] == 200 and e["laeufe"] == 1600
        assert 0.75 < e["verhaeltnis_median"] < 0.85 and e["differenz"] > 8 * e["differenz_se"]  # die größte Marge gewinnt deutlich ...
        assert 0 < e["besser"] < 25  # ... aber nicht immer: in einigen Instanzen ist das Perceptron besser
        assert e["fehler_perceptron"] > 2 * e["fehler_marge"]
    assert E["A"]["luecke"] == 0.25 and E["B"]["luecke"] == [0.05, 0.8] and E["C"]["luecke"] == 0.25
    eA = E["A"]
    for s in (f(eA["verhaeltnis_median"]), f(eA["verhaeltnis_q10"]), f(eA["verhaeltnis_min"]), f(eA["winkel_median"], 1), f(eA["fehler_marge"], 4), f(eA["fehler_perceptron"], 4), f(eA["differenz"], 4)):
        assert s in README, s
    assert f"je Instanz Perceptron besser {eA['besser']}, schlechter {eA['schlechter']} von 200" in README
    assert f"in {E['A']['besser']} und {E['C']['besser']} von 200 Instanzen" in README and f"in {E['B']['besser']} von 200 bei unsymmetrischer Lücke" in README


def test_ringe_tabelle_maximum_bei_sigma_0_8_und_testfehler_null():
    G = D["G"]
    assert G["linear_trennbar"] == 0 and G["n_seeds"] == 20 and G["sigmas"] == [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    z = G["zeilen"]
    meds = [z[str(s)]["marge_median"] for s in G["sigmas"]]
    assert G["sigmas"][meds.index(max(meds))] == 0.8 and meds[0] < max(meds) and meds[-1] < max(meds)  # die Vermutung „wächst mit kleiner werdendem σ“ ist widerlegt
    assert all(z[str(s)]["testfehler_mittel"] == 0.0 for s in (0.4, 0.8, 1.6, 3.2, 6.4)) and z["0.2"]["testfehler_mittel"] > 0  # Testfehler 0 von σ = 0,4 bis 6,4
    assert z["6.4"]["verfehlt_1e3"] == 0 and 10 <= z["6.4"]["verfehlt_1e5"] <= 20 and all(z[str(s)]["warnung"] == 0 for s in G["sigmas"])
    for s in G["sigmas"]:
        line = next(ln for ln in README.splitlines() if ln.startswith(f"| {g(s, 2)} |"))
        assert f(z[str(s)]["marge_median"], 4) in line and f(z[str(s)]["sv_median"], 2) in line and f(z[str(s)]["testfehler_mittel"], 4) in line
    assert f"Marge 2/√60 = {f(G['grenzwert_klein_sigma'], 4)}" in README


def test_regler_der_app_halten_die_zusagen_der_readme():
    assert 1e6 not in C.E4_C_OPTIONS and C.E2_RANDOM_N == 6 and "fest n = 6" in README
    assert C.HARD_MAX_STEPS == 200000 and "Schrittgrenze 200 000" in README


def test_keine_verbotenen_woerter_in_readme_und_texten():
    import hmg_texts

    text = README + hmg_texts.INTRO + hmg_texts.HOW + hmg_texts.FORMULAS + hmg_texts.assumptions_md()
    for wort in ("Saatwert", "trägt", "tragen", "Phase 0", "Stück ", "TODO"):
        assert wort not in text, wort
    assert "Zufalls-Seed" in text and "Seeds" not in re.sub(r"Zufalls-Seeds?", "", text)
