"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jeder Zweig der Art steht hier; dazu je Preset die erwartete Verdikt-Zeile
(ein Preset ohne Effekt darf keinen Effekt melden, „XOR linear“ meldet nie eine Marge)."""

import numpy as np
import pytest

import hmg_constants as C
import hmg_evaluation as ev
import hmg_scenario as Sc


def test_zahlenformat_mit_komma_und_nicht_definiert():
    assert ev.fmt(1.0) == "1,000" and ev.fmt(0.25, 2) == "0,25" and ev.fmt(None) == "nicht definiert" and ev.fmt(float("nan")) == "nicht definiert"
    assert ev.fmt_g(0.00012345, 2) == "0,00012" and ev.fmt_g(1234.5, 3) == "1,23e+03"


def test_kern_parse_aller_reglerwerte():
    assert ev.kern_parse("linear") == ("linear", 1.0) and ev.kern_parse("Polynom Grad 2") == ("poly2", 1.0)
    assert [ev.kern_parse(k) for k in C.KERN_CHOICES[2:]] == [("gauss", s) for s in (0.1, 0.2, 0.4, 0.8, 1.0, 1.6, 3.2, 6.4)]


def test_stuetzvektoren_nach_schwelle():
    assert ev.stuetz([0.0, 2.0, 1e-12, 0.5]) == [1, 3] and ev.stuetz([0.0, 0.0]) == [] and ev.stuetz([]) == []


def _r1(m_sel, ratios=()):
    return dict(m_max=1.0, m_sel=m_sel, lines=[dict(converged=True, ratio=r) for r in ratios])


def test_verdikt_exp1_nennt_marge_verhaeltnis_und_die_studie_mit_voraussetzung():
    kind, text = ev.verdict_exp1(_r1(0.5, (0.4, 0.9)))
    assert kind == "info"
    assert "Größte Marge 2/‖w‖ = 1,000" in text and "die gewählte Richtung hat 0,500 (50,0 % der größten)" in text
    assert "Die 2 Perceptron-Geraden erreichen in ihrer Richtung 40 bis 90 % der größten Marge (diese Instanz, diese Reihenfolgen)" in text
    a = ev.studies()["E"]["saetze"]["A"]
    assert f"im Median {ev.fmt(a['verhaeltnis_median'], 2)} der größten Marge" in text and f"je Instanz (Mittel über 8 Reihenfolgen) ist das Perceptron in {a['besser']} von 200 besser" in text
    assert "Zwei Gebiete mit Puffer, n = 20, Lücke 0,25, 200 Instanzen" in text and "für andere Daten ist nichts behauptet" in text


def test_verdikt_exp1_richtung_trennt_nicht_und_ohne_perceptron():
    kind, text = ev.verdict_exp1(_r1(-0.3))
    assert "die gewählte Richtung hat 0,000 (sie trennt die Klassen nicht)" in text and "Perceptron-Geraden erreichen" not in text


def test_verdikt_exp3_drei_aktionen_und_die_studienzahlen():
    base = dict(margin=1.0, sv=[0, 1, 2])
    r = dict(base=base, remove=dict(margin=1.2, ratio=1.2, angle=3.0), move=None, add=dict(margin=0.9, ratio=0.9, angle=0.0))
    kind, text = ev.verdict_exp3(r)
    assert "Vorher: Marge 1,000, 3 Stützvektoren." in text and "Entfernen: Marge 1,200 (1,200-fach), Drehung 3,0 Grad" in text
    assert "Verschieben: nicht trennbar, keine harte Marge" in text and "Hinzufügen: Marge 0,900 (0,900-fach), Drehung 0,0 Grad" in text
    c = ev.studies()["C"]
    assert f"je {c['c2']['verletzungen']} und {c['c3']['verletzungen']} Verletzungen in {c['instanzen']} Instanzen" in text and "nicht Gegenstand dieser Demo" in text


def _trap(sep_ok, violators, converged=True, margin=0.5, cert=None, message="", C_val=100, name="Gauß-Klassen"):
    return dict(inst=Sc.Instance(np.zeros((3, 2)), np.array([1.0, -1.0, 1.0]), name), C=C_val,
                sep=dict(separable=sep_ok, reason="lineares Programm " + ("zulässig" if sep_ok else "unzulässig"), certificate=cert),
                sol=dict(violators=violators, converged=converged, margin=margin, steps=123, at_C=2, errors=1, message=message))


def test_verdikt_falle_drei_zweige():
    kind, text = ev.verdict_trap(_trap(True, 0))
    assert kind == "success" and "erreicht die harte Bedingung" in text and "Marge 2/‖w‖ = 0,500" in text and "123 Iterationen" in text
    kind, text = ev.verdict_trap(_trap(True, 4, message="Schrittgrenze 99 erreicht"))
    assert kind == "warning" and "verletzt die harte Bedingung bei 4 Punkten" in text and "Multiplikatoren an der Schranke C: 2" in text and "keine Nicht-Trennbarkeit" in text
    kind, text = ev.verdict_trap(_trap(False, 6, margin=None, cert=np.array([0.25, 0.25, 0.25, 0.25])), note=" Hinweis.")
    assert kind == "error" and text.startswith("Gauß-Klassen: NICHT trennbar") and "es gibt keine harte Marge" in text
    assert "Zertifikat α = (1, 1, 1, 1) (bis auf einen Faktor): α ≥ 0, α ≠ 0, Σαy = 0 und Σαyx = 0." in text
    assert "„Marge“ 2/‖w‖ = nicht definiert (w = 0), 6 Punkte mit y f < 1, 1 falsch klassifiziert, 123 Iterationen" in text and text.endswith(" Hinweis.")


def test_hinweise_zu_libsvm_nennen_die_zahlen_der_studie():
    s = ev.studies()
    c4 = [s["B2"]["saetze"][lb]["c"]["10000"]["ohne_warnung_mit_verletzern"] for lb in "ABC"]
    assert ev.note_overlap() == f"`SVC` linear mit C = 1e4 antwortet bei je 100 solchen Instanzen in {c4[0]} / {c4[1]} / {c4[2]} ohne Warnung (Studie unten)."
    assert ev.note_xor().endswith("ohne Warnung.") and "w = 0 und alle α = C" in ev.note_xor()
    assert ev.note_conflict() == f"`SVC` mit Gauß-Kern σ = 1 und C = 1e4 gibt in {30 - s['B4']['warnung']} von 30 solchen Instanzen keine Warnung (Studie unten)."


def test_verdikt_b2_nennt_drei_saetze_paarweise_anteile_und_die_voraussetzung():
    kind, text = ev.verdict_b2()
    b = ev.studies()["B2"]
    c4 = [b["saetze"][lb]["c"]["10000"]["ohne_warnung_mit_verletzern"] for lb in "ABC"]
    assert kind == "error" and f"in {c4[0]} / {c4[1]} / {c4[2]} von 100 Instanzen ohne Warnung" in text
    assert "`SVC` mit tol 1e-3, linearer Kern, diesen Erzeuger" in text and "drei Sätze Zufalls-Seeds" in text
    assert "auf den hier gemessenen trennbaren Instanzen" in text and "tol 1e-10" in text and "19 von 20" in text and "im Gauß-Kern (3 bis 5 von 100)" in text
    p24 = b["paarweise"]["100->10000"]["alle"]
    assert f"ändern sich {ev.fmt(100 * p24['unter_1_prozent'], 0)} % der {p24['instanzen']} Instanzen um weniger als 1 %" in text
    rows = ev.b2_table()
    assert [r["C"] for r in rows] == ["1", "1e2", "1e4", "1e6"] and all(len(r) == 6 for r in rows)


def test_verdikt_exp5_nicht_trennbar_und_trennbar():
    r = ev.exp5("Ringe mit Lücke", 3, "linear")
    kind, text = ev.verdict_exp5(r)
    assert kind == "warning" and "nicht trennbar" in text and "keine harte Marge" in text
    r = ev.exp5("Ringe mit Lücke", 3, "Gauß σ = 0,8")
    kind, text = ev.verdict_exp5(r)
    assert kind == "info" and "trennbar, harte Marge 2/‖w‖ =" in text and "Testfehler" in text and "an 2000 frischen Punkten" in text
    g = ev.studies()["G"]
    assert f"linear trennbar in {g['linear_trennbar']} von {g['n_seeds']}" in text and "eine größere Marge zeigt nicht ein besseres σ" in text


# --- je Preset die erwartete Verdikt-Zeile -----------------------------------------------------------------------------------------------------
def _setting(preset, key, state_key):
    from hmg_presets import SETTING_SPECS

    return C.PRESETS[preset].get(key, SETTING_SPECS[state_key].default)


def test_preset_fuenf_punkte_meldet_marge_eins_und_eine_von_21_mengen():
    r = ev.exp2(_setting("Fünf Punkte in der Ebene (Handbeispiel)", "e2", "e2_set"), 7, "linear")
    kind, text = ev.verdict_exp2(r)
    assert kind == "success" and text.startswith("Von 21 gemischten Stützmengen ist 1 zulässig; Marge 2/‖w‖ = 1,000, Dualwert 2,000")


def test_preset_viele_geraden_meldet_die_perceptron_geraden():
    p = "Viele Geraden"
    r = ev.exp1(20, 0.25, 7, _setting(p, "theta", "theta_off"), _setting(p, "perc", "perc_k"))
    kind, text = ev.verdict_exp1(r)
    assert len(r["lines"]) == 8 and "Die 8 Perceptron-Geraden erreichen" in text and r["m_sel"] < r["m_max"]  # Abweichung 20 Grad: die gewählte Richtung hat weniger Marge


def test_preset_rechteck_meldet_alpha_nicht_eindeutig():
    r = ev.exp2(_setting("Rechteck (α nicht eindeutig)", "e2", "e2_set"), 7, "linear")
    kind, text = ev.verdict_exp2(r)
    assert "2 zulässig" in text and "nicht eindeutig" in text and "Marge 2/‖w‖ = 2,000" in text


def test_preset_xor_linear_meldet_nie_eine_marge():
    p = "XOR linear: nicht trennbar"
    r = ev.exp4_xor("linear", _setting(p, "e4_c", "e4_c"), 1e-3, 20000)
    kind, text = ev.verdict_trap(r, ev.note_xor())
    assert kind == "error" and "NICHT trennbar" in text and "„Marge“ 2/‖w‖ = nicht definiert (w = 0)" in text and "Marge 2/‖w‖ = 0" not in text
    assert ev.verdict_exp2(ev.exp2("XOR (vier Punkte)", 7, "linear"))[0] == "warning"


def test_preset_ueberlappende_klassen_meldet_nicht_trennbar_und_verletzer():
    p = "Überlappende Klassen, C = 1e4"
    r = ev.exp4_overlap(20, 2.0, 13005, "linear", _setting(p, "e4_c", "e4_c"), 1e-3, 200000)
    kind, text = ev.verdict_trap(r, ev.note_overlap())
    assert kind == "error" and "NICHT trennbar" in text and r["C"] == 10000 and r["sol"]["violators"] >= 1 and "Studie unten" in text


def test_preset_xor_mit_gauss_kern_meldet_trennbar_und_die_gauss_marge():
    p = "XOR mit Gauß-Kern"
    kern = _setting(p, "kern", "kern_choice")
    r = ev.exp2(_setting(p, "e2", "e2_set"), 7, kern)
    kind, text = ev.verdict_exp2(r)
    assert kind == "success" and "Marge 2/‖w‖ = 0,865" in text  # 1 - exp(-2) = 0,8647
    kind5, text5 = ev.verdict_exp5(ev.exp5(_setting(p, "e5", "e5_data"), 7, kern))
    assert kind5 == "info" and "XOR (vier Punkte): trennbar, harte Marge 2/‖w‖ = 0,865" in text5


def test_preset_ein_punkt_in_der_marge_meldet_neun_zehntel_ohne_drehung():
    p = "Ein Punkt in der Marge"
    r = ev.exp3(_setting(p, "e3", "e3_data"), 20, 0.25, 7, _setting(p, "e3_point", "e3_point"), _setting(p, "e3_dist", "e3_dist"), _setting(p, "e3_pos", "e3_pos"))
    kind, text = ev.verdict_exp3(r)
    assert "Vorher: Marge 1,000, 3 Stützvektoren." in text and "Hinzufügen: Marge 0,900 (0,900-fach), Drehung 0,0 Grad" in text
    assert "Entfernen: Marge 1,200 (1,200-fach)" in text  # (3,1) ist ein Stützvektor: ohne ihn ||w||^2 = 25/9, Marge 6/5 (Handbeispiel)


def test_preset_zwei_ringe_meldet_trennbar_mit_gauss_kern():
    p = "Zwei Ringe, Gauß-Kern"
    kern = _setting(p, "kern", "kern_choice")
    kind, text = ev.verdict_exp5(ev.exp5(_setting(p, "e5", "e5_data"), 7, kern))
    assert kind == "info" and "Ringe mit Lücke: trennbar" in text and "Testfehler 0,00 %" in text
