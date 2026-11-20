"""`hmg_b2.json`: Konsistenz der eingefrorenen Studien und eine kleine Neuberechnung (`tools/hmg_make_b2.py --check`, scikit-learn nur hier und im Werkzeug).

Die Neuberechnung ist auf der gleichen Plattform exakt für Zählungen und ein Band für Gleitkommazahlen; unter Linux sind `n_iter_` und Defizite nur als Band gleich (Linux-Lauf steht aus)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
D = json.loads((ROOT / "hmg_b2.json").read_text(encoding="utf-8"))


def test_kopf_nennt_zufallsquelle_und_hinweis_auf_die_vormessung():
    k = D["kopf"]
    assert k["zufallsquelle"] == "SplitMix64 (hmg_rng.py)" and "default_rng" in k["hinweis"] and k["python"] and k["scikit_learn"]


def test_b2_zaehlungen_sind_in_sich_stimmig():
    for lb in "ABC":
        s = D["B2"]["saetze"][lb]
        assert s["nicht_trennbar"] == 100 and s["gezogen"] >= 100
        for ck, c in s["c"].items():
            assert len(c["verletzer"]) == len(c["schritte"]) == len(c["marge"]) == 100
            assert c["ohne_warnung_mit_verletzern"] <= 100 - c["warnung"] and c["ohne_warnung_mit_verletzern"] >= sum(1 for v in c["verletzer"] if v > 0) - c["warnung"]
            assert c["verletzer_min"] == min(c["verletzer"]) and c["verletzer_max"] == max(c["verletzer"]) and c["schritte_median"] == float(np.median(c["schritte"]))
        assert s["c"]["1"]["warnung"] == s["c"]["100"]["warnung"] == 0  # unter der Grenze: keine Warnung möglich
        assert all(sch == 2_000_000 for sch, v in zip(s["c"]["1000000"]["schritte"], s["c"]["1000000"]["verletzer"]) if sch >= 2_000_000)


def test_paarweise_verhaeltnisse_stimmen_mit_den_einzelwerten_ueberein():
    S = D["B2"]["saetze"]
    for c1, c2, key in (("1", "100", "1->100"), ("100", "10000", "100->10000")):
        a = np.concatenate([S[lb]["c"][c1]["marge"] for lb in "ABC"])
        b = np.concatenate([S[lb]["c"][c2]["marge"] for lb in "ABC"])
        r = b / a
        p = D["B2"]["paarweise"][key]["alle"]
        assert p["median"] == pytest.approx(float(np.median(r))) and p["unter_1_prozent"] == pytest.approx(float(np.mean(np.abs(r - 1) < 0.01))) and p["instanzen"] == 300


def test_e_und_a_und_d_zaehlungen_summieren_sich():
    for lb, e in D["E"]["saetze"].items():
        assert e["besser"] + e["gleich"] + e["schlechter"] == 200 and e["laeufe_besser"] + e["laeufe_gleich"] + e["laeufe_schlechter"] == e["laeufe"] == 1600
    for lb, a in D["A"]["saetze"].items():
        assert sum(a["stuetzvektoren"].values()) == a["eindeutig"] == 200
    for lb, s in D["D"]["saetze"].items():
        assert s["eindeutig"] == 100 and all(0 <= v <= 100 for row in s["gleich"].values() for v in row.values())
        for nm, row in s["gleich"].items():
            vals = [row[str(t)] for t in D["D"]["schwellen"]]
            assert vals[1] >= vals[0] or nm.startswith("SLSQP") or nm.startswith("trust")  # eine Schwelle 1e-12 verliert nie Stützvektoren, die α > 0 schon richtig hat (nur Rundungsreste können mehr zählen)


def test_check_des_werkzeugs_rechnet_teilmengen_neu_und_findet_keine_abweichung(capsys):
    pytest.importorskip("sklearn")
    spec = importlib.util.spec_from_file_location("hmg_make_b2", ROOT / "tools" / "hmg_make_b2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.check() == 0
    assert "stimmt mit der Neuberechnung" in capsys.readouterr().out
