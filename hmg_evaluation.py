"""Rechnungen und Verdikte der fünf Experimente. Die Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`.

Zähleinheiten stehen in jedem Satz: Instanzen, Zufalls-Seeds, Aktualisierungsschritte. Die Marge heißt 2/||w||. Zahlen aus den eingefrorenen Studien (`hmg_b2.json`,
Werkzeug `tools/hmg_make_b2.py`) kommen über `studies()`; die App rechnet nicht mit libsvm."""

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

import hmg_constants as C
import hmg_margin as M
import hmg_perceptron as Pe
import hmg_scenario as Sc
import hmg_separable as P
import hmg_smo as S

STUDY_PATH = Path(__file__).resolve().parent / C.STUDY_FILE


@lru_cache(maxsize=1)
def studies() -> dict:
    """Eingefrorene Studien (leeres Wörterbuch, wenn die Datei fehlt)."""
    try:
        return json.loads(STUDY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def kern_parse(choice: str):
    """„linear“, „Polynom Grad 2“ oder „Gauß σ = 0,8“ -> (Art, sigma)."""
    if choice == "linear":
        return "linear", 1.0
    if choice.startswith("Polynom"):
        return "poly2", 1.0
    return "gauss", float(choice.split("=")[1].strip().replace(",", "."))


def fmt(x, nd: int = 3) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nicht definiert"
    return f"{x:.{nd}f}".replace(".", ",")


def fmt_g(x, nd: int = 3) -> str:
    return f"{x:.{nd}g}".replace(".", ",")


def stuetz(alpha, rel: float = 1e-9):
    """Indizes der Stützvektoren: alpha > rel * max(alpha) (Schwelle im Text nennen)."""
    a = np.asarray(alpha, float)
    if a.size == 0 or a.max() <= 0:
        return []
    return [int(i) for i in np.where(a > rel * a.max())[0]]


# --- gemeinsame Lösung der harten Marge im Merkmalsraum -----------------------------------------------------------------------------------------
def solve_hard(X, y, kind: str = "linear", sigma: float = 1.0, eps: float = C.SMO_EPS_HARD, max_steps: int = C.HARD_MAX_STEPS):
    """Harte Marge mit dem eigenen SMO (C = unendlich). Gibt dict mit alpha, b, K, margin (2/||w||, None bei w = 0), dual, steps, converged, message zurück."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    K = M.kernel_matrix(X, kind, sigma)
    r = S.smo(K, y, np.inf, eps=eps, max_steps=max_steps)
    return dict(alpha=r.alpha, b=r.b, K=K, margin=M.margin_from_alpha(K, y, r.alpha), dual=r.dual, steps=r.steps, converged=r.converged, message=r.message, sv=stuetz(r.alpha))


# --- Experiment 1: viele Geraden, eine größte Marge ------------------------------------------------------------------------------------------
def exp1(n: int, puffer: float, seed: int, theta_off: int, perc_k: int) -> dict:
    ins = Sc.puffer(n, puffer, seed)
    X, y = ins.X, ins.y
    sol = solve_hard(X, y, eps=1e-10)
    w = (sol["alpha"] * y) @ X
    m_max = sol["margin"]
    mg, th_star = M.max_margin_by_directions(X, y)
    hull = M.hull_distance(X, y)
    thetas = np.radians(np.arange(-180, 181, 1.0))
    curve = M.margin_curve(X, y, thetas)
    th_sel = th_star + math.radians(theta_off)
    m_sel = M.direction_margin(X, y, th_sel)
    u = np.array([math.cos(th_sel), math.sin(th_sel)])
    p = X @ u
    lo_neg, hi_pos = p[y < 0].max(), p[y > 0].min()
    lines = []
    for wp, bp, upd, ep, ok, order in Pe.perceptron_lines(X, y, perc_k, seed + 1000) if perc_k > 0 else []:
        ang = math.atan2(wp[1], wp[0])
        lines.append(dict(w=wp, b=bp, updates=upd, converged=ok, margin=M.direction_margin(X, y, ang), ratio=M.direction_margin(X, y, ang) / m_max if ok else None))
    return dict(inst=ins, w=w, b=sol["b"], m_max=m_max, grid_max=mg, hull=hull, th_star=th_star, curve=curve, thetas_deg=np.degrees(thetas), th_sel=th_sel, m_sel=m_sel, u=u,
                strip=(lo_neg, hi_pos), sv=sol["sv"], lines=lines, steps=sol["steps"], converged=sol["converged"])


def verdict_exp1(r: dict):
    """Verdikt zu Experiment 1 (Marge der gewählten Richtung gegen die größte; Anker aus der Studie)."""
    st = studies().get("E", {}).get("saetze", {})
    parts = [f"Größte Marge 2/‖w‖ = {fmt(r['m_max'])}; die gewählte Richtung hat {fmt(max(r['m_sel'], 0.0))}"]
    if r["m_sel"] <= 0:
        parts[0] += " (sie trennt die Klassen nicht)"
    else:
        parts[0] += f" ({fmt(100 * r['m_sel'] / r['m_max'], 1)} % der größten)"
    ok = [ln for ln in r["lines"] if ln["converged"]]
    if ok:
        ratios = sorted(ln["ratio"] for ln in ok)
        parts.append(f"Die {len(ok)} Perceptron-Geraden erreichen in ihrer Richtung {fmt(100 * ratios[0], 0)} bis {fmt(100 * ratios[-1], 0)} % der größten Marge (diese Instanz, diese Reihenfolgen)")
    text = ". ".join(parts) + "."
    if st:
        a = st["A"]
        text += (f" Studie in diesem Erzeuger (Zwei Gebiete mit Puffer, n = 20, Lücke 0,25, 200 Instanzen mit je 8 Reihenfolgen): die Richtung des Perceptrons erreicht im Median "
                 f"{fmt(a['verhaeltnis_median'], 2)} der größten Marge (10 % der Läufe unter {fmt(a['verhaeltnis_q10'], 2)}), der Testfehler ist im Mittel {fmt(a['fehler_perceptron'], 4)} gegen {fmt(a['fehler_marge'], 4)}; "
                 f"je Instanz (Mittel über 8 Reihenfolgen) ist das Perceptron in {a['besser']} von 200 besser. Der Erzeuger kommt der Marge durch seine Lücke entgegen; für andere Daten ist nichts behauptet.")
    return "info", text


# --- Experiment 2: das duale Problem von Hand ------------------------------------------------------------------------------------------------
def exp2_instance(set_name: str, seed: int):
    if set_name == C.E2_SETS[0]:
        return Sc.fuenf_punkte()
    if set_name == C.E2_SETS[1]:
        return Sc.rechteck()
    if set_name == C.E2_SETS[2]:
        return Sc.xor4()
    return Sc.puffer(C.E2_RANDOM_N, 0.2, seed)


def exp2(set_name: str, seed: int, kern: str) -> dict:
    ins = exp2_instance(set_name, seed)
    kind, sigma = kern_parse(kern)
    K = M.kernel_matrix(ins.X, kind, sigma)
    en = M.enumerate_support_sets(K, ins.y)
    sep = P.separability(ins.X, ins.y, kind)
    sol = solve_hard(ins.X, ins.y, kind, sigma, eps=1e-10, max_steps=20000 if sep["separable"] else 2000)
    feas = en.feasible
    res = dict(inst=ins, kind=kind, sigma=sigma, K=K, enum=en, solution=sol, feasible=feas, separable=sep["separable"], reason=sep["reason"], certificate=sep["certificate"])
    if feas:
        r0 = feas[0]
        res["margin"] = M.margin_from_alpha(K, ins.y, r0.alpha)
        res["dual"] = S.dual_value(K, ins.y, r0.alpha)
        ap, an = r0.alpha * (ins.y > 0), r0.alpha * (ins.y < 0)
        res["fusspunkte"] = ((ap @ ins.X) / ap.sum(), (an @ ins.X) / an.sum()) if kind == "linear" else None
        res["hull"] = M.hull_distance(ins.X, ins.y) if kind == "linear" else None
        res["grid"] = M.max_margin_by_directions(ins.X, ins.y)[0] if kind == "linear" else None
    return res


def verdict_exp2(r: dict):
    en = r["enum"]
    nfe = len(r["feasible"])
    if nfe == 0:
        if not r["separable"]:
            return "warning", (f"Von {en.tested} gemischten Stützmengen ist keine zulässig: die Daten sind in diesem Kern nicht trennbar ({r['reason']}); es gibt keine harte Marge. "
                               f"Das eigene SMO meldet das nach {r['solution']['steps']} Aktualisierungsschritten nicht als Fehler, sondern hält an der Schrittgrenze oder mit einem Lösungsvektor an, der die Bedingung verletzt.")
        return "warning", f"Von {en.tested} gemischten Stützmengen ist keine zulässig, obwohl der Test „trennbar“ sagt: die Zahlenschwelle der Aufzählung ist zu eng (Kondition)."
    w_set = {tuple(np.round(M.primal_wb(r["inst"].X, r["inst"].y, f.alpha, f.b)[0], 6)) for f in r["feasible"]} if r["kind"] == "linear" else set()
    head = f"Von {en.tested} gemischten Stützmengen {'ist' if nfe == 1 else 'sind'} {nfe} zulässig; Marge 2/‖w‖ = {fmt(r['margin'])}, Dualwert {fmt(r['dual'])} (= 2/Marge² im linearen Kern)."
    if nfe > 1:
        same = "demselben w und b" if len(w_set) <= 1 else f"{len(w_set)} verschiedenen w"
        head += f" Die KKT-Lösung ist nicht eindeutig: {nfe} Stützmengen mit {same}, aber verschiedenen α (Entartung: Punkte auf den Randgeraden, mehr als zum Festlegen nötig)."
    tail = " Vergleich der Rechenwege (200 Instanzen, n = 4 bis 8, Zwei Gebiete mit Puffer): "
    a = studies().get("A", {}).get("saetze", {}).get("A")
    if a:
        tail += (f"Gitter über 3 600 Richtungen, Abstand der konvexen Hüllen, Primal, Dual, eigenes SMO und `SVC` geben dieselbe Marge (größte relative Abweichung {fmt_g(a['abweichung']['grid'], 2)} "
                 f"bis {fmt_g(a['abweichung']['svc'], 2)}); in {a['eindeutig']} von {a['instanzen']} Instanzen gibt es genau eine KKT-Lösung.")
    return "success", head + tail


# --- Experiment 3: wie viel hängt an einzelnen Punkten ---------------------------------------------------------------------------------------
def _hard_or_none(X, y):
    """Harte Marge, falls trennbar (lineares Programm), sonst None."""
    if P.one_class(y) or P.lp_separable(X, y) is None:
        return None
    sol = solve_hard(X, y, eps=1e-10, max_steps=50000)
    w = (sol["alpha"] * y) @ X
    return dict(w=w, b=sol["b"], margin=sol["margin"], sv=sol["sv"], alpha=sol["alpha"])


def exp3_instance(data: str, n: int, puffer: float, seed: int):
    return Sc.fuenf_punkte() if data == C.E3_SETS[1] else Sc.puffer(n, puffer, seed)


def exp3(data: str, n: int, puffer: float, seed: int, k: int, dist: float, pos: float) -> dict:
    ins = exp3_instance(data, n, puffer, seed)
    X, y = ins.X, ins.y
    N = len(y)
    k = min(max(k, 1), N) - 1
    base = _hard_or_none(X, y)
    wn = base["w"] / np.linalg.norm(base["w"])
    tdir = np.array([-wn[1], wn[0]])
    foot = -base["b"] * base["w"] / (base["w"] @ base["w"])
    half = base["margin"] / 2.0
    lab = y[k]
    out = dict(inst=ins, base=base, k=k, label=lab)
    # (a) entfernen
    keep = np.arange(N) != k
    out["remove"] = _hard_or_none(X[keep], y[keep]) if N > 2 else None
    # (b) verschieben: tangentiale Lage bleibt, Abstand y f = dist
    t_k = float((X[k] - foot) @ tdir)
    p_move = foot + t_k * tdir + lab * dist * half * wn
    Xm = X.copy()
    Xm[k] = p_move
    out["move"] = _hard_or_none(Xm, y)
    out["p_move"] = p_move
    # (c) hinzufügen: Label des gewählten Punkts, Lage pos längs der Geraden, Abstand y f = dist
    p_add = foot + pos * tdir + lab * dist * half * wn
    out["add"] = _hard_or_none(np.vstack([X, p_add]), np.r_[y, lab])
    out["p_add"] = p_add
    for key in ("remove", "move", "add"):
        r = out[key]
        if r is not None:
            r["angle"] = M.angle_between_deg(r["w"], base["w"])
            r["ratio"] = r["margin"] / base["margin"]
    return out


def verdict_exp3(r: dict):
    b = r["base"]
    rows = []
    for key, name in (("remove", "Entfernen"), ("move", "Verschieben"), ("add", "Hinzufügen")):
        x = r[key]
        if x is None:
            rows.append(f"{name}: nicht trennbar, keine harte Marge")
        else:
            rows.append(f"{name}: Marge {fmt(x['margin'])} ({fmt(x['ratio'], 3)}-fach), Drehung {fmt(x['angle'], 1)} Grad")
    text = f"Vorher: Marge {fmt(b['margin'])}, {len(b['sv'])} Stützvektoren. " + "; ".join(rows) + "."
    c = studies().get("C")
    if c:
        text += (f" Studie (200 Instanzen, n = 20, Zwei Gebiete mit Puffer): Entfernen senkt die Marge nie und Hinzufügen hebt sie nie (je {c['c2']['verletzungen']} und {c['c3']['verletzungen']} Verletzungen in {c['instanzen']} Instanzen); "
                 f"ein Nicht-Stützvektor ändert w um höchstens {fmt_g(c['c1']['w_aenderung_max'], 2)}; ein richtig etikettierter Punkt bei y·f = 0,25 (zufällig längs der Geraden) senkt sie im Median auf {fmt(c['c4']['median'], 2)} "
                 f"und dreht die Gerade im Median um {fmt(c['c4']['winkel_median'], 1)} Grad (90 % der Instanzen unter {fmt(c['c4']['winkel_q90'], 1)}). Der Preis je Verletzung (weiche Marge) ist nicht Gegenstand dieser Demo.")
    return "info", text


# --- Experiment 4: nicht trennbar, und der Löser meldet nichts --------------------------------------------------------------------------------
def solve_soft(X, y, kind: str, sigma: float, C_val: float, eps: float, max_steps: int, trace: bool = True):
    K = M.kernel_matrix(X, kind, sigma)
    r = S.smo(K, y, float(C_val), eps=eps, max_steps=max_steps, trace=trace, trace_every=25)
    f = K @ (r.alpha * y) + r.b
    yf = y * f
    nviol, nerr = P.violations(yf)
    atC = int((r.alpha >= C_val * (1 - 1e-9)).sum())
    return dict(alpha=r.alpha, b=r.b, K=K, steps=r.steps, converged=r.converged, message=r.message, gap=r.gap, dual=r.dual, trace=r.trace, margin=M.margin_from_alpha(K, y, r.alpha),
                violators=nviol, errors=nerr, yf=yf, at_C=atC, defizit=float(max(0.0, 1.0 - yf.min())))


def exp4_case(ins, kern: str, C_val: int, eps: float, max_steps: int) -> dict:
    kind, sigma = kern_parse(kern)
    sep = P.separability(ins.X, ins.y, kind)
    sol = solve_soft(ins.X, ins.y, kind, sigma, C_val, eps, max_steps)
    return dict(inst=ins, kind=kind, sigma=sigma, sep=sep, sol=sol, C=C_val, eps=eps, max_steps=max_steps)


def exp4_overlap(n, abstand, seed, kern, C_val, eps, max_steps):
    return exp4_case(Sc.gauss_classes(n, abstand, seed), kern, C_val, eps, max_steps)


def exp4_xor(kern, C_val, eps, max_steps):
    return exp4_case(Sc.xor4(), kern, C_val, eps, max_steps)


def exp4_conflict(n, seed, kern, C_val, eps, max_steps):
    return exp4_case(Sc.widerspruch(n, seed), kern, C_val, eps, max_steps)


def note_overlap() -> str:
    b = studies().get("B2")
    if not b:
        return ""
    c4 = [b["saetze"][lb]["c"]["10000"]["ohne_warnung_mit_verletzern"] for lb in "ABC"]
    return f"`SVC` linear mit C = 1e4 antwortet bei je 100 solchen Instanzen in {c4[0]} / {c4[1]} / {c4[2]} ohne Warnung (Studie unten)."


def note_xor() -> str:
    b = studies().get("B1")
    if not b:
        return ""
    return "`SVC` linear (tol 1e-3) gibt bei C = 1, 1e2 und 1e4 w = 0 und alle α = C" + (" ohne Warnung." if not any(v["warnung"] for v in b["svc"].values()) else " (mit Warnung).")


def note_conflict() -> str:
    b = studies().get("B4")
    if not b:
        return ""
    return f"`SVC` mit Gauß-Kern σ = 1 und C = 1e4 gibt in {b['instanzen'] - b['warnung']} von {b['instanzen']} solchen Instanzen keine Warnung (Studie unten)."


def verdict_trap(r: dict, note: str = ""):
    """Verdikt eines Falls aus Experiment 4: erst die Voraussetzung „trennbar“, dann was der Löser liefert. Die App meldet, was libsvm nicht meldet."""
    sep, sol = r["sep"], r["sol"]
    ins = r["inst"]
    nm = ins.name
    marge = "nicht definiert (w = 0)" if sol["margin"] is None else fmt(sol["margin"])
    if sep["separable"]:
        head = f"{nm}: trennbar ({sep['reason']}). "
        if sol["violators"] == 0 and sol["converged"]:
            return "success", head + f"Das eigene SMO mit C = {r['C']} erreicht die harte Bedingung (kein Punkt mit y f < 1), Marge 2/‖w‖ = {marge}, {sol['steps']} Aktualisierungsschritte."
        return "warning", head + (f"Das eigene SMO mit C = {r['C']} verletzt die harte Bedingung bei {sol['violators']} Punkten (Multiplikatoren an der Schranke C: {sol['at_C']}), {sol['steps']} Aktualisierungsschritte"
                                  f"{'; ' + sol['message'] if sol['message'] else ''}: C ist hier zu klein für die Marge dieser Instanz, das ist eine Grenze des Lösers, keine Nicht-Trennbarkeit.")
    head = f"{nm}: NICHT trennbar ({sep['reason']}), es gibt keine harte Marge. "
    cert = ""
    if sep["certificate"] is not None and len(sep["certificate"]) <= 12:
        c0 = np.asarray(sep["certificate"], float)
        cert = "Zertifikat α = (" + ", ".join(fmt_g(float(a), 3) for a in c0 / c0.max()) + ") (bis auf einen Faktor): α ≥ 0, α ≠ 0, Σαy = 0 und Σαyx = 0. "
    body = (f"Was der Löser trotzdem liefert: eine Gerade mit „Marge“ 2/‖w‖ = {marge}, {sol['violators']} Punkte mit y f < 1, {sol['errors']} falsch klassifiziert, {sol['steps']} Aktualisierungsschritte, "
            f"Multiplikatoren an der Schranke C = {r['C']}: {sol['at_C']}. ")
    if sol["message"]:
        body += sol["message"] + ". "
    return "error", head + cert + body + note


def b2_table():
    """Zeilen der eingefrorenen Tabelle B2 (drei Sätze) für die Anzeige."""
    b = studies().get("B2")
    if not b:
        return []
    rows = []
    for Ckey in ("1", "100", "10000", "1000000"):
        cells = [b["saetze"][lb]["c"][Ckey] for lb in "ABC"]
        rows.append({
            "C": f"{int(Ckey):.0e}".replace("e+0", "e") if int(Ckey) >= 100 else Ckey,
            "Warnung (Grenze erreicht), von 100": " / ".join(str(c["warnung"]) for c in cells),
            "ohne Warnung und mit Verletzern (y f < 1), von 100": " / ".join(str(c["ohne_warnung_mit_verletzern"]) for c in cells),
            "Verletzer je Instanz, Median (Satz A)": f"{fmt_g(cells[0]['verletzer_median'], 3)} [{cells[0]['verletzer_min']}, {cells[0]['verletzer_max']}]",
            "Aktualisierungsschritte, Median": " / ".join(f"{int(c['schritte_median']):,}".replace(",", " ") for c in cells),
            "Marge 2/‖w‖, Median": " / ".join(fmt_g(c["marge_median"], 4) for c in cells),
        })
    return rows


def verdict_b2():
    b = studies().get("B2")
    if not b:
        return "info", "Eingefrorene Studie nicht gefunden."
    c4 = [b["saetze"][lb]["c"]["10000"]["ohne_warnung_mit_verletzern"] for lb in "ABC"]
    c4v = b["saetze"]["A"]["c"]["10000"]
    p12 = b["paarweise"]["1->100"]["alle"]
    p24 = b["paarweise"]["100->10000"]["alle"]
    b3 = studies().get("B3") or b.get("B3", {})
    txt = (f"Je 100 nicht trennbare Instanzen (n = 20, Gauß-Klassen mit Abstand 2, drei Sätze Zufalls-Seeds): `SVC` linear mit C = 1e4 antwortet in {c4[0]} / {c4[1]} / {c4[2]} von 100 Instanzen ohne Warnung "
           f"mit einer Geraden und einer „Marge“ 2/‖w‖ (Median {fmt_g(c4v['marge_median'], 3)} im Satz A); im Median verletzen {fmt_g(c4v['verletzer_median'], 2)} Punkte die harte Bedingung (Satz A). "
           f"Je Instanz sinkt die „Marge“ von C = 1 nach 100 im Median auf das {fmt(p12['median'], 2)}fache, von C = 100 nach 1e4 ändern sich {fmt(100 * p24['unter_1_prozent'], 0)} % der {p24['instanzen']} Instanzen um weniger als 1 %. "
           f"Zum Ersatz der harten Marge durch ein großes C: auf den hier gemessenen trennbaren Instanzen (Zwei Gebiete mit Puffer, linearer Kern, Merkmale der Größenordnung 1) genügt C = 1e4 (200 von 200 Instanzen richtig, Abweichung der Marge unter 1e-6 bei `SVC` mit tol 1e-10); "
           f"bei Merkmalen um 0,01 skaliert (19 von 20 Instanzen verfehlen die harte Bedingung) oder im Gauß-Kern (3 bis 5 von 100) genügt es nicht, auf nicht trennbaren Daten ebenso wenig. Die Warnungs- und Verletzerzahlen oben gelten für `SVC` mit tol 1e-3, linearer Kern, diesen Erzeuger.")
    return "error", txt


# --- Experiment 5: das Skalarprodukt tauschen ------------------------------------------------------------------------------------------------
def exp5_instance(data: str, seed: int):
    if data == C.E5_SETS[0]:
        return Sc.rings(C.RINGE_N_PER, seed)
    if data == C.E5_SETS[1]:
        return Sc.xor4()
    return Sc.liefergebiet(C.LIEFER_N, seed)


def exp5(data: str, seed: int, kern: str) -> dict:
    ins = exp5_instance(data, seed)
    kind, sigma = kern_parse(kern)
    sep = P.separability(ins.X, ins.y, kind)
    lin = P.lp_separable(ins.X, ins.y) is not None
    res = dict(inst=ins, kind=kind, sigma=sigma, sep=sep, linear_separable=lin)
    if not sep["separable"]:
        return res
    sol = solve_hard(ins.X, ins.y, kind, sigma, eps=1e-6, max_steps=C.HARD_MAX_STEPS)
    res["sol"] = sol
    res["sv_share"] = len(sol["sv"]) / len(ins.y)
    f_train = sol["K"] @ (sol["alpha"] * ins.y) + sol["b"]
    res["yf_min"] = float((ins.y * f_train).min())
    if data != C.E5_SETS[1]:
        te = Sc.rings(1000, seed + 500) if data == C.E5_SETS[0] else Sc.liefergebiet(2000, seed + 500)
        ft = M.kernel_cross(te.X, ins.X, kind, sigma) @ (sol["alpha"] * ins.y) + sol["b"]
        res["test_error"] = float((np.sign(ft) != te.y).mean())
        res["n_test"] = len(te.y)
    return res


def verdict_exp5(r: dict):
    ins = r["inst"]
    if not r["sep"]["separable"]:
        return "warning", (f"{ins.name}, {C.KERN_CHOICES[0] if r['kind'] == 'linear' else 'dieser Kern'}: nicht trennbar ({r['sep']['reason']}); es gibt keine harte Marge. "
                           f"Mit einem anderen Skalarprodukt (Gauß-Kern) wird dieselbe Punktmenge trennbar: verschiedene Punkte heißen dort immer trennbar.")
    sol = r["sol"]
    txt = (f"{ins.name}: trennbar, harte Marge 2/‖w‖ = {fmt(sol['margin'])} im Merkmalsraum, {len(sol['sv'])} von {ins.n} Punkten sind Stützvektoren ({fmt(100 * r['sv_share'], 0)} %), "
           f"{sol['steps']} Aktualisierungsschritte" + (f"; {sol['message']}" if sol["message"] else "") + ".")
    if "test_error" in r:
        txt += f" Testfehler {fmt(100 * r['test_error'], 2)} % an {r['n_test']} frischen Punkten desselben Modells."
    g = studies().get("G")
    if g and ins.name == "Ringe mit Lücke":
        z = g["zeilen"]
        best = max(g["sigmas"], key=lambda s: z[str(s)]["marge_median"])
        txt += (f" Studie ({g['n_seeds']} Zufalls-Seeds, 2 × 30 Punkte): linear trennbar in {g['linear_trennbar']} von {g['n_seeds']}; die mediane Marge hat ihr Maximum bei σ = {fmt_g(best, 2)} "
                f"({fmt(z[str(best)]['marge_median'])}), der Testfehler ist von σ = 0,4 bis 6,4 gleich 0: eine größere Marge zeigt nicht ein besseres σ.")
    return "info", txt
