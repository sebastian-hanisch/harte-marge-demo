"""Harte Marge und duales QP: welche trennende Gerade? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünf Experimente: viele trennende Geraden und die größte Marge, das duale Problem von Hand (Aufzählung der Stützmengen), die Wirkung einzelner Punkte, die Falle „nicht trennbar,
und der Löser meldet nichts“, das Skalarprodukt tauschen. Aufbau nach DEMO-PLAYBOOK §2.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import hmg_constants as C
import hmg_evaluation as ev
import hmg_texts as T
from hmg_layout import dunkles_design, kopf_css
from hmg_presets import (
    aktives_preset,
    bounds,
    center_theta,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from hmg_visualization import fig_boundary, fig_exp1, fig_margin_curve, fig_points_lines, fig_sigma_table, fig_trace

st.set_page_config(page_title="Harte Marge und duales QP – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _exp1(n, puffer, seed, theta, perc):
    return ev.exp1(n, puffer, seed, theta, perc)


@st.cache_data(show_spinner=False)
def _exp2(set_name, seed, kern):
    return ev.exp2(set_name, seed, kern)


@st.cache_data(show_spinner=False)
def _exp3(data, n, puffer, seed, k, dist, pos):
    return ev.exp3(data, n, puffer, seed, k, dist, pos)


@st.cache_data(show_spinner=False)
def _exp4_overlap(n, abstand, seed, kern, c, eps, steps):
    return ev.exp4_overlap(n, abstand, seed, kern, c, eps, steps)


@st.cache_data(show_spinner=False)
def _exp4_xor(kern, c, eps, steps):
    return ev.exp4_xor(kern, c, eps, steps)


@st.cache_data(show_spinner=False)
def _exp4_conflict(n, seed, kern, c, eps, steps):
    return ev.exp4_conflict(n, seed, kern, c, eps, steps)


@st.cache_data(show_spinner=False)
def _exp5(data, seed, kern):
    return ev.exp5(data, seed, kern)


def _cfmt(c):
    return f"{c:g}" if c < 100 else f"{c:.0e}".replace("e+0", "e")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in hmg_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Harte Marge und duales QP: welche trennende Gerade?")
st.markdown(T.INTRO)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/harte-marge-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich zwei Dörfer auf einem Plan vor, das eine mit roten, das andere mit blauen Häusern, getrennt durch freies Feld. Viele gerade Linien trennen die beiden Dörfer. Gesucht wird die Gerade in der Mitte der **breitesten Straße**, die zwischen den Dörfern Platz hat. Die Breite dieser Straße heißt **Marge**. Nur die Häuser am Rand der Straße, die **Stützvektoren**, legen sie fest: Wer ein anderes Haus wegnimmt, ändert nichts. Das setzt voraus, dass sich die Dörfer überhaupt durch eine Gerade trennen lassen. Ist das nicht der Fall, gibt es keine solche Straße, und manche Rechenprogramme liefern trotzdem ohne Warnung eine Gerade. Darum prüft die App zuerst, ob es eine Straße gibt.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/harte-marge-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die harte Marge", expanded=True):
    st.markdown(T.HOW)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Zufall")
    seed = st.number_input("Zufalls-Seed der Instanzen", *bounds("seed_input"), key="seed_input", step=1, help="Legt alle zufällig erzeugten Punktmengen und die Reihenfolgen des Perceptrons fest.")
    st.button("🎲 Neue Instanzen generieren", width="stretch", on_click=randomize_seed)
    st.header("📐 Experiment 1: Geraden")
    n1 = st.slider("Punktezahl (Zwei Gebiete mit Puffer)", *bounds("n1_slider"), key="n1_slider", help="Wie viele Punkte die zwei Gebiete beiderseits der Grenze haben (Experiment 1 und 3).")
    puffer = st.select_slider("Pufferbreite (Lücke je Seite)", options=C.PUFFER_OPTIONS, key="puffer_select", help="Abstand der Gebiete von der wahren Grenze: größer heißt leichter trennbar, größere Marge.")
    theta_off = st.slider("Richtung: Abweichung von der besten (Grad)", *bounds("theta_off"), key="theta_off", help="0 Grad ist die Richtung der größten Marge; je Richtung liegt die Gerade am besten in der Mitte des leeren Streifens.")
    st.button("Richtung der größten Marge", width="stretch", on_click=center_theta)
    perc_k = st.slider("Perceptron-Geraden (Reihenfolgen)", *bounds("perc_k"), key="perc_k", help="So viele Reihenfolgen der Punkte; jede gibt eine eigene trennende Gerade. 0 blendet sie aus.")
    st.header("🔢 Experiment 2: Duales Problem")
    e2_set = st.selectbox("Punktmenge", C.E2_SETS, key="e2_set", help="Die Aufzählung aller Stützmengen läuft bis n = 8; die Zufallspunkte sind 6 Punkte des Erzeugers „Zwei Gebiete mit Puffer“.")
    st.header("🧮 Experiment 3: Einzelne Punkte")
    e3_data = st.selectbox("Punktmenge ", C.E3_SETS, key="e3_data")
    e3_point = st.slider("Punkt Nr.", *bounds("e3_point"), key="e3_point", help="Der Punkt, der entfernt oder verschoben wird; sein Label bekommt auch der neue Punkt. Größere Nummern als die Punktezahl werden auf die letzte begrenzt.")
    e3_dist = st.select_slider("Abstand y·f des verschobenen und des neuen Punkts", options=C.E3_DIST_OPTIONS, key="e3_dist", help="0 liegt auf der Geraden, 1 auf der Randgeraden, darunter in der Marge.")
    e3_pos = st.select_slider("Lage des neuen Punkts längs der Geraden", options=C.E3_POS_OPTIONS, key="e3_pos", help="Einheit wie die Daten, vom Fußpunkt der Geraden aus gemessen.")
    st.header("⚠️ Experiment 4: Nicht trennbar")
    e4_abstand = st.select_slider("Klassenabstand (Gauß-Klassen)", options=C.E4_ABSTAND_OPTIONS, key="e4_abstand", help="Abstand der Klassenmitten in Einheiten der Streuung; 0 heißt vollständig überlappend.")
    e4_n = st.slider("Punktezahl (Gauß-Klassen)", *bounds("e4_n"), key="e4_n")
    e4_c = st.select_slider("C (Ersatz der harten Marge)", options=C.E4_C_OPTIONS, key="e4_c", format_func=_cfmt, help="Obergrenze der Multiplikatoren; die Größe C = 1e6 gibt es nur in der eingefrorenen Tabelle.")
    e4_tol = st.checkbox("Enge Toleranz (eps = 1e-8) statt der Voreinstellung 1e-3", key="e4_tol", help="Das eigene SMO hält an, wenn die KKT-Lücke unter eps liegt; 1e-3 ist die Voreinstellung von libsvm.")
    e4_steps = st.select_slider("Schrittgrenze des Lösers", options=C.E4_STEPS_OPTIONS, key="e4_steps", help="Bei Erreichen hält das SMO an und sagt es.")
    st.header("🔀 Kern und Experiment 5")
    kern = st.selectbox("Kern (Experimente 2, 4 und 5)", C.KERN_CHOICES, key="kern_choice", help="Der Gauß-Kern hat die Bandbreite σ (Längenmaß), γ = 1/(2σ²).")
    e5_data = st.selectbox("Punktmenge  ", C.E5_SETS, key="e5_data")

sync_query_params({k: st.session_state[k] for k in (
    "seed_input", "n1_slider", "puffer_select", "theta_off", "perc_k", "e2_set", "e3_data", "e3_point", "e3_dist", "e3_pos", "e4_abstand", "e4_n", "e4_c", "e4_tol", "e4_steps", "kern_choice", "e5_data")})

seed, n1, theta_off, perc_k, e3_point, e4_n = int(seed), int(n1), int(theta_off), int(perc_k), int(e3_point), int(e4_n)
eps4 = C.EPS_TIGHT if e4_tol else C.EPS_LIBSVM

aktiv = aktives_preset()
if aktiv:
    st.info(f"Preset „{aktiv}“: {C.PRESET_HELP.get(aktiv, '')}")

r1 = _exp1(n1, puffer, seed, theta_off, perc_k)
r2 = _exp2(e2_set, seed, kern)
r3 = _exp3(e3_data, n1, puffer, seed, e3_point, e3_dist, e3_pos)

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------------
st.markdown("## ❓ Kernfragen")
st.markdown("\n".join(f"* {q}" for q in T.KERNFRAGEN))
m1, m2, m3, m4 = st.columns(4)
m1.metric("Größte Marge 2/‖w‖ (Exp. 1)", ev.fmt(r1["m_max"]), delta=f"gewählte Richtung: {ev.fmt(max(r1['m_sel'], 0.0))}", delta_color="off")
m2.metric("Zulässige Stützmengen (Exp. 2)", f"{len(r2['feasible'])} von {r2['enum'].tested}", delta=("Marge " + ev.fmt(r2["margin"])) if r2["feasible"] else "keine harte Marge", delta_color="off")
m3.metric("Marge nach Hinzufügen (Exp. 3)", ev.fmt(r3["add"]["margin"]) if r3["add"] else "nicht trennbar", delta=f"vorher {ev.fmt(r3['base']['margin'])}", delta_color="off")
m4.metric("Stützvektoren in Exp. 1", f"{len(r1['sv'])} von {r1['inst'].n}", delta="2 oder 3 in 2D", delta_color="off")

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 1: Viele trennende Geraden, eine größte Marge")
st.caption(f"Zwei Gebiete mit Puffer, {n1} Punkte, Pufferbreite {ev.fmt(puffer, 2)}, Zufalls-Seed {seed}; Zähleinheit Instanzen. Schwarz: die Gerade der größten Marge mit ihren Randgeraden, "
           "blau: die gewählte Richtung (bester Streifen), grau: Perceptron-Geraden aus zufälligen Reihenfolgen. Stützvektoren sind umkreist.")
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(fig_exp1(r1), width="stretch", key="chart_exp1")
with c2:
    st.plotly_chart(fig_margin_curve(r1), width="stretch", key="chart_curve")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Marge der gewählten Richtung", ev.fmt(max(r1["m_sel"], 0.0)))
k2.metric("Größte Marge (SMO)", ev.fmt(r1["m_max"]), delta=f"Richtungsgitter: {ev.fmt(r1['grid_max'])}", delta_color="off")
k3.metric("Verhältnis", ev.fmt(max(r1["m_sel"], 0.0) / r1["m_max"], 3))
k4.metric("Abstand der Hüllen", ev.fmt(r1["hull"]), delta="gleiche Zahl, anderer Weg", delta_color="off")
kind, text = ev.verdict_exp1(r1)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 2: Das duale Problem von Hand")
st.caption(f"{r2['inst'].name}, Kern {kern}; Zähleinheit gemischte Stützmengen (Mengen mit Punkten beider Klassen). Je Menge ein lineares KKT-System, kein Optimierer.")
ca, cb = st.columns([3, 2])
with ca:
    en = r2["enum"]
    show = sorted(en.rows, key=lambda r: (r.status != "zulässig", len(r.support), r.support))[:60]
    df = pd.DataFrame({
        "Stützmenge (Punkte)": [", ".join(str(i + 1) for i in r.support) for r in show],
        "Ergebnis": [r.status for r in show],
        "Begründung": [r.detail for r in show],
        "b": [("–" if r.b is None else ev.fmt(r.b, 3)) for r in show],
    })
    st.dataframe(df, hide_index=True, width="stretch", height=300)
    if len(en.rows) > len(show):
        st.caption(f"Angezeigt: die ersten {len(show)} von {len(en.rows)} Mengen, zulässige zuerst.")
with cb:
    if r2["feasible"]:
        f0 = r2["feasible"][0]
        if r2["kind"] == "linear":
            from hmg_margin import primal_wb

            w0, b0 = primal_wb(r2["inst"].X, r2["inst"].y, f0.alpha, f0.b)
            st.plotly_chart(fig_points_lines(r2["inst"], w0, b0, sv=[i for i in f0.support], title="Lösung der zulässigen Stützmenge"), width="stretch", key="chart_exp2")
        else:
            st.plotly_chart(fig_boundary(r2["inst"], r2["kind"], r2["sigma"], f0.alpha, f0.b, sv=list(f0.support), title="f = 0 und f = ±1 im Eingaberaum"), width="stretch", key="chart_exp2")
    else:
        st.plotly_chart(fig_points_lines(r2["inst"], None, 0.0, title="keine trennende Gerade", note="Keine zulässige Stützmenge: nicht trennbar."), width="stretch", key="chart_exp2")
if r2["feasible"]:
    f0 = r2["feasible"][0]
    st.markdown("**α:** " + ", ".join(f"α{i + 1} = {ev.fmt(a, 4)}" for i, a in enumerate(f0.alpha) if a > 0) + f"; **b** = {ev.fmt(f0.b, 4)}; **Dualwert** = {ev.fmt(r2['dual'], 4)}.")
    if r2.get("fusspunkte") is not None:
        fp, fn = r2["fusspunkte"]
        st.caption(f"Fußpunkte der Hüllen (Σα/Σα·x je Klasse): ({ev.fmt(fp[0], 3)}; {ev.fmt(fp[1], 3)}) und ({ev.fmt(fn[0], 3)}; {ev.fmt(fn[1], 3)}), ihr Abstand ist die Marge. "
                   f"Gitter über 3 600 Richtungen: {ev.fmt(r2['grid'], 4)}; Abstand der konvexen Hüllen: {ev.fmt(r2['hull'], 4)}; eigenes SMO: {ev.fmt(r2['solution']['margin'], 4)} nach {r2['solution']['steps']} Aktualisierungsschritten.")
kind, text = ev.verdict_exp2(r2)
getattr(st, kind)(text)
d = ev.studies().get("D")
if d:
    st.markdown("**Stützvektoren zählen: Löser und Schwelle** (eingefroren, 100 Instanzen je Satz, n = 8, Zwei Gebiete mit Puffer): in wie vielen Instanzen die Menge {α > Schwelle · max α} gleich der exakten ist (Satz A / Satz B).")
    thr = d["schwellen"]
    rows = []
    for name in d["saetze"]["A"]["gleich"]:
        row = {"Löser (Einstellungen)": name}
        for t in thr:
            row["α > 0" if t == 0 else f"Schwelle {t:g}"] = f"{d['saetze']['A']['gleich'][name][str(t)]} / {d['saetze']['B']['gleich'][name][str(t)]}"
        row["Marge relativ, größte Abweichung"] = f"{d['saetze']['A']['marge_rel_max'][name]:.1e} / {d['saetze']['B']['marge_rel_max'][name]:.1e}".replace(".", ",")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("SLSQP: Dual, Startwert 0, ftol 1e-15, maxiter 1000 (bzw. scipy-Voreinstellungen, Startwert 0,1); trust-constr: gtol 1e-12, xtol 1e-14, maxiter 3000 (bzw. Voreinstellungen); "
               "`SVC` und eigenes SMO mit C = 1e4. Die Zahl bei α > 0 hängt an Rundungsresten und an den Einstellungen des Lösers; mit Schwelle 1e-3 zählen alle acht Einstellungen in 99 bis 100 von 100 Instanzen richtig, "
               "die Genauigkeit der Marge schwankt aber zwischen 5e-11 und 6e-4. `SVC` (libsvm) liefert nur Stützvektoren, das ist kein Vergleich der Löserqualität bei α > 0. "
               "Die Erfolgsmeldung von SLSQP sagt über die Richtigkeit in diesen Instanzen nichts: Primal `success` in "
               + " und ".join(str(ev.studies()["A"]["saetze"][lb]["primal_success"]) for lb in "AB") + " von je 200 Instanzen, die Marge ist in "
               + " und ".join(str(ev.studies()["A"]["saetze"][lb]["primal_richtig"]) for lb in "AB") + " richtig.")

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 3: Wie viel hängt an einzelnen Punkten?")
st.caption(f"{r3['inst'].name}, linearer Kern; gewählt ist Punkt {r3['k'] + 1} (Label {'+1' if r3['label'] > 0 else '−1'}). Zähleinheit Instanzen. "
           "Entfernen, Verschieben auf den Abstand y·f und Hinzufügen eines Punkts mit demselben Label sind drei getrennte Rechnungen auf der Ausgangsinstanz.")
cc1, cc2 = st.columns(2)
with cc1:
    st.plotly_chart(fig_points_lines(r3["inst"], r3["base"]["w"], r3["base"]["b"], sv=r3["base"]["sv"], title="vorher"), width="stretch", key="chart_exp3a")
with cc2:
    if r3["add"] is not None:
        Xn = np.vstack([r3["inst"].X, r3["p_add"]])
        yn = np.r_[r3["inst"].y, r3["label"]]
        from hmg_scenario import Instance

        st.plotly_chart(fig_points_lines(Instance(Xn, yn, "nachher"), r3["add"]["w"], r3["add"]["b"], sv=r3["add"]["sv"], title="nachher: Punkt hinzugefügt (Raute)", extra_points=r3["p_add"]), width="stretch", key="chart_exp3b")
    else:
        st.plotly_chart(fig_points_lines(r3["inst"], None, 0.0, title="nachher: nicht trennbar", note="Der neue Punkt liegt auf der falschen Seite: nicht trennbar.", extra_points=r3["p_add"]), width="stretch", key="chart_exp3b")
rows3 = []
for key, name in (("remove", "Punkt entfernen"), ("move", "Punkt verschieben"), ("add", "Punkt hinzufügen")):
    x = r3[key]
    rows3.append({"Aktion": name, "Marge vorher": ev.fmt(r3["base"]["margin"]), "Marge nachher": ev.fmt(x["margin"]) if x else "nicht trennbar",
                  "Verhältnis": ev.fmt(x["ratio"], 3) if x else "–", "Drehung der Geraden (Grad)": ev.fmt(x["angle"], 1) if x else "–",
                  "Stützvektoren vorher → nachher": f"{len(r3['base']['sv'])} → {len(x['sv'])}" if x else "–"})
st.dataframe(pd.DataFrame(rows3), hide_index=True, width="stretch")
kind, text = ev.verdict_exp3(r3)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 4: Die Falle, nicht trennbar, und der Löser meldet nichts")
st.caption(f"Zähleinheit Instanzen und Aktualisierungsschritte. **Zuerst der Test „trennbar?“** (lineares Programm in Gleitkomma, bei winzigen Margen nicht zuverlässig, siehe „Wo die Annahmen enden“; im Gauß-Kern: verschiedene Punkte heißen immer trennbar), dann das, was das eigene SMO "
           f"mit C = {_cfmt(e4_c)}, eps = {eps4:g} und Schrittgrenze {e4_steps} liefert. Die App meldet, was libsvm nicht meldet.")
with st.spinner(f"Das eigene SMO rechnet (bis zu {e4_steps} Aktualisierungsschritte je Fall) …"):
    r4a = _exp4_overlap(e4_n, e4_abstand, seed, kern, e4_c, eps4, e4_steps)
    r4b = _exp4_xor(kern, e4_c, eps4, e4_steps)
    r4c = _exp4_conflict(11, seed, kern, e4_c, eps4, e4_steps)
st.markdown("**Fall 1: überlappende Gauß-Klassen**")
cA, cB = st.columns([3, 2])
with cA:
    sol = r4a["sol"]
    if r4a["kind"] == "linear":
        w4 = (sol["alpha"] * r4a["inst"].y) @ r4a["inst"].X
        fig4 = fig_points_lines(r4a["inst"], w4, sol["b"], sv=[i for i in range(r4a["inst"].n) if sol["alpha"][i] > 1e-9], title=f"Gerade des Lösers mit C = {_cfmt(e4_c)} (Randgeraden y·f = ±1)",
                                note="w = 0: keine Gerade")
    else:
        fig4 = fig_boundary(r4a["inst"], r4a["kind"], r4a["sigma"], sol["alpha"], sol["b"], sv=[i for i in range(r4a["inst"].n) if sol["alpha"][i] > 1e-9], title=f"f = 0 und f = ±1 mit C = {_cfmt(e4_c)}")
    st.plotly_chart(fig4, width="stretch", key="chart_exp4a")
with cB:
    if sol["trace"] is not None and len(sol["trace"]["step"]) > 1:
        st.plotly_chart(fig_trace(sol["trace"], eps4, e4_steps), width="stretch", key="chart_exp4_trace")
kind, text = ev.verdict_trap(r4a, ev.note_overlap() if r4a['kind'] == 'linear' else '')
getattr(st, kind)(text)
st.markdown("**Fall 2: XOR (vier Punkte)**")
cA2, cB2 = st.columns([2, 3])
with cA2:
    sol_b = r4b["sol"]
    if r4b["kind"] == "linear":
        w4b = (sol_b["alpha"] * r4b["inst"].y) @ r4b["inst"].X
        fig4b = fig_points_lines(r4b["inst"], w4b, sol_b["b"], title="XOR: w = 0, keine Gerade", note="w = 0: keine Gerade, „Marge“ nicht definiert")
    else:
        fig4b = fig_boundary(r4b["inst"], r4b["kind"], r4b["sigma"], sol_b["alpha"], sol_b["b"], title="XOR mit diesem Kern", height=340)
    st.plotly_chart(fig4b, width="stretch", key="chart_exp4b")
with cB2:
    kind, text = ev.verdict_trap(r4b, ev.note_xor() if r4b['kind'] == 'linear' else '')
    getattr(st, kind)(text)
    b1 = ev.studies().get("B1")
    if b1:
        st.caption("libsvm (`SVC`, linear, tol 1e-3, eingefroren): " + "; ".join(
            f"C = {_cfmt(float(c))}: w = ({', '.join(ev.fmt(v, 0) for v in b1['svc'][c]['w'])}), alle α = {ev.fmt_g(b1['svc'][c]['alpha'][0], 3)}, {b1['svc'][c]['n_iter']} Aktualisierungsschritte, "
            f"Warnung: {'ja' if b1['svc'][c]['warnung'] else 'nein'}" for c in b1["svc"]))
st.markdown("**Fall 3: ein Punkt mit beiden Labels**")
kind, text = ev.verdict_trap(r4c, ev.note_conflict() if r4c['kind'] == 'gauss' else '')
getattr(st, kind)(text)
b4 = ev.studies().get("B4")
if b4:
    st.caption(f"Studie (eingefroren, {b4['instanzen']} Instanzen, Gauß-Kern σ = 1, C = 1e4): `SVC` gibt in {b4['instanzen'] - b4['warnung']} von {b4['instanzen']} Instanzen keine Warnung, "
               f"{ev.fmt_g(b4['verletzer_min'], 2)} bis {ev.fmt_g(b4['verletzer_max'], 2)} Punkte verletzen die Bedingung, {int(b4['schritte_min'])} bis {int(b4['schritte_max'])} Aktualisierungsschritte.")
st.markdown("**Studie mit libsvm (eingefroren): nicht trennbare Gauß-Klassen, drei Sätze zu je 100 Instanzen** (Spalten: Satz A / B / C)")
st.dataframe(pd.DataFrame(ev.b2_table()), hide_index=True, width="stretch")
b5 = ev.studies().get("B5")
if b5:
    st.caption("Gegenzeile (Messung eines anderen Bausteins der Reihe, nicht hier neu gemessen; Gauß-Kern σ = 1, Gauß-Klassen mit 10 % vertauschten Labels, n = 60, `SVC` mit Schrittgrenze 100 000): "
               "Warnung bei C = 1e2 / 1e4 / 1e6 / 1e8 in " + " / ".join(str(v) for v in b5["warnung_gepoolt"]) + f" von {b5['von']} Zufalls-Seeds. Andere Grenze, anderer Kern, anderes n, andere Daten als die Tabelle darüber: verschieden, nicht widersprüchlich. "
               "Gemeinsam gilt: bei C = 1e4 warnt der Löser kaum, bei C ≥ 1e6 an der Grenze oft, und der Rat „C = 1e4 nehmen“ gilt nur für trennbare Daten.")
kind, text = ev.verdict_b2()
getattr(st, kind)(text)
st.caption(T.live_vs_studie())

st.markdown("---")

# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 5: Das Skalarprodukt tauschen")
st.caption(f"{e5_data}, Kern {kern}; Zähleinheit Zufalls-Seeds (Studie) und Aktualisierungsschritte (live). Die Marge 2/‖w‖ ist die im Merkmalsraum des Kerns.")
with st.spinner("Das eigene SMO rechnet …"):
    r5 = _exp5(e5_data, seed, kern)
if r5["sep"]["separable"]:
    sol5 = r5["sol"]
    st.plotly_chart(fig_boundary(r5["inst"], r5["kind"], r5["sigma"], sol5["alpha"], sol5["b"], sv=sol5["sv"], title="f = 0 und f = ±1 im Eingaberaum"), width="stretch", key="chart_exp5")
else:
    st.plotly_chart(fig_points_lines(r5["inst"], None, 0.0, title="nicht trennbar in diesem Kern", note="Keine harte Marge: nicht trennbar."), width="stretch", key="chart_exp5")
kind, text = ev.verdict_exp5(r5)
getattr(st, kind)(text)
g = ev.studies().get("G")
if g:
    cG1, cG2 = st.columns([2, 3])
    with cG1:
        st.plotly_chart(fig_sigma_table(g), width="stretch", key="chart_sigma")
    with cG2:
        st.dataframe(pd.DataFrame([{"σ": ev.fmt_g(s, 2), "Marge (Median)": ev.fmt(g["zeilen"][str(s)]["marge_median"], 4), "Anteil Stützvektoren (Median)": ev.fmt(g["zeilen"][str(s)]["sv_median"], 2),
                                    "Testfehler (Mittel)": ev.fmt(g["zeilen"][str(s)]["testfehler_mittel"], 4)} for s in g["sigmas"]]), hide_index=True, width="stretch")
        st.caption(f"Grenzwert σ → 0: Gram-Matrix → Einheitsmatrix, alle α = 1, Marge 2/√60 = {ev.fmt(g['grenzwert_klein_sigma'], 4)}. Die Vorab-Vermutung „die Marge wächst mit kleiner werdendem σ“ ist widerlegt. "
                   f"Bei σ = 6,4 ist die harte Bedingung auf 1e-3 erreicht ({g['zeilen']['6.4']['verfehlt_1e3']} von {g['n_seeds']} verfehlen sie), auf 1e-5 verfehlen sie {g['zeilen']['6.4']['verfehlt_1e5']} von {g['n_seeds']}: "
                   "Löser-Genauigkeit, keine Warnung. Zahl der Läufe klein: Bänder, keine Rangfolge.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.assumptions_md())

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/harte-marge-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind Punkte $x_i$ mit Etiketten $y_i \in \{-1,+1\}$ und ein Kern $K$, das Skalarprodukt im Merkmalsraum. Gesucht ist die trennende Funktion $f$ mit der größten Marge; mit linearem Kern ist das eine Gerade. Das Primalproblem sucht sie direkt, das duale Problem über einen Multiplikator je Punkt.

| Symbol | Bedeutung |
|---|---|
| $x_i$, $y_i$ | Punkt $i$ und sein Etikett, $-1$ oder $+1$ |
| $\varphi(x)$, $K$ | Abbildung in den Merkmalsraum und Kern $K(x,z)=\varphi(x)\cdot\varphi(z)$ (linear: $\varphi(x)=x$) |
| $w$, $b$ | Normalenvektor und Verschiebung der Trennfunktion $f(x)=w\cdot\varphi(x)+b$ |
| $\alpha_i$ | Multiplikator (Dualvariable) des Punkts $i$; ein positives $\alpha_i$ heißt Stützvektor |
| $Q_{ij}$ | $y_i y_j K(x_i,x_j)$ |
| Marge | $2/\lVert w\rVert$: Abstand der Randgeraden $y f=\pm1$ (nicht $1/\lVert w\rVert$) |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/harte-marge-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(T.FORMULAS)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kernmethoden: Skalarprodukte tauschen](https://sebastianhanisch.net/konzepte-kernmethoden.html)."
)
