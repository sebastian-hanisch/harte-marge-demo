"""Texte der App: Einleitung, Erklärungen, Annahmen, Formeln. Zahlen aus den eingefrorenen Studien kommen über `hmg_evaluation.studies()`; die Zahl-gegen-Test-Tabelle der
README und `tests/test_claims.py` halten sie fest. Die Annahmentabelle steht wortgleich in der README (ein Test vergleicht beide Fassungen)."""

import hmg_evaluation as ev

INTRO = """
Zwei Gruppen von Punkten lassen sich oft durch viele verschiedene Geraden trennen. Diese Demo zeigt, **welche davon die größte Marge hat**, warum sie (bei trennbaren Daten) eindeutig ist und nur an wenigen
**Stützvektoren** hängt, und **was ein Löser verschweigt**, wenn die Voraussetzung „trennbar“ gar nicht gilt. Alle Punkte sind **ausgedachte Modelle** (zwei Gebiete mit Pufferstreifen,
überlappende Gauß-Klassen, zwei Ringe), keine Messdaten. Die **Marge heißt hier 2/‖w‖**, der Abstand der beiden Randgeraden; viele Lehrtexte meinen 1/‖w‖, die halbe Breite.

Weiter unten: fünf Experimente, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

HOW = """
1. **Trennbar** heißt: es gibt eine Gerade (allgemein: eine Funktion f im Merkmalsraum eines Kerns) mit y·f ≥ 1 für jeden Punkt. Das prüft ein lineares Programm (in Gleitkomma mit den Toleranzen des Lösers; bei winzigen Margen kann es fälschlich „nicht trennbar“ melden, siehe „Wo die Annahmen enden“).
2. **Harte Marge:** unter allen trennenden Geraden die mit dem kleinsten ‖w‖, also der größten Marge 2/‖w‖. Das Problem ist in w streng konvex, deshalb gibt es (bei trennbaren Daten) genau ein w und ein b.
3. **Das duale Problem** hat einen Multiplikator α je Punkt; Punkte mit α > 0 sind die **Stützvektoren** und liegen genau auf den Randgeraden. Außer ihnen hängt die Lösung an keinem Punkt.
4. **Das Skalarprodukt tauschen:** ersetzt man x·z durch eine Kernfunktion K(x, z), rechnet dasselbe Verfahren mit einer Gerade im Merkmalsraum; die Gram-Matrix K ist alles, was der Löser sieht.
"""

KERNFRAGEN = (
    "1. Welche trennende Gerade hat die größte Marge? (Experiment 1)",
    "2. Was rechnet das duale Problem, und wie viele Punkte sind Stützvektoren? (Experiment 2)",
    "3. Wie viel hängt an einzelnen Punkten? (Experiment 3)",
    "4. Was passiert, wenn es keine trennende Gerade gibt? (Experiment 4)",
    "5. Was ändert ein anderes Skalarprodukt? (Experiment 5)",
)

LIVE_VS_STUDIE = (
    "**Live-Modell ≠ Studie.** Die App rechnet mit einem eigenen SMO (zwei Multiplikatoren je Schritt, Arbeitsmenge zweiter Ordnung) und mit einer Schrittgrenze; die eingefrorenen Tabellen "
    "stammen von libsvm (`SVC`, in der App nicht enthalten, nur im Messwerkzeug). Bei konvergiertem SMO geben beide denselben Dualwert (relativ 1e-9, 25 von 30 Instanzen); an der Schrittgrenze (5 von 30) liegt der Dualwert des SMO um 0,04 bis 0,5 % tiefer. "
    "Verschieden sind die Zahl der Aktualisierungsschritte und die Meldung an der Grenze: {smo_text}"
)


def live_vs_studie() -> str:
    b = (ev.studies().get("B2") or {}).get("smo")
    if not b:
        return LIVE_VS_STUDIE.format(smo_text="(Studie nicht gefunden)")
    s = b["10000"]
    w = ev.studies()["B2"]["saetze"]["A"]["c"]["10000"]["warnung"]
    t = (f"bei C = 1e4 auf 30 nicht trennbaren Instanzen (n = 20) hält das eigene SMO in {s['nicht_konvergiert']} von {s['instanzen']} an der Schrittgrenze 200 000 an und sagt es, "
         f"`SVC` (Grenze 2 000 000) warnt in {w} von 100 Instanzen (Satz A); die „Marge“ 2/‖w‖ beider Löser liegt je Instanz im Verhältnis {ev.fmt(s['marge_verhaeltnis_min'], 3)} bis "
         f"{ev.fmt(s['marge_verhaeltnis_max'], 3)} (Median {ev.fmt(s['marge_verhaeltnis_median'], 4)}).")
    return LIVE_VS_STUDIE.format(smo_text=t)


def assumptions_md() -> str:
    st = ev.studies()
    b2 = st.get("B2", {})
    c4 = [b2["saetze"][lb]["c"]["10000"]["ohne_warnung_mit_verletzern"] for lb in "ABC"] if b2 else ["?", "?", "?"]
    d = st.get("D", {}).get("saetze", {})
    sl = [d[lb]["gleich"]["SLSQP (dual)"]["0.0"] for lb in "AB"] if d else ["?", "?"]
    slv = [d[lb]["gleich"]["SLSQP (Voreinstellungen)"]["0.0"] for lb in "AB"] if d else ["?", "?"]
    sv = [d[lb]["gleich"]["SVC tol=1e-10"]["0.0"] for lb in "AB"] if d else ["?", "?"]
    sm = [d[lb]["gleich"]["SMO eps=1e-10"]["0.0"] for lb in "AB"] if d else ["?", "?"]
    sm3 = [d[lb]["gleich"]["SMO eps=1e-3"]["0.0"] for lb in "AB"] if d else ["?", "?"]
    return f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Trennbar** (alle y·f ≥ 1 erreichbar) | `SVC` mit C = 1e4 antwortet bei je 100 nicht trennbaren Instanzen (n = 20, Gauß-Klassen mit Abstand 2) in {c4[0]} / {c4[1]} / {c4[2]} von 100 ohne Warnung mit einer Geraden; die App prüft die Voraussetzung selbst (Experiment 4). | die weiche Marge ist nicht Gegenstand dieser Demo |
| **Eine eindeutige Lösung** | w und b sind eindeutig, α nur in Allgemeinlage: beim Rechteck (vier Punkte auf den Randgeraden) ist α nicht eindeutig: eine Strecke von Lösungen mit demselben w und b; die Aufzählung über Stützmengen findet deren zwei Endpunkte. | – |
| **Stützvektoren zählen** | Die Zahl hängt an den Einstellungen des Lösers: bei α > 0 zählt SLSQP in {sl[0]} und {sl[1]} von je 100 Instanzen richtig (zwei Sätze, Startwert 0), mit scipy-Voreinstellungen in {slv[0]} und {slv[1]}; libsvm in {sv[0]} und {sv[1]}, eigenes SMO mit eps 1e-10 in {sm[0]} und {sm[1]}, mit eps 1e-3 (Einstellung der App) in {sm3[0]} und {sm3[1]} (libsvm liefert nur Stützvektoren, ein Vergleich der Löserqualität bei α > 0 ist das nicht). | – |
| **Trennbarkeitstest** | Das lineare Programm (HiGHS, Toleranzen der Voreinstellung) rechnet in Gleitkomma. Gegenbeispiel: die Punkte (0, ε), (1, 1+ε), (0, −ε), (1, 1−ε) mit y = +, +, −, − sind für jedes ε > 0 trennbar; der Test meldet bei ε = 1e-3 und 1e-6 richtig „trennbar“, bei ε = 1e-9 und 1e-12 fälschlich „nicht trennbar“. Bei den Instanzen und Reglern der Demo (Marge nicht unter etwa 0,01) ohne Auswirkung. | – |
| **Ein zulässiger Kern** | Linearer, Polynom-2- und Gauß-Kern sind zulässig; ob eine Funktion überhaupt ein Kern ist, wird hier nicht geprüft. | – |
| **Zwei Dimensionen, höchstens acht Punkte** | Hüllenabstand und Richtungsgitter als unabhängige Rechenwege gibt es nur in 2D, die Aufzählung der Stützmengen nur bis n = 8 (2ⁿ Mengen); größere Instanzen werden nur gegen Fremdlöser geprüft. | – |
| **Ein Erzeuger mit Lücke** | „Perceptron gegen größte Marge“ ist nur an dem Erzeuger „Zwei Gebiete mit Puffer“ gemessen; seine Lücke kommt der Marge entgegen. Für andere Daten ist nichts behauptet. | – |
| **Nutzen im Betrieb** | Ob eine Marge-Methode in Logistik, Lager oder Fertigung bessere Entscheidungen liefert, ist in den gelesenen Quellen nicht gemessen und nicht Gegenstand. | – |

**Nicht gebaut und nicht gemessen** (kein Regler, kein Versprechen): die Einheit eines Merkmals (Skalierung), Dimension über 2 für die unabhängigen Rechenwege, `SVC` ohne Schrittgrenze auf nicht
trennbaren Daten (die Laufzeit ist dort unbeschränkt), die Einstellungen `shrinking` und `cache_size` von `SVC`, ein zweiter Erzeuger ohne Puffer und mit Ausreißern für „Perceptron gegen
größte Marge“, ν-SVM, `class_weight` und Platt-Skalierung.
"""


FORMULAS = r"""
**Lesehilfe.** Gegeben sind Punkte $x_i$ mit Labels $y_i \in \{-1, +1\}$ ($i = 1, \dots, n$, beide Klassen kommen vor) und ein Kern $K$ (Skalarprodukt im Merkmalsraum). Gesucht ist die trennende
Funktion $f$ mit der größten Marge.

| Symbol | Bedeutung |
|---|---|
| $\varphi(x)$ | Abbildung in den Merkmalsraum, $K(x, z) = \varphi(x) \cdot \varphi(z)$ (linear: $\varphi(x) = x$) |
| $w, b$ | Normalenvektor und Verschiebung; $f(x) = w \cdot \varphi(x) + b$ |
| $\alpha_i$ | Multiplikator (Dualvariable) des Punkts $i$; $\alpha_i > 0$ heißt Stützvektor |
| $Q_{ij}$ | $y_i y_j K(x_i, x_j)$ |
| Marge | $2/\lVert w \rVert$, Abstand der Randgeraden $y f = \pm 1$ (nicht $1/\lVert w \rVert$) |

**Trennbar** heißt: es gibt $(w, b)$ mit $y_i f(x_i) \ge 1$ für alle $i$. **Primal** (harte Marge):

$$\min_{w,\,b} \; \tfrac12 \lVert w \rVert^2 \quad \text{unter} \quad y_i \,(w \cdot \varphi(x_i) + b) \ge 1, \quad i = 1, \dots, n.$$

**Dual:**

$$\max_{\alpha} \; \sum_i \alpha_i - \tfrac12 \sum_{i,j} \alpha_i \alpha_j \, Q_{ij} \quad \text{unter} \quad \alpha_i \ge 0, \quad \sum_i y_i \alpha_i = 0.$$

**Zusammenhang und KKT-Bedingung:**

$$w = \sum_i \alpha_i y_i \varphi(x_i), \qquad \alpha_i \,\bigl(y_i f(x_i) - 1\bigr) = 0 .$$

Der Dualwert ist $\tfrac12 \lVert w \rVert^2 = 2/\text{Marge}^2$. **Nicht trennbar** (Zertifikat, Alternativsatz von Gordan als eigene Rechnung, nicht aus einer gelesenen Quelle): genau dann, wenn es
$\alpha \ge 0$, $\alpha \ne 0$ gibt mit

$$\sum_i \alpha_i y_i = 0 \quad \text{und} \quad \sum_i \alpha_i y_i \varphi(x_i) = 0 ;$$

dann gilt $\alpha^\top Q \alpha = 0$ und das Dualziel wächst längs $t\alpha$ ohne Grenze (beim XOR mit linearem Kern: $\alpha = (1, 1, 1, 1)$).

Das Modell ist vollständig für die harte Marge; die weiche Marge (Schlupfvariablen, Schranke $C$) fehlt bewusst. **Quellen** (nur der gelesene Umfang, in eigenen Worten): Cortes und Vapnik 1995
(Nebenbedingung, Marge $2/|w|$, Dual, KKT); Platt 1998 (KKT-Bedingungen, SMO); Fan, Chen und Lin 2005 (Arbeitsmenge zweiter Ordnung). Die Aussage „Dual = nächste Punkte der konvexen Hüllen“ ist hier
nur als eigene Rechnung bestätigt.
"""
