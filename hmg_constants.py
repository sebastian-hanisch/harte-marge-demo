"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

SEED_MAX, DEFAULT_SEED = 999999, 7

# Experiment 1 und 3: Zwei Gebiete mit Puffer (linear)
N1_MIN, N1_MAX, DEFAULT_N1 = 6, 40, 20
PUFFER_OPTIONS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)
DEFAULT_PUFFER = 0.25
THETA_MIN, THETA_MAX, DEFAULT_THETA = -90, 90, 0
PERC_MIN, PERC_MAX, DEFAULT_PERC = 0, 8, 0

# Experiment 2: duales Problem (Aufzählung, n <= 8)
E2_SETS = ("Fünf Punkte in der Ebene (Handbeispiel)", "Rechteck (α nicht eindeutig)", "XOR (vier Punkte)", "Zufallspunkte (n = 6)")
DEFAULT_E2 = E2_SETS[0]
E2_RANDOM_N = 6

# Experiment 3: wie viel hängt an einzelnen Punkten
E3_SETS = ("Zwei Gebiete mit Puffer (Experiment 1)", "Fünf Punkte in der Ebene (Handbeispiel)")
DEFAULT_E3 = E3_SETS[0]
E3_POINT_MIN, E3_POINT_MAX, DEFAULT_E3_POINT = 1, 40, 2
E3_DIST_OPTIONS = tuple(round(0.1 * k, 1) for k in range(0, 31))  # y f(x) des neuen bzw. verschobenen Punkts: 0 = auf der Geraden, 1 = auf der Randgeraden
DEFAULT_E3_DIST = 0.3
E3_POS_OPTIONS = tuple(round(-4.0 + 0.1 * k, 1) for k in range(0, 81))  # Lage längs der Geraden, Einheit wie die Daten
DEFAULT_E3_POS = 0.0

# Experiment 4: nicht trennbar, und der Löser meldet nichts
E4_ABSTAND_OPTIONS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
DEFAULT_E4_ABSTAND = 2.0
E4_N_MIN, E4_N_MAX, DEFAULT_E4_N = 10, 40, 20
E4_C_OPTIONS = (1, 100, 10000)
DEFAULT_E4_C = 100
E4_STEPS_OPTIONS = (10000, 50000, 100000, 200000)
DEFAULT_E4_STEPS = 100000
EPS_LIBSVM, EPS_TIGHT = 1e-3, 1e-8

# Experiment 5 und Kern (auch Experiment 2 und 4): ein Regler für Kern und Bandbreite
KERN_CHOICES = ("linear", "Polynom Grad 2", "Gauß σ = 0,1", "Gauß σ = 0,2", "Gauß σ = 0,4", "Gauß σ = 0,8", "Gauß σ = 1", "Gauß σ = 1,6", "Gauß σ = 3,2", "Gauß σ = 6,4")
DEFAULT_KERN = KERN_CHOICES[0]
E5_SETS = ("Ringe mit Lücke", "XOR (vier Punkte)", "Liefergebiet (Kreis)")
DEFAULT_E5 = E5_SETS[0]
LIEFER_N = 60
RINGE_N_PER = 30

SMO_EPS_HARD = 1e-9  # harte Marge in der App: Abbruch bei KKT-Lücke unter 1e-9
HARD_MAX_STEPS = 200000

# eingefrorene Studien (tools/hmg_make_b2.py)
STUDY_FILE = "hmg_b2.json"

COLOR_POS = "#2F6B65"
COLOR_NEG = "#B5543A"
COLOR_LINE = "#222222"
COLOR_MARGIN = "#7a7a7a"
COLOR_ALT = "#4c78a8"

PRESETS = {
    "Fünf Punkte in der Ebene (Handbeispiel)": {},
    "Viele Geraden": {"perc": 8, "theta": 20},
    "Rechteck (α nicht eindeutig)": {"e2": E2_SETS[1]},
    "XOR linear: nicht trennbar": {"e4_c": 100, "e4_abstand": 0.0},
    "Überlappende Klassen, C = 1e4": {"e4_c": 10000},
    "XOR mit Gauß-Kern": {"e2": E2_SETS[2], "kern": "Gauß σ = 1", "e5": "XOR (vier Punkte)"},
    "Ein Punkt in der Marge": {"e3": E3_SETS[1], "e3_point": 2, "e3_dist": 0.8, "e3_pos": 3.7},
    "Zwei Ringe, Gauß-Kern": {"e5": E5_SETS[0], "kern": "Gauß σ = 0,8"},
}
PRESET_HELP = {
    "Fünf Punkte in der Ebene (Handbeispiel)": "Experiment 2: fünf Punkte, 21 gemischte Stützmengen, genau eine zulässig (α = 6/5, 2, 4/5, 0, 0), Marge 1. Von Hand in Brüchen nachrechenbar.",
    "Viele Geraden": "Experiment 1: acht Reihenfolgen des Perceptrons geben verschiedene trennende Geraden, die größte Marge ist eine einzige.",
    "Rechteck (α nicht eindeutig)": "Experiment 2: vier Punkte auf den Randgeraden: w und b eindeutig, α nicht (eine Strecke von Lösungen; die Aufzählung findet deren zwei Endpunkte).",
    "XOR linear: nicht trennbar": "Experiment 4: vier Punkte, die keine Gerade trennt. Die App prüft die Voraussetzung und zeigt das Zertifikat; libsvm gibt w = 0 ohne Warnung.",
    "Überlappende Klassen, C = 1e4": "Experiment 4: nicht trennbare Gauß-Klassen mit C = 1e4 als Ersatz der harten Marge; das eigene SMO rechnet live, die Studie mit libsvm steht eingefroren darunter.",
    "XOR mit Gauß-Kern": "Experiment 2 und 5: dasselbe XOR, aber mit Gauß-Kern σ = 1 ist es trennbar; die Marge im Merkmalsraum ist 1 − exp(−2/σ²).",
    "Ein Punkt in der Marge": "Experiment 3: ein richtig etikettierter Punkt in der Marge senkt die Marge im Handbeispiel von 1 auf 9/10, ohne die Gerade zu drehen.",
    "Zwei Ringe, Gauß-Kern": "Experiment 5: kein linearer Schnitt, mit Gauß-Kern σ = 0,8 trennbar; die mediane Marge hat dort ihr Maximum, ist aber kein Gütemaß.",
}
