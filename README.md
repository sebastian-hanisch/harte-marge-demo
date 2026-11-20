# Harte Marge und duales QP: welche trennende Gerade? – Streamlit-Demo

Diese Demo ist die zweite der Reihe **kernmethoden** („Skalarprodukte tauschen“) in der „Konzepte“-Reihe der Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie knüpft an das Perceptron an:
Dort genügt irgendeine trennende Gerade, und welche herauskommt, hängt an der Reihenfolge der Punkte. Hier gibt es bei trennbaren Daten unter allen trennenden Geraden **genau eine mit der größten Marge** (w und b eindeutig,
α nur in Allgemeinlage), sie hängt nur an den **Stützvektoren**, und das duale Problem lässt sich mit allen Zwischenschritten zeigen. Zweiter Teil der Demo: die Falle **„nicht trennbar, und der Löser meldet nichts“**.
Die Punkte sind **ausgedachte Modelle** (keine Messdaten); gezählt werden Instanzen, Zufalls-Seeds und Aktualisierungsschritte. Die **Marge heißt hier 2/‖w‖**, der Abstand der beiden Randgeraden
(`perceptron-demo` rechnet mit dem Abstand zur Ebene, der **halben** Marge: beim Vergleichen umrechnen).

## Kernfrage

Zwei Gruppen von Punkten lassen sich oft durch viele Geraden trennen. Welche ist die beste, was rechnet das duale Problem, wie viel hängt an einzelnen Punkten, und was passiert, wenn es gar keine
trennende Gerade gibt? Fünf Experimente beantworten das live; die Zahlen der Studien stehen eingefroren darunter.

## Befunde und Korrekturen gegenüber der Vormessung

Die Vormessung der Planung lief mit `numpy.random.default_rng`; die App zieht aus SplitMix64 (`hmg_rng.py`). Darum sind alle Zahlen dieser README **im Bau neu gemessen** (Bänder über frische Zufalls-Seeds,
`tools/hmg_make_b2.py` mit scikit-learn nur im Werkzeug) und weichen leicht ab. Wichtigste Abweichungen: (1) Perceptron besser als die größte Marge in 18 und 18 von 200 Instanzen
(zwei Sätze, symmetrische Lücke; eine Instanz zählt hier als Mittel über 8 Reihenfolgen) und in 3 von 200 bei unsymmetrischer Lücke (Vormessung 16, 20 und 11): die größte Marge gewinnt etwas deutlicher als vermutet, „immer“ gilt trotzdem nicht.
(2) `SVC` antwortet bei C = 1e4 in 98 / 100 / 99 von 100 Instanzen ohne Warnung (Vormessung 98 / 99 / 98).
(3) Das eigene SMO (das Live-Modell der App) hält bei C = 1e4 auf 30 nicht trennbaren Instanzen in 5 an der Schrittgrenze 200 000 an; `SVC` hat dort die Grenze 2 000 000 und warnt in
2 von 100: **Live-Modell ≠ Studie**, die App sagt es. (4) Bei α > 0 zählt das eigene SMO mit eps 1e-3 in 99 und 100 von je 100 Instanzen
die richtigen Stützvektoren (die Vormessung hatte 100). (5) Der Regler „n“ der Aufzählung entfällt: die Zufallspunkte haben fest n = 6 (ein Regler, der bei den übrigen Punktmengen wirkungslos wäre, wird
nicht angeboten). (6) Das frühere Experiment „Einheit eines Merkmals“ ist gestrichen. (7) Die Regler-Presets stehen in der Seitenleiste (Schnellstart).

## Modell

Daten (xᵢ, yᵢ) mit yᵢ ∈ {−1, +1}; **trennbar** heißt: es gibt (w, b) mit yᵢ(w·φ(xᵢ) + b) ≥ 1 für alle i. **Harte Marge:** min ½‖w‖² unter dieser Bedingung; **Dual:** max Σαᵢ − ½ΣΣ αᵢαⱼ yᵢyⱼ K(xᵢ, xⱼ)
unter αᵢ ≥ 0 und Σyᵢαᵢ = 0; w = Σαᵢyᵢφ(xᵢ); KKT: αᵢ(yᵢf(xᵢ) − 1) = 0. **Stützvektor** heißt αᵢ > 0 (bei der harten Marge liegen sie genau auf den Randgeraden); Dualwert = ½‖w‖² = 2/Marge².
Der Gauß-Kern hat die **Bandbreite σ** (Längenmaß, γ = 1/(2σ²)). Vollständiges Modell mit Lesehilfe im 📐-Expander der App. Namensfallen: „Marge“ heißt in `adaboost-demo` etwas anderes (Voting-Marge),
„Stützvektor“ findet die Suche der Website nicht (Suchindex 0 Treffer), die Schreibweise der App ist „Perceptron“ wie im Bestand.

## Methodik

* **Vier Rechenwege für die harte Marge** (Experiment 1 und 2, `tests/test_oracle_harte_marge.py`): (1) **Aufzählung aller gemischten Stützmengen**, je Menge ein lineares KKT-System, kein Optimierer (n ≤ 8, 2ⁿ Mengen);
  (2) **Abstand der konvexen Hüllen** (Punkt gegen Strecke, 2D); (3) **Richtungsgitter** (3 600 Richtungen mit Verfeinerung, 2D; größte Breite des leeren Streifens je Richtung); (4) das **eigene SMO**
  (`hmg_smo.py`: zwei Multiplikatoren je Schritt, Arbeitsmenge zweiter Ordnung wie libsvm, Schranke C oder unendlich, Schrittgrenze mit Meldung, Spur mit Dualwert und KKT-Lücke je Schritt).
  Fremdlöser nur im Werkzeug und in den Tests: `SVC`, SLSQP (Primal und Dual), `trust-constr`. In der Befunde-Tabelle stehen gegen die Aufzählung (1) sechs Rechenwege: Richtungsgitter, Hüllenabstand und eigenes SMO (zusammen mit der Aufzählung die vier) sowie die Fremdlöser Primal (SLSQP), Dual (SLSQP) und `SVC`.
* **Trennbarkeit** (`hmg_separable.py`): lineares Programm (scipy HiGHS; als Verfahren exakt, gerechnet wird in Gleitkomma mit den Toleranzen von HiGHS, Randfall unter „Ehrliche Grenzen“); bei „nein“ das **Zertifikat** α ≥ 0, α ≠ 0 mit Σαᵢyᵢ = 0 und Σαᵢyᵢφ(xᵢ) = 0 (Alternativsatz von Gordan, eigene Rechnung, nicht aus einer
  gelesenen Quelle). Im Gauß-Kern gilt: verschiedene Punkte heißen immer trennbar (Gram-Matrix positiv definit, K c = y löst yᵢf(xᵢ) = 1), ein Punkt mit beiden Etiketten nie.
* **Erzeuger** (`hmg_scenario.py`, SplitMix64): „Zwei Gebiete mit Puffer“ (x₁ = y·(Lücke + |N(0,1)|), x₂ ~ N(0; 1,5), zufällig gedreht), „Gauß-Klassen“ (N((±Abstand/2, 0), I)), „Ringe mit Lücke“ (Radius 1 und 2,
  Breite 0,1), „Liefergebiet (Kreis)“ (Kunden im Quadrat, Kreis mit halber Fläche), dazu die Handbeispiele „Fünf Punkte in der Ebene“, „XOR (vier Punkte)“ und „Rechteck“. Normalwerte der Instanzen aus `normal()` der
  Vorlage (plattformfest); nur die Testmengen der Werkzeuge nutzen eine schnelle Umkehrfunktion.
* **Eingefroren** (`hmg_b2.json`, erzeugt von `tools/hmg_make_b2.py`, geprüft mit `--check`): die Tabellen mit libsvm (Fremdlöser, Schwellen, C-Tabelle, Perceptron gegen Marge, Ringe). Sie laufen in der App nicht live.
  Zusammengesetzt aus Python 3.14.6, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, Windows; Linux-Lauf (`demo_linux_check.py`) steht vor dem ersten Push aus.

## Befunde (gemessen, keine Behauptungen)

Zählung in Instanzen (Orakel, Falle), Zufalls-Seeds (Ringe) und Aktualisierungsschritten. Voraussetzungen stehen in jedem Satz; die Spalte „Beleg“ nennt den Test oder die Messung (Tabelle „Zahl → Test“ unten).

| Frage | Befund | Beleg |
|---|---|---|
| Handbeispiel „Fünf Punkte in der Ebene“ (exakte Brüche) | Von 21 gemischten Stützmengen ist genau eine zulässig, α = (6/5, 2, 4/5, 0, 0), b = −1, w = (6/5, −8/5), Marge 2/‖w‖ = 1, Dualwert 2; Fußpunkte der Hüllen (3, 1) und (12/5, 9/5) | `test_algorithm.py` |
| Eindeutigkeit der KKT-Lösung | In zwei Sätzen zu je 200 Instanzen (n = 4 bis 8, Zwei Gebiete mit Puffer, linearer Kern) gibt es in 200 und 200 von 200 genau eine KKT-Lösung; beim Rechteck (vier Punkte auf den Randgeraden) ist α nicht eindeutig: eine Strecke von Lösungen α = (½ − t, t, ½ − t, t) mit t in [0, ½] und demselben w und b; die Aufzählung über Stützmengen findet deren zwei Endpunkte | Messung, `test_algorithm.py` (Rechteck) |
| Zahl der Stützvektoren in 2D | 2 in 57 und 41, 3 in 143 und 159 von je 200 Instanzen (zwei Sätze, n = 4 bis 8, Zwei Gebiete mit Puffer, Lücke 0,2, linearer Kern; erzeugerabhängig) | Messung |
| Sechs Rechenwege gegen die Aufzählung | Größte relative Abweichung der Marge: Richtungsgitter 3,8e-15, Hüllenabstand 1,3e-15, Primal (SLSQP) 1,1e-14, eigenes SMO 5,0e-11, Dual (SLSQP) 6,3e-8, `SVC` 7,3e-7; Dualwert gegen 2/Marge² 2,9e-15 | `test_oracle_harte_marge.py` (40 Instanzen), Messung (400) |
| Meldung „Erfolg“ von SLSQP | Primal 196 und 199, Dual 191 und 192 von je 200 Instanzen; die Marge ist im Primal in 200 und 200 richtig: die Meldung sagt über die Richtigkeit in diesen Instanzen nichts (4 und 1 Fehlmeldungen bei richtiger Marge) | Messung |
| Stützvektoren zählen | Bei α > 0 zählt SLSQP (Dual, Startwert 0, ftol 1e-15) in 30 und 31 von je 100 Instanzen richtig (zwei Sätze, n = 8), mit scipy-Voreinstellungen in 14 und 12, `trust-constr` in 0 und 0; `SVC` in 100 und 100 (libsvm liefert nur Stützvektoren, ein Vergleich der Löserqualität bei α > 0 ist das nicht), eigenes SMO mit eps 1e-10 in 100 und 100, mit eps 1e-3 (Einstellung der App) in 99 und 100. Die Zahl bei α > 0 hängt an Rundungsresten und an den Einstellungen des Lösers; mit Schwelle 1e-3 zählen alle acht Einstellungen in 99 bis 100 von 100 Instanzen richtig, die Genauigkeit der Marge schwankt aber zwischen 5e-11 und 6e-4 | Messung (Tabelle unten) |
| Perceptron gegen größte Marge | Zwei Gebiete mit Puffer, n = 20, Lücke 0,25, 200 Instanzen mit je 8 Reihenfolgen, 4 000 Testpunkte: Marge der Perceptron-Richtung durch die größte Marge im Median 0,802 (10-%-Quantil 0,423, kleinstes 0,062), Winkel zur besten Richtung im Median 6,4 Grad (90 %: 15,5, größter 44,5); Testfehler größte Marge 0,0086 (SE 0,0013), Perceptron 0,0268, Differenz 0,0183 (SE 0,0016); je Instanz Perceptron besser 18, schlechter 182 von 200 (Instanz heißt hier Mittel über 8 Reihenfolgen; je Einzellauf ist das Perceptron in 320 von 1 600 Läufen besser, im Satz A). Zweiter Satz: Differenz 0,0182, besser 18, schlechter 182. Unsymmetrische Lücke (0,05 / 0,8): Marge-Verhältnis 0,811, Testfehler 0,0024 gegen 0,0255, besser 3. Der Erzeuger kommt der Marge durch seine Lücke entgegen (Kontrollmessung ohne Lücke, 0,0: 100 Instanzen, n = 20, kein Vorteil der größten Marge; nicht in `hmg_b2.json`, nicht im Test); für andere Daten ist nichts behauptet | Messung |
| Entfernen und Hinzufügen | 200 Instanzen (n = 20): Entfernen eines Stützvektors senkt die Marge nie (0 Verletzungen; Verhältnis im Median 1,110, größtes 2,387), Hinzufügen hebt sie nie (0 Verletzungen; kleinstes Verhältnis 0,020, in 22 von 200 unter 0,5); ein Nicht-Stützvektor ändert w um höchstens 1,4e-10 | Messung, `test_eigenschaften_harte_marge.py` |
| Ein Punkt in der Marge | Ein Punkt bei y·f = 0,25 (ein Viertel der halben Marge, zufällig längs der Geraden): Marge nachher durch vorher im Median 0,679 (kleinstes 0,625, 90-%-Quantil 0,883); die Gerade dreht sich im Median um 3,5 Grad (90 %: 8,3, größte 23,6); der Punkt wird in 200 von 200 Instanzen Stützvektor. Im Handbeispiel sinkt die Marge von 1 auf 9/10 ohne Drehung | Messung, `test_algorithm.py` |
| XOR mit linearem Kern | nicht trennbar (Zertifikat α = (1, 1, 1, 1)); `SVC` (tol 1e-3) gibt bei C = 1 / 1e2 / 1e4 w = 0, alle α = C, 3 / 201 / 20 001 Aktualisierungsschritte, keine Warnung; die „Marge“ ist nicht definiert | Messung, `test_algorithm.py` |
| Falle: `SVC` bei nicht trennbaren Daten | Je 100 nicht trennbare Instanzen (n = 20, Gauß-Klassen mit Abstand 2; drei Sätze): bei C = 1e4 ohne Warnung und mit Verletzern in 98 / 100 / 99; im Median 6 Punkte mit y f < 1 (Satz A); Tabelle unten | Messung |
| Wandert die „Marge“ mit C? | Je Instanz (300 Instanzen) von C = 1 nach 1e2 im Median das 0,73fache (87 % sinken um mehr als 5 %); von 1e2 nach 1e4 ändern sich 92 % um weniger als 1 % (7 % sinken um mehr als 5 %, 0 % steigen); mit tol 1e-8: 81 % und 87 % unter 1 % (Satz A und B). Bis etwa C = 1e2 hängt sie an C, von 1e2 bis 1e4 in den meisten Instanzen nicht mehr (darüber hinaus nicht aussagekräftig, da die Läufe an der Grenze anhalten; Deutung „exakte Strafe“: nicht gemessen) | Messung |
| Gauß-Kern auf denselben Instanzen | Alle 300 Instanzen sind im Merkmalsraum trennbar (Gram-Matrix positiv definit 100 / 100 / 100 von 100, K c = y löst y f = 1 in 100 / 100 / 100); `SVC` mit C = 1e4 erreicht die harte Bedingung (Schwelle 1e-3) in 97 / 95 / 95 von 100, die übrigen sind Löser-Grenzen (Multiplikatoren an der Schranke C), keine Nicht-Trennbarkeit; mit der Schwelle 1e-5 verfehlen sie 27 / 32 / 40 (Löser-Genauigkeit) von je 100 | Messung |
| Ein Punkt mit beiden Etiketten | 30 Instanzen (Gauß-Kern σ = 1, 11 Punkte, C = 1e4): in jedem Kern nicht trennbar, keine Warnung in 30 von 30, 1 bis 2 Verletzer, 36 bis 140 Aktualisierungsschritte | Messung, `test_verdicts.py` |
| Gegenzeile „C als Ersatz“ (Messung eines anderen Bausteins der Reihe) | Gauß-Kern σ = 1, 10 % vertauschte Etiketten, n = 60, `SVC` mit Schrittgrenze 100 000: Warnung bei C = 1e2 / 1e4 / 1e6 / 1e8 in 0 / 1 / 76 / 78 von 180 Zufalls-Seeds; andere Grenze, anderer Kern, anderes n, andere Daten als die Tabelle: verschieden, nicht widersprüchlich. Nicht hier neu gemessen | übernommen, kein Test |
| Gauß-Kern auf zwei Ringen | 20 Zufalls-Seeds, 2 × 30 Punkte, C = 1e4: linear trennbar in 0 von 20; die mediane Marge hat ihr Maximum bei σ = 0,8 (0,5051), Testfehler 0 von σ = 0,4 bis 6,4: eine größere Marge zeigt nicht ein besseres σ; Tabelle unten | Messung |

**Falle, drei Sätze Zufalls-Seeds** (Spalten: Satz A / B / C; `SVC` linear, tol 1e-3, `max_iter` 2 000 000; je Satz 100 nicht trennbare Instanzen, gezogen 125 / 124 / 121 Zufalls-Seeds):

| C | Warnung (Grenze erreicht), von 100 | ohne Warnung und mit Verletzern (y f < 1), von 100 | Verletzer je Instanz, Median [Min, Max] im Satz A | Aktualisierungsschritte, Median | Marge 2/‖w‖, Median |
|---|---|---|---|---|---|
| 1 | 0 / 0 / 0 | 100 / 100 / 100 | 8 [4, 16] | 22 / 20 / 22 | 1,616 / 1,776 / 1,586 |
| 1e2 | 0 / 0 / 0 | 100 / 100 / 100 | 7 [1, 15] | 742 / 825 / 650 | 1,155 / 1,372 / 1,047 |
| 1e4 | 2 / 0 / 1 | 98 / 100 / 99 | 6 [2, 13] | 72 580 / 79 188 / 62 391 | 1,147 / 1,371 / 1,034 |
| 1e6 (Grenze erreicht, nicht konvergiert) | 85 / 86 / 83 | 15 / 14 / 17 | 7 [1, 20] | 2 000 000 / 2 000 000 / 2 000 000 | 1,137 / 1,215 / 1,039 |

Bei C = 1e6 hält nur die Grenze `max_iter` den Lauf an; der Vergleich 1e4 → 1e6 sagt deshalb nichts über C (nicht konvergierte Läufe, 28 % sinken, 46 % steigen um mehr als 5 %). Ohne Grenze gäbe es keine Warnung, sondern langes Warten: nicht gemessen.
Das Live-Modell (eigenes SMO, eps 1e-3, Schrittgrenze 200 000) auf den ersten 30 Instanzen des Satzes A: C = 1 / 1e2 / 1e4 Aktualisierungsschritte im Median 23 / 680 / 63 643;
an der Schrittgrenze 0 / 0 / 5 von 30; „Marge“ des eigenen SMO durch die von `SVC` je Instanz 0,920 bis 1,009 (Median 1,0003) bei C = 1e4.

**Stützvektoren zählen** (Satz A / Satz B, je 100 Instanzen, n = 8, in wie vielen die Menge {α > Schwelle · max α} gleich der exakten ist; Marge relativ: größte Abweichung):

| Löser (Einstellungen) | α > 0 | 1e-12 | 1e-9 | 1e-6 | 1e-3 | 1e-2 | Marge relativ |
|---|---|---|---|---|---|---|---|
| SLSQP (dual) | 30 / 31 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 98 / 99 | 2,5e-8 / 3,3e-8 |
| SLSQP (Voreinstellungen) | 14 / 12 | 99 / 100 | 99 / 100 | 99 / 100 | 99 / 100 | 97 / 99 | 6,0e-4 / 5,4e-4 |
| trust-constr | 0 / 0 | 0 / 0 | 0 / 1 | 86 / 80 | 100 / 100 | 98 / 99 | 1,2e-6 / 2,0e-6 |
| trust-constr (Voreinstellungen) | 0 / 0 | 0 / 0 | 0 / 0 | 4 / 7 | 99 / 100 | 98 / 99 | 1,0e-4 / 1,9e-4 |
| SVC tol=1e-10 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 98 / 99 | 4,4e-7 / 6,9e-7 |
| SVC tol=1e-3 (Voreinstellung) | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 99 / 100 | 97 / 99 | 4,9e-4 / 5,0e-4 |
| SMO eps=1e-10 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 98 / 99 | 5,0e-11 / 5,0e-11 |
| SMO eps=1e-3 | 99 / 100 | 99 / 100 | 99 / 100 | 99 / 100 | 99 / 100 | 96 / 99 | 4,9e-4 / 4,9e-4 |

**Ringe, Gauß-Kern** (20 Zufalls-Seeds, C = 1e4, `SVC` tol 1e-10; Grenzwert σ → 0: alle α = 1, Marge 2/√60 = 0,2582):

| σ | Marge (Median) | Anteil Stützvektoren (Median) | Testfehler (Mittel) | Zufalls-Seeds mit verfehlter Bedingung (1e-3 / 1e-5) |
|---|---|---|---|---|
| 0,1 | 0,3159 | 0,93 | 0,0919 | 0 / 0 |
| 0,2 | 0,3857 | 0,80 | 0,0172 | 0 / 0 |
| 0,4 | 0,4909 | 0,58 | 0,0000 | 0 / 0 |
| 0,8 | 0,5051 | 0,32 | 0,0000 | 0 / 0 |
| 1,6 | 0,3797 | 0,18 | 0,0000 | 0 / 0 |
| 3,2 | 0,1487 | 0,13 | 0,0000 | 0 / 0 |
| 6,4 | 0,0418 | 0,11 | 0,0000 | 0 / 19 |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Trennbar** (alle y·f ≥ 1 erreichbar) | `SVC` mit C = 1e4 antwortet bei je 100 nicht trennbaren Instanzen (n = 20, Gauß-Klassen mit Abstand 2) in 98 / 100 / 99 von 100 ohne Warnung mit einer Geraden; die App prüft die Voraussetzung selbst (Experiment 4). | die weiche Marge ist nicht Gegenstand dieser Demo |
| **Eine eindeutige Lösung** | w und b sind eindeutig, α nur in Allgemeinlage: beim Rechteck (vier Punkte auf den Randgeraden) ist α nicht eindeutig: eine Strecke von Lösungen mit demselben w und b; die Aufzählung über Stützmengen findet deren zwei Endpunkte. | – |
| **Stützvektoren zählen** | Die Zahl hängt an den Einstellungen des Lösers: bei α > 0 zählt SLSQP in 30 und 31 von je 100 Instanzen richtig (zwei Sätze, Startwert 0), mit scipy-Voreinstellungen in 14 und 12; libsvm in 100 und 100, eigenes SMO mit eps 1e-10 in 100 und 100, mit eps 1e-3 (Einstellung der App) in 99 und 100 (libsvm liefert nur Stützvektoren, ein Vergleich der Löserqualität bei α > 0 ist das nicht). | – |
| **Trennbarkeitstest** | Das lineare Programm (HiGHS, Toleranzen der Voreinstellung) rechnet in Gleitkomma. Gegenbeispiel: die Punkte (0, ε), (1, 1+ε), (0, −ε), (1, 1−ε) mit y = +, +, −, − sind für jedes ε > 0 trennbar; der Test meldet bei ε = 1e-3 und 1e-6 richtig „trennbar“, bei ε = 1e-9 und 1e-12 fälschlich „nicht trennbar“. Bei den Instanzen und Reglern der Demo (Marge nicht unter etwa 0,01) ohne Auswirkung. | – |
| **Ein zulässiger Kern** | Linearer, Polynom-2- und Gauß-Kern sind zulässig; ob eine Funktion überhaupt ein Kern ist, wird hier nicht geprüft. | – |
| **Zwei Dimensionen, höchstens acht Punkte** | Hüllenabstand und Richtungsgitter als unabhängige Rechenwege gibt es nur in 2D, die Aufzählung der Stützmengen nur bis n = 8 (2ⁿ Mengen); größere Instanzen werden nur gegen Fremdlöser geprüft. | – |
| **Ein Erzeuger mit Lücke** | „Perceptron gegen größte Marge“ ist nur an dem Erzeuger „Zwei Gebiete mit Puffer“ gemessen; seine Lücke kommt der Marge entgegen. Für andere Daten ist nichts behauptet. | – |
| **Nutzen im Betrieb** | Ob eine Marge-Methode in Logistik, Lager oder Fertigung bessere Entscheidungen liefert, ist in den gelesenen Quellen nicht gemessen und nicht Gegenstand. | – |

**Nicht gebaut und nicht gemessen** (kein Regler, kein Versprechen): die Einheit eines Merkmals (Skalierung), Dimension über 2 für die unabhängigen Rechenwege, `SVC` ohne Schrittgrenze auf nicht
trennbaren Daten (die Laufzeit ist dort unbeschränkt), die Einstellungen `shrinking` und `cache_size` von `SVC`, ein zweiter Erzeuger ohne Puffer und mit Ausreißern für „Perceptron gegen
größte Marge“, ν-SVM, `class_weight` und Platt-Skalierung.

* **Live-Modell ≠ Studie:** Die App rechnet mit einem eigenen SMO (zwei Multiplikatoren je Schritt, Arbeitsmenge zweiter Ordnung) und mit einer Schrittgrenze; die eingefrorenen Tabellen stammen von libsvm (`SVC`, in der App nicht enthalten, nur im Messwerkzeug). Bei konvergiertem SMO geben beide denselben Dualwert (relativ 1e-9, 25 von 30 Instanzen); an der Schrittgrenze (5 von 30) liegt der Dualwert des SMO um 0,04 bis 0,5 % tiefer. Verschieden sind die Zahl der Aktualisierungsschritte und die Meldung an der Grenze: bei C = 1e4 auf 30 nicht trennbaren Instanzen (n = 20) hält das eigene SMO in 5 von 30 an der Schrittgrenze 200 000 an und sagt es, `SVC` (Grenze 2 000 000) warnt in 2 von 100 Instanzen (Satz A); die „Marge“ 2/‖w‖ beider Löser liegt je Instanz im Verhältnis 0,920 bis 1,009 (Median 1,0003).
* **Die Zahlen sind Bänder, keine Rangfolgen:** die Läufe sind klein (20 Zufalls-Seeds bei den Ringen, 30 Instanzen beim eigenen SMO), die Streuung zwischen den Sätzen steht in den Tabellen. Plattform: gemessen unter Windows, Python 3.14.6; `n_iter_`, Defizite und Löser-Abweichungen sind plattformabhängig (die Tests prüfen Bänder); der Linux-Lauf steht aus.
* **Zwei Fassungen desselben Textes:** die Annahmentabelle steht in der App und hier wortgleich; `tests/test_claims.py` vergleicht beide.

## Zahl → Test

| Zahl in dieser README | Beleg |
|---|---|
| Handbeispiel (21 Mengen, α, Marge 1, Dualwert 2, Fußpunkte, 25/9, 400/81, 9/10, 12 Geraden, Rechteck, XOR mit a = 1/4, 1 − exp(−2/σ²)) | exakt in `tests/test_algorithm.py` |
| Aufzählung, Gitter, Hülle, SMO, `SVC`, SLSQP gleiche Marge (40 und 20 Instanzen) | Band in `tests/test_oracle_harte_marge.py` |
| Entfernen, Hinzufügen, Verschieben, Drehen, Skalieren (Eigenschaften) | `tests/test_eigenschaften_harte_marge.py` |
| SMO-Handbeispiele (C = 1/8 bis 100), Schrittgrenze, Spur, Kopier-Hash | `tests/test_smo.py` |
| Alle Zahlen der Tabellen „Befunde“, „Falle“, „Stützvektoren zählen“, „Ringe“ | gleich dem Inhalt von `hmg_b2.json` in `tests/test_claims.py`; `hmg_b2.json` selbst: Teilmenge neu gerechnet in `tests/test_studies.py` (`--check`) |
| Texte der Verdikte und je Preset die erwartete Zeile | `tests/test_verdicts.py` |
| Erzeuger, eingefrorene Werte je Zufalls-Seed | `tests/test_scenario.py` |
| Zahlen aus der Vormessung der Planung (16, 20, 11 von 200; 98 / 99 / 98) | Messung dieser README, nicht im Test (nur als Abweichung genannt) |
| Gegenzeile C = 1e2 bis 1e8 (0 / 1 / 76 / 78 von 180) | übernommen, nicht im Test |
| Kontrollmessung ohne Lücke; `SVC` mit C = 1e4 bei auf 0,01 skalierten Merkmalen (19 von 20 verfehlen die harte Bedingung, bei Skalen 1, 0,1 und 0,03: 0 von 20; trennbare Instanzen n = 8); Dualwert von SMO und `SVC` bei konvergiertem und abgebrochenem SMO (1e-9, 25 von 30; 0,04 bis 0,5 %); Gegenbeispiel zum Trennbarkeitstest (ε = 1e-3 bis 1e-12) | Messung nach dem Bau, nicht in `hmg_b2.json`, nicht im Test |

## Tests

169 Tests (`pytest --collect-only -q tests`), grün unter Windows (parallel mit `pytest -n 8`). Zusammensetzung: Handbeispiele in Brüchen, Orakel (vier Rechenwege, Trennbarkeit gegen Zertifikat, Fremdlöser mit `importorskip`),
Eigenschaften (je 20 bis 40 Instanzen), Randfälle, SMO (Schrittgrenze, Spur, Arbeitsmengen), Erzeuger (eingefrorene Werte als Band), Verdikte je Preset, Presets, Layout, Laufzeit-Importe, AppTest-Rauchtest, Footer, Kopier-Tests
(`hmg_rng.py` bytegleich mit der Vorlage, Hash der Funktion `smo` in `hmg_smo.py`), Studien-Konsistenz. Kein Mutationstest.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore`, `.streamlit/config.toml` | venv, Caches, Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Annahmen, Formeln |
| `hmg_rng.py` | SplitMix64 (Kopie der Vorlage, Kopier-Test) |
| `hmg_smo.py` | eigenes SMO: Schranke C, eps, Schrittgrenze mit Meldung, Spur mit Dualwert und KKT-Lücke (Quelle für die Folge-Demos der Reihe; der Hash der Funktion steht im Test) |
| `hmg_scenario.py` | Erzeuger und Handbeispiele |
| `hmg_margin.py` | Gram-Matrix, Aufzählung der Stützmengen, Hüllenabstand, Richtungsgitter |
| `hmg_separable.py` | Trennbarkeit (lineares Programm), Zertifikat, Gauß-Regel |
| `hmg_perceptron.py` | Perceptron mit Bias in festen Reihenfolgen |
| `hmg_evaluation.py` | Rechnung und Verdikte der Experimente (reine Funktionen), Zugriff auf die Studien |
| `hmg_texts.py` | Einleitung, Annahmen, Formeln |
| `hmg_constants.py`, `hmg_presets.py`, `hmg_layout.py`, `hmg_visualization.py` | Regler, Presets mit Permalink, fester Kopf, Plotly-Abbildungen |
| `hmg_b2.json` | eingefrorene Studien (libsvm, SLSQP, trust-constr) |
| `tools/hmg_make_b2.py` | Messwerkzeug: rechnet die Studien, `--merge`, `--check` |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, scipy, pandas, plotly); dazu pytest und scikit-learn (nur Tests und Werkzeug) |
| `tests/` | siehe „Tests“ |

## Bewusst nicht umgesetzt

Einheit eines Merkmals und Skalierung; Dimension über 2 für Hüllenabstand und Richtungsgitter, n über 8 für die Aufzählung; `SVC` ohne Schrittgrenze auf nicht trennbaren Daten, `shrinking` und `cache_size`;
ein zweiter Erzeuger ohne Puffer und mit Ausreißern für „Perceptron gegen größte Marge“; ν-SVM, `class_weight`, Platt-Skalierung; die weiche Marge und der Preis je Verletzung; Prüfung, ob eine Funktion ein Kern ist;
ein Nutzenbeleg in Logistik, Lager oder Fertigung (in den gelesenen Quellen nicht gemessen).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kernmethoden: Skalarprodukte tauschen](https://sebastianhanisch.net/konzepte-kernmethoden.html).
