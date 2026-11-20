"""Perceptron mit Bias (Primalform) in einer festen Reihenfolge der Punkte, für den Vergleich „viele trennende Geraden, welche?“.

Regel: für jeden Punkt in der Reihenfolge, falls y (w.x + b) <= 0: w += y x, b += y. Ein Durchlauf aller Punkte heißt Epoche; das Verfahren hält, wenn eine Epoche
ohne Korrektur endet. Bei trennbaren Daten hält es nach endlich vielen Korrekturen (Novikoff, hier nicht wiederholt); bei nicht trennbaren nie (Grenze `max_epochs`)."""

import numpy as np

from hmg_rng import SplitMix64


def perceptron(X, y, order, max_epochs: int = 20000):
    """Gibt (w, b, Korrekturen, Epochen, konvergiert) zurück."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    b = 0.0
    updates = 0
    for ep in range(1, max_epochs + 1):
        mist = 0
        for i in order:
            if y[i] * (X[i] @ w + b) <= 0:
                w = w + y[i] * X[i]
                b += y[i]
                mist += 1
        updates += mist
        if mist == 0:
            return w, b, updates, ep, True
    return w, b, updates, max_epochs, False


def random_orders(n: int, k: int, seed: int):
    """k Reihenfolgen der Punkte 0 .. n-1 (Mischen mit SplitMix64, fest je Zufalls-Seed)."""
    rng = SplitMix64(seed)
    out = []
    for _ in range(k):
        o = list(range(n))
        rng.shuffle(o)
        out.append(o)
    return out


def perceptron_lines(X, y, k: int, seed: int, max_epochs: int = 20000):
    """Das Perceptron in k zufälligen Reihenfolgen: Liste von (w, b, Korrekturen, Epochen, konvergiert, Reihenfolge)."""
    return [perceptron(X, y, o, max_epochs) + (o,) for o in random_orders(len(y), k, seed)]
