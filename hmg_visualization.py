"""Plotly-Figuren: alle Achsen fest (`fixedrange`), Beschriftungen als Annotationen, gleicher Maßstab auf beiden Achsen über `scaleanchor` mit automatischem Bereich."""

import math

import numpy as np
import plotly.graph_objects as go

import hmg_constants as C
import hmg_margin as M


def _box(X, extra=None, pad: float = 0.25):
    pts = np.asarray(X, float)
    if extra is not None and len(extra):
        pts = np.vstack([pts, np.atleast_2d(extra)])
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-9)
    return lo - pad * span, hi + pad * span


def clip_line(normal, c, lo, hi):
    """Schnitt der Geraden normal . x = c mit dem Rechteck [lo, hi]: zwei Endpunkte oder None."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    cc = c / np.linalg.norm(normal)
    t = np.array([-n[1], n[0]])
    p0 = n * cc
    pts = []
    for ax in (0, 1):
        for edge in (lo[ax], hi[ax]):
            if abs(t[ax]) > 1e-12:
                s = (edge - p0[ax]) / t[ax]
                q = p0 + s * t
                other = 1 - ax
                if lo[other] - 1e-9 <= q[other] <= hi[other] + 1e-9:
                    pts.append(q)
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    d = [(np.linalg.norm(pts[i] - pts[j]), i, j) for i in range(len(pts)) for j in range(i + 1, len(pts))]
    _, i, j = max(d)
    return pts[i], pts[j]


def _line(fig, normal, c, lo, hi, color, width=2.0, dash=None, name=None, opacity=1.0, showlegend=False):
    seg = clip_line(normal, c, lo, hi)
    if seg is None:
        return
    fig.add_trace(go.Scatter(x=[seg[0][0], seg[1][0]], y=[seg[0][1], seg[1][1]], mode="lines", line=dict(color=color, width=width, dash=dash), opacity=opacity, name=name or "",
                             showlegend=showlegend, hoverinfo="skip"))


def _layout(fig, lo, hi, title=None, height=430):
    fig.add_trace(go.Scatter(x=[lo[0], hi[0]], y=[lo[1], hi[1]], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=36 if title else 10, b=10), title=dict(text=title, font=dict(size=14)) if title else None,
                      legend=dict(orientation="h", y=-0.12), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(fixedrange=True, scaleanchor="y", scaleratio=1, showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


def _points(fig, X, y, sv=None, labels=None, size=9):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    for lab, col, nm in ((1.0, C.COLOR_POS, "Klasse +1"), (-1.0, C.COLOR_NEG, "Klasse −1")):
        m = y == lab
        fig.add_trace(go.Scatter(x=X[m, 0], y=X[m, 1], mode="markers", marker=dict(color=col, size=size, line=dict(width=1, color="white")), name=nm,
                                 text=[f"Punkt {i + 1}" for i in np.where(m)[0]], hovertemplate="%{text}<br>(%{x:.2f}; %{y:.2f})<extra></extra>"))
    if sv:
        fig.add_trace(go.Scatter(x=X[sv, 0], y=X[sv, 1], mode="markers", marker=dict(symbol="circle-open", size=size + 9, line=dict(width=2, color=C.COLOR_LINE)), name="Stützvektor",
                                 hoverinfo="skip"))
    if labels is not False:
        for i in range(len(y)):
            if len(y) <= 12:
                fig.add_annotation(x=X[i, 0], y=X[i, 1], text=str(i + 1), showarrow=False, yshift=-14, font=dict(size=10))


def fig_exp1(r: dict):
    ins = r["inst"]
    X, y = ins.X, ins.y
    lo, hi = _box(X)
    fig = go.Figure()
    for ln in r["lines"]:
        if ln["converged"]:
            _line(fig, ln["w"], -ln["b"], lo, hi, "rgba(120,120,120,0.55)", 1.2, name="Perceptron-Geraden", showlegend=False)
    w, b = r["w"], r["b"]
    _line(fig, w, -b, lo, hi, C.COLOR_LINE, 2.5, name="größte Marge")
    _line(fig, w, 1 - b, lo, hi, C.COLOR_LINE, 1.2, "dash")
    _line(fig, w, -1 - b, lo, hi, C.COLOR_LINE, 1.2, "dash")
    u = r["u"]
    lo_neg, hi_pos = r["strip"]
    if hi_pos > lo_neg:
        mid = 0.5 * (hi_pos + lo_neg)
        _line(fig, u, mid, lo, hi, C.COLOR_ALT, 2.0, name="gewählte Richtung")
        _line(fig, u, hi_pos, lo, hi, C.COLOR_ALT, 1.0, "dot")
        _line(fig, u, lo_neg, lo, hi, C.COLOR_ALT, 1.0, "dot")
    else:
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, text="Diese Richtung trennt die Klassen nicht.", showarrow=False, font=dict(color=C.COLOR_NEG), xanchor="left", yanchor="top")
    _points(fig, X, y, r["sv"])
    return _layout(fig, lo, hi, "Punkte, größte Marge (schwarz), gewählte Richtung (blau), Perceptron-Geraden (grau)")


def fig_margin_curve(r: dict):
    th = r["thetas_deg"]
    curve = np.maximum(r["curve"], 0.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=th, y=curve, mode="lines", line=dict(color=C.COLOR_ALT, width=2), name="Marge der Richtung", hovertemplate="θ = %{x:.0f}°<br>Marge %{y:.3f}<extra></extra>"))
    th_star = math.degrees(r["th_star"])
    th_star = ((th_star + 180) % 360) - 180
    fig.add_trace(go.Scatter(x=[th_star], y=[r["grid_max"]], mode="markers", marker=dict(color=C.COLOR_LINE, size=11, symbol="star"), name="größte Marge"))
    th_sel = ((math.degrees(r["th_sel"]) + 180) % 360) - 180
    fig.add_trace(go.Scatter(x=[th_sel], y=[max(r["m_sel"], 0.0)], mode="markers", marker=dict(color=C.COLOR_NEG, size=11), name="gewählte Richtung"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=36, b=10), title=dict(text="Marge in Abhängigkeit von der Richtung θ (0, wo die Richtung nicht trennt)", font=dict(size=14)),
                      legend=dict(orientation="h", y=-0.25), xaxis_title="Richtung θ des Normalenvektors (Grad)", yaxis_title="Breite des leeren Streifens")
    fig.update_xaxes(fixedrange=True, range=[-180, 180])
    fig.update_yaxes(fixedrange=True, rangemode="tozero")
    return fig


def fig_boundary(ins, kind, sigma, alpha, b, sv=None, title=None, grid_n: int = 110, height: int = 430):
    """Entscheidungsfunktion f = 0 und die Randlinien f = ±1 im Eingaberaum (Höhenlinien), Punkte, Stützvektoren."""
    X, y = ins.X, ins.y
    lo, hi = _box(X, pad=0.3)
    gx = np.linspace(lo[0], hi[0], grid_n)
    gy = np.linspace(lo[1], hi[1], grid_n)
    G = np.array([(a, c) for c in gy for a in gx])
    f = (M.kernel_cross(G, X, kind, sigma) @ (np.asarray(alpha) * y) + b).reshape(grid_n, grid_n)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=gx, y=gy, z=f, showscale=False, contours=dict(start=-1, end=1, size=1, coloring="none"), line=dict(color=C.COLOR_LINE, width=1.6),
                             hoverinfo="skip", name="f = −1, 0, +1", showlegend=False))
    _points(fig, X, y, sv)
    return _layout(fig, lo, hi, title, height)


def fig_points_lines(ins, w, b, sv=None, title=None, extra_points=None, height: int = 380, note=None):
    """Linearer Kern: Gerade f = 0 und Randgeraden f = ±1 aus (w, b)."""
    X, y = ins.X, ins.y
    lo, hi = _box(X, extra_points)
    fig = go.Figure()
    if w is not None and np.linalg.norm(w) > 1e-12:
        _line(fig, w, -b, lo, hi, C.COLOR_LINE, 2.5)
        _line(fig, w, 1 - b, lo, hi, C.COLOR_LINE, 1.2, "dash")
        _line(fig, w, -1 - b, lo, hi, C.COLOR_LINE, 1.2, "dash")
    elif note:
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, text=note, showarrow=False, font=dict(color=C.COLOR_NEG), xanchor="left", yanchor="top")
    _points(fig, X, y, sv)
    if extra_points is not None and len(extra_points):
        ep = np.atleast_2d(extra_points)
        fig.add_trace(go.Scatter(x=ep[:, 0], y=ep[:, 1], mode="markers", marker=dict(symbol="diamond", size=13, color="rgba(0,0,0,0)", line=dict(width=2.5, color=C.COLOR_ALT)), name="neuer Punkt"))
    return _layout(fig, lo, hi, title, height)


def fig_trace(trace, eps: float, max_steps: int):
    """KKT-Lücke je Iteration (logarithmisch) mit der Abbruchschwelle eps; der Dualwert steht als zweite Kurve."""
    fig = go.Figure()
    steps = trace["step"]
    fig.add_trace(go.Scatter(x=steps, y=np.maximum(trace["gap"], 1e-16), mode="lines", line=dict(color=C.COLOR_NEG, width=1.6), name="KKT-Lücke"))
    fig.add_trace(go.Scatter(x=[0, max(int(steps[-1]), 1)], y=[eps, eps], mode="lines", line=dict(color=C.COLOR_MARGIN, dash="dash"), name=f"Abbruch bei {eps:g}"))
    fig.add_trace(go.Scatter(x=steps, y=trace["dual"], mode="lines", line=dict(color=C.COLOR_ALT, width=1.6), name="Dualwert", yaxis="y2"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=36, b=10), title=dict(text="Verlauf des Lösers: KKT-Lücke und Dualwert je Iteration", font=dict(size=14)),
                      legend=dict(orientation="h", y=-0.28), xaxis_title="Iterationen", yaxis=dict(title="KKT-Lücke", type="log", fixedrange=True),
                      yaxis2=dict(title="Dualwert", overlaying="y", side="right", fixedrange=True))
    fig.update_xaxes(fixedrange=True)
    return fig


def fig_sigma_table(g: dict):
    """Mediane Marge gegen die Bandbreite sigma (eingefrorene Studie G)."""
    sig = g["sigmas"]
    z = g["zeilen"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sig, y=[z[str(s)]["marge_median"] for s in sig], mode="lines+markers", line=dict(color=C.COLOR_ALT), name="Marge 2/‖w‖ (Median)"))
    fig.add_trace(go.Scatter(x=sig, y=[z[str(s)]["testfehler_mittel"] for s in sig], mode="lines+markers", line=dict(color=C.COLOR_NEG), name="Testfehler (Mittel)", yaxis="y2"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=36, b=10), title=dict(text=f"Ringe mit Lücke, Gauß-Kern, {g['n_seeds']} Zufalls-Seeds (eingefroren)", font=dict(size=14)),
                      legend=dict(orientation="h", y=-0.3), xaxis=dict(title="Bandbreite σ", type="log", fixedrange=True), yaxis=dict(title="Marge", fixedrange=True, rangemode="tozero"),
                      yaxis2=dict(title="Testfehler", overlaying="y", side="right", fixedrange=True, rangemode="tozero"))
    return fig
