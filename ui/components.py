"""Shared visual components: theme, KPI cards, badges, score cards, checklists, graphs."""

import html

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fraud.graph_analysis import NODE_STYLE, RELATION_COLOR
from fraud.risk_engine import BAND_COLORS, band

ACTION_COLORS = {"ALLOW": "#22c55e", "MONITOR": "#eab308", "STEP-UP AUTHENTICATION": "#f97316",
                 "BLOCK + INVESTIGATE": "#ef4444", "NO ACTION": "#22c55e",
                 "STEP-UP + RESTRICT": "#f97316", "FREEZE + INVESTIGATE": "#ef4444",
                 "NORMAL": "#22c55e", "REVIEW": "#eab308", "STEP-UP": "#f97316",
                 "BLOCK": "#ef4444"}
STATUS_ICON = {"NORMAL": "🟢 NORMAL", "REVIEW": "🟡 REVIEW", "STEP-UP": "🟠 STEP-UP",
               "BLOCK": "🔴 BLOCK"}
PLOT_BG = "rgba(0,0,0,0)"
GRID = "rgba(148,163,184,0.12)"

CSS = """
<style>
:root { --card:#0f1b2d; --card2:#13223a; --line:#1e3352; --muted:#8aa0bd; --text:#e2e8f0; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }
[data-testid="stSidebar"] { background: #0a1424; border-right: 1px solid var(--line); }
h1, h2, h3 { letter-spacing: -0.01em; }
.fi-hero { padding: 14px 18px; border:1px solid var(--line); border-radius: 12px;
  background: linear-gradient(135deg, #0f1f38 0%, #0b1526 60%, #170f1f 100%); margin-bottom: 14px; }
.fi-hero h1 { font-size: 1.55rem; margin: 0; color: var(--text); }
.fi-hero p { margin: 4px 0 0 0; color: var(--muted); font-size: 0.92rem; }
.fi-kpi { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 10px 14px 9px 14px; height: 100%; }
.fi-kpi .lbl { color: var(--muted); font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: .06em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fi-kpi .val { color: var(--text); font-size: 1.55rem; font-weight: 700; line-height: 1.25; }
.fi-kpi .sub { color: var(--muted); font-size: 0.72rem; }
.fi-badge { display:inline-block; padding: 2px 10px; border-radius: 999px; font-weight: 700;
  font-size: 0.78rem; color: #0b1220; letter-spacing: .02em; }
.fi-score { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 12px 14px; text-align: center; }
.fi-score .lbl { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em; }
.fi-score .val { font-size: 2.1rem; font-weight: 800; line-height: 1.15; }
.fi-score .bar { height: 6px; border-radius: 4px; background: #1e293b; margin-top: 6px; overflow: hidden; }
.fi-score .bar > div { height: 100%; border-radius: 4px; }
.fi-alert { border-radius: 10px; padding: 10px 14px; margin: 6px 0; border: 1px solid; }
.fi-alert.crit { background: rgba(239,68,68,.10); border-color: rgba(239,68,68,.55); }
.fi-alert.high { background: rgba(249,115,22,.10); border-color: rgba(249,115,22,.5); }
.fi-alert.ok { background: rgba(34,197,94,.08); border-color: rgba(34,197,94,.45); }
.fi-alert.info { background: rgba(96,165,250,.08); border-color: rgba(96,165,250,.45); }
.fi-alert b { color: var(--text); }
.fi-check { padding: 5px 10px; border-radius: 8px; margin: 3px 0; font-size: 0.92rem; }
.fi-check.hit { background: rgba(239,68,68,.10); border-left: 3px solid #ef4444; }
.fi-check.miss { color: #64748b; border-left: 3px solid #1e293b; }
.fi-check small { color: var(--muted); }
.fi-card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px 16px; }
.fi-chip { display:inline-block; background: var(--card2); border:1px solid var(--line);
  border-radius: 999px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: .82rem; }
.fi-muted { color: var(--muted); font-size: .85rem; }
.fi-innov { background: var(--card); border:1px solid var(--line); border-radius: 10px; padding: 10px 12px; height: 100%; }
.fi-innov b { color: #93c5fd; }
.fi-innov div { color: var(--muted); font-size: .8rem; margin-top: 3px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def badge(text, color=None):
    color = color or ACTION_COLORS.get(text) or BAND_COLORS.get(text, "#94a3b8")
    return f"<span class='fi-badge' style='background:{color}'>{esc(text)}</span>"


def hero(title, subtitle):
    st.markdown(f"<div class='fi-hero'><h1>{esc(title)}</h1><p>{subtitle}</p></div>",
                unsafe_allow_html=True)


def kpi(col, label, value, sub="", color=None):
    style = f" style='color:{color}'" if color else ""
    col.markdown(f"<div class='fi-kpi'><div class='lbl'>{esc(label)}</div>"
                 f"<div class='val'{style}>{value}</div><div class='sub'>{sub}</div></div>",
                 unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        kpi(col, *item)


def score_card(col, label, score, sub=""):
    score = float(score or 0)
    c = BAND_COLORS[band(score)]
    col.markdown(f"<div class='fi-score'><div class='lbl'>{esc(label)}</div>"
                 f"<div class='val' style='color:{c}'>{score:.0f}<span style='font-size:1rem;"
                 f"color:#64748b'>/100</span></div>"
                 f"<div>{badge(band(score))}</div>"
                 f"<div class='bar'><div style='width:{min(score, 100):.0f}%;background:{c}'></div></div>"
                 f"<div class='fi-muted' style='margin-top:4px'>{sub}</div></div>",
                 unsafe_allow_html=True)


def alert(text, kind="crit"):
    st.markdown(f"<div class='fi-alert {kind}'>{text}</div>", unsafe_allow_html=True)


def checklist(items, show_misses=True):
    out = []
    for it in sorted(items, key=lambda i: not i["hit"]):
        if it["hit"]:
            out.append(f"<div class='fi-check hit'>✓ <b>{esc(it['label'])}</b> "
                       f"<small>— {esc(it['detail'])}</small></div>")
        elif show_misses:
            out.append(f"<div class='fi-check miss'>✗ {esc(it['label'])} "
                       f"<small>— not observed</small></div>")
    st.markdown("".join(out), unsafe_allow_html=True)


def chips(values, color=None):
    style = f" style='border-color:{color}'" if color else ""
    return "".join(f"<span class='fi-chip'{style}>{esc(v)}</span>" for v in values if v)


def style_fig(fig, height=320, legend=True, margin=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      height=height, margin=margin or dict(l=10, r=10, t=36, b=10),
                      font=dict(color="#cbd5e1", size=12), showlegend=legend,
                      legend=dict(orientation="h", y=-0.18, bgcolor=PLOT_BG),
                      title_font=dict(size=14, color="#e2e8f0"))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def money(x):
    x = float(x or 0)
    if x >= 1e7:
        return f"₹{x / 1e7:,.2f} Cr"
    if x >= 1e5:
        return f"₹{x / 1e5:,.2f} L"
    return f"₹{x:,.0f}"


# --------------------------------------------------------------------------- #
# Network figure
# --------------------------------------------------------------------------- #

def network_figure(H, height=600, highlight=None, communities=None, seed=7, title=None):
    """Interactive Plotly network. Node ids are attached as customdata for selection."""
    fig = go.Figure()
    if H.number_of_nodes() == 0:
        fig.add_annotation(text="No connections to show", showarrow=False,
                           font=dict(size=16, color="#94a3b8"))
        return style_fig(fig, height=height, legend=False), []
    k = 1.8 / max(np.sqrt(H.number_of_nodes()), 1)
    pos = nx.spring_layout(H, seed=seed, k=k, iterations=120)
    highlight = set(highlight or [])
    comm_of = {}
    for i, c in enumerate(communities or []):
        for a in c:
            comm_of[a] = i
    for rel, color in RELATION_COLOR.items():
        xs, ys = [], []
        for a, b, d in H.edges(data=True):
            if d.get("relation") == rel:
                xs += [pos[a][0], pos[b][0], None]
                ys += [pos[a][1], pos[b][1], None]
        if xs:
            width = 2.6 if rel == "CONNECTED_TO" else 1.1
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                     line=dict(color=color, width=width),
                                     opacity=0.85 if rel == "CONNECTED_TO" else 0.45,
                                     name=rel.replace("_", " ").title(), legendgroup="edges"))
    order = []
    for typ, (color, symbol, label) in NODE_STYLE.items():
        nodes = [n for n, d in H.nodes(data=True) if d.get("type") == typ]
        if not nodes:
            continue
        risks = [H.nodes[n].get("risk", 0) for n in nodes]
        sizes = [(26 if typ == "account" else 18) + (10 if n in highlight else 0) +
                 (8 if r >= 81 else 0) for n, r in zip(nodes, risks)]
        line_col = ["#ffffff" if n in highlight else ("#ef4444" if r >= 81 else
                                                      "#f97316" if r >= 61 else "#0b1220")
                    for n, r in zip(nodes, risks)]
        if typ == "account":
            fill = [BAND_COLORS[band(r)] for r in risks]
        else:
            fill = color
        text = [H.nodes[n].get("label", n) if (typ == "account" or n in highlight or r >= 61)
                else "" for n, r in zip(nodes, risks)]
        hover = []
        for n, r in zip(nodes, risks):
            d = H.nodes[n]
            extra = f"<br>ring {d['ring']}" if d.get("ring") else ""
            extra += f"<br>{d['n_accounts']} accounts" if d.get("n_accounts") else ""
            extra += f"<br>cluster #{comm_of[d['label']] + 1}" if d.get("label") in comm_of else ""
            hover.append(f"<b>{label}: {esc(d.get('label', n))}</b><br>risk {r:.0f}{extra}")
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
            text=text, textposition="top center", textfont=dict(size=10, color="#cbd5e1"),
            marker=dict(size=sizes, color=fill, symbol=symbol,
                        line=dict(color=line_col, width=2)),
            customdata=nodes, hovertext=hover, hoverinfo="text", name=label))
        order += nodes
    style_fig(fig, height=height)
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), dragmode="pan",
                      hovermode="closest", title=title,
                      legend=dict(orientation="h", y=-0.04, font=dict(size=11)))
    return fig, order


def selected_node(event):
    """Extract the clicked node id from a plotly selection event."""
    try:
        pts = event["selection"]["points"] if isinstance(event, dict) else event.selection.points
    except Exception:
        return None
    for p in pts or []:
        cd = p.get("customdata") if isinstance(p, dict) else None
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if cd:
            return cd
    return None
