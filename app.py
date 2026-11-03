"""ARC Episode Viewer v3 — pré-cálculo vetorizado, vistas isoladas e gráficos de diagnóstico.

Principais ideias de performance:
  1. Cada episódio é transformado UMA vez em DataFrames "longos" (cache_resource, sem hashing do timeline).
  2. Só a vista ativa é calculada (st.tabs executa TODAS as abas em cada frame; segmented_control não).
  3. O avanço da animação acontece DENTRO do fragmento (antes de criar o slider) -> sem race condition.
  4. Por frame só se faz slicing de DataFrames; cores/tooltips/tracejados já vêm pré-calculados.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit.errors import StreamlitAPIException

from arc_student import BundleValidationError, load_bundle, load_bundle_bytes

alt.data_transformers.disable_max_rows()

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_bundles"

# --------------------------------------------------------------------------------------
# CONSTANTES VISUAIS
# --------------------------------------------------------------------------------------
V_STATES = ["idle", "transit", "on_site", "repairing", "returning"]
V_COLORS = ["#555555", "#0078ff", "#00dcdc", "#ff8c00", "#9b59b6"]
I_STATES = ["open", "on_site", "repairing", "resolved"]
I_COLORS = ["#ff2b2b", "#00dcdc", "#ff8c00", "#5a5a5a"]
I_LABEL = {
    "open": "Aberto (a aguardar)",
    "on_site": "Navio no local",
    "repairing": "Em reparação",
    "resolved": "Resolvido",
}
OUTCOMES = ["accepted", "rejected", "duplicate"]
OUTCOME_COLORS = ["#00dcdc", "#ff4b4b", "#ff8c00"]
GREY = [120, 120, 120, 60]
VIEWS = ["🗺️ Mapa", "🩺 Diagnóstico", "🎯 Foco (navio)", "⚖️ Comparação"]
DIAG_SECTIONS = ["Backlog e capacidade", "Porque ficaram por resolver", "Recompensa", "Frota", "Decisões"]


# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------
def bundle_label(bundle: dict) -> str:
    ep = bundle["episode"]
    return f"{ep['scenario']['id']} | {ep['policy']['label']} | seed {ep['scenario']['world_seed']}"


def metric(metrics: dict, name: str, default=0):
    value = metrics.get(name, default)
    return default if value is None else value


def fmt_id(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return str(int(x)) if float(x).is_integer() else str(x)


def haversine_np(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def ensure(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    for c, default in cols.items():
        if c not in df.columns:
            df[c] = default
    return df


def add_rgba(df: pd.DataFrame, colors: list[list[int]]) -> None:
    arr = np.array(colors, dtype=int).reshape(-1, 4)
    for j, c in enumerate("rgba"):
        df[c] = arr[:, j]


def dashed_segments(pts: list[tuple], dash: float = 0.55, gap: float = 0.40) -> list[tuple]:
    """pts = [(lon, lat, k)]. Devolve [(x1, y1, x2, y2, k)] com k = passo em que o segmento passa a existir."""
    clean = pts[:1]
    for p in pts[1:]:
        if abs(p[0] - clean[-1][0]) > 1e-5 or abs(p[1] - clean[-1][1]) > 1e-5:
            clean.append(p)
    out, drawing, remaining = [], True, dash
    for a, b in zip(clean, clean[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        ux, uy, done, cx, cy = dx / length, dy / length, 0.0, a[0], a[1]
        while done < length:
            step = min(remaining, length - done)
            nx, ny = cx + ux * step, cy + uy * step
            if drawing:
                out.append((cx, cy, nx, ny, b[2]))
            cx, cy, done, remaining = nx, ny, done + step, remaining - step
            if remaining <= 1e-6:
                drawing = not drawing
                remaining = dash if drawing else gap
    return out


def run_blocks(df: pd.DataFrame, ent: str, col: str, t_end: float) -> pd.DataFrame:
    """Run-length encoding vetorizado: blocos contínuos de estado por entidade."""
    d = df[[ent, col, "i", "time_h"]].rename(columns={ent: "ent", col: "est"}).sort_values(["ent", "i"], kind="stable")
    if d.empty:
        return pd.DataFrame(columns=["ent", "est", "ini", "fim", "dur"])
    d["blk"] = (d["est"] != d.groupby("ent")["est"].shift()).cumsum()
    b = d.groupby("blk").agg(ent=("ent", "first"), est=("est", "first"), ini=("time_h", "min"))
    b["fim"] = b.groupby("ent")["ini"].shift(-1).fillna(t_end)
    b["dur"] = b["fim"] - b["ini"]
    return b.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# INFRAESTRUTURA ESTÁTICA
# --------------------------------------------------------------------------------------
def build_infra(map_data: dict, faded: bool) -> tuple[dict, pd.DataFrame]:
    feats, rows = [], []
    alpha = 45 if faded else 255
    for cable in map_data.get("cables", map_data.get("submarine_cables", [])):
        coords = cable.get("coordinates", cable.get("path", []))
        if coords:
            feats.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"color": [0, 180, 216, 25 if faded else 90],
                               "tooltip_text": f"Cabo Submarino: {cable.get('id', 'Cabo')}", "radius": 0},
            })
    for key, label, color, radius in (("ports", "Porto", [255, 255, 255], 55000),
                                      ("landing_points", "Landing Point", [155, 89, 182], 50000)):
        for p in map_data.get(key, []):
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                "properties": {"color": color + [alpha], "tooltip_text": f"ID: {p['id']}\n{label}\nEstado: Static",
                               "radius": radius},
            })
            rows.append({"Nome": p["id"], "Tipo": label, "Coords": f"{p['lat']:.3f}, {p['lon']:.3f}"})
    df = pd.DataFrame(rows, columns=["Nome", "Tipo", "Coords"])
    return {"type": "FeatureCollection", "features": feats}, df


# --------------------------------------------------------------------------------------
# CARREGAMENTO COM CACHE (sem cópias: cache_resource devolve sempre o mesmo objeto)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_sample(path: str, mtime: float) -> dict:
    return load_bundle(Path(path))


@st.cache_resource(show_spinner=False)
def _load_upload(data: bytes, name: str) -> dict:
    return load_bundle_bytes(data, name)


# --------------------------------------------------------------------------------------
# PRÉ-CÁLCULO ÚNICO POR EPISÓDIO
# --------------------------------------------------------------------------------------
def ep_key(ep: dict) -> str:
    tl = ep["timeline"]
    return f"{ep['id']}|{len(tl)}|{tl[-1].get('cumulative_reward', 0) if tl else 0}"


@st.cache_resource(show_spinner="A pré-calcular o episódio…")
def _prepare(_ep: dict, key: str) -> SimpleNamespace:
    tl = _ep["timeline"]
    T = len(tl)
    t_end = float(tl[-1]["time_h"])
    vrows, irows, drows, erows = [], [], [], []
    for i, t in enumerate(tl):
        th = float(t["time_h"])
        tag = {"i": i, "time_h": th}
        lab = {"Passo": t.get("step", i), "Tempo": f"{th:.1f}h"}
        vrows += [{**v, **tag} for v in t.get("vessels", [])]
        irows += [{**x, **tag} for x in t.get("incidents", [])]
        drows += [{**d, **tag, **lab} for d in t.get("decisions", []) if d.get("action_type") != "hold"]
        erows += [{**e, **tag, **lab} for e in t.get("events", [])]
    idx = pd.RangeIndex(T, name="i")

    # ---------- navios (formato longo) ----------
    ves = ensure(pd.DataFrame(vrows), {"type": "unknown", "assigned_incident_id": np.nan})
    ves["type"] = ves["type"].fillna("unknown").astype(str)
    ves["aid"] = pd.to_numeric(ves["assigned_incident_id"], errors="coerce")
    has = ves["aid"].notna()
    aid_s = ves["aid"].map(fmt_id)
    s = ves["status"]
    ves["Tipo"] = ves["type"].str.capitalize()
    ves["Estado"] = np.select(
        [(s == "transit") & has, (s == "on_site") & has, (s == "repairing") & has, s == "returning", s == "idle"],
        [("Em trânsito para o Incidente " + aid_s).to_numpy(), ("No local do Incidente " + aid_s).to_numpy(),
         ("A reparar o Incidente " + aid_s).to_numpy(), "A regressar ao Porto", "Inativo (a aguardar ordens)"],
        default=s.astype(str).str.capitalize().to_numpy(),
    )
    ves["Coords"] = ves["lat"].map("{:.3f}".format) + ", " + ves["lon"].map("{:.3f}".format)
    ves["tooltip_text"] = "ID: " + ves["id"].astype(str) + "\nNavio (" + ves["Tipo"] + ")\nEstado: " + ves["Estado"]
    vt = {"patrol": [0, 120, 255, 255], "repair": [255, 140, 0, 255]}
    add_rgba(ves, [vt.get(t, [150, 150, 150, 255]) for t in ves["type"]])

    # ---------- incidentes (formato longo) ----------
    inc = ensure(pd.DataFrame(irows), {"id": np.nan, "status": "", "lat": np.nan, "lon": np.nan,
                                         "severity": np.nan, "i": 0, "time_h": 0.0})
    inc["severity"] = pd.to_numeric(inc["severity"], errors="coerce")
    inc["Estado"] = inc["status"].map(lambda x: I_LABEL.get(x, str(x).capitalize()))
    inc["Incidente"] = "incident_" + inc["id"].astype(str)
    inc["Severidade"] = inc["severity"]
    inc["Coords"] = inc["lat"].map("{:.3f}".format) + ", " + inc["lon"].map("{:.3f}".format)
    sev_txt = inc["severity"].map(lambda x: "" if pd.isna(x) else f"\nSeveridade: {x:.2f}")
    inc["tooltip_text"] = "ID: " + inc["Incidente"] + "\nIncidente\nEstado: " + inc["Estado"] + sev_txt
    ic = {"open": [255, 40, 40, 255], "on_site": [0, 220, 220, 255], "repairing": [0, 220, 220, 255]}
    add_rgba(inc, [ic.get(x, [100, 100, 100, 120]) for x in inc["status"]])

    # ---------- séries por passo ----------
    steps = pd.DataFrame({
        "i": idx, "time_h": [float(t["time_h"]) for t in tl],
        "reward": [float(t.get("reward", 0.0)) for t in tl],
        "cumulative_reward": [float(t.get("cumulative_reward", 0.0)) for t in tl],
    })
    cnt = (inc.groupby(["i", "status"]).size().unstack(fill_value=0)
           .reindex(index=idx, columns=I_STATES, fill_value=0))
    steps["abertos"] = cnt["open"].to_numpy()
    steps["em_curso"] = (cnt["on_site"] + cnt["repairing"]).to_numpy()
    steps["resolvidos"] = cnt["resolved"].to_numpy()
    steps["idle"] = (ves[ves["status"] == "idle"].groupby("i").size().reindex(idx, fill_value=0)).to_numpy()
    # incidente aberto sem nenhum navio atribuído nesse passo
    pairs = ves.loc[has, ["i", "aid"]].drop_duplicates().rename(columns={"aid": "id"}).astype({"id": float})
    oi = inc.loc[inc["status"] == "open", ["i", "id"]].copy()
    oi["id"] = pd.to_numeric(oi["id"], errors="coerce").astype(float)
    mrg = oi.merge(pairs, on=["i", "id"], how="left", indicator=True)
    steps["sem_alvo"] = (mrg[mrg["_merge"] == "left_only"].groupby("i").size()
                         .reindex(idx, fill_value=0)).to_numpy()
    steps["perdida"] = np.minimum(steps["idle"], steps["sem_alvo"])
    lbl_b = {"abertos": "Aberto (a aguardar)", "em_curso": "Em curso (no local / a reparar)", "resolvidos": "Resolvido"}
    backlog_long = steps.melt(id_vars="time_h", value_vars=list(lbl_b), var_name="k", value_name="n")
    backlog_long["Estado"] = backlog_long["k"].map(lbl_b)
    lbl_c = {"idle": "Navios inativos", "sem_alvo": "Incidentes abertos sem navio atribuído"}
    cap_long = steps.melt(id_vars="time_h", value_vars=list(lbl_c), var_name="k", value_name="n")
    cap_long["Série"] = cap_long["k"].map(lbl_c)

    # ---------- resumo por incidente ----------
    if inc.empty:
        inc_sum = pd.DataFrame(columns=["Incidente", "Rotulo", "Severidade", "Surgiu (h)", "Espera (h)",
                                        "Idade (h)", "Resultado", "Estado final", "first"])
    else:
        g = inc.sort_values(["id", "i"]).groupby("id")
        inc_sum = pd.DataFrame({"first": g["time_h"].min(), "final": g["status"].last(), "sev": g["severity"].last()})
        inc_sum["resp"] = inc[inc["status"] != "open"].groupby("id")["time_h"].min()
        inc_sum["res"] = inc[inc["status"] == "resolved"].groupby("id")["time_h"].min()
        inc_sum["Espera (h)"] = inc_sum["resp"].fillna(t_end) - inc_sum["first"]
        inc_sum["Idade (h)"] = inc_sum["res"].fillna(t_end) - inc_sum["first"]
        inc_sum["Resultado"] = np.where(inc_sum["final"] == "resolved", "Resolvido", "Por resolver")
        inc_sum = inc_sum.reset_index()
        inc_sum["Incidente"] = "Inc " + inc_sum["id"].map(fmt_id)
        inc_sum["Rotulo"] = inc_sum["Incidente"] + np.where(inc_sum["Resultado"] == "Por resolver", " ⚠", "")
        inc_sum["Severidade"] = inc_sum["sev"]
        inc_sum["Surgiu (h)"] = inc_sum["first"]
        inc_sum["Estado final"] = inc_sum["final"].map(lambda x: I_LABEL.get(x, x))
        inc_sum = inc_sum.sort_values(["first", "id"]).reset_index(drop=True)
    lab_map = dict(zip(inc_sum.get("id", []), inc_sum["Rotulo"]))
    inc_blocks = run_blocks(inc, "id", "status", t_end)
    inc_blocks["Incidente"] = inc_blocks["ent"].map(lab_map)
    inc_blocks = inc_blocks.rename(columns={"est": "Estado", "ini": "Início", "fim": "Fim", "dur": "Duracao_num"})
    top = inc_sum.sort_values("Idade (h)", ascending=False).head(25)
    wait_long = pd.concat([
        pd.DataFrame({"Rotulo": top["Rotulo"], "Fase": "Espera pela 1.ª resposta", "Horas": top["Espera (h)"]}),
        pd.DataFrame({"Rotulo": top["Rotulo"], "Fase": "Tratamento / pendente",
                      "Horas": (top["Idade (h)"] - top["Espera (h)"]).clip(lower=0)}),
    ])

    # ---------- Gantt e ocupação da frota ----------
    gantt = run_blocks(ves, "id", "status", t_end).rename(
        columns={"ent": "Navio", "est": "Estado", "ini": "Início", "fim": "Fim", "dur": "Duracao_num"})
    gantt["Duração"] = gantt["Duracao_num"].map("{:.1f}h".format)
    util = gantt.groupby(["Navio", "Estado"], as_index=False)["Duracao_num"].sum()

    # ---------- distâncias ----------
    vs = ves.sort_values(["id", "i"]).copy()
    gb = vs.groupby("id")
    vs["step_km"] = haversine_np(gb["lat"].shift(), gb["lon"].shift(), vs["lat"], vs["lon"])
    vs["cum_km"] = vs["step_km"].fillna(0).groupby(vs["id"]).cumsum()
    cumkm = vs.pivot_table(index="i", columns="id", values="cum_km", aggfunc="last").reindex(idx).ffill().fillna(0)
    visited = (ves[ves["status"].isin(["on_site", "repairing"]) & has].groupby("id")["aid"].nunique())

    # ---------- rasto tracejado (pré-calculado, filtrado por k <= passo) ----------
    drows_s = []
    for vid, gv in vs.groupby("id"):
        patrol = gv["type"].iloc[0] == "patrol"
        rgb = (100, 195, 255) if patrol else (255, 180, 80)
        for x1, y1, x2, y2, k in dashed_segments(list(zip(gv["lon"], gv["lat"], gv["i"]))):
            drows_s.append((vid, x1, y1, x2, y2, *rgb, 175, k))
    dash = pd.DataFrame(drows_s, columns=["vid", "sx", "sy", "tx", "ty", "r", "g", "b", "a", "k"])

    # ---------- decisões e eventos ----------
    dec = ensure(pd.DataFrame(drows), {"agent_id": None, "action_type": None, "target_incident_id": None,
                                         "outcome": None, "i": 0, "time_h": 0.0, "Passo": 0, "Tempo": ""})
    ev = pd.DataFrame(erows) if erows else pd.DataFrame(columns=["Passo", "Tempo", "i", "time_h"])

    geo_n, df_infra = build_infra(_ep.get("map", {}), False)
    geo_f, _ = build_infra(_ep.get("map", {}), True)
    return SimpleNamespace(
        T=T, t_end=t_end, timeline=tl, steps=steps, backlog_long=backlog_long, cap_long=cap_long,
        ves=ves, inc=inc, ves_by_i={k: g for k, g in ves.groupby("i")}, inc_by_i={k: g for k, g in inc.groupby("i")},
        empty_ves=ves.iloc[0:0], empty_inc=inc.iloc[0:0], inc_sum=inc_sum, inc_blocks=inc_blocks, wait_long=wait_long,
        gantt=gantt, util=util, cumkm=cumkm, visited=visited, dash=dash, dec=dec, ev=ev,
        geo_n=geo_n, geo_f=geo_f, df_infra=df_infra, vessel_ids=list(dict.fromkeys(ves["id"])),
    )


def prep(ep: dict) -> SimpleNamespace:
    return _prepare(ep, ep_key(ep))


# --------------------------------------------------------------------------------------
# MAPA (apenas slicing; tudo o resto já está pré-calculado)
# --------------------------------------------------------------------------------------
def build_deck(P, i, focus=None, isolate=False, lines=True, trails=False, height=500):
    i = min(i, P.T - 1)
    ves = P.ves_by_i.get(i, P.empty_ves)
    inc = P.inc_by_i.get(i, P.empty_inc)
    fv = ves[ves["id"] == focus] if focus else ves.iloc[0:0]
    target = fv["aid"].iloc[0] if len(fv) else np.nan
    layers = []

    if not (focus and isolate) and P.geo_n["features"]:
        geo = P.geo_f if focus else P.geo_n
        layers.append(pdk.Layer(
            "GeoJsonLayer", id="infra", data=geo, pickable=True, stroked=True, filled=True,
            point_radius_min_pixels=6, point_radius_max_pixels=12, get_point_radius="properties.radius",
            get_fill_color="properties.color", get_line_color="properties.color", line_width_min_pixels=2))

    inc_m = inc
    if focus:
        mask = inc["id"] != target if pd.notna(target) else pd.Series(True, index=inc.index)
        inc_m = inc[~mask].copy() if isolate else inc.copy()
        if not isolate and mask.any():
            inc_m.loc[mask.to_numpy(), ["r", "g", "b", "a"]] = GREY
    ves_m = ves
    if focus:
        mask_v = ves["id"] != focus
        ves_m = ves[~mask_v].copy() if isolate else ves.copy()
        if not isolate:
            ves_m.loc[mask_v.to_numpy(), ["r", "g", "b", "a"]] = GREY

    if trails and i > 0 and not P.dash.empty:
        d = P.dash[P.dash["k"] <= i]
        if focus:
            d = d[d["vid"] == focus]
        if not d.empty:
            layers.append(pdk.Layer("LineLayer", id="trails", data=d[["sx", "sy", "tx", "ty", "r", "g", "b", "a"]],
                                    get_source_position="[sx, sy]", get_target_position="[tx, ty]",
                                    get_color="[r, g, b, a]", get_width=2.5, pickable=False))

    if lines:
        tr = ves[(ves["status"] == "transit") & ves["aid"].notna()]
        if focus:
            tr = tr[tr["id"] == focus]
        if not tr.empty and not inc.empty:
            tgt = inc[["id", "lon", "lat"]].rename(columns={"id": "aid", "lon": "tx", "lat": "ty"}).astype({"aid": float})
            m = tr.merge(tgt, on="aid").rename(columns={"lon": "sx", "lat": "sy"})
            if not m.empty:
                p = (m["type"] == "patrol").to_numpy()
                m["r"], m["g"], m["b"], m["a"] = np.where(p, 0, 255), np.where(p, 195, 165), np.where(p, 255, 0), 210
                layers.append(pdk.Layer("LineLayer", id="lines", data=m[["sx", "sy", "tx", "ty", "r", "g", "b", "a"]],
                                        get_source_position="[sx, sy]", get_target_position="[tx, ty]",
                                        get_color="[r, g, b, a]", get_width=3, pickable=False))

    cols = ["lon", "lat", "r", "g", "b", "a", "tooltip_text"]
    if not inc_m.empty:
        layers.append(pdk.Layer("ScatterplotLayer", id="inc", data=inc_m[cols], get_position="[lon, lat]",
                                get_fill_color="[r, g, b, a]", get_radius=62000, radius_min_pixels=8,
                                radius_max_pixels=17, pickable=True))
    if not ves_m.empty:
        layers.append(pdk.Layer("ScatterplotLayer", id="ves", data=ves_m[cols], get_position="[lon, lat]",
                                get_fill_color="[r, g, b, a]", get_line_color=[255, 255, 255, 200], stroked=True,
                                line_width_min_pixels=1.5, get_radius=55000, radius_min_pixels=8,
                                radius_max_pixels=16, pickable=True))

    tooltip = {"text": "{tooltip_text}", "style": {
        "backgroundColor": "#222", "color": "#fff", "border": "1px solid #444", "borderRadius": "4px",
        "fontSize": "0.85em", "padding": "12px", "whiteSpace": "pre-wrap", "width": "max-content",
        "maxWidth": "none", "height": "auto", "maxHeight": "none", "overflow": "visible"}}
    lat, lon, zoom = 20, -40, 2
    if len(fv):
        lat, lon, zoom = float(fv["lat"].iloc[0]), float(fv["lon"].iloc[0]), 3.4
    return pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=0),
                    tooltip=tooltip, map_style="dark", height=height)


def tables_at(P, i):
    i = min(i, P.T - 1)
    ves = P.ves_by_i.get(i, P.empty_ves)
    inc = P.inc_by_i.get(i, P.empty_inc)
    return (ves[["id", "Estado", "Tipo", "Coords"]].rename(columns={"id": "Navio"}),
            inc[["Incidente", "Estado", "Severidade", "Coords"]])


def legend():
    st.markdown(
        "<div style='font-size:.88em;color:#a0a0a0;padding:8px 0;text-align:center'>"
        "<span style='color:rgb(0,120,255)'>●</span> Patrulha &nbsp;&nbsp;"
        "<span style='color:rgb(255,140,0)'>●</span> Reparação &nbsp;&nbsp;"
        "<span style='color:rgb(255,40,40)'>●</span> Alerta aberto &nbsp;&nbsp;"
        "<span style='color:rgb(0,220,220)'>●</span> No local / a reparar &nbsp;&nbsp;"
        "<span style='color:#fff'>●</span> Porto &nbsp;&nbsp;"
        "<span style='color:rgb(155,89,182)'>●</span> Landing Point</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# GRÁFICOS
# --------------------------------------------------------------------------------------
def rule(t: float, field: str = "time_h", color: str = "red", w: int = 2):
    return (alt.Chart(pd.DataFrame({field: [t]})).mark_rule(color=color, strokeWidth=w)
            .encode(x=alt.X(f"{field}:Q", title="Tempo (h)")))


def legend_top():
    return alt.Legend(orient="top", title=None)


def ch_backlog(P, t):
    base = alt.Chart(P.backlog_long).mark_area(opacity=0.85, interpolate="step-after").encode(
        x=alt.X("time_h:Q", title="Tempo (h)"), y=alt.Y("n:Q", title="Incidentes"),
        color=alt.Color("Estado:N", legend=legend_top(), scale=alt.Scale(
            domain=["Aberto (a aguardar)", "Em curso (no local / a reparar)", "Resolvido"],
            range=["#ff2b2b", "#ff8c00", "#3a3a3a"])),
        tooltip=["Estado", "n", alt.Tooltip("time_h:Q", format=".1f")])
    return (base + rule(t)).properties(height=260, title="Composição dos incidentes ao longo do tempo")


def ch_capacity(P, t):
    lines = alt.Chart(P.cap_long).mark_line(interpolate="step-after", strokeWidth=2).encode(
        x=alt.X("time_h:Q", title="Tempo (h)"), y=alt.Y("n:Q", title="Quantidade"),
        color=alt.Color("Série:N", legend=legend_top(),
                        scale=alt.Scale(range=["#8a8a8a", "#ff2b2b"])))
    area = alt.Chart(P.steps).mark_area(color="#ff2b2b", opacity=0.3, interpolate="step-after").encode(
        x="time_h:Q", y=alt.Y("perdida:Q", title="Quantidade"),
        tooltip=[alt.Tooltip("time_h:Q", format=".1f"), alt.Tooltip("perdida:Q", title="Oportunidades perdidas")])
    return (area + lines + rule(t)).properties(
        height=260, title="Capacidade ociosa vs. incidentes sem navio (área = oportunidade perdida)")


def ch_incident_gantt(P, t):
    if P.inc_blocks.empty:
        return None
    order = list(P.inc_sum["Rotulo"])
    base = alt.Chart(P.inc_blocks).mark_bar(cornerRadius=2, size=14).encode(
        x=alt.X("Início:Q", title="Tempo (h)"), x2="Fim:Q",
        y=alt.Y("Incidente:N", sort=order, title=None, axis=alt.Axis(grid=False)),
        color=alt.Color("Estado:N", scale=alt.Scale(domain=I_STATES, range=I_COLORS), legend=legend_top()),
        tooltip=["Incidente", "Estado", alt.Tooltip("Início:Q", format=".1f"), alt.Tooltip("Fim:Q", format=".1f"),
                 alt.Tooltip("Duracao_num:Q", title="Duração (h)", format=".1f")])
    h = max(140, min(900, len(order) * 24 + 60))
    return (base + rule(t, "Início")).properties(height=h, title="Ciclo de vida de cada incidente (⚠ = ficou por resolver)")


def ch_wait(P):
    if P.wait_long.empty:
        return None
    order = list(P.wait_long.groupby("Rotulo")["Horas"].sum().sort_values(ascending=False).index)
    return alt.Chart(P.wait_long).mark_bar().encode(
        x=alt.X("Horas:Q", title="Horas desde que surgiu", stack=True),
        y=alt.Y("Rotulo:N", sort=order, title=None),
        color=alt.Color("Fase:N", legend=legend_top(), scale=alt.Scale(
            domain=["Espera pela 1.ª resposta", "Tratamento / pendente"], range=["#ff2b2b", "#0078ff"])),
        tooltip=["Rotulo", "Fase", alt.Tooltip("Horas:Q", format=".1f")],
    ).properties(height=max(140, len(order) * 22 + 50), title="Incidentes que mais tempo demoraram")


def ch_triage(P):
    d = P.inc_sum
    if d.empty or d["Severidade"].notna().sum() == 0:
        return None
    return alt.Chart(d).mark_circle(size=110, opacity=0.9).encode(
        x=alt.X("Severidade:Q", title="Severidade", scale=alt.Scale(zero=False)),
        y=alt.Y("Idade (h):Q", title="Idade até resolver / fim (h)"),
        color=alt.Color("Resultado:N", legend=legend_top(),
                        scale=alt.Scale(domain=["Resolvido", "Por resolver"], range=["#00dcdc", "#ff2b2b"])),
        tooltip=["Incidente", alt.Tooltip("Severidade:Q", format=".2f"), alt.Tooltip("Espera (h):Q", format=".1f"),
                 alt.Tooltip("Idade (h):Q", format=".1f"), "Resultado"],
    ).properties(height=280, title="Triagem: os mais graves esperam menos? (pontos altos à direita = problema)")


def ch_reward_cum(P, t):
    base = alt.Chart(P.steps).mark_line(color="#0078ff").encode(
        x=alt.X("time_h:Q", title="Tempo (h)"), y=alt.Y("cumulative_reward:Q", title="Recompensa acumulada"))
    return (base + rule(t)).properties(height=240, title="Recompensa acumulada")


def ch_reward_step(P, t):
    base = alt.Chart(P.steps).mark_bar(color="#ff4b4b", size=2.5).encode(
        x=alt.X("time_h:Q", title="Tempo (h)"), y=alt.Y("reward:Q", title="Recompensa no passo"),
        tooltip=[alt.Tooltip("time_h:Q", title="Tempo (h)", format=".1f"), alt.Tooltip("reward:Q", format=".4f")])
    return (base + rule(t, color="white")).properties(height=240, title="Sangramento de pontos por passo")


def ch_donut(vals: dict[str, float], height=240, title="Peso absoluto das falhas"):
    d = pd.DataFrame({"Componente": list(vals), "Valor": list(vals.values())})
    d = d[d["Valor"] > 0]
    return alt.Chart(d).mark_arc(innerRadius=50).encode(
        theta="Valor:Q", color=alt.Color("Componente:N", scale=alt.Scale(scheme="set2"),
                                         legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Componente", alt.Tooltip("Valor:Q", format=".2f")]).properties(height=height, title=title)


def ch_fleet_gantt(P, t, gantt=None, ids=None):
    g = P.gantt if gantt is None else gantt
    if g.empty:
        return None
    ids = ids or P.vessel_ids
    base = alt.Chart(g).mark_bar(cornerRadius=4, size=24).encode(
        x=alt.X("Início:Q", title="Tempo (h)"), x2="Fim:Q",
        y=alt.Y("Navio:N", sort=ids, title=None, axis=alt.Axis(grid=False, tickSize=0, domain=False)),
        color=alt.Color("Estado:N", scale=alt.Scale(domain=V_STATES, range=V_COLORS), legend=legend_top()),
        tooltip=["Navio", "Estado", alt.Tooltip("Início:Q", format=".1f"), alt.Tooltip("Fim:Q", format=".1f"), "Duração:N"])
    return (base + rule(t, "Início")).properties(height=max(120, len(ids) * 46 + 60),
                                                  title="Cronograma da frota")


def ch_util(P):
    if P.util.empty:
        return None
    return alt.Chart(P.util).mark_bar().encode(
        x=alt.X("Duracao_num:Q", stack="normalize", title="% do episódio", axis=alt.Axis(format="%")),
        y=alt.Y("Navio:N", sort=P.vessel_ids, title=None),
        color=alt.Color("Estado:N", scale=alt.Scale(domain=V_STATES, range=V_COLORS), legend=legend_top()),
        tooltip=["Navio", "Estado", alt.Tooltip("Duracao_num:Q", title="Horas", format=".1f")],
    ).properties(height=max(120, len(P.vessel_ids) * 40 + 60), title="Ocupação por navio (onde há cinzento há desperdício)")


def ch_dec_scatter(dec, t):
    base = alt.Chart(dec).mark_circle(size=80, opacity=0.9).encode(
        x=alt.X("time_h:Q", title="Tempo (h)"), y=alt.Y("agent_id:N", title=None),
        color=alt.Color("outcome:N", title="Resultado", scale=alt.Scale(domain=OUTCOMES, range=OUTCOME_COLORS),
                        legend=legend_top()),
        tooltip=["Passo", "Tempo", "agent_id", "action_type", "target_incident_id", "outcome"])
    return (base + rule(t, w=1)).properties(height=200, title="Mapa de decisões")


def ch_dec_bars(dec):
    d = dec.groupby(["agent_id", "outcome"]).size().reset_index(name="n")
    return alt.Chart(d).mark_bar().encode(
        x=alt.X("n:Q", title="Nº de ações"), y=alt.Y("agent_id:N", title=None),
        color=alt.Color("outcome:N", title="Resultado", scale=alt.Scale(domain=OUTCOMES, range=OUTCOME_COLORS),
                        legend=legend_top()),
        tooltip=["agent_id", "outcome", "n"]).properties(height=200, title="Resultado das decisões por navio")


def style_outcome(df: pd.DataFrame):
    def f(v):
        return {"rejected": "color:#ff4b4b;font-weight:bold", "duplicate": "color:#ff8c00;font-weight:bold",
                "accepted": "color:#00dcdc"}.get(v, "")
    return df.style.map(f, subset=["outcome"])


def decision_panel(dec: pd.DataFrame, key: str):
    """Taxa de erro + feed das últimas ações."""
    if dec.empty:
        st.info("O agente ainda não executou ações de despacho.")
        return
    err = int(dec["outcome"].isin(["rejected", "duplicate"]).sum())
    a, b = st.columns(2)
    a.metric("Taxa de erro", f"{100 * err / len(dec):.1f}%", f"{err} falhas", delta_color="inverse")
    b.metric("Total de ações", len(dec))
    show = dec.iloc[::-1].head(15)[["Passo", "Tempo", "agent_id", "action_type", "target_incident_id", "outcome"]]
    st.dataframe(style_outcome(show), hide_index=True, width="stretch", height=250, key=key)


# --------------------------------------------------------------------------------------
# PÁGINA
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="ARC Episode Viewer", layout="wide")
st.title("ARC Episode Viewer")
st.caption("Plataforma operacional e de auditoria de políticas de Reinforcement Learning (cabos submarinos).")
st.markdown("""<style>
.mapboxgl-ctrl-bottom-left,.mapboxgl-ctrl-bottom-right{display:none!important}
div[data-baseweb="select"]{height:auto!important}
div[data-baseweb="select"] *{white-space:normal!important;word-break:break-word!important;text-overflow:unset!important}
</style>""", unsafe_allow_html=True)

bundles: dict[str, dict] = {}
errors: list[str] = []
for path in sorted(SAMPLE_DIR.glob("*.json")):
    try:
        b = _load_sample(str(path), path.stat().st_mtime)
        bundles[b["episode"]["id"]] = b
    except BundleValidationError as e:
        errors.append(f"{path.name}: {e}")
for up in st.sidebar.file_uploader("Adicionar episode bundles", type=["json"], accept_multiple_files=True):
    try:
        b = _load_upload(up.getvalue(), up.name)
        bundles[b["episode"]["id"]] = b
    except BundleValidationError as e:
        errors.append(f"{up.name}: {e}")
for e in errors:
    st.error(e)
if not bundles:
    st.warning("Não existem bundles válidos para apresentar.")
    st.stop()

episode_id = st.sidebar.selectbox("Episódio principal", options=list(bundles),
                                  format_func=lambda v: bundle_label(bundles[v]))
episode = bundles[episode_id]["episode"]
scenario, policy, metrics = episode["scenario"], episode["policy"], episode["metrics"]
P = prep(episode)
max_frame = P.T - 1

st.sidebar.divider()
st.sidebar.caption("**📋 DETALHES DO EPISÓDIO ATIVO**")
st.sidebar.caption(f"**Cenário:** {scenario['id']}")
st.sidebar.caption(f"**Política:** {policy['label']} ({policy['kind']})")
st.sidebar.caption(f"**Duração:** {P.t_end:.1f}h | **Passos:** {P.T}")
st.sidebar.caption(f"**Seed:** {scenario['world_seed']} | **Split:** {scenario['split']}")
st.sidebar.caption(f"**Banco:** {scenario['bank_id']}")

ss = st.session_state
ss.setdefault("step", 0)
ss.setdefault("playing", False)
if ss.get("ep_seen") != episode_id:
    ss.step, ss.playing, ss.ep_seen = 0, False, episode_id
ss.step = min(ss.step, max_frame)


def click_play():
    if ss.step >= max_frame:
        ss.step = 0
    ss.playing = not ss.playing


def click_reset():
    ss.step, ss.playing = 0, False


def on_slider():
    ss.step = ss.ui_step
    ss.playing = False  # arrastar o slider pausa a reprodução


# ---------------------------------- VISTAS ----------------------------------
def view_map(i: int):
    now = P.steps.iloc[i]
    prev = P.steps.iloc[max(i - 1, 0)]
    k = st.columns(5)
    k[0].metric("Abertos", int(now["abertos"]), int(now["abertos"] - prev["abertos"]), delta_color="inverse")
    k[1].metric("Em curso", int(now["em_curso"]), int(now["em_curso"] - prev["em_curso"]), delta_color="off")
    k[2].metric("Resolvidos", int(now["resolvidos"]), int(now["resolvidos"] - prev["resolvidos"]))
    k[3].metric("Navios inativos", int(now["idle"]), f"{int(now['perdida'])} c/ incidente sem alvo", delta_color="off")
    k[4].metric("Recompensa acumulada", f"{now['cumulative_reward']:.2f}", f"{now['reward']:+.4f}")
    o1, o2, _ = st.columns([1.5, 1.5, 7])
    lines = o1.checkbox("Trajetórias", True, key="chk_lin_g")
    trails = o2.checkbox("Rastos", False, key="chk_rst_g")
    df_nav, df_inc = tables_at(P, i)
    left, right = st.columns([2, 1])
    with left:
        st.pydeck_chart(build_deck(P, i, lines=lines, trails=trails), key="deck_vg")
        legend()
    with right:
        st.markdown("**Estado no instante atual**")
        t1, t2, t3 = st.tabs([f"Navios ({len(df_nav)})", f"Incidentes ({len(df_inc)})", f"Infra ({len(P.df_infra)})"])
        t1.dataframe(df_nav, hide_index=True, width="stretch", height=380)
        t2.dataframe(df_inc, hide_index=True, width="stretch", height=380)
        t3.dataframe(P.df_infra, hide_index=True, width="stretch", height=380)


def view_diag(i: int):
    t = float(P.steps["time_h"].iloc[i])
    sec = st.segmented_control("Secção", DIAG_SECTIONS, default=DIAG_SECTIONS[0], key="diag_sec",
                               label_visibility="collapsed") or DIAG_SECTIONS[0]

    if sec == DIAG_SECTIONS[0]:
        pct = 100 * float((P.steps["perdida"] > 0).mean())
        a, b, c = st.columns(3)
        a.metric("% dos passos com oportunidade perdida", f"{pct:.1f}%",
                 help="Passos em que havia navio inativo E incidente aberto sem nenhum navio atribuído.")
        b.metric("Pico de incidentes sem alvo", int(P.steps["sem_alvo"].max()))
        c.metric("Taxa de hold", f"{(lambda h: h * 100 if h <= 1 else h)(float(metric(metrics, 'hold_action_rate'))):.1f}%")
        l, r = st.columns(2)
        l.altair_chart(ch_backlog(P, t), width="stretch")
        r.altair_chart(ch_capacity(P, t), width="stretch")

    elif sec == DIAG_SECTIONS[1]:
        g = ch_incident_gantt(P, t)
        if g is not None:
            st.altair_chart(g, width="stretch")
        l, r = st.columns(2)
        w = ch_wait(P)
        if w is not None:
            l.altair_chart(w, width="stretch")
        tri = ch_triage(P)
        if tri is not None:
            r.altair_chart(tri, width="stretch")
        pend = P.inc_sum[P.inc_sum["Resultado"] == "Por resolver"] if not P.inc_sum.empty else P.inc_sum
        st.markdown(f"**Incidentes por resolver no fim do episódio ({len(pend)})**")
        if len(pend):
            st.dataframe(pend.sort_values("Severidade", ascending=False)[
                ["Incidente", "Severidade", "Surgiu (h)", "Espera (h)", "Idade (h)", "Estado final"]],
                hide_index=True, width="stretch")
        else:
            st.success("Todos os incidentes foram resolvidos.")

    elif sec == DIAG_SECTIONS[2]:
        va, vt_, vd = (abs(metric(metrics, k)) for k in
                       ("incident_age_penalty", "transit_penalty", "unassessed_adversarial_penalty"))
        tot = va + vt_ + vd or 1
        m = st.columns(4)
        m[0].metric("Recompensa total", f"{metric(metrics, 'total_reward'):.2f}")
        for col, lab, key, v in ((m[1], "Penalidade: idade SLA", "incident_age_penalty", va),
                                 (m[2], "Penalidade: trânsito", "transit_penalty", vt_),
                                 (m[3], "Penalidade: adversarial", "unassessed_adversarial_penalty", vd)):
            col.metric(lab, f"{metric(metrics, key):.2f}", f"{100 * v / tot:.1f}% do total", delta_color="off")
        l, r = st.columns(2)
        l.altair_chart(ch_reward_cum(P, t), width="stretch")
        r.altair_chart(ch_reward_step(P, t), width="stretch")
        st.altair_chart(ch_donut({"Idade do incidente": va, "Trânsito": vt_, "Risco adversarial": vd}),
                        width="stretch")

    elif sec == DIAG_SECTIONS[3]:
        g = ch_fleet_gantt(P, t)
        if g is not None:
            st.altair_chart(g, width="stretch")
        u = ch_util(P)
        if u is not None:
            st.altair_chart(u, width="stretch")

    else:
        dec = P.dec[P.dec["i"] <= i]
        ev = P.ev[P.ev["i"] <= i]
        if not dec.empty:
            l, r = st.columns([2, 1])
            l.altair_chart(ch_dec_scatter(dec, t), width="stretch")
            r.altair_chart(ch_dec_bars(dec), width="stretch")
        l, r = st.columns(2)
        with l:
            st.markdown("🎯 **Eficácia de despacho e feed de ações**")
            decision_panel(dec, "dec_diag")
        with r:
            st.markdown("🚨 **Log do ambiente (últimos eventos)**")
            if ev.empty:
                st.info("Sem eventos registados até ao momento.")
            else:
                st.dataframe(ev.iloc[::-1].head(15).drop(columns=["i", "time_h"], errors="ignore"),
                             hide_index=True, width="stretch", height=300)


def view_focus(i: int):
    f1, f2, f3, f4 = st.columns([2.5, 2.2, 2.2, 2.2])
    vid = f1.selectbox("Navio", P.vessel_ids, key="focus_vessel", label_visibility="collapsed")
    iso = f2.toggle("Ocultar restantes", True, key="tog_iso")
    lines = f3.checkbox("Trajetórias", True, key="chk_lin_f")
    trails = f4.checkbox("Rastos", True, key="chk_rst_f")

    row = P.ves_by_i[i][P.ves_by_i[i]["id"] == vid]
    estado = row["Estado"].iloc[0] if len(row) else "N/A"
    tipo = row["Tipo"].iloc[0] if len(row) else "N/A"
    gn = P.gantt[P.gantt["Navio"] == vid]
    hrs = gn.groupby("Estado")["Duracao_num"].sum()
    idle_h, tr_h = float(hrs.get("idle", 0)), float(hrs.get("transit", 0))
    work_h = float(hrs.get("on_site", 0) + hrs.get("repairing", 0))
    total_h = max(1.0, P.t_end)
    act_h = total_h - idle_h
    km_now, km_tot = float(P.cumkm.at[i, vid]), float(P.cumkm[vid].iloc[-1])

    k = st.columns(6)
    k[0].metric(f"Estado ({tipo})", estado)
    k[1].metric("Distância navegada", f"{km_now:.0f} km", f"Total: {km_tot:.0f} km", delta_color="off")
    k[2].metric("Taxa de ocupação", f"{100 * act_h / total_h:.1f}%", f"{act_h:.1f}h ativas", delta_color="off")
    k[3].metric("Em trânsito", f"{tr_h:.1f} h", f"{100 * tr_h / total_h:.1f}% do ep.", delta_color="off")
    k[4].metric("No local", f"{work_h:.1f} h", f"{int(P.visited.get(vid, 0))} inc. visitados", delta_color="off")
    k[5].metric("Horas inativas", f"{idle_h:.1f} h")

    l, r = st.columns([2, 1])
    with l:
        st.pydeck_chart(build_deck(P, i, focus=vid, isolate=iso, lines=lines, trails=trails, height=460),
                        key="deck_mf")
    with r:
        st.markdown(f"**Telemetria de {vid}**")
        if not gn.empty:
            d = gn.groupby("Estado", as_index=False)["Duracao_num"].sum()
            st.altair_chart(alt.Chart(d).mark_arc(innerRadius=45).encode(
                theta="Duracao_num:Q",
                color=alt.Color("Estado:N", scale=alt.Scale(domain=V_STATES, range=V_COLORS),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["Estado", alt.Tooltip("Duracao_num:Q", format=".1f")]).properties(height=220),
                width="stretch")
    g = ch_fleet_gantt(P, float(P.steps["time_h"].iloc[i]), gantt=gn, ids=[vid])
    if g is not None:
        st.altair_chart(g, width="stretch")
    st.markdown(f"**Decisões de {vid} até ao instante atual**")
    decision_panel(P.dec[(P.dec["agent_id"] == vid) & (P.dec["i"] <= i)], "dec_focus")


def view_compare(i: int):
    compat = {k: c for k, c in bundles.items()
              if c["episode"]["scenario"]["id"] == scenario["id"]
              and c["episode"]["scenario"]["world_seed"] == scenario["world_seed"]}
    pool = compat if len(compat) >= 2 else bundles
    keys = list(pool)
    b0 = keys.index(episode_id) if episode_id in keys else 0
    c1, c2 = st.columns(2)
    base_id = c1.selectbox("Episódio base", keys, index=b0, format_func=lambda v: bundle_label(pool[v]), key="cmp_b")
    tgt_id = c2.selectbox("Episódio alvo", keys, index=1 if (len(keys) > 1 and b0 == 0) else 0,
                          format_func=lambda v: bundle_label(pool[v]), key="cmp_t")
    eb, et = pool[base_id]["episode"], pool[tgt_id]["episode"]
    Pb, Pt = prep(eb), prep(et)
    mb, mt = eb["metrics"], et["metrics"]
    g = lambda m, n: float(metric(m, n))  # noqa: E731
    d = st.columns(6)
    d[0].metric("Recompensa total", f"{g(mt, 'total_reward'):.2f}", f"{g(mt, 'total_reward') - g(mb, 'total_reward'):+.2f}")
    d[1].metric("Taxa de resolução", f"{100 * g(mt, 'resolution_rate'):.1f}%",
                f"{100 * (g(mt, 'resolution_rate') - g(mb, 'resolution_rate')):+.1f}%")
    for col, lab, n in ((d[2], "Primeira resposta", "mean_first_response_time_h"),
                        (d[3], "Tempo de resolução", "mean_time_to_resolution_h")):
        col.metric(lab, f"{g(mt, n):.1f} h", f"{g(mt, n) - g(mb, n):+.1f} h", delta_color="inverse")
    d[4].metric("Despachos duplicados", int(g(mt, "duplicate_dispatch_count")),
                f"{int(g(mt, 'duplicate_dispatch_count') - g(mb, 'duplicate_dispatch_count')):+d}", delta_color="inverse")
    d[5].metric("Severidade por resolver", f"{g(mt, 'unresolved_severity'):.2f}",
                f"{g(mt, 'unresolved_severity') - g(mb, 'unresolved_severity'):+.2f}", delta_color="inverse")

    lb, lt = f"Base: {eb['policy']['label']}", f"Alvo: {et['policy']['label']}"
    if lb[6:] == lt[6:]:
        lb, lt = lb + " (base)", lt + " (alvo)"
    both = pd.concat([Pb.steps.assign(Política=lb), Pt.steps.assign(Política=lt)])
    dom, rng = [lb, lt], ["#8a8a8a", "#ff8c00"]
    t = float(Pb.steps["time_h"].iloc[min(i, Pb.T - 1)])

    def overlay(col, title, interp="linear"):
        ln = alt.Chart(both).mark_line(interpolate=interp, strokeWidth=2).encode(
            x=alt.X("time_h:Q", title="Tempo (h)"), y=alt.Y(f"{col}:Q", title=None),
            color=alt.Color("Política:N", scale=alt.Scale(domain=dom, range=rng), legend=legend_top()))
        return (ln + rule(t)).properties(height=240, title=title)

    c1, c2, c3 = st.columns(3)
    c1.altair_chart(overlay("cumulative_reward", "Recompensa acumulada"), width="stretch")
    c2.altair_chart(overlay("abertos", "Incidentes abertos (backlog)", "step-after"), width="stretch")
    c3.altair_chart(overlay("perdida", "Oportunidades perdidas (ocioso + incidente sem alvo)", "step-after"),
                    width="stretch")

    s1 = Pb.inc_sum[["Incidente", "Espera (h)", "Idade (h)"]].assign(Política=lb)
    s2 = Pt.inc_sum[["Incidente", "Espera (h)", "Idade (h)"]].assign(Política=lt)
    inc_cmp = pd.concat([s1, s2])
    l, r = st.columns(2)
    if not inc_cmp.empty:
        order = list(Pb.inc_sum["Incidente"])
        l.altair_chart(alt.Chart(inc_cmp).mark_bar().encode(
            y=alt.Y("Incidente:N", sort=order, title=None), yOffset="Política:N",
            x=alt.X("Espera (h):Q", title="Espera pela 1.ª resposta (h)"),
            color=alt.Color("Política:N", scale=alt.Scale(domain=dom, range=rng), legend=legend_top()),
            tooltip=["Incidente", "Política", alt.Tooltip("Espera (h):Q", format=".1f"),
                     alt.Tooltip("Idade (h):Q", format=".1f")],
        ).properties(height=max(160, len(order) * 40 + 40), title="Espera por incidente (mesma seed)"), width="stretch")
    dd = pd.concat([Pb.dec.assign(Política=lb), Pt.dec.assign(Política=lt)])
    if not dd.empty:
        cnt = dd.groupby(["Política", "outcome"]).size().reset_index(name="n")
        r.altair_chart(alt.Chart(cnt).mark_bar().encode(
            x=alt.X("n:Q", title="Nº de ações"), y=alt.Y("Política:N", title=None, sort=dom),
            color=alt.Color("outcome:N", title="Resultado", scale=alt.Scale(domain=OUTCOMES, range=OUTCOME_COLORS),
                            legend=legend_top()), tooltip=["Política", "outcome", "n"],
        ).properties(height=200, title="Resultado das decisões"), width="stretch")

    if st.toggle("Mostrar mapas lado a lado (mais pesado)", value=False, key="cmp_maps"):
        o1, o2, _ = st.columns([1.5, 1.5, 7])
        lines = o1.checkbox("Trajetórias", True, key="chk_lin_c")
        trails = o2.checkbox("Rastos", False, key="chk_rst_c")
        m1, m2 = st.columns(2)
        m1.markdown(f"**{lb}**")
        m1.pydeck_chart(build_deck(Pb, i, lines=lines, trails=trails, height=430), key="deck_cb")
        m2.markdown(f"**{lt}**")
        m2.pydeck_chart(build_deck(Pt, i, lines=lines, trails=trails, height=430), key="deck_ct")


# ---------------------------------- WORKSPACE (fragmento animado) ----------------------------------
@st.fragment
def workspace():
    # 1) avançar o passo ANTES de instanciar o slider (elimina a race condition)
    if ss.playing:
        nxt = ss.step + int(ss.get("speed", 1))
        if nxt >= max_frame:
            ss.step, ss.playing = max_frame, False
        else:
            ss.step = nxt
    ss.ui_step = ss.step
    i = ss.step
    frame = P.timeline[i]

    st.subheader(f"Cenário: {scenario['id']} — Política: {policy['label']}", divider="gray")
    k = st.columns(6)
    k[0].metric("Resolvidos", f"{int(metric(metrics, 'n_resolved'))}/{int(metric(metrics, 'n_incidents'))}")
    k[1].metric("Taxa de resolução", f"{100 * float(metric(metrics, 'resolution_rate')):.1f}%")
    k[2].metric("Primeira resposta", f"{float(metric(metrics, 'mean_first_response_time_h')):.1f} h")
    k[3].metric("Tempo de resolução", f"{float(metric(metrics, 'mean_time_to_resolution_h')):.1f} h")
    k[4].metric("Despachos duplicados", int(metric(metrics, "duplicate_dispatch_count")))
    k[5].metric("Severidade por resolver", f"{float(metric(metrics, 'unresolved_severity')):.2f}")

    c1, c2, c3, c4, c5 = st.columns([1.3, 1.1, 1.0, 2.2, 5.0])
    c1.button("⏸ Pausa" if ss.playing else "▶ Reproduzir", on_click=click_play, width="stretch", key="btn_play")
    c2.button("⏮ Reset", on_click=click_reset, width="stretch", key="btn_reset")
    c3.selectbox("Velocidade", [1, 2, 5, 10], key="speed", format_func=lambda v: f"{v}×", label_visibility="collapsed")
    c4.markdown(f"<div style='padding-top:8px;font-weight:500'>⏱️ Passo {frame.get('step', i)}/{max_frame} "
                f"({frame['time_h']:.1f}h)</div>", unsafe_allow_html=True)
    c5.slider("Navegação temporal", 0, max_frame, key="ui_step", on_change=on_slider, label_visibility="collapsed")

    view = st.segmented_control("Vista", VIEWS, default=VIEWS[0], key="view", label_visibility="collapsed") or VIEWS[0]
    st.divider()
    {VIEWS[0]: view_map, VIEWS[1]: view_diag, VIEWS[2]: view_focus, VIEWS[3]: view_compare}[view](i)

    # 2) loop de reprodução: só re-executa este fragmento
    if ss.playing:
        time.sleep(0.03)
        try:
            st.rerun(scope="fragment")
        except StreamlitAPIException:  # execução completa do script (ex.: mudança na sidebar a meio do play)
            st.rerun()


workspace()