"""ARC Episode Viewer v3.1 — pré-cálculo vetorizado, vistas isoladas e gráficos de diagnóstico.

Esta versão é a v3 + passagem final de correções VISUAIS / DE LAYOUT (a lógica de dados e o uso do
`arc_student` não foram alterados). Resumo das correções no fim do ficheiro ("NOTAS DE LAYOUT").

Principais ideias de performance (inalteradas):
  1. Cada episódio é transformado UMA vez em DataFrames "longos" (cache_resource, sem hashing do timeline).
  2. Só a vista ativa é calculada (st.tabs executa TODAS as abas em cada frame; segmented_control não).
  3. O avanço da animação acontece DENTRO do fragmento (antes de criar o slider) -> sem race condition.
  4. Por frame só se faz slicing de DataFrames; cores/tooltips/tracejados já vêm pré-calculados.
"""

from __future__ import annotations

import html
import math
import time
import re
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
GREY_UI = "#7a828e"

V_STATES = ["idle", "transit", "on_site", "repairing", "returning"]
V_COLORS = [GREY_UI, "#0078ff", "#00dcdc", "#ff8c00", "#9b59b6"]
I_STATES = ["open", "on_site", "repairing", "resolved"]
I_COLORS = ["#ff2b2b", "#00dcdc", "#ff8c00", GREY_UI]
I_LABEL = {
    "open": "Aberto (a aguardar)",
    "on_site": "Navio no local",
    "repairing": "Em reparação",
    "resolved": "Resolvido",
}

BACKLOG_LABELS = {"abertos": "Aberto (a aguardar)", "em_curso": "Em curso (no local / a reparar)",
                  "resolvidos": "Resolvido"}
BACKLOG_COLORS = ["#ff2b2b", "#ff8c00", GREY_UI]
CAP_LABELS = {"idle": "Navios inativos", "sem_alvo": "Incidentes abertos sem navio atribuído"}
CAP_COLORS = ["#9aa0a6", "#ff2b2b"]

OUTCOMES = ["accepted", "rejected", "duplicate", "duplicate_committed"]
OUTCOME_COLORS = ["#00dcdc", "#ff4b4b", "#ff8c00", "#e056c1"]
ERROR_OUTCOMES = {"rejected", "duplicate", "duplicate_committed"}
GREY = [120, 120, 120, 60]
VIEWS = ["🗺️ Mapa", "🩺 Diagnóstico", "🎯 Foco (navio)", "⚖️ Comparação"]
DIAG_SECTIONS = ["Backlog e capacidade", "Resolução", "Recompensa", "Frota", "Decisões"]

MAP_H, FOCUS_MAP_H, CMP_MAP_H = 520, 480, 460
TABLE_H = MAP_H - 90  # painel de tabelas à direita do mapa 
H_STD = 260           # gráficos temporais emparelhados
H_PAIR = 320          # gráficos de barras/dispersão emparelhados
PANEL_H = 430         # painéis (métricas + tabela) lado a lado
MAX_INC_ROWS_INLINE = 30  # Gantt de incidentes com scroll 

CC = st.column_config
NAV_CFG = {"Navio": CC.TextColumn("Navio", width="small"), "Estado": CC.TextColumn("Estado", width="large"),
           "Tipo": CC.TextColumn("Tipo", width="small"), "Coords": CC.TextColumn("Coords", width="medium")}
INC_CFG = {"Incidente": CC.TextColumn("Incidente", width="small"), "Estado": CC.TextColumn("Estado", width="medium"),
           "Severidade": CC.NumberColumn("Severidade", format="%.2f", width="small"),
           "Coords": CC.TextColumn("Coords", width="medium")}
PEND_CFG = {"Incidente": CC.TextColumn("Incidente", width="small"),
            "Severidade": CC.NumberColumn("Severidade", format="%.2f"),
            "Surgiu (h)": CC.NumberColumn("Surgiu (h)", format="%.1f"),
            "Espera (h)": CC.NumberColumn("Espera (h)", format="%.1f"),
            "Idade (h)": CC.NumberColumn("Idade (h)", format="%.1f"),
            "Estado final": CC.TextColumn("Estado final", width="medium")}


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

def _nat_key(s: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(s))]


def fleet_order(ids) -> list:
    items = list(dict.fromkeys(ids))
    repairs = sorted([x for x in items if str(x).lower().startswith("repair")], key=_nat_key)
    patrols = sorted([x for x in items if str(x).lower().startswith("patrol")], key=_nat_key)
    others = sorted([x for x in items if x not in set(repairs) and x not in set(patrols)], key=_nat_key)
    return repairs + patrols + others


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
# CARREGAMENTO COM CACHE 
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False) #devolve sempre o mesmo objeto
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
    ves = ensure(pd.DataFrame(vrows), {
        "type": "unknown",
        "assigned_incident_id": np.nan,
        "lat": np.nan,
        "lon": np.nan,
    })
    ves["type"] = ves["type"].fillna("unknown").astype(str)
    ves["lat"] = pd.to_numeric(ves["lat"], errors="coerce")
    ves["lon"] = pd.to_numeric(ves["lon"], errors="coerce")

    is_patrol = ves["type"] == "patrol"
    is_repair = ves["type"] == "repair"
    ves.loc[is_repair & (ves["status"] == "on_site"), "status"] = "repairing"
    ves.loc[is_patrol & (ves["status"] == "repairing"), "status"] = "on_site"

    ves["aid"] = pd.to_numeric(ves["assigned_incident_id"], errors="coerce")
    has = ves["aid"].notna()
    aid_s = ves["aid"].map(fmt_id)
    s = ves["status"]
    ves["Tipo"] = ves["type"].str.capitalize()
    ves["Estado"] = np.select(
        [(s == "transit") & has, (s == "on_site") & has, (s == "repairing") & has, s == "returning", s == "idle"],
        [
            ("Em trânsito para o Incidente " + aid_s).to_numpy(),
            ("No local do Incidente " + aid_s).to_numpy(),
            ("A reparar o Incidente " + aid_s).to_numpy(),
            "A regressar ao Porto",
            "Inativo (a aguardar ordens)",
        ],
        default=s.astype(str).str.capitalize().to_numpy(),
    )
    ves["Coords"] = np.where(
        ves["lat"].notna() & ves["lon"].notna(),
        ves["lat"].map("{:.3f}".format) + ", " + ves["lon"].map("{:.3f}".format),
        "",
    )
    ves["tooltip_text"] = "ID: " + ves["id"].astype(str) + "\nNavio (" + ves["Tipo"] + ")\nEstado: " + ves["Estado"]
    vt = {"patrol": [0, 120, 255, 255], "repair": [255, 140, 0, 255]}
    add_rgba(ves, [vt.get(t, [150, 150, 150, 255]) for t in ves["type"]])

    # ---------- incidentes (formato longo) ----------
    inc = ensure(pd.DataFrame(irows), {
        "id": np.nan,
        "status": "",
        "lat": np.nan,
        "lon": np.nan,
        "severity": np.nan,
        "i": 0,
        "time_h": 0.0,
    })
    inc["lat"] = pd.to_numeric(inc["lat"], errors="coerce")
    inc["lon"] = pd.to_numeric(inc["lon"], errors="coerce")
    inc["severity"] = pd.to_numeric(inc["severity"], errors="coerce")
    inc["Estado"] = inc["status"].map(lambda x: I_LABEL.get(x, str(x).capitalize()))
    inc["Incidente"] = "incident_" + inc["id"].astype(str)
    inc["Severidade"] = inc["severity"]
    inc["Coords"] = np.where(
        inc["lat"].notna() & inc["lon"].notna(),
        inc["lat"].map("{:.3f}".format) + ", " + inc["lon"].map("{:.3f}".format),
        "",
    )
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
    backlog_long = steps.melt(id_vars="time_h", value_vars=list(BACKLOG_LABELS), var_name="k", value_name="n")
    backlog_long["Estado"] = backlog_long["k"].map(BACKLOG_LABELS)
    cap_long = steps.melt(id_vars="time_h", value_vars=list(CAP_LABELS), var_name="k", value_name="n")
    cap_long["Série"] = cap_long["k"].map(CAP_LABELS)

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

    # ---------- rasto tracejado ----------
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
# MAPA 
# --------------------------------------------------------------------------------------
def build_deck(P, i, focus=None, isolate=False, lines=True, trails=False, height=MAP_H):
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
            point_radius_min_pixels=10, point_radius_max_pixels=16, get_point_radius="properties.radius",
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
                                get_fill_color="[r, g, b, a]", get_radius=62000, radius_min_pixels=12,
                                radius_max_pixels=20, pickable=True))
    if not ves_m.empty:
        layers.append(pdk.Layer("ScatterplotLayer", id="ves", data=ves_m[cols], get_position="[lon, lat]",
                                get_fill_color="[r, g, b, a]", get_line_color="[255, 255, 255, a]", stroked=True,
                                line_width_min_pixels=1.5, get_radius=55000, radius_min_pixels=7,
                                radius_max_pixels=13, pickable=True))

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
    ves = ensure(P.ves_by_i.get(i, P.empty_ves).copy(), {"id": "", "Estado": "", "Tipo": "", "Coords": ""})
    inc = ensure(P.inc_by_i.get(i, P.empty_inc).copy(), {"Incidente": "", "Estado": "", "Severidade": np.nan, "Coords": ""})
    return (
        ves[["id", "Estado", "Tipo", "Coords"]].rename(columns={"id": "Navio"}),
        inc[["Incidente", "Estado", "Severidade", "Coords"]],
    )


# --------------------------------------------------------------------------------------
# LEGENDAS HTML
# --------------------------------------------------------------------------------------
def sw(label: str, color: str, kind: str = "square") -> str:
    """Uma entrada de legenda. kind: square | dot | line | dash."""
    shape = {
        "square": f"width:12px;height:12px;border-radius:2px;background:{color}",
        "dot": f"width:11px;height:11px;border-radius:50%;background:{color}",
        "line": f"width:20px;height:0;border-top:3px solid {color}",
        "dash": f"width:20px;height:0;border-top:3px dashed {color}",
    }[kind]
    return (f"<span style='display:inline-flex;align-items:center;gap:6px;white-space:nowrap'>"
            f"<span style='display:inline-block;{shape}'></span>{html.escape(str(label))}</span>")


def grp(label: str) -> str:
    return f"<b style='color:#c8c8c8'>{label}</b>"


def legend_row(items: list[str], center: bool = False) -> None:
    just = "center" if center else "flex-start"
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;justify-content:{just};gap:6px 18px;" #flex-wrap não trunca nem altera a altura da área de desenho dos gráficos
        f"font-size:.86em;color:#a0a0a0;padding:2px 0 8px'>{''.join(items)}</div>", unsafe_allow_html=True)


def legend():
    """Legenda do mapa (agora inclui incidente resolvido, trajetória e rasto, que já aparecem no mapa)."""
    legend_row([
        sw("Patrulha", "rgb(0,120,255)", "dot"), sw("Reparação", "rgb(255,140,0)", "dot"),
        sw("Incidente aberto", "rgb(255,40,40)", "dot"), sw("No local / a reparar", "rgb(0,220,220)", "dot"),
        sw("Resolvido", "rgb(100,100,100)", "dot"), sw("Porto", "#ffffff", "dot"),
        sw("Landing Point", "rgb(155,89,182)", "dot"), sw("Trajetória navio → incidente", "rgb(0,195,255)", "line"),
        sw("Rasto", "rgb(100,195,255)", "dash"),
    ], center=True)


# --------------------------------------------------------------------------------------
# GRÁFICOS
# --------------------------------------------------------------------------------------
def ttl(main: str, sub: str | None = None) -> alt.TitleParams:
    """Título curto + subtítulo explicativo, alinhado à esquerda (evita títulos longos cortados em colunas estreitas)."""
    kw = dict(text=main, anchor="start", fontSize=14, offset=8)
    if sub:
        kw.update(subtitle=sub, subtitleFontSize=11, subtitleColor="#9aa0a6", subtitlePadding=3)
    return alt.TitleParams(**kw)


def _legend(orient: str) -> alt.Legend:
    return alt.Legend(orient=orient, title=None, labelLimit=0, labelFontSize=11, symbolSize=90,
                      columnPadding=14, rowPadding=4, padding=2)


def legend_top() -> alt.Legend:
    return _legend("top")


def legend_bottom() -> alt.Legend:
    return _legend("bottom")


def rule(t: float, t_end: float | None = None, color: str = "#ff2b2b", w: int = 3):
    scale = alt.Scale(domain=[0, t_end], nice=False) if t_end else alt.Undefined
    return (
        alt.Chart(pd.DataFrame({"t_ref": [float(t)]}))
        .mark_rule(color=color, strokeWidth=w, opacity=0.95)
        .encode(x=alt.X("t_ref:Q", title="Tempo (h)", scale=scale))
    )


def ch_backlog(P, t):
    base = alt.Chart(P.backlog_long).mark_area(opacity=0.9, interpolate="step-after").encode(
        x=alt.X("time_h:Q", title="Tempo (h)"), y=alt.Y("n:Q", title="Incidentes", axis=alt.Axis(tickMinStep=1, format='d')),
        color=alt.Color("Estado:N", legend=None, scale=alt.Scale(domain=list(BACKLOG_LABELS.values()),
                                                                  range=BACKLOG_COLORS)),
        tooltip=["Estado", "n", alt.Tooltip("time_h:Q", format=".1f")])
    return (base + rule(t)).properties(
        height=H_STD, title=ttl("Composição dos incidentes", "Quantos estão abertos, em curso ou resolvidos em cada instante"))


def ch_capacity(P, t):
    lines = alt.Chart(P.cap_long).mark_line(interpolate="step-after", strokeWidth=2).encode(
        x=alt.X("time_h:Q", title="Tempo (h)"), y=alt.Y("n:Q", title="Quantidade", axis=alt.Axis(tickMinStep=1, format='d')),
        # domínio explícito: sem ele o Vega ordena alfabeticamente e as cores ficavam trocadas
        color=alt.Color("Série:N", legend=None, scale=alt.Scale(domain=list(CAP_LABELS.values()), range=CAP_COLORS)),
        tooltip=["Série", "n", alt.Tooltip("time_h:Q", format=".1f")])
    area = alt.Chart(P.steps).mark_area(color="#ff2b2b", opacity=0.3, interpolate="step-after").encode(
        x=alt.X("time_h:Q", title="Tempo (h)"), y=alt.Y("perdida:Q", title="Quantidade", axis=alt.Axis(tickMinStep=1, format='d')),
        tooltip=[alt.Tooltip("time_h:Q", title="Tempo (h)", format=".1f"),
                 alt.Tooltip("perdida:Q", title="Oportunidades perdidas")])
    return (area + lines + rule(t)).properties(
        height=H_STD, title=ttl("Capacidade ociosa vs. incidentes sem navio", "Área a vermelho = oportunidade perdida"))


def legend_backlog_capacity() -> None:
    legend_row([
        grp("Composição:"), sw(BACKLOG_LABELS["abertos"], BACKLOG_COLORS[0]),
        sw(BACKLOG_LABELS["em_curso"], BACKLOG_COLORS[1]), sw(BACKLOG_LABELS["resolvidos"], BACKLOG_COLORS[2]),
        grp("&nbsp;Capacidade:"), sw(CAP_LABELS["idle"], CAP_COLORS[0], "line"),
        sw(CAP_LABELS["sem_alvo"], CAP_COLORS[1], "line"), sw("Oportunidade perdida", "rgba(255,43,43,.35)"),
    ])


def ch_incident_gantt(P, t):
    if P.inc_blocks.empty:
        return None
    order = list(P.inc_sum["Rotulo"])
    base = alt.Chart(P.inc_blocks).mark_bar(cornerRadius=2, size=14).encode(
        x=alt.X("Início:Q", title="Tempo (h)"), x2="Fim:Q",
        y=alt.Y("Incidente:N", sort=order, title=None, axis=alt.Axis(grid=False, labelLimit=0)),
        color=alt.Color("Estado:N", scale=alt.Scale(domain=I_STATES, range=I_COLORS), legend=legend_top()),
        tooltip=["Incidente", "Estado", alt.Tooltip("Início:Q", format=".1f"), alt.Tooltip("Fim:Q", format=".1f"),
                 alt.Tooltip("Duracao_num:Q", title="Duração (h)", format=".1f")])
    # sem teto de altura: antes min(900, …) fazia as barras (14 px) sobrepor-se a partir de ~35 incidentes
    return (base + rule(t)).properties(
        height=max(140, len(order) * 22 + 40),
        title=ttl("Ciclo de vida de cada incidente", "⚠ = ficou por resolver no fim do episódio"))


def ch_wait(P):
    if P.wait_long.empty:
        return None
    order = list(P.wait_long.groupby("Rotulo")["Horas"].sum().sort_values(ascending=False).index)[:10]
    d = P.wait_long[P.wait_long["Rotulo"].isin(order)]
    return alt.Chart(d).mark_bar().encode(
        x=alt.X("Horas:Q", title="Horas desde que surgiu", stack=True),
        y=alt.Y("Rotulo:N", sort=order, title=None, axis=alt.Axis(labelLimit=0)),
        color=alt.Color("Fase:N", legend=legend_top(), scale=alt.Scale(
            domain=["Espera pela 1.ª resposta", "Tratamento / pendente"], range=["#ff2b2b", "#0078ff"])),
        tooltip=["Rotulo", "Fase", alt.Tooltip("Horas:Q", format=".1f")],
    ).properties(height=H_PAIR, title=ttl("Top 10 incidentes mais demorados", "Horas desde que surgiu, por fase"))


def ch_triage(P):
    d = P.inc_sum
    if d.empty or d["Severidade"].notna().sum() == 0:
        return None
    return alt.Chart(d).mark_circle(size=110, opacity=0.85, stroke="#ffffff", strokeWidth=0.6).encode(
        x=alt.X("Severidade:Q", title="Severidade", scale=alt.Scale(zero=False)),
        y=alt.Y("Idade (h):Q", title="Idade até resolver / fim (h)"),
        color=alt.Color("Resultado:N", legend=legend_top(),
                        scale=alt.Scale(domain=["Resolvido", "Por resolver"], range=["#00dcdc", "#ff2b2b"])),
        tooltip=["Incidente", alt.Tooltip("Severidade:Q", format=".2f"), alt.Tooltip("Espera (h):Q", format=".1f"),
                 alt.Tooltip("Idade (h):Q", format=".1f"), "Resultado"],
    ).properties(height=H_PAIR, title=ttl("Triagem: gravidade vs. idade", "Pontos altos à direita = graves a esperar"))


def ch_reward_cum(P, t):
    base = alt.Chart(P.steps).mark_line(color="#0078ff").encode(
        x=alt.X("time_h:Q", title="Tempo (h)"), y=alt.Y("cumulative_reward:Q", title="Recompensa acumulada"))
    return (base + rule(t)).properties(height=240, title=ttl("Recompensa acumulada", "Soma da recompensa desde o passo 0"))


def ch_reward_step(P, t):
    base = alt.Chart(P.steps).mark_bar(color="#ff4b4b", size=2.5).encode(
        x=alt.X("time_h:Q", title="Tempo (h)"), y=alt.Y("reward:Q", title="Recompensa no passo"),
        tooltip=[alt.Tooltip("time_h:Q", title="Tempo (h)", format=".1f"), alt.Tooltip("reward:Q", format=".4f")])
    return (base + rule(t)).properties(height=240, title=ttl("Sangramento de pontos por passo", "Recompensa obtida em cada passo"))


def ch_donut(vals: dict[str, float], height=240, title="Peso absoluto das falhas"):
    d = pd.DataFrame({"Componente": list(vals), "Valor": list(vals.values())})
    d = d[d["Valor"] > 0]
    return alt.Chart(d).mark_arc(innerRadius=50).encode(
        theta="Valor:Q", color=alt.Color("Componente:N", scale=alt.Scale(scheme="set2"), legend=legend_bottom()),
        tooltip=["Componente", alt.Tooltip("Valor:Q", format=".2f")]).properties(height=height, title=title)

def _dec_height(dec, vessel_ids) -> int:
    n_rows = len(vessel_ids) if vessel_ids else len(dec["agent_id"].unique())
    return max(240, n_rows * 48 + 70)


def ch_fleet_gantt(P, t, gantt=None, ids=None, title="Cronograma da frota"):
    g = P.gantt if gantt is None else gantt
    if g.empty:
        return None
    ids = fleet_order(ids) if ids is not None else fleet_order(P.vessel_ids)
    base = alt.Chart(g).mark_bar(cornerRadius=4, size=24).encode(
        x=alt.X("Início:Q", title="Tempo (h)"), x2="Fim:Q",
        y=alt.Y(
            "Navio:N",
            sort=ids,
            title=None,
            axis=alt.Axis(grid=False, tickSize=0, domain=False, labelLimit=0, labelOverlap=False, labelPadding=8),
        ),
        color=alt.Color("Estado:N", scale=alt.Scale(domain=V_STATES, range=V_COLORS), legend=legend_top()),
        tooltip=["Navio", "Estado", alt.Tooltip("Início:Q", format=".1f"), alt.Tooltip("Fim:Q", format=".1f"), "Duração:N"],
    )
    return (base + rule(t)).properties(
        height=max(160, len(ids) * 52 + 50),
        padding={"top": 6, "bottom": 6, "left": 8, "right": 8},
        title=ttl(title, "Estado de cada navio ao longo do episódio"),
    )


def ch_util(P):
    if P.util.empty:
        return None
    order = fleet_order(P.vessel_ids)
    return alt.Chart(P.util).mark_bar(size=24).encode(
        x=alt.X("Duracao_num:Q", stack="normalize", title="% do episódio", axis=alt.Axis(format="%")),
        y=alt.Y("Navio:N", sort=order, title=None, axis=alt.Axis(labelLimit=0, labelOverlap=False, labelPadding=8)),
        color=alt.Color("Estado:N", scale=alt.Scale(domain=V_STATES, range=V_COLORS), legend=legend_top()),
        tooltip=["Navio", "Estado", alt.Tooltip("Duracao_num:Q", title="Horas", format=".1f")],
    ).properties(
        height=max(160, len(order) * 48 + 50),
        padding={"top": 6, "bottom": 6, "left": 8, "right": 8},
        title=ttl("Ocupação por navio", "Onde há cinzento há desperdício (navio inativo)"),
    )


def ch_dec_scatter(dec, t, t_end, vessel_ids=None):
    y_sort = fleet_order(vessel_ids) if vessel_ids else alt.SortField("agent_id", order="ascending")
    base = alt.Chart(dec).mark_circle(size=85, opacity=0.9).encode(
        x=alt.X("time_h:Q", title="Tempo (h)", scale=alt.Scale(domain=[0, t_end], nice=False)),
        y=alt.Y("agent_id:N", sort=y_sort, title=None, axis=alt.Axis(labelLimit=0, labelOverlap=False, labelPadding=8)),
        color=alt.Color("outcome:N", title="Resultado", scale=alt.Scale(domain=OUTCOMES, range=OUTCOME_COLORS),
                        legend=legend_top()),
        tooltip=["Passo", "Tempo", "agent_id", "action_type", "target_incident_id", "outcome"],
    )
    return (base + rule(t, t_end)).properties(
        height=_dec_height(dec, vessel_ids),
        padding={"top": 6, "bottom": 6, "left": 8, "right": 8},
        title=ttl("Mapa de decisões", "Cada ponto é uma ação de despacho de um navio"),
    )


def ch_dec_bars(dec, vessel_ids=None):
    d = dec.groupby(["agent_id", "outcome"]).size().reset_index(name="n")
    y_sort = fleet_order(vessel_ids) if vessel_ids else alt.SortField("agent_id", order="ascending")
    return alt.Chart(d).mark_bar(size=22).encode(
        x=alt.X("n:Q", title="Nº de ações", axis=alt.Axis(tickMinStep=1, format="d")),
        y=alt.Y("agent_id:N", sort=y_sort, title=None, axis=alt.Axis(labelLimit=0, labelOverlap=False, labelPadding=8)),
        color=alt.Color("outcome:N", title="Resultado", scale=alt.Scale(domain=OUTCOMES, range=OUTCOME_COLORS),
                        legend=legend_top()),
        tooltip=["agent_id", "outcome", "n"],
    ).properties(
        height=_dec_height(dec, vessel_ids),
        padding={"top": 6, "bottom": 6, "left": 8, "right": 8},
        title=ttl("Resultado das decisões por navio", "Total acumulado até ao instante atual"),
    )


def style_outcome(df: pd.DataFrame):
    def f(v):
        s = str(v) if v is not None else ""
        if s == "rejected":
            return "color:#ff4b4b;font-weight:bold"
        if s.startswith("duplicate"):
            return "color:#ff8c00;font-weight:bold"
        if s == "accepted":
            return "color:#00dcdc"
        return ""
    return df.style.map(f, subset=["outcome"])


def decision_panel(dec: pd.DataFrame, key: str, height: int = 250):
    """Taxa de erro + feed das últimas ações."""
    if dec.empty:
        st.info("O agente ainda não executou ações de despacho.")
        return
    err = int(dec["outcome"].astype(str).isin(ERROR_OUTCOMES).sum())
    a, b = st.columns(2)
    a.metric("Taxa de erro", f"{100 * err / len(dec):.1f}%", f"{err} falhas", delta_color="inverse")
    b.metric("Total de ações", len(dec))
    show = dec.iloc[::-1].head(15)[["Passo", "Tempo", "agent_id", "action_type", "target_incident_id", "outcome"]]
    st.dataframe(style_outcome(show), hide_index=True, width="stretch", height=height, key=key)


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
/* métricas: rótulos/deltas passam a quebrar linha em vez de "…"; rótulo reserva 2 linhas para os valores alinharem */
[data-testid="stMetricLabel"]{min-height:2.5em;display:flex;align-items:flex-end}
[data-testid="stMetricLabel"] *,[data-testid="stMetricDelta"] *{white-space:normal!important;overflow:visible!important;text-overflow:unset!important;line-height:1.25!important}
[data-testid="stMetricValue"]{font-size:1.7rem!important}
[data-testid="stMetricValue"] *{white-space:normal!important;overflow:visible!important;text-overflow:unset!important}
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
st.sidebar.caption("** DETALHES DO EPISÓDIO **")
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
    ss.playing = False


# ---------------------------------- VISTAS ----------------------------------
def view_map(i: int):
    now = P.steps.iloc[i]
    prev = P.steps.iloc[max(i - 1, 0)]
    st.caption("Instante atual")
    k = st.columns(5)
    k[0].metric("Abertos", int(now["abertos"]), int(now["abertos"] - prev["abertos"]), delta_color="inverse")
    k[1].metric("Em curso", int(now["em_curso"]), int(now["em_curso"] - prev["em_curso"]), delta_color="off")
    k[2].metric("Resolvidos", int(now["resolvidos"]), int(now["resolvidos"] - prev["resolvidos"]))
    k[3].metric("Navios inativos", int(now["idle"]), f"{int(now['perdida'])} c/ incidente sem alvo", delta_color="off")
    k[4].metric("Recompensa acumulada", f"{now['cumulative_reward']:.2f}", f"{now['reward']:+.4f}")
    o1, o2, _ = st.columns([1.5, 1.5, 7], vertical_alignment="center")
    lines = o1.checkbox("Trajetórias", True, key="chk_lin_g")
    trails = o2.checkbox("Rastos", False, key="chk_rst_g")
    df_nav, df_inc = tables_at(P, i)
    left, right = st.columns([2, 1])
    with left:
        st.pydeck_chart(build_deck(P, i, lines=lines, trails=trails, height=MAP_H), key="deck_vg")
        legend()
    with right:
        st.markdown("**Estado · instante atual**")
        t1, t2, t3 = st.tabs([f"Navios ({len(df_nav)})", f"Incidentes ({len(df_inc)})", f"Infra ({len(P.df_infra)})"])
        t1.dataframe(df_nav, hide_index=True, width="stretch", height=TABLE_H, column_config=NAV_CFG)
        t2.dataframe(df_inc, hide_index=True, width="stretch", height=TABLE_H, column_config=INC_CFG)
        t3.dataframe(P.df_infra, hide_index=True, width="stretch", height=TABLE_H)


def view_diag(i: int):
    t = float(P.steps["time_h"].iloc[i])
    sec = st.segmented_control("Secção", DIAG_SECTIONS, default=DIAG_SECTIONS[0], key="diag_sec",
                               label_visibility="collapsed") or DIAG_SECTIONS[0]

    if sec == DIAG_SECTIONS[0]:
        pct = 100 * float((P.steps["perdida"] > 0).mean())
        a, b, c = st.columns(3)
        a.metric("% dos passos com oportunidade perdida", f"{pct:.1f}%")
        b.metric("Pico de incidentes sem alvo", int(P.steps["sem_alvo"].max()))
        c.metric("Taxa de hold", f"{(lambda h: h * 100 if h <= 1 else h)(float(metric(metrics, 'hold_action_rate'))):.1f}%")
        legend_backlog_capacity()
        l, r = st.columns(2)
        l.altair_chart(ch_backlog(P, t), width="stretch")
        r.altair_chart(ch_capacity(P, t), width="stretch")

    elif sec == DIAG_SECTIONS[1]:
        g = ch_incident_gantt(P, t)
        if g is not None:
            if len(P.inc_sum) > MAX_INC_ROWS_INLINE:
                st.caption(f"{len(P.inc_sum)} incidentes ")
                with st.container(height=640, border=True):
                    st.altair_chart(g, width="stretch")
            else:
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
                hide_index=True, width="stretch", column_config=PEND_CFG)
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
            l, r = st.columns(2)
            l.altair_chart(ch_dec_scatter(dec, t, P.t_end, vessel_ids=P.vessel_ids), width="stretch")
            r.altair_chart(ch_dec_bars(dec, vessel_ids=P.vessel_ids), width="stretch")
        l, r = st.columns(2)
        with l, st.container(border=True, height=PANEL_H):
            st.markdown(" **Eficácia de despacho e feed de ações**")
            st.caption("Até ao instante atual")
            decision_panel(dec, "dec_diag", height=PANEL_H - 190)
        with r, st.container(border=True, height=PANEL_H):
            st.markdown(" **Log do ambiente**")
            st.caption("Últimos eventos até ao instante atual")
            if ev.empty:
                st.info("Sem eventos registados até ao momento.")
            else:
                st.dataframe(ev.iloc[::-1].head(15).drop(columns=["i", "time_h"], errors="ignore"),
                             hide_index=True, width="stretch", height=PANEL_H - 130)


def view_focus(i: int):
    f1, f2, f3, f4 = st.columns([2.5, 2.2, 2.2, 2.2], vertical_alignment="center")
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

    k = st.columns(5)
    k[0].metric("Distância (agora)", f"{km_now:.0f} km", f"Total: {km_tot:.0f} km", delta_color="off")
    k[1].metric("Taxa de ocupação", f"{100 * act_h / total_h:.1f}%", f"{act_h:.1f}h ativas", delta_color="off")
    k[2].metric("Em trânsito", f"{tr_h:.1f} h", f"{100 * tr_h / total_h:.1f}% do ep.", delta_color="off")
    k[3].metric("No local", f"{work_h:.1f} h", f"{int(P.visited.get(vid, 0))} inc. visitados", delta_color="off")
    k[4].metric("Horas inativas", f"{idle_h:.1f} h")

    l, r = st.columns([2, 1])
    with l:
        st.caption(f"Mapa · **instante atual** — {vid} ({tipo}): {estado}")
        st.pydeck_chart(build_deck(P, i, focus=vid, isolate=iso, lines=lines, trails=trails, height=FOCUS_MAP_H),
                        key="deck_mf")
    with r:
        st.markdown(f"**Telemetria de {vid}**")
        if not gn.empty:
            d = gn.groupby("Estado", as_index=False)["Duracao_num"].sum()
            st.altair_chart(alt.Chart(d).mark_arc(innerRadius=55).encode(
                theta="Duracao_num:Q",
                color=alt.Color("Estado:N", scale=alt.Scale(domain=V_STATES, range=V_COLORS), legend=legend_bottom()),
                tooltip=["Estado", alt.Tooltip("Duracao_num:Q", format=".1f")]).properties(height=FOCUS_MAP_H - 120),
                width="stretch")
    g = ch_fleet_gantt(P, float(P.steps["time_h"].iloc[i]), gantt=gn, ids=[vid], title=f"Cronograma de {vid}")
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
    lb, lt = f"Base: {eb['policy']['label']}", f"Alvo: {et['policy']['label']}"
    if lb[6:] == lt[6:]:
        lb, lt = lb + " (base)", lt + " (alvo)"
    dom, rng = [lb, lt], ["#9aa0a6", "#ff8c00"]

    o0, o1, o2, _ = st.columns([2.6, 1.4, 1.4, 5], vertical_alignment="center")
    mostrar_mapas = o0.toggle("Mapas lado a lado (sincronizados)", value=False, key="cmp_maps")
    if mostrar_mapas:
        lines = o1.checkbox("Trajetórias", True, key="chk_lin_c")
        trails = o2.checkbox("Rastos", False, key="chk_rst_c")
        m1, m2 = st.columns(2)
        m1.markdown(f"**{lb}**")
        m1.pydeck_chart(build_deck(Pb, i, lines=lines, trails=trails, height=CMP_MAP_H), key="deck_cb")
        m2.markdown(f"**{lt}**")
        m2.pydeck_chart(build_deck(Pt, i, lines=lines, trails=trails, height=CMP_MAP_H), key="deck_ct")
        legend()
    st.divider()

    g = lambda m, n: float(metric(m, n)) 
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
    st.divider()

    both = pd.concat([Pb.steps.assign(Política=lb), Pt.steps.assign(Política=lt)])
    t = float(Pb.steps["time_h"].iloc[min(i, Pb.T - 1)])
    t_max = float(max(Pb.t_end, Pt.t_end))

    def overlay(col, title, sub, interp="linear"):
        ln = alt.Chart(both).mark_line(interpolate=interp, strokeWidth=2).encode(
            x=alt.X("time_h:Q", title="Tempo (h)"),
            y=alt.Y(f"{col}:Q", title=None, axis=alt.Axis(tickMinStep=1, format="d") if col != "cumulative_reward" else alt.Axis()),
            color=alt.Color("Política:N", scale=alt.Scale(domain=dom, range=rng), legend=None),
            tooltip=["Política", alt.Tooltip("time_h:Q", title="Tempo (h)", format=".1f"), alt.Tooltip(f"{col}:Q", format=".2f")])
        return (ln + rule(t)).properties(height=H_STD, title=ttl(title, sub))

    legend_row([sw(lb, rng[0], "line"), sw(lt, rng[1], "line")])
    c1, c2, c3 = st.columns(3)
    c1.altair_chart(overlay("cumulative_reward", "Recompensa acumulada", "Maior = melhor"), width="stretch")
    c2.altair_chart(overlay("abertos", "Incidentes abertos", "Backlog por instante", "step-after"), width="stretch")
    c3.altair_chart(overlay("perdida", "Oportunidades perdidas", "Navio ocioso + incidente sem alvo", "step-after"),
                    width="stretch")
    st.divider()

    st.markdown("Espera por incidente vs. resultados das decisões")
    s1 = Pb.inc_sum[["Incidente", "Espera (h)", "Idade (h)"]].assign(Política=lb)
    s2 = Pt.inc_sum[["Incidente", "Espera (h)", "Idade (h)"]].assign(Política=lt)
    inc_cmp = pd.concat([s1, s2])
    dd = pd.concat([Pb.dec.assign(Política=lb), Pt.dec.assign(Política=lt)])
    legend_row([grp("Política:"), sw(lb, rng[0]), sw(lt, rng[1]), grp("&nbsp;Resultado da decisão:")]
               + [sw(o, c) for o, c in zip(OUTCOMES, OUTCOME_COLORS)])
    l, r = st.columns(2)
    if not inc_cmp.empty:
        order = list(inc_cmp.groupby("Incidente")["Espera (h)"].max().sort_values(ascending=False).index[:10])
        l.altair_chart(alt.Chart(inc_cmp[inc_cmp["Incidente"].isin(order)]).mark_bar().encode(
            y=alt.Y("Incidente:N", sort=order, title=None, axis=alt.Axis(labelLimit=0)),
            yOffset=alt.YOffset("Política:N", sort=dom),
            x=alt.X("Espera (h):Q", title="Espera pela 1.ª resposta (h)"),
            color=alt.Color("Política:N", scale=alt.Scale(domain=dom, range=rng), legend=None),
            tooltip=["Incidente", "Política", alt.Tooltip("Espera (h):Q", format=".1f"),
                     alt.Tooltip("Idade (h):Q", format=".1f")],
        ).properties(height=H_PAIR, title=ttl("Top 10 incidentes com maior espera", "Pior caso entre as duas políticas (mesma seed)")),
            width="stretch")
    if not dd.empty:
        cnt = dd.groupby(["Política", "outcome"]).size().reset_index(name="n")
        r.altair_chart(alt.Chart(cnt).mark_bar(size=60).encode(
            x=alt.X("n:Q", title="Nº de ações", axis=alt.Axis(tickMinStep=1, format='d')),
            y=alt.Y("Política:N", title=None, sort=dom, axis=alt.Axis(labelLimit=0)),
            color=alt.Color("outcome:N", title="Resultado", scale=alt.Scale(domain=OUTCOMES, range=OUTCOME_COLORS),
                            legend=None), tooltip=["Política", "outcome", "n"],
        ).properties(height=H_PAIR, title=ttl("Resultado das decisões", "Total de ações de despacho por política")),
            width="stretch")


# ---------------------------------- WORKSPACE (fragmento animado) ----------------------------------
@st.fragment
def workspace():
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
    st.caption("Resultado final do episódio")
    k = st.columns(6)
    k[0].metric("Resolvidos", f"{int(metric(metrics, 'n_resolved'))}/{int(metric(metrics, 'n_incidents'))}")
    k[1].metric("Taxa de resolução", f"{100 * float(metric(metrics, 'resolution_rate')):.1f}%")
    k[2].metric("Primeira resposta", f"{float(metric(metrics, 'mean_first_response_time_h')):.1f} h")
    k[3].metric("Tempo de resolução", f"{float(metric(metrics, 'mean_time_to_resolution_h')):.1f} h")
    k[4].metric("Despachos duplicados", int(metric(metrics, "duplicate_dispatch_count")))
    k[5].metric("Severidade por resolver", f"{float(metric(metrics, 'unresolved_severity')):.2f}")

    c1, c2, c3, c4, c5 = st.columns([1.4, 1.1, 1.0, 2.3, 5.0], vertical_alignment="center")
    c1.button("⏸ Pausa" if ss.playing else "▶ Reproduzir", on_click=click_play, width="stretch", key="btn_play")
    c2.button("⏮ Reset", on_click=click_reset, width="stretch", key="btn_reset")
    c3.selectbox("Velocidade", [1, 2, 5, 10], key="speed", format_func=lambda v: f"{v}×", label_visibility="collapsed")
    c4.markdown(f"<div style='font-weight:500;white-space:nowrap'>⏱️ Passo {frame.get('step', i)}/{max_frame} "
                f"({frame['time_h']:.1f}h)</div>", unsafe_allow_html=True)
    c5.slider("Navegação temporal", 0, max_frame, key="ui_step", on_change=on_slider, label_visibility="collapsed")

    view = st.segmented_control("Vista", VIEWS, default=VIEWS[0], key="view", label_visibility="collapsed") or VIEWS[0]
    st.divider()
    {VIEWS[0]: view_map, VIEWS[1]: view_diag, VIEWS[2]: view_focus, VIEWS[3]: view_compare}[view](i)

    if ss.playing:
        time.sleep(0.03)
        try:
            st.rerun(scope="fragment")
        except StreamlitAPIException:
            st.rerun()


workspace()