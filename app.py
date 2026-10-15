"""ARC Episode Viewer - Dashboard Operacional e Auditoria de RL para Cabos Submarinos."""

from __future__ import annotations

import math
import time
from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from arc_student import BundleValidationError, load_bundle, load_bundle_bytes

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_bundles"


# --- FUNÇÕES AUXILIARES E MATEMÁTICAS ---
def bundle_label(bundle: dict) -> str:
    episode = bundle["episode"]
    scenario = episode["scenario"]
    policy = episode["policy"]
    return f"{scenario['id']} | {policy['label']} | seed {scenario['world_seed']}"


def metric(metrics: dict, name: str, default=0):
    value = metrics.get(name, default)
    return default if value is None else value


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula a distância em km entre duas coordenadas geográficas."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_dashed_segments(coords: list[list[float]], color: list[int], dash_deg: float = 0.55, gap_deg: float = 0.40) -> list[dict]:
    """Converte uma lista de coordenadas históricas num rasto tracejado (dashes) uniforme."""
    segments = []
    clean_coords = [coords[0]]
    for pt in coords[1:]:
        if abs(pt[0] - clean_coords[-1][0]) > 1e-5 or abs(pt[1] - clean_coords[-1][1]) > 1e-5:
            clean_coords.append(pt)

    if len(clean_coords) < 2:
        return segments

    drawing = True
    remaining = dash_deg

    for i in range(len(clean_coords) - 1):
        x1, y1 = clean_coords[i]
        x2, y2 = clean_coords[i + 1]
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            continue

        ux, uy = dx / seg_len, dy / seg_len
        dist_covered = 0.0
        cx, cy = x1, y1

        while dist_covered < seg_len:
            step = min(remaining, seg_len - dist_covered)
            nx, ny = cx + ux * step, cy + uy * step
            if drawing:
                segments.append({"start": [cx, cy], "end": [nx, ny], "color": color})
            cx, cy = nx, ny
            dist_covered += step
            remaining -= step
            if remaining <= 1e-6:
                drawing = not drawing
                remaining = dash_deg if drawing else gap_deg

    return segments


@st.cache_data(show_spinner=False)
def precompute_episode_series(episode_id: str, timeline: list[dict], vessel_ids: list[str]):
    """Pré-calcula séries temporais e blocos de Gantt em cache para acelerar a reprodução."""
    chart_rows = [
        {
            "time_h": item["time_h"],
            "open": sum(inc["status"] != "resolved" for inc in item["incidents"]),
            "resolved": sum(inc["status"] == "resolved" for inc in item["incidents"]),
            "cumulative_reward": item["cumulative_reward"],
        }
        for item in timeline
    ]
    df_chart = pd.DataFrame(chart_rows)

    step_rewards = [{"Tempo (h)": t["time_h"], "Recompensa no Passo": t["reward"]} for t in timeline]
    df_step = pd.DataFrame(step_rewards)

    gantt_blocks = []
    for v_id in vessel_ids:
        current_status = None
        start_time = 0.0
        for t in timeline:
            v = next((x for x in t["vessels"] if x["id"] == v_id), None)
            if v:
                status = v["status"]
                if status != current_status:
                    if current_status is not None:
                        duracao = t["time_h"] - start_time
                        gantt_blocks.append({
                            "Navio": v_id,
                            "Estado": current_status,
                            "Início": start_time,
                            "Fim": t["time_h"],
                            "Duração": f"{duracao:.1f}h",
                            "Duracao_num": duracao,
                        })
                    current_status = status
                    start_time = t["time_h"]

        if current_status is not None and timeline:
            duracao = timeline[-1]["time_h"] - start_time
            gantt_blocks.append({
                "Navio": v_id,
                "Estado": current_status,
                "Início": start_time,
                "Fim": timeline[-1]["time_h"],
                "Duração": f"{duracao:.1f}h",
                "Duracao_num": duracao,
            })

    df_gantt = pd.DataFrame(gantt_blocks)
    return df_chart, df_step, df_gantt


def criar_grafico_com_regua(df: pd.DataFrame, y_cols: list[str], title: str, color_range: list[str], current_time_h: float):
    base = (
        alt.Chart(df)
        .transform_fold(y_cols, as_=["Variável", "Valor"])
        .mark_line()
        .encode(
            x=alt.X("time_h:Q", title="Tempo (h)"),
            y=alt.Y("Valor:Q", title=""),
            color=alt.Color("Variável:N", scale=alt.Scale(range=color_range)),
        )
        .properties(title=title, height=240)
    )
    regua = alt.Chart(pd.DataFrame({"x": [current_time_h]})).mark_rule(color="red", strokeWidth=2).encode(x="x:Q")
    return base + regua


# --- MOTOR DE RENDERIZAÇÃO DO MAPA (REUTILIZÁVEL NAS 3 TABS) ---
def build_map_and_tables(
    episode: dict,
    timeline: list[dict],
    step_idx: int,
    navio_focado: str | None = None,
    isolar_entidades: bool = False,
    mostrar_linhas: bool = True,
    mostrar_rastos: bool = False,
    map_height: int = 500,
):
    """Constrói o objeto PyDeck e os DataFrames de estado para qualquer episódio e modo."""
    safe_idx = min(step_idx, len(timeline) - 1)
    frame = timeline[safe_idx]

    dados_navio_focado = None
    incidente_alvo_do_foco = None
    if navio_focado:
        dados_navio_focado = next((v for v in frame["vessels"] if v["id"] == navio_focado), None)
        if dados_navio_focado:
            incidente_alvo_do_foco = dados_navio_focado.get("assigned_incident_id")

    location_groups = {}

    def add_to_group(lat, lon, item_data):
        key = (round(lat, 4), round(lon, 4))
        if key not in location_groups:
            location_groups[key] = []
        location_groups[key].append(item_data)

    # 1. Navios
    for vessel in frame["vessels"]:
        v_type = vessel.get("type", "unknown")
        raw_status = vessel["status"]
        assigned_id = vessel.get("assigned_incident_id")

        if raw_status == "transit" and assigned_id is not None:
            status_detail = f"Em trânsito para o Incidente {assigned_id}"
        elif raw_status == "on_site" and assigned_id is not None:
            status_detail = f"No local do Incidente {assigned_id}"
        elif raw_status == "repairing" and assigned_id is not None:
            status_detail = f"A reparar o Incidente {assigned_id}"
        elif raw_status == "returning":
            status_detail = "A regressar ao Porto"
        elif raw_status == "idle":
            status_detail = "Inativo (A aguardar ordens)"
        else:
            status_detail = raw_status.capitalize() + (f" (Incidente {assigned_id})" if assigned_id else "")

        color = (
            [0, 120, 255, 255]
            if v_type == "patrol"
            else [255, 140, 0, 255]
            if v_type == "repair"
            else [150, 150, 150, 255]
        )

        ignorar = False
        if navio_focado and vessel["id"] != navio_focado:
            if isolar_entidades:
                ignorar = True
            else:
                color = [120, 120, 120, 60]

        if not ignorar:
            add_to_group(
                vessel["lat"],
                vessel["lon"],
                {
                    "id": vessel["id"],
                    "kind": f"Navio ({v_type.capitalize()})",
                    "status": status_detail,
                    "color": color,
                    "priority": 2,
                },
            )

    # 2. Incidentes
    for incident in frame["incidents"]:
        raw_status = incident["status"]
        if raw_status == "open":
            color, status_detail = [255, 40, 40, 255], "Aberto (A aguardar)"
        elif raw_status in ["on_site", "repairing"]:
            color, status_detail = [0, 220, 220, 255], "Navio no local" if raw_status == "on_site" else "Em reparação"
        elif raw_status == "resolved":
            color, status_detail = [100, 100, 100, 120], "Resolvido"
        else:
            color, status_detail = [100, 100, 100, 120], raw_status.capitalize()

        ignorar = False
        if navio_focado and incident["id"] != incidente_alvo_do_foco:
            if isolar_entidades:
                ignorar = True
            else:
                color = [120, 120, 120, 60]

        if not ignorar:
            add_to_group(
                incident["lat"],
                incident["lon"],
                {
                    "id": f"incident_{incident['id']}",
                    "kind": "Incidente",
                    "status": status_detail,
                    "color": color,
                    "priority": 1,
                },
            )

    # 3. Portos e Landing Points
    for port in episode["map"].get("ports", []):
        ignorar = False
        color = [255, 255, 255, 255]
        if navio_focado:
            if isolar_entidades:
                ignorar = True
            else:
                color = [150, 150, 150, 40]
        if not ignorar:
            add_to_group(
                port["lat"],
                port["lon"],
                {"id": port["id"], "kind": "Porto", "status": "Static", "color": color, "priority": 3},
            )

    for point in episode["map"].get("landing_points", []):
        ignorar = False
        color = [155, 89, 182, 255]
        if navio_focado:
            if isolar_entidades:
                ignorar = True
            else:
                color = [120, 80, 150, 40]
        if not ignorar:
            add_to_group(
                point["lat"],
                point["lon"],
                {"id": point["id"], "kind": "Landing Point", "status": "Static", "color": color, "priority": 4},
            )

    map_rows, table_rows, line_rows, trail_dash_rows = [], [], [], []

    for (lat, lon), items in location_groups.items():
        items.sort(key=lambda x: x["priority"])
        tooltip_parts = [f"ID: {it['id']}\n{it['kind']}\nEstado: {it['status']}" for it in items]
        for it in items:
            table_rows.append({"lat": lat, "lon": lon, "id": it["id"], "kind": it["kind"], "status": it["status"]})
        map_rows.append({
            "lat": lat,
            "lon": lon,
            "count": str(len(items)) if len(items) > 1 else "",
            "color": items[0]["color"],
            "tooltip_text": "\n-------------------------\n".join(tooltip_parts),
        })

    # --- TRAJETÓRIA PROJETADA: LINHA CONTÍNUA COMPLETA (SEM TARGET RING) ---
    if mostrar_linhas:
        for vessel in frame["vessels"]:
            if navio_focado and vessel["id"] != navio_focado:
                continue
            assigned_id = vessel.get("assigned_incident_id")
            if vessel["status"] == "transit" and assigned_id is not None:
                alvo = next((inc for inc in frame["incidents"] if inc["id"] == assigned_id), None)
                if alvo:
                    v_type = vessel.get("type", "unknown")
                    line_color = [0, 195, 255, 200] if v_type == "patrol" else [255, 165, 0, 200]
                    line_rows.append({
                        "start": [vessel["lon"], vessel["lat"]],
                        "end": [alvo["lon"], alvo["lat"]],
                        "color": line_color,
                    })

    # --- RASTO HISTÓRICO (TRAIL): COM DASHES (TRACEJADO) ---
    if mostrar_rastos and safe_idx > 0:
        trails_data = {}
        vessel_types = {v["id"]: v.get("type", "unknown") for v in frame["vessels"]}
        for t in timeline[: safe_idx + 1]:
            for v in t["vessels"]:
                if navio_focado and v["id"] != navio_focado:
                    continue
                vid = v["id"]
                if vid not in trails_data:
                    trails_data[vid] = []
                trails_data[vid].append([v["lon"], v["lat"]])

        for vid, coords in trails_data.items():
            if len(coords) > 1:
                v_type = vessel_types.get(vid, "unknown")
                dash_color = [100, 195, 255, 160] if v_type == "patrol" else [255, 180, 80, 160]
                trail_dash_rows.extend(build_dashed_segments(coords, dash_color))

    df_map = pd.DataFrame(map_rows)
    df_table = pd.DataFrame(table_rows)
    df_lines = pd.DataFrame(line_rows)
    df_trails = pd.DataFrame(trail_dash_rows)

    layers_list = []
    if not df_trails.empty:
        layers_list.append(
            pdk.Layer(
                "LineLayer",
                data=df_trails,
                get_source_position="start",
                get_target_position="end",
                get_color="color",
                get_width=2.5,
                pickable=False,
            )
        )
    if not df_lines.empty:
        layers_list.append(
            pdk.Layer(
                "LineLayer",
                data=df_lines,
                get_source_position="start",
                get_target_position="end",
                get_color="color",
                get_width=3,
                pickable=False,
            )
        )
    if not df_map.empty:
        layers_list.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=60000,
                radius_min_pixels=8,
                radius_max_pixels=18,
                pickable=True,
            )
        )
        layers_list.append(
            pdk.Layer(
                "TextLayer",
                data=df_map,
                get_position="[lon, lat]",
                get_text="count",
                get_size=13,
                get_color=[255, 255, 255, 255],
                get_text_anchor='"middle"',
                get_alignment_baseline='"center"',
                pickable=False,
            )
        )

    tooltip = {
        "text": "{tooltip_text}",
        "style": {
            "backgroundColor": "#222222",
            "color": "#ffffff",
            "border": "1px solid #444",
            "borderRadius": "4px",
            "fontSize": "0.85em",
            "padding": "12px",
            "whiteSpace": "pre-wrap",
            "wordBreak": "normal",
            "width": "max-content",
            "maxWidth": "none",
            "height": "auto",
            "maxHeight": "none",
            "overflow": "visible",
        },
    }

    center_lat, center_lon, map_zoom = 20, -40, 2
    if navio_focado and dados_navio_focado:
        center_lat, center_lon = dados_navio_focado["lat"], dados_navio_focado["lon"]
        map_zoom = 3.4

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=map_zoom, pitch=0)
    deck = pdk.Deck(layers=layers_list, initial_view_state=view_state, tooltip=tooltip, map_style="dark", height=map_height)
    return deck, df_table, frame


def render_legenda_mapa():
    st.markdown(
        """
        <div style="font-size: 0.88em; color: #a0a0a0; padding-top: 8px; padding-bottom: 8px; text-align: center;">
            <span style="color: rgb(0, 120, 255);">●</span> Patrulha &nbsp;&nbsp;
            <span style="color: rgb(255, 140, 0);">●</span> Reparação &nbsp;&nbsp;
            <span style="color: rgb(255, 40, 40);">●</span> Alerta Aberto &nbsp;&nbsp;
            <span style="color: rgb(0, 220, 220);">●</span> No Local / A Reparar &nbsp;&nbsp;
            <span style="color: rgb(255, 255, 255);">●</span> Porto &nbsp;&nbsp;
            <span style="color: rgb(155, 89, 182);">●</span> Landing Point
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- CONFIGURAÇÃO DA PÁGINA E CSS (INCLUINDO TABS LARGAS COM BORDA E LETRA MAIOR) ---
st.set_page_config(page_title="ARC Episode Viewer", layout="wide")
st.title("ARC Episode Viewer")
st.caption("Plataforma Operacional e de Auditoria de Políticas de Reinforcement Learning (Cabos Submarinos).")

st.markdown(
    """
    <style>
    .mapboxgl-ctrl-bottom-left, .mapboxgl-ctrl-bottom-right { display: none !important; }
    div[data-baseweb="select"] { height: auto !important; }
    div[data-baseweb="select"] * { white-space: normal !important; word-break: break-word !important; text-overflow: unset !important; }
    div[data-baseweb="popover"] ul[role="listbox"] li, 
    div[data-baseweb="popover"] ul[role="listbox"] li * { white-space: normal !important; word-break: break-word !important; text-overflow: unset !important; padding-top: 8px; padding-bottom: 8px; }

    /* Estilo dos Separadores (Tabs) a ocupar a horizontal toda, com borda e letra maior */
    div[data-baseweb="tab-list"] {
        display: flex !important;
        width: 100% !important;
        gap: 12px !important;
        background-color: transparent !important;
        padding-bottom: 6px !important;
    }
    button[data-baseweb="tab"] {
        flex: 1 1 0% !important;
        width: 100% !important;
        justify-content: center !important;
        border: 1px solid #3d4452 !important;
        border-radius: 8px !important;
        padding: 14px 20px !important;
        background-color: #161920 !important;
        transition: all 0.2s ease-in-out !important;
    }
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] div,
    button[data-baseweb="tab"] span {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"]:hover {
        border-color: #6c7893 !important;
        background-color: #1e222b !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border: 2px solid #ff4b4b !important;
        background-color: #232834 !important;
        color: #ffffff !important;
    }
    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CARREGAMENTO DE DADOS ---
bundles: dict[str, dict] = {}
errors = []
for path in sorted(SAMPLE_DIR.glob("*.json")):
    try:
        b = load_bundle(path)
        bundles[b["episode"]["id"]] = b
    except BundleValidationError as error:
        errors.append(f"{path.name}: {error}")

uploaded_files = st.sidebar.file_uploader("Adicionar episode bundles", type=["json"], accept_multiple_files=True)
for uploaded in uploaded_files:
    try:
        b = load_bundle_bytes(uploaded.getvalue(), uploaded.name)
        bundles[b["episode"]["id"]] = b
    except BundleValidationError as error:
        errors.append(f"{uploaded.name}: {error}")

for error in errors:
    st.error(error)
if not bundles:
    st.warning("Não existem bundles válidos para apresentar.")
    st.stop()

episode_id = st.sidebar.selectbox(
    "Episódio Principal", options=list(bundles), format_func=lambda value: bundle_label(bundles[value])
)
bundle = bundles[episode_id]
episode = bundle["episode"]
scenario = episode["scenario"]
policy = episode["policy"]
metrics = episode["metrics"]
timeline = episode["timeline"]
max_frame = len(timeline) - 1

# --- ABA LATERAL: DETALHES DO EPISÓDIO ---
st.sidebar.divider()
st.sidebar.caption("**📋 DETALHES DO EPISÓDIO ATIVO**")
st.sidebar.caption(f"**Cenário:** {scenario['id']}")
st.sidebar.caption(f"**Política:** {policy['label']} ({policy['kind']})")
st.sidebar.caption(f"**Duração:** {timeline[-1]['time_h']:.1f}h | **Passos:** {len(timeline)}")
st.sidebar.caption(f"**Seed:** {scenario['world_seed']} | **Split:** {scenario['split']}")
st.sidebar.caption(f"**Banco:** {scenario['bank_id']}")

# =====================================================================
# CABEÇALHO E MÉTRICAS GLOBAIS NO TOPO (COMO NA TUA IMAGEM)
# =====================================================================
st.subheader(f"Cenário: {scenario['id']} — Política: {policy['label']}", divider="gray")

columns = st.columns(6)
columns[0].metric("Resolvidos", f"{int(metric(metrics, 'n_resolved'))}/{int(metric(metrics, 'n_incidents'))}")
columns[1].metric("Taxa de resolução", f"{100 * float(metric(metrics, 'resolution_rate')):.1f}%")
columns[2].metric("Primeira resposta", f"{float(metric(metrics, 'mean_first_response_time_h')):.1f} h")
columns[3].metric("Tempo de resolução", f"{float(metric(metrics, 'mean_time_to_resolution_h')):.1f} h")
columns[4].metric("Despachos duplicados", int(metric(metrics, "duplicate_dispatch_count")))
columns[5].metric("Severidade por resolver", f"{float(metric(metrics, 'unresolved_severity')):.2f}")
st.divider()

# --- CONTROLADOR TEMPORAL GLOBAL ---
if "step_slider" not in st.session_state:
    st.session_state.step_slider = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

if st.session_state.step_slider > max_frame:
    st.session_state.step_slider = max_frame

if st.session_state.playing:
    if st.session_state.step_slider < max_frame:
        st.session_state.step_slider += 1
    else:
        st.session_state.playing = False


def click_play():
    if st.session_state.step_slider >= max_frame:
        st.session_state.step_slider = 0
    st.session_state.playing = not st.session_state.playing


def click_reset():
    st.session_state.step_slider = 0
    st.session_state.playing = False


frame = timeline[st.session_state.step_slider]
vessel_ids = [v["id"] for v in frame["vessels"]]

ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.3, 1.1, 2.0, 5.6])
with ctrl_col1:
    st.button("⏸ Pausa" if st.session_state.playing else "▶ Reproduzir", on_click=click_play, width="stretch")
with ctrl_col2:
    st.button("⏮ Reset", on_click=click_reset, width="stretch")
with ctrl_col3:
    st.markdown(
        f"<div style='padding-top: 8px; font-weight: 600;'>⏱️ Passo {frame['step']}/{max_frame} ({frame['time_h']:.1f}h)</div>",
        unsafe_allow_html=True,
    )
with ctrl_col4:
    st.slider("Navegação Temporal", min_value=0, max_value=max_frame, key="step_slider", label_visibility="collapsed")

# Pré-cálculo em cache das séries temporais e Gantt do episódio principal
df_chart, df_step, df_gantt = precompute_episode_series(episode["id"], timeline, vessel_ids)

# =====================================================================
# SEPARADORES LARGOS COM BORDA ABAIXO DO CABEÇALHO DE MÉTRICAS
# =====================================================================
tab_geral, tab_foco, tab_comp = st.tabs([
    "🌐 Visão Geral",
    "🎯 Modo Foco (Análise de Navio)",
    "⚖️ Comparador de Episódios",
])

# =====================================================================
# TAB 1: VISÃO GERAL (GENERAL VIEW)
# =====================================================================
with tab_geral:
    c_opt1, c_opt2, _ = st.columns([2.4, 2.4, 5.2])
    with c_opt1:
        mostrar_linhas_glob = st.checkbox("Trajetórias projetadas (Linha)", value=True, key="chk_lin_glob")
    with c_opt2:
        mostrar_rastos_glob = st.checkbox("Rasto histórico (Tracejado)", value=False, key="chk_rst_glob")

    deck_geral, df_table_geral, _ = build_map_and_tables(
        episode=episode,
        timeline=timeline,
        step_idx=st.session_state.step_slider,
        navio_focado=None,
        isolar_entidades=False,
        mostrar_linhas=mostrar_linhas_glob,
        mostrar_rastos=mostrar_rastos_glob,
    )

    left, right = st.columns([2, 1])
    with left:
        st.pydeck_chart(deck_geral)
        render_legenda_mapa()

    with right:
        st.markdown("**Estado no instante atual**")
        df_navios = df_table_geral[df_table_geral["kind"].str.contains("Navio")].copy() if not df_table_geral.empty else pd.DataFrame()
        df_incidentes = df_table_geral[df_table_geral["kind"] == "Incidente"].copy() if not df_table_geral.empty else pd.DataFrame()
        df_infra = df_table_geral[df_table_geral["kind"].isin(["Porto", "Landing Point"])].copy() if not df_table_geral.empty else pd.DataFrame()

        for df in [df_navios, df_incidentes, df_infra]:
            if not df.empty:
                df["Coords"] = df.apply(lambda row: f"{row['lat']:.3f}, {row['lon']:.3f}", axis=1)

        df_navios = (
            df_navios[["id", "status", "kind", "Coords"]].rename(columns={"id": "Navio", "status": "Estado", "kind": "Tipo"})
            if not df_navios.empty
            else pd.DataFrame(columns=["Navio", "Estado", "Tipo", "Coords"])
        )
        if not df_navios.empty:
            df_navios["Tipo"] = df_navios["Tipo"].str.replace(r"Navio \(\vert{}\)", "", regex=True)

        df_incidentes = (
            df_incidentes[["id", "status", "Coords"]].rename(columns={"id": "Incidente", "status": "Estado"})
            if not df_incidentes.empty
            else pd.DataFrame(columns=["Incidente", "Estado", "Coords"])
        )
        df_infra = (
            df_infra[["id", "kind", "Coords"]].rename(columns={"id": "Nome", "kind": "Tipo"})
            if not df_infra.empty
            else pd.DataFrame(columns=["Nome", "Tipo", "Coords"])
        )

        t_nav, t_inc, t_inf = st.tabs([
            f"⛴️ Navios ({len(df_navios)})",
            f"🚨 Incidentes ({len(df_incidentes)})",
            f"🏢 Infra ({len(df_infra)})",
        ])
        with t_nav:
            st.dataframe(df_navios, hide_index=True, width="stretch", height=400)
        with t_inc:
            st.dataframe(df_incidentes, hide_index=True, width="stretch", height=400)
        with t_inf:
            st.dataframe(df_infra, hide_index=True, width="stretch", height=400)

    st.divider()

    # --- GRÁFICOS DINÂMICOS ---
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.altair_chart(
            criar_grafico_com_regua(df_chart, ["open", "resolved"], "Incidentes ao longo do tempo", ["#ff2b2b", "#00dcdc"], frame["time_h"]),
            width="stretch",
        )
    with chart_right:
        st.altair_chart(
            criar_grafico_com_regua(df_chart, ["cumulative_reward"], "Recompensa Acumulada", ["#0078ff"], frame["time_h"]),
            width="stretch",
        )

    st.divider()

    # --- DIAGNÓSTICO DO AGENTE (RECOMPENSAS) ---
    st.markdown("**Diagnóstico de Decisão do Agente**")
    val_age = abs(metric(metrics, "incident_age_penalty"))
    val_transit = abs(metric(metrics, "transit_penalty"))
    val_adv = abs(metric(metrics, "unassessed_adversarial_penalty"))
    total_pen = val_age + val_transit + val_adv

    pct_age = f"{(val_age / total_pen) * 100:.1f}% do total" if total_pen > 0 else "0%"
    pct_transit = f"{(val_transit / total_pen) * 100:.1f}% do total" if total_pen > 0 else "0%"
    pct_adv = f"{(val_adv / total_pen) * 100:.1f}% do total" if total_pen > 0 else "0%"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recompensa Total", f"{metric(metrics, 'total_reward'):.2f}")
    m2.metric("Penalidade: Idade SLA", f"{metric(metrics, 'incident_age_penalty'):.2f}", pct_age, delta_color="off")
    m3.metric("Penalidade: Trânsito", f"{metric(metrics, 'transit_penalty'):.2f}", pct_transit, delta_color="off")
    m4.metric("Penalidade: Adversarial", f"{metric(metrics, 'unassessed_adversarial_penalty'):.2f}", pct_adv, delta_color="off")

    base_step = (
        alt.Chart(df_step)
        .mark_bar(color="#ff4b4b", size=2.5)
        .encode(
            x=alt.X("Tempo (h):Q", title="Tempo (h)"),
            y=alt.Y("Recompensa no Passo:Q", title="Penalização Instantânea"),
            tooltip=[alt.Tooltip("Tempo (h)"), alt.Tooltip("Recompensa no Passo", format=".4f")],
        )
        .properties(height=250, title="Sangramento de Pontos por Passo (Pulso)")
    )
    regua_step = alt.Chart(pd.DataFrame({"x": [frame["time_h"]]})).mark_rule(color="white", strokeWidth=2).encode(x="x:Q")

    df_donut = pd.DataFrame({
        "Componente": ["Idade do Incidente", "Trânsito", "Risco Adversarial"],
        "Valor": [val_age, val_transit, val_adv],
    })
    df_donut = df_donut[df_donut["Valor"] > 0]

    donut_chart = (
        alt.Chart(df_donut)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta(field="Valor", type="quantitative"),
            color=alt.Color(
                field="Componente",
                type="nominal",
                scale=alt.Scale(scheme="set2"),
                legend=alt.Legend(orient="bottom", title=""),
            ),
            tooltip=["Componente", alt.Tooltip("Valor", format=".2f")],
        )
        .properties(height=250, title="Peso Absoluto das Falhas")
    )

    c1, c2 = st.columns([2.5, 1.5])
    with c1:
        st.altair_chart(base_step + regua_step, width="stretch")
    with c2:
        st.altair_chart(donut_chart, width="stretch")

    st.divider()

    # --- GRÁFICO DE GANTT GLOBAL ---
    st.markdown("**Cronograma de Operações da Frota (Gantt)**")
    if not df_gantt.empty:
        domain_estados = ["idle", "transit", "on_site", "repairing", "returning"]
        range_cores = ["#555555", "#0078ff", "#00dcdc", "#ff8c00", "#9b59b6"]
        altura_dinamica = max(150, len(vessel_ids) * 50 + 60)

        gantt_base = (
            alt.Chart(df_gantt)
            .mark_bar(cornerRadius=4, size=24)
            .encode(
                x=alt.X("Início:Q", title="Tempo (h)"),
                x2="Fim:Q",
                y=alt.Y("Navio:N", title="", sort=vessel_ids, axis=alt.Axis(grid=False, tickSize=0, domain=False, labelPadding=10, labelFontSize=12)),
                color=alt.Color("Estado:N", scale=alt.Scale(domain=domain_estados, range=range_cores), legend=alt.Legend(title="", orient="top", labelFontSize=12)),
                tooltip=[
                    alt.Tooltip("Navio:N", title="Navio"),
                    alt.Tooltip("Estado:N", title="Estado"),
                    alt.Tooltip("Início:Q", title="Início (h)", format=".1f"),
                    alt.Tooltip("Fim:Q", title="Fim (h)", format=".1f"),
                    alt.Tooltip("Duração:N", title="Duração total"),
                ],
            )
            .properties(height=altura_dinamica)
        )
        regua_gantt = alt.Chart(pd.DataFrame({"x": [frame["time_h"]]})).mark_rule(color="red", strokeWidth=2).encode(x="x:Q")
        st.altair_chart(gantt_base + regua_gantt, width="stretch")

    # --- TABELAS HISTÓRICAS DE ACONTECIMENTOS ---
    hist_decisions, hist_events = [], []
    for t in timeline[: st.session_state.step_slider + 1]:
        for d in t.get("decisions", []):
            if d.get("action_type") != "hold":
                hist_decisions.append({"Passo": t["step"], "Tempo": f"{t['time_h']:.1f}h", **d})
        for e in t.get("events", []):
            hist_events.append({"Passo": t["step"], "Tempo": f"{t['time_h']:.1f}h", **e})

    df_dec = pd.DataFrame(hist_decisions)
    if not df_dec.empty:
        df_dec = df_dec.iloc[::-1][["Passo", "Tempo", "agent_id", "action_type", "target_incident_id", "outcome"]]
    df_ev = pd.DataFrame(hist_events)
    if not df_ev.empty:
        df_ev = df_ev.iloc[::-1]

    details_left, details_right = st.columns(2)
    with details_left:
        st.markdown("**Últimas Decisões Relevantes** *(Ações de despacho)*")
        st.dataframe(df_dec, hide_index=True, width="stretch", height=250)
    with details_right:
        st.markdown("**Registo de Acontecimentos** *(Surgimento de Alarmes)*")
        st.dataframe(df_ev, hide_index=True, width="stretch", height=250)

    with st.expander("Metadados e métricas completas do Episódio"):
        st.json({"generator": bundle["generator"], "scenario": scenario, "policy": policy, "metrics": metrics})


# =====================================================================
# TAB 2: MODO FOCO (ANÁLISE EXCLUSIVA DE NAVIO)
# =====================================================================
with tab_foco:
    f_col1, f_col2, f_col3, f_col4 = st.columns([2.5, 2.2, 2.2, 2.2])
    with f_col1:
        navio_escolhido = st.selectbox("🎯 Selecionar Navio para Auditoria", options=vessel_ids, key="tab_focus_vessel")
    with f_col2:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        isolar_foco = st.toggle("Ocultar restantes entidades", value=True, key="tog_iso_foco")
    with f_col3:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        linhas_foco = st.checkbox("Trajetória Projetada (Linha)", value=True, key="chk_lin_foco")
    with f_col4:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        rastos_foco = st.checkbox("Rasto Histórico (Tracejado)", value=True, key="chk_rst_foco")

    dados_nav = next((v for v in frame["vessels"] if v["id"] == navio_escolhido), None)
    v_type_str = dados_nav.get("type", "unknown").capitalize() if dados_nav else "N/A"
    raw_st = dados_nav["status"] if dados_nav else "unknown"
    alvo_id = dados_nav.get("assigned_incident_id") if dados_nav else None

    if raw_st == "transit":
        estado_atual_str = f"Em trânsito ➔ Inc {alvo_id}"
    elif raw_st == "on_site":
        estado_atual_str = f"No local ➔ Inc {alvo_id}"
    elif raw_st == "repairing":
        estado_atual_str = f"A reparar ➔ Inc {alvo_id}"
    elif raw_st == "returning":
        estado_atual_str = "Regresso ao Porto"
    elif raw_st == "idle":
        estado_atual_str = "Inativo (Porto)"
    else:
        estado_atual_str = raw_st.capitalize()

    dist_atual_km, dist_total_km = 0.0, 0.0
    incidentes_atendidos = set()
    prev_lat, prev_lon = None, None
    for idx_t, t in enumerate(timeline):
        v_step = next((x for x in t["vessels"] if x["id"] == navio_escolhido), None)
        if v_step:
            if prev_lat is not None and prev_lon is not None:
                step_d = haversine_km(prev_lat, prev_lon, v_step["lat"], v_step["lon"])
                dist_total_km += step_d
                if idx_t <= st.session_state.step_slider:
                    dist_atual_km += step_d
            prev_lat, prev_lon = v_step["lat"], v_step["lon"]
            if v_step["status"] in ["on_site", "repairing"] and v_step.get("assigned_incident_id") is not None:
                incidentes_atendidos.add(v_step.get("assigned_incident_id"))

    df_gantt_nav = df_gantt[df_gantt["Navio"] == navio_escolhido].copy() if not df_gantt.empty else pd.DataFrame()
    total_ep_h = max(1.0, float(timeline[-1]["time_h"]))
    horas_idle = df_gantt_nav[df_gantt_nav["Estado"] == "idle"]["Duracao_num"].sum() if not df_gantt_nav.empty else 0.0
    horas_transit = df_gantt_nav[df_gantt_nav["Estado"] == "transit"]["Duracao_num"].sum() if not df_gantt_nav.empty else 0.0
    horas_work = df_gantt_nav[df_gantt_nav["Estado"].isin(["on_site", "repairing"])]["Duracao_num"].sum() if not df_gantt_nav.empty else 0.0
    horas_ativas = total_ep_h - horas_idle
    taxa_ocupacao = (horas_ativas / total_ep_h) * 100.0

    decisoes_navio = []
    for t in timeline:
        for d in t.get("decisions", []):
            if d.get("agent_id") == navio_escolhido and d.get("action_type") != "hold":
                decisoes_navio.append({
                    "Passo": t["step"],
                    "Tempo (h)": f"{t['time_h']:.1f}h",
                    "Ação": d.get("action_type"),
                    "Incidente Alvo": d.get("target_incident_id"),
                    "Resultado": d.get("outcome"),
                })

    st.divider()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(f"Estado ({v_type_str})", estado_atual_str)
    k2.metric("Distância Navegada", f"{dist_atual_km:.0f} km", f"Total Ep: {dist_total_km:.0f} km", delta_color="off")
    k3.metric("Taxa de Ocupação", f"{taxa_ocupacao:.1f}%", f"{horas_ativas:.1f}h ativas", delta_color="off")
    k4.metric("Tempo em Trânsito", f"{horas_transit:.1f} h", f"{(horas_transit / total_ep_h) * 100:.1f}% do ep.", delta_color="off")
    k5.metric("Tempo no Local/Reparo", f"{horas_work:.1f} h", f"{len(incidentes_atendidos)} inc. visitados", delta_color="off")
    k6.metric("Ordens de Despacho", len(decisoes_navio), f"Inativo: {horas_idle:.1f}h", delta_color="off")

    deck_foco, _, _ = build_map_and_tables(
        episode=episode,
        timeline=timeline,
        step_idx=st.session_state.step_slider,
        navio_focado=navio_escolhido,
        isolar_entidades=isolar_foco,
        mostrar_linhas=linhas_foco,
        mostrar_rastos=rastos_foco,
        map_height=460,
    )

    f_map_col, f_stat_col = st.columns([2, 1])
    with f_map_col:
        st.pydeck_chart(deck_foco)
        render_legenda_mapa()

    with f_stat_col:
        st.markdown(f"**Telemetria e Perfil de `{navio_escolhido}`**")
        if dados_nav:
            st.caption(
                f"📍 **Coordenadas Atuais:** `{dados_nav['lat']:.3f}, {dados_nav['lon']:.3f}` | "
                f"🎯 **Alvo Atual:** `{alvo_id if alvo_id is not None else 'Nenhum'}`"
            )

        if not df_gantt_nav.empty:
            df_dist_est = df_gantt_nav.groupby("Estado", as_index=False)["Duracao_num"].sum()
            domain_estados = ["idle", "transit", "on_site", "repairing", "returning"]
            range_cores = ["#555555", "#0078ff", "#00dcdc", "#ff8c00", "#9b59b6"]
            donut_nav = (
                alt.Chart(df_dist_est)
                .mark_arc(innerRadius=45)
                .encode(
                    theta=alt.Theta("Duracao_num:Q", title="Horas"),
                    color=alt.Color("Estado:N", scale=alt.Scale(domain=domain_estados, range=range_cores), legend=alt.Legend(orient="bottom", title="")),
                    tooltip=[alt.Tooltip("Estado:N"), alt.Tooltip("Duracao_num:Q", title="Horas", format=".1f")],
                )
                .properties(height=220, title="Distribuição de Tempo por Estado (h)")
            )
            st.altair_chart(donut_nav, width="stretch")

        st.markdown("**Histórico de Despachos do Navio**")
        df_dec_nav = pd.DataFrame(decisoes_navio)
        if not df_dec_nav.empty:
            st.dataframe(df_dec_nav.iloc[::-1], hide_index=True, width="stretch", height=190)
        else:
            st.info("Este navio não recebeu ordens de despacho no episódio.")

    if not df_gantt_nav.empty:
        st.markdown(f"**Linha Temporal de Estados (`{navio_escolhido}`)**")
        domain_estados = ["idle", "transit", "on_site", "repairing", "returning"]
        range_cores = ["#555555", "#0078ff", "#00dcdc", "#ff8c00", "#9b59b6"]
        gantt_single = (
            alt.Chart(df_gantt_nav)
            .mark_bar(cornerRadius=4, size=30)
            .encode(
                x=alt.X("Início:Q", title="Tempo (h)"),
                x2="Fim:Q",
                y=alt.Y("Navio:N", title="", axis=alt.Axis(grid=False, tickSize=0, domain=False)),
                color=alt.Color("Estado:N", scale=alt.Scale(domain=domain_estados, range=range_cores), legend=None),
                tooltip=["Estado:N", alt.Tooltip("Início:Q", format=".1f"), alt.Tooltip("Fim:Q", format=".1f"), "Duração:N"],
            )
            .properties(height=90)
        )
        regua_nav = alt.Chart(pd.DataFrame({"x": [frame["time_h"]]})).mark_rule(color="red", strokeWidth=2).encode(x="x:Q")
        st.altair_chart(gantt_single + regua_nav, width="stretch")


# =====================================================================
# TAB 3: COMPARADOR DE EPISÓDIOS (AUDITORIA DELTA + LADO A LADO)
# =====================================================================
with tab_comp:
    st.subheader("Auditoria de Performance: Frente-a-Frente", divider="gray")

    compatible = {
        k: c
        for k, c in bundles.items()
        if c["episode"]["scenario"]["id"] == scenario["id"]
        and c["episode"]["scenario"]["world_seed"] == scenario["world_seed"]
    }
    comp_pool = compatible if len(compatible) >= 2 else bundles
    pool_keys = list(comp_pool.keys())

    default_base_idx = pool_keys.index(episode_id) if episode_id in pool_keys else 0
    default_target_idx = 1 if (len(pool_keys) > 1 and default_base_idx == 0) else 0

    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        base_id = st.selectbox(
            "📌 Episódio Base (Referência)",
            options=pool_keys,
            index=default_base_idx,
            format_func=lambda v: bundle_label(comp_pool[v]),
            key="comp_base_sel",
        )
    with sel_col2:
        target_id = st.selectbox(
            "🚀 Episódio Alvo (Política em Auditoria)",
            options=pool_keys,
            index=default_target_idx,
            format_func=lambda v: bundle_label(comp_pool[v]),
            key="comp_target_sel",
        )

    ep_base = comp_pool[base_id]["episode"]
    ep_target = comp_pool[target_id]["episode"]
    m_base = ep_base["metrics"]
    m_target = ep_target["metrics"]

    # --- 1. ANÁLISE QUANTITATIVA: PAINEL DE MÉTRICAS DELTA ---
    st.markdown("**1. Variação de Performance (Alvo vs. Base)**")

    rew_base, rew_target = float(metric(m_base, "total_reward")), float(metric(m_target, "total_reward"))
    rate_base, rate_target = 100 * float(metric(m_base, "resolution_rate")), 100 * float(metric(m_target, "resolution_rate"))
    resp_base, resp_target = float(metric(m_base, "mean_first_response_time_h")), float(metric(m_target, "mean_first_response_time_h"))
    res_base, res_target = float(metric(m_base, "mean_time_to_resolution_h")), float(metric(m_target, "mean_time_to_resolution_h"))
    dup_base, dup_target = int(metric(m_base, "duplicate_dispatch_count")), int(metric(m_target, "duplicate_dispatch_count"))
    sev_base, sev_target = float(metric(m_base, "unresolved_severity")), float(metric(m_target, "unresolved_severity"))

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Recompensa Total", f"{rew_target:.2f}", f"{rew_target - rew_base:+.2f}", delta_color="normal")
    d2.metric("Taxa de Resolução", f"{rate_target:.1f}%", f"{rate_target - rate_base:+.1f}%", delta_color="normal")
    d3.metric("Primeira Resposta", f"{resp_target:.1f} h", f"{resp_target - resp_base:+.1f} h", delta_color="inverse")
    d4.metric("Tempo de Resolução", f"{res_target:.1f} h", f"{res_target - res_base:+.1f} h", delta_color="inverse")
    d5.metric("Despachos Duplicados", dup_target, f"{dup_target - dup_base:+d}", delta_color="inverse")
    d6.metric("Severidade p/ Resolver", f"{sev_target:.2f}", f"{sev_target - sev_base:+.2f}", delta_color="inverse")

    st.divider()

    # --- 2. ANÁLISE QUALITATIVA: REPRODUÇÃO LADO A LADO ---
    st.markdown("**2. Reprodução Sincronizada Lado a Lado**")
    cc1, cc2, _ = st.columns([2.4, 2.4, 5.2])
    with cc1:
        comp_linhas = st.checkbox("Trajetórias nos 2 Mapas (Linha)", value=True, key="chk_lin_comp")
    with cc2:
        comp_rastos = st.checkbox("Rastos nos 2 Mapas (Tracejado)", value=False, key="chk_rst_comp")

    deck_base, df_tab_base, frame_base = build_map_and_tables(
        episode=ep_base,
        timeline=ep_base["timeline"],
        step_idx=st.session_state.step_slider,
        mostrar_linhas=comp_linhas,
        mostrar_rastos=comp_rastos,
        map_height=430,
    )
    deck_target, df_tab_target, frame_target = build_map_and_tables(
        episode=ep_target,
        timeline=ep_target["timeline"],
        step_idx=st.session_state.step_slider,
        mostrar_linhas=comp_linhas,
        mostrar_rastos=comp_rastos,
        map_height=430,
    )

    map_col_base, map_col_target = st.columns(2)
    with map_col_base:
        st.markdown(f"##### 📌 Base: `{ep_base['policy']['label']}`")
        open_b = sum(inc["status"] != "resolved" for inc in frame_base["incidents"])
        res_b = sum(inc["status"] == "resolved" for inc in frame_base["incidents"])
        st.caption(
            f"⏱️ **Hora:** {frame_base['time_h']:.1f}h | 🚨 **Abertos:** {open_b} | "
            f"✅ **Resolvidos:** {res_b} | 🏆 **Recompensa Acum.:** {frame_base['cumulative_reward']:.2f}"
        )
        st.pydeck_chart(deck_base)
        df_nav_b = df_tab_base[df_tab_base["kind"].str.contains("Navio")][["id", "status"]].rename(columns={"id": "Navio", "status": "Estado no Passo"})
        st.dataframe(df_nav_b, hide_index=True, width="stretch", height=175)

    with map_col_target:
        st.markdown(f"##### 🚀 Alvo: `{ep_target['policy']['label']}`")
        open_t = sum(inc["status"] != "resolved" for inc in frame_target["incidents"])
        res_t = sum(inc["status"] == "resolved" for inc in frame_target["incidents"])
        st.caption(
            f"⏱️ **Hora:** {frame_target['time_h']:.1f}h | 🚨 **Abertos:** {open_t} | "
            f"✅ **Resolvidos:** {res_t} | 🏆 **Recompensa Acum.:** {frame_target['cumulative_reward']:.2f}"
        )
        st.pydeck_chart(deck_target)
        df_nav_t = df_tab_target[df_tab_target["kind"].str.contains("Navio")][["id", "status"]].rename(columns={"id": "Navio", "status": "Estado no Passo"})
        st.dataframe(df_nav_t, hide_index=True, width="stretch", height=175)

    render_legenda_mapa()
    st.divider()

    # --- 3. TABELA RESUMO MULTI-EPISÓDIO ---
    st.markdown("**3. Tabela Resumo de Todos os Episódios Compatíveis**")
    selected_for_comparison = st.multiselect(
        "Episódios com o mesmo cenário e seed",
        options=list(comp_pool),
        default=list(comp_pool),
        format_func=lambda value: comp_pool[value]["episode"]["policy"]["label"],
    )
    comparison_rows = [
        {
            "Política": comp_pool[sel]["episode"]["policy"]["label"],
            "Recompensa Total": metric(comp_pool[sel]["episode"]["metrics"], "total_reward", None),
            "Taxa Resolução (%)": round(100 * float(metric(comp_pool[sel]["episode"]["metrics"], "resolution_rate", 0)), 1),
            "1ª Resposta (h)": metric(comp_pool[sel]["episode"]["metrics"], "mean_first_response_time_h", None),
            "Tempo Resolução (h)": metric(comp_pool[sel]["episode"]["metrics"], "mean_time_to_resolution_h", None),
            "Severidade p/ Resolver": metric(comp_pool[sel]["episode"]["metrics"], "unresolved_severity", None),
            "Despachos Duplicados": metric(comp_pool[sel]["episode"]["metrics"], "duplicate_dispatch_count", None),
        }
        for sel in selected_for_comparison
    ]
    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")


# --- GATILHO DA ANIMAÇÃO ---
if st.session_state.playing:
    time.sleep(0.1)
    st.rerun()