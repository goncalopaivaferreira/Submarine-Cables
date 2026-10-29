"""ARC Episode Viewer - Dashboard Operacional Otimizado (WebGL + Cache + Vectorized DataFrames)."""

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
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_dashed_segments(coords: list[list[float]], color: list[int], dash_deg: float = 0.55, gap_deg: float = 0.40) -> list[dict]:
    """Gera segmentos tracejados (dashes) para o rasto histórico."""
    segments = []
    if not coords:
        return segments
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


# =====================================================================
# NÍVEL 2 & GPU: PRÉ-CÁLCULO EM CACHE DE INFRAESTRUTURA (GEOJSON) E TRIPS
# =====================================================================
@st.cache_data(show_spinner=False)
def get_static_infra_geojson(episode_id: str, map_data: dict, faded: bool = False) -> tuple[dict, pd.DataFrame]:
    """
    Converte Portos, Landing Points e Cabos Submarinos num GeoJSON estático em Cache.
    Calculado apenas 1 vez por episódio (Zero custo de CPU durante a animação!).
    """
    features = []
    infra_rows = []
    alpha = 45 if faded else 255

    # 1. Cabos submarinos (caso existam ou venham a existir no JSON como segmentos/linhas)
    for cable in map_data.get("cables", map_data.get("submarine_cables", [])):
        coords = cable.get("coordinates", cable.get("path", []))
        if coords:
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "color": [0, 180, 216, 90 if not faded else 25],
                    "tooltip_text": f"Cabo Submarino: {cable.get('id', 'Cabo')}",
                    "radius": 0,
                },
            })

    # 2. Portos
    for port in map_data.get("ports", []):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [port["lon"], port["lat"]]},
            "properties": {
                "color": [255, 255, 255, alpha],
                "tooltip_text": f"ID: {port['id']}\nPorto\nEstado: Static",
                "radius": 55000,
            },
        })
        infra_rows.append({"Nome": port["id"], "Tipo": "Porto", "Coords": f"{port['lat']:.3f}, {port['lon']:.3f}"})

    # 3. Landing Points (Pontos estáticos dos cabos submarinos)
    for point in map_data.get("landing_points", []):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [point["lon"], point["lat"]]},
            "properties": {
                "color": [155, 89, 182, alpha],
                "tooltip_text": f"ID: {point['id']}\nLanding Point\nEstado: Static",
                "radius": 50000,
            },
        })
        infra_rows.append({"Nome": point["id"], "Tipo": "Landing Point", "Coords": f"{point['lat']:.3f}, {point['lon']:.3f}"})

    geojson_obj = {"type": "FeatureCollection", "features": features}
    df_infra = pd.DataFrame(infra_rows) if infra_rows else pd.DataFrame(columns=["Nome", "Tipo", "Coords"])
    return geojson_obj, df_infra


@st.cache_data(show_spinner=False)
def precompute_episode_series(episode_id: str, timeline: list[dict], vessel_ids: list[str]):
    """Pré-calcula séries temporais, Gantt e dados de TripsLayer (GPU) em cache."""
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
    trips_dict = {vid: {"path": [], "timestamps": [], "type": "unknown"} for vid in vessel_ids}

    for t in timeline:
        th = float(t["time_h"])
        for v in t["vessels"]:
            vid = v["id"]
            if vid in trips_dict:
                trips_dict[vid]["path"].append([v["lon"], v["lat"]])
                trips_dict[vid]["timestamps"].append(th)
                trips_dict[vid]["type"] = v.get("type", "unknown")

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

    trips_rows = []
    for vid, tdata in trips_dict.items():
        if len(tdata["path"]) > 1:
            color = [80, 190, 255] if tdata["type"] == "patrol" else [255, 170, 50]
            trips_rows.append({
                "id": vid,
                "path": tdata["path"],
                "timestamps": tdata["timestamps"],
                "color": color,
            })

    df_gantt = pd.DataFrame(gantt_blocks)
    df_trips = pd.DataFrame(trips_rows)
    return df_chart, df_step, df_gantt, df_trips


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


# =====================================================================
# NÍVEL 3: MOTOR DE MAPA VETORIZADO COM PANDAS DATAFRAMES + GEOJSON LAYER
# =====================================================================
def build_fast_map_and_tables(
    episode: dict,
    timeline: list[dict],
    step_idx: int,
    navio_focado: str | None = None,
    isolar_entidades: bool = False,
    mostrar_linhas: bool = True,
    mostrar_rastos: bool = False,
    map_height: int = 500,
):
    safe_idx = min(step_idx, len(timeline) - 1)
    frame = timeline[safe_idx]
    ep_id = episode["id"]

    dados_navio_focado = None
    incidente_alvo_do_foco = None
    if navio_focado:
        dados_navio_focado = next((v for v in frame["vessels"] if v["id"] == navio_focado), None)
        if dados_navio_focado:
            incidente_alvo_do_foco = dados_navio_focado.get("assigned_incident_id")

    layers_list = []

    # 1. CAMADA ESTÁTICA DE INFRAESTRUTURA (GeoJsonLayer)
    geojson_infra, df_infra = get_static_infra_geojson(
        ep_id, episode.get("map", {}), faded=(navio_focado is not None and not isolar_entidades)
    )
    if not (navio_focado and isolar_entidades) and geojson_infra["features"]:
        layers_list.append(
            pdk.Layer(
                "GeoJsonLayer",
                id=f"static_infra_{ep_id}",
                data=geojson_infra,
                pickable=True,
                stroked=True,
                filled=True,
                point_radius_min_pixels=6,
                point_radius_max_pixels=12,
                get_point_radius="properties.radius",
                get_fill_color="properties.color",
                get_line_color="properties.color",
                line_width_min_pixels=2,
            )
        )

    # 2. DATAFRAME DIRETO DE INCIDENTES
    inc_list = frame.get("incidents", [])
    if inc_list:
        df_inc_raw = pd.DataFrame(inc_list)
        if navio_focado and isolar_entidades:
            df_inc_raw = df_inc_raw[df_inc_raw["id"] == incidente_alvo_do_foco].copy()

        if not df_inc_raw.empty:
            status_map = {
                "open": "Aberto (A aguardar)",
                "on_site": "Navio no local",
                "repairing": "Em reparação",
                "resolved": "Resolvido",
            }
            df_inc_raw["Estado"] = df_inc_raw["status"].map(lambda s: status_map.get(s, str(s).capitalize()))
            df_inc_raw["Incidente"] = "incident_" + df_inc_raw["id"].astype(str)
            df_inc_raw["Coords"] = df_inc_raw["lat"].map("{:.3f}".format) + ", " + df_inc_raw["lon"].map("{:.3f}".format)

            def get_inc_color(row):
                if navio_focado and row["id"] != incidente_alvo_do_foco:
                    return [120, 120, 120, 60]
                st_val = row["status"]
                if st_val == "open":
                    return [255, 40, 40, 255]
                if st_val in ("on_site", "repairing"):
                    return [0, 220, 220, 255]
                return [100, 100, 100, 120]

            df_inc_raw["color"] = df_inc_raw.apply(get_inc_color, axis=1)
            df_inc_raw["tooltip_text"] = (
                "ID: " + df_inc_raw["Incidente"] + "\nIncidente\nEstado: " + df_inc_raw["Estado"]
            )
            df_incidentes = df_inc_raw[["Incidente", "Estado", "Coords"]].copy()
        else:
            df_incidentes = pd.DataFrame(columns=["Incidente", "Estado", "Coords"])
    else:
        df_inc_raw = pd.DataFrame()
        df_incidentes = pd.DataFrame(columns=["Incidente", "Estado", "Coords"])

    # 3. DATAFRAME DIRETO DE NAVIOS
    ves_list = frame.get("vessels", [])
    if ves_list:
        df_ves_raw = pd.DataFrame(ves_list)
        if navio_focado and isolar_entidades:
            df_ves_raw = df_ves_raw[df_ves_raw["id"] == navio_focado].copy()

        if not df_ves_raw.empty:
            def format_ves_status(row):
                st_val = row["status"]
                aid = row.get("assigned_incident_id")
                has_aid = aid is not None and not pd.isna(aid)
                aid_str = int(aid) if has_aid and isinstance(aid, (int, float)) else aid
                if st_val == "transit" and has_aid:
                    return f"Em trânsito para o Incidente {aid_str}"
                if st_val == "on_site" and has_aid:
                    return f"No local do Incidente {aid_str}"
                if st_val == "repairing" and has_aid:
                    return f"A reparar o Incidente {aid_str}"
                if st_val == "returning":
                    return "A regressar ao Porto"
                if st_val == "idle":
                    return "Inativo (A aguardar ordens)"
                return str(st_val).capitalize()

            def get_ves_color(row):
                if navio_focado and row["id"] != navio_focado:
                    return [120, 120, 120, 60]
                vt = row.get("type", "unknown")
                if vt == "patrol":
                    return [0, 120, 255, 255]
                if vt == "repair":
                    return [255, 140, 0, 255]
                return [150, 150, 150, 255]

            df_ves_raw["Estado"] = df_ves_raw.apply(format_ves_status, axis=1)
            df_ves_raw["Tipo"] = df_ves_raw["type"].astype(str).str.capitalize()
            df_ves_raw["Coords"] = df_ves_raw["lat"].map("{:.3f}".format) + ", " + df_ves_raw["lon"].map("{:.3f}".format)
            df_ves_raw["color"] = df_ves_raw.apply(get_ves_color, axis=1)
            df_ves_raw["tooltip_text"] = (
                "ID: " + df_ves_raw["id"] + "\nNavio (" + df_ves_raw["Tipo"] + ")\nEstado: " + df_ves_raw["Estado"]
            )
            df_navios = df_ves_raw[["id", "Estado", "Tipo", "Coords"]].rename(columns={"id": "Navio"}).copy()
        else:
            df_navios = pd.DataFrame(columns=["Navio", "Estado", "Tipo", "Coords"])
    else:
        df_ves_raw = pd.DataFrame()
        df_navios = pd.DataFrame(columns=["Navio", "Estado", "Tipo", "Coords"])

    # 4. RASTO HISTÓRICO (TRACEJADO / DASHES)
    if mostrar_rastos and safe_idx > 0:
        trail_dash_rows = []
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
                dash_color = [100, 195, 255, 175] if v_type == "patrol" else [255, 180, 80, 175]
                trail_dash_rows.extend(build_dashed_segments(coords, dash_color))

        if trail_dash_rows:
            df_trails = pd.DataFrame(trail_dash_rows)
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

    # 5. TRAJETÓRIA PROJETADA: LINHA CONTÍNUA COMPLETA (SEM ANEL DE ALVO)
    if mostrar_linhas:
        line_rows = []
        inc_lookup = {inc["id"]: inc for inc in inc_list}
        for vessel in ves_list:
            if navio_focado and vessel["id"] != navio_focado:
                continue
            assigned_id = vessel.get("assigned_incident_id")
            if vessel["status"] == "transit" and assigned_id is not None:
                alvo = inc_lookup.get(assigned_id)
                if alvo:
                    v_type = vessel.get("type", "unknown")
                    line_color = [0, 195, 255, 210] if v_type == "patrol" else [255, 165, 0, 210]
                    line_rows.append({
                        "start": [vessel["lon"], vessel["lat"]],
                        "end": [alvo["lon"], alvo["lat"]],
                        "color": line_color,
                    })
        if line_rows:
            df_lines = pd.DataFrame(line_rows)
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

    # 6. CAMADAS VETORIZADAS GERAIS (Scatters)
    if not df_inc_raw.empty:
        layers_list.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=df_inc_raw[["lon", "lat", "color", "tooltip_text"]],
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=62000,
                radius_min_pixels=8,
                radius_max_pixels=17,
                pickable=True,
            )
        )

    if not df_ves_raw.empty:
        layers_list.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=df_ves_raw[["lon", "lat", "color", "tooltip_text"]],
                get_position="[lon, lat]",
                get_fill_color="color",
                get_line_color=[255, 255, 255, 200],
                stroked=True,
                line_width_min_pixels=1.5,
                get_radius=55000,
                radius_min_pixels=8,
                radius_max_pixels=16,
                pickable=True,
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
    return deck, df_navios, df_incidentes, df_infra, frame


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


# --- CONFIGURAÇÃO DA PÁGINA E CSS ---
st.set_page_config(page_title="ARC Episode Viewer", layout="wide")
st.title("ARC Episode Viewer")
st.caption("Plataforma Operacional e de Auditoria de Políticas de Reinforcement Learning (Cabos Submarinos).")

st.markdown("""
    <style>
    .mapboxgl-ctrl-bottom-left, .mapboxgl-ctrl-bottom-right { display: none !important; }
    div[data-baseweb="select"] { height: auto !important; }
    div[data-baseweb="select"] * { white-space: normal !important; word-break: break-word !important; text-overflow: unset !important; }
    
    /* Tabs Minimalistas e Centradas */
    div[data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        border-bottom: 1px solid #333 !important;
        padding-bottom: 0px !important;
        gap: 0px !important;
    }
    button[data-baseweb="tab"] {
        flex: 1;
        text-align: center;
        font-size: 1.15rem !important;
        font-weight: 400 !important;
        color: #777 !important;
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 12px 0px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #fff !important;
        border-bottom: 2px solid #fff !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="tab-highlight"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

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
vessel_ids = [v["id"] for v in timeline[0]["vessels"]] if timeline else []

# --- ABA LATERAL: DETALHES DO EPISÓDIO ---
st.sidebar.divider()
st.sidebar.caption("**📋 DETALHES DO EPISÓDIO ATIVO**")
st.sidebar.caption(f"**Cenário:** {scenario['id']}")
st.sidebar.caption(f"**Política:** {policy['label']} ({policy['kind']})")
st.sidebar.caption(f"**Duração:** {timeline[-1]['time_h']:.1f}h | **Passos:** {len(timeline)}")
st.sidebar.caption(f"**Seed:** {scenario['world_seed']} | **Split:** {scenario['split']}")
st.sidebar.caption(f"**Banco:** {scenario['bank_id']}")

# --- ESTADO GLOBAL DA ANIMAÇÃO ---
if "step_slider" not in st.session_state: st.session_state.step_slider = 0
if "playing" not in st.session_state: st.session_state.playing = False

if st.session_state.step_slider > max_frame: st.session_state.step_slider = max_frame
if st.session_state.playing:
    if st.session_state.step_slider < max_frame: st.session_state.step_slider += 1
    else: st.session_state.playing = False

def click_play():
    if st.session_state.step_slider >= max_frame: st.session_state.step_slider = 0
    st.session_state.playing = not st.session_state.playing

def click_reset():
    st.session_state.step_slider = 0
    st.session_state.playing = False

def update_slider(key):
    st.session_state.step_slider = st.session_state[key]

df_chart, df_step, df_gantt, _ = precompute_episode_series(episode["id"], timeline, vessel_ids)

@st.fragment
def render_interactive_workspace():
    frame = timeline[st.session_state.step_slider]

    # 1. TABS NO TOPO ABSOLUTO (Isto resolve a duplicação)
    tab_geral, tab_foco, tab_comp = st.tabs([
        "   Visão Geral   ", 
        "   Modo Foco   ", 
        "   Comparação de Episódios   "
    ])

    # 2. FUNÇÃO AUXILIAR: Desenha o Cabeçalho e Slider abaixo da aba ativa
    def render_header_and_controls(tab_id):
        st.subheader(f"Cenário: {scenario['id']} — Política: {policy['label']}", divider="gray")
        columns = st.columns(6)
        columns[0].metric("Resolvidos", f"{int(metric(metrics, 'n_resolved'))}/{int(metric(metrics, 'n_incidents'))}")
        columns[1].metric("Taxa de resolução", f"{100 * float(metric(metrics, 'resolution_rate')):.1f}%")
        columns[2].metric("Primeira resposta", f"{float(metric(metrics, 'mean_first_response_time_h')):.1f} h")
        columns[3].metric("Tempo de resolução", f"{float(metric(metrics, 'mean_time_to_resolution_h')):.1f} h")
        columns[4].metric("Despachos duplicados", int(metric(metrics, "duplicate_dispatch_count")))
        columns[5].metric("Severidade por resolver", f"{float(metric(metrics, 'unresolved_severity')):.2f}")
        st.markdown("<br>", unsafe_allow_html=True)

        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.3, 1.1, 2.2, 5.4])
        with ctrl_col1: st.button("⏸ Pausa" if st.session_state.playing else "▶ Reproduzir", on_click=click_play, width="stretch", key=f"play_{tab_id}")
        with ctrl_col2: st.button("⏮ Reset", on_click=click_reset, width="stretch", key=f"reset_{tab_id}")
        with ctrl_col3: st.markdown(f"<div style='padding-top: 8px; font-weight: 500;'>⏱️ Passo {frame['step']}/{max_frame} ({frame['time_h']:.1f}h)</div>", unsafe_allow_html=True)
        with ctrl_col4: st.slider("Navegação Temporal", min_value=0, max_value=max_frame, value=st.session_state.step_slider, key=f"slider_{tab_id}", on_change=update_slider, args=(f"slider_{tab_id}",), label_visibility="collapsed")
        st.divider()

    # -----------------------------------------------------------------
    # ABA 1: VISÃO GERAL
    # -----------------------------------------------------------------
    with tab_geral:
        render_header_and_controls("vg")
        
        c_opt1, c_opt2, _ = st.columns([1.5, 1.5, 7])
        with c_opt1: mostrar_linhas_glob = st.checkbox("Trajetórias", value=True, key="chk_lin_glob")
        with c_opt2: mostrar_rastos_glob = st.checkbox("Rastos", value=False, key="chk_rst_glob")

        deck_geral, df_navios, df_incidentes, df_infra, _ = build_fast_map_and_tables(
            episode=episode, timeline=timeline, step_idx=st.session_state.step_slider,
            mostrar_linhas=mostrar_linhas_glob, mostrar_rastos=mostrar_rastos_glob
        )

        left, right = st.columns([2, 1])
        with left:
            st.pydeck_chart(deck_geral, key="deck_vg")
            render_legenda_mapa()

        with right:
            st.markdown("**Estado no instante atual**")
            t_nav, t_inc, t_inf = st.tabs([f"Navios ({len(df_navios)})", f"Incidentes ({len(df_incidentes)})", f"Infra ({len(df_infra)})"])
            with t_nav: st.dataframe(df_navios, hide_index=True, width="stretch", height=400)
            with t_inc: st.dataframe(df_incidentes, hide_index=True, width="stretch", height=400)
            with t_inf: st.dataframe(df_infra, hide_index=True, width="stretch", height=400)

        st.divider()

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.altair_chart(criar_grafico_com_regua(df_chart, ["open", "resolved"], "Incidentes ao longo do tempo", ["#ff2b2b", "#00dcdc"], frame["time_h"]), width="stretch")
        with chart_right:
            st.altair_chart(criar_grafico_com_regua(df_chart, ["cumulative_reward"], "Recompensa Acumulada", ["#0078ff"], frame["time_h"]), width="stretch")

        st.divider()

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

        base_step = alt.Chart(df_step).mark_bar(color="#ff4b4b", size=2.5).encode(
            x=alt.X("Tempo (h):Q", title="Tempo (h)"),
            y=alt.Y("Recompensa no Passo:Q", title="Penalização Instantânea"),
            tooltip=[alt.Tooltip("Tempo (h)"), alt.Tooltip("Recompensa no Passo", format=".4f")]
        ).properties(height=250, title="Sangramento de Pontos por Passo")

        regua_step = alt.Chart(pd.DataFrame({"x": [frame["time_h"]]})).mark_rule(color="white", strokeWidth=2).encode(x="x:Q")

        df_donut = pd.DataFrame({"Componente": ["Idade do Incidente", "Trânsito", "Risco Adversarial"], "Valor": [val_age, val_transit, val_adv]})
        df_donut = df_donut[df_donut["Valor"] > 0]

        donut_chart = alt.Chart(df_donut).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Valor", type="quantitative"),
            color=alt.Color(field="Componente", type="nominal", scale=alt.Scale(scheme="set2"), legend=alt.Legend(orient="bottom", title="")),
            tooltip=["Componente", alt.Tooltip("Valor", format=".2f")]
        ).properties(height=250, title="Peso Absoluto das Falhas")

        c1, c2 = st.columns([2.5, 1.5])
        with c1: st.altair_chart(base_step + regua_step, width="stretch")
        with c2: st.altair_chart(donut_chart, width="stretch")

        st.divider()

        st.markdown("**Cronograma de Operações da Frota (Gantt)**")
        if not df_gantt.empty:
            domain_estados = ["idle", "transit", "on_site", "repairing", "returning"]
            range_cores = ["#555555", "#0078ff", "#00dcdc", "#ff8c00", "#9b59b6"]
            altura_dinamica = max(150, len(vessel_ids) * 50 + 60)

            gantt_base = alt.Chart(df_gantt).mark_bar(cornerRadius=4, size=24).encode(
                x=alt.X("Início:Q", title="Tempo (h)"),
                x2="Fim:Q",
                y=alt.Y("Navio:N", title="", sort=vessel_ids, axis=alt.Axis(grid=False, tickSize=0, domain=False, labelPadding=10, labelFontSize=12)),
                color=alt.Color("Estado:N", scale=alt.Scale(domain=domain_estados, range=range_cores), legend=alt.Legend(title="", orient="top", labelFontSize=12)),
                tooltip=["Navio", "Estado", alt.Tooltip("Início:Q", format=".1f"), alt.Tooltip("Fim:Q", format=".1f"), "Duração:N"]
            ).properties(height=altura_dinamica)
            regua_gantt = alt.Chart(pd.DataFrame({"x": [frame["time_h"]]})).mark_rule(color="red", strokeWidth=2).encode(x="x:Q")
            st.altair_chart(gantt_base + regua_gantt, width="stretch")

        st.divider()
        
        # O novo formato que pediste (Taxa de Erro e Feed em tempo real)
        st.markdown("**Auditoria de Comportamento da Política de RL**")
        hist_decisions, hist_events = [], []
        for t in timeline[: st.session_state.step_slider + 1]:
            for d in t.get("decisions", []):
                if d.get("action_type") != "hold":
                    hist_decisions.append({"Passo": t["step"], "time_h": t['time_h'], "Tempo": f"{t['time_h']:.1f}h", **d})
            for e in t.get("events", []):
                hist_events.append({"Passo": t["step"], "time_h": t['time_h'], "Tempo": f"{t['time_h']:.1f}h", **e})

        df_dec = pd.DataFrame(hist_decisions)
        df_ev = pd.DataFrame(hist_events)

        if not df_dec.empty:
            scatter_dec = alt.Chart(df_dec).mark_circle(size=80, opacity=0.9).encode(
                x=alt.X('time_h:Q', title='Tempo (h)'),
                y=alt.Y('agent_id:N', title='Navio'),
                color=alt.Color('outcome:N', title='Resultado', scale=alt.Scale(domain=['accepted', 'rejected', 'duplicate'], range=['#00dcdc', '#ff4b4b', '#ff8c00'])),
                tooltip=['Passo', 'Tempo', 'agent_id', 'action_type', 'target_incident_id', 'outcome']
            ).properties(height=200, title="Mapeamento de Decisões")
            regua_dec = alt.Chart(pd.DataFrame({"x": [frame["time_h"]]})).mark_rule(color="red", strokeWidth=1).encode(x="x:Q")
            st.altair_chart(scatter_dec + regua_dec, use_container_width=True)

        details_left, details_right = st.columns(2)
        with details_left:
            st.markdown("🎯 **Eficácia de Despacho e Feed de Ações**")
            if not df_dec.empty:
                total_dec = len(df_dec)
                err_count = len(df_dec[df_dec["outcome"].isin(["rejected", "duplicate"])])
                err_rate = (err_count / total_dec) * 100
                
                c1, c2 = st.columns(2)
                c1.metric("Taxa de Erro (RL)", f"{err_rate:.1f}%", f"{err_count} falhas", delta_color="inverse")
                c2.metric("Total de Ações", total_dec)
                
                df_dec_show = df_dec.iloc[::-1].head(15)[["Passo", "Tempo", "agent_id", "outcome", "target_incident_id"]]
                def color_outcome(val):
                    if val == 'rejected': return 'color: #ff4b4b; font-weight: bold;'
                    elif val == 'duplicate': return 'color: #ff8c00; font-weight: bold;'
                    elif val == 'accepted': return 'color: #00dcdc;'
                    return ''
                st.dataframe(df_dec_show.style.map(color_outcome, subset=['outcome']), hide_index=True, width="stretch", height=210)
            else:
                st.info("O agente ainda não executou ações de despacho.")

        with details_right:
            st.markdown("🚨 **Log do Ambiente (Últimos Eventos)**")
            if not df_ev.empty:
                df_ev_show = df_ev.iloc[::-1].head(15).drop(columns=['time_h'], errors='ignore')
                st.dataframe(df_ev_show, hide_index=True, width="stretch", height=280)
            else:
                st.info("Sem eventos registados até ao momento.")

    # -----------------------------------------------------------------
    # ABA 2: MODO FOCO
    # -----------------------------------------------------------------
    with tab_foco:
        render_header_and_controls("mf")
        
        f_col1, f_col2, f_col3, f_col4 = st.columns([2.5, 2.2, 2.2, 2.2])
        with f_col1: navio_escolhido = st.selectbox("Selecionar Navio", options=vessel_ids, key="tab_focus_vessel", label_visibility="collapsed")
        with f_col2: isolar_foco = st.toggle("Ocultar restantes", value=True, key="tog_iso_foco")
        with f_col3: linhas_foco = st.checkbox("Trajetórias", value=True, key="chk_lin_foco")
        with f_col4: rastos_foco = st.checkbox("Rastos", value=True, key="chk_rst_foco")

        dados_nav = next((v for v in frame["vessels"] if v["id"] == navio_escolhido), None)
        v_type_str = dados_nav.get("type", "unknown").capitalize() if dados_nav else "N/A"
        raw_st = dados_nav["status"] if dados_nav else "unknown"
        alvo_id = dados_nav.get("assigned_incident_id") if dados_nav else None

        if raw_st == "transit": estado_atual_str = f"Em trânsito ➔ Inc {alvo_id}"
        elif raw_st == "on_site": estado_atual_str = f"No local ➔ Inc {alvo_id}"
        elif raw_st == "repairing": estado_atual_str = f"A reparar ➔ Inc {alvo_id}"
        elif raw_st == "returning": estado_atual_str = "Regresso ao Porto"
        elif raw_st == "idle": estado_atual_str = "Inativo (Porto)"
        else: estado_atual_str = raw_st.capitalize()

        dist_atual_km, dist_total_km = 0.0, 0.0
        incidentes_atendidos = set()
        prev_lat, prev_lon = None, None
        for idx_t, t in enumerate(timeline):
            v_step = next((x for x in t["vessels"] if x["id"] == navio_escolhido), None)
            if v_step:
                if prev_lat is not None and prev_lon is not None:
                    step_d = haversine_km(prev_lat, prev_lon, v_step["lat"], v_step["lon"])
                    dist_total_km += step_d
                    if idx_t <= st.session_state.step_slider: dist_atual_km += step_d
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

        st.divider()

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric(f"Estado ({v_type_str})", estado_atual_str)
        k2.metric("Distância Navegada", f"{dist_atual_km:.0f} km", f"Total: {dist_total_km:.0f} km", delta_color="off")
        k3.metric("Taxa de Ocupação", f"{taxa_ocupacao:.1f}%", f"{horas_ativas:.1f}h ativas", delta_color="off")
        k4.metric("Tempo em Trânsito", f"{horas_transit:.1f} h", f"{(horas_transit / total_ep_h) * 100:.1f}% do ep.", delta_color="off")
        k5.metric("Tempo no Local", f"{horas_work:.1f} h", f"{len(incidentes_atendidos)} inc. visitados", delta_color="off")
        k6.metric("Horas Inativas", f"{horas_idle:.1f} h", delta_color="off")

        deck_foco, _, _, _, _ = build_fast_map_and_tables(
            episode=episode, timeline=timeline, step_idx=st.session_state.step_slider,
            navio_focado=navio_escolhido, isolar_entidades=isolar_foco,
            mostrar_linhas=linhas_foco, mostrar_rastos=rastos_foco, map_height=460
        )

        f_map_col, f_stat_col = st.columns([2, 1])
        with f_map_col:
            st.pydeck_chart(deck_foco, key="deck_mf")

        with f_stat_col:
            st.markdown(f"**Telemetria de {navio_escolhido}**")
            if not df_gantt_nav.empty:
                df_dist_est = df_gantt_nav.groupby("Estado", as_index=False)["Duracao_num"].sum()
                donut_nav = alt.Chart(df_dist_est).mark_arc(innerRadius=45).encode(
                    theta=alt.Theta("Duracao_num:Q"),
                    color=alt.Color("Estado:N", scale=alt.Scale(domain=["idle", "transit", "on_site", "repairing", "returning"], range=["#555555", "#0078ff", "#00dcdc", "#ff8c00", "#9b59b6"]), legend=alt.Legend(orient="bottom", title="")),
                    tooltip=["Estado", "Duracao_num"]
                ).properties(height=220)
                st.altair_chart(donut_nav, width="stretch")

    # -----------------------------------------------------------------
    # ABA 3: COMPARAÇÃO
    # -----------------------------------------------------------------
    with tab_comp:
        render_header_and_controls("ce")
        
        compatible = {k: c for k, c in bundles.items() if c["episode"]["scenario"]["id"] == scenario["id"] and c["episode"]["scenario"]["world_seed"] == scenario["world_seed"]}
        comp_pool = compatible if len(compatible) >= 2 else bundles
        pool_keys = list(comp_pool.keys())

        default_base_idx = pool_keys.index(episode_id) if episode_id in pool_keys else 0
        default_target_idx = 1 if (len(pool_keys) > 1 and default_base_idx == 0) else 0

        sel_col1, sel_col2 = st.columns(2)
        with sel_col1: base_id = st.selectbox("Episódio Base", options=pool_keys, index=default_base_idx, format_func=lambda v: bundle_label(comp_pool[v]), key="comp_base_sel")
        with sel_col2: target_id = st.selectbox("Episódio Alvo", options=pool_keys, index=default_target_idx, format_func=lambda v: bundle_label(comp_pool[v]), key="comp_target_sel")

        ep_base = comp_pool[base_id]["episode"]
        ep_target = comp_pool[target_id]["episode"]
        m_base = ep_base["metrics"]
        m_target = ep_target["metrics"]

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

        cc1, cc2, _ = st.columns([1.5, 1.5, 7])
        with cc1: comp_linhas = st.checkbox("Trajetórias", value=True, key="chk_lin_comp")
        with cc2: comp_rastos = st.checkbox("Rastos", value=False, key="chk_rst_comp")

        deck_base, df_nav_b, _, _, frame_base = build_fast_map_and_tables(
            episode=ep_base, timeline=ep_base["timeline"], step_idx=st.session_state.step_slider,
            mostrar_linhas=comp_linhas, mostrar_rastos=comp_rastos, map_height=430
        )
        deck_target, df_nav_t, _, _, frame_target = build_fast_map_and_tables(
            episode=ep_target, timeline=ep_target["timeline"], step_idx=st.session_state.step_slider,
            mostrar_linhas=comp_linhas, mostrar_rastos=comp_rastos, map_height=430
        )

        map_col_base, map_col_target = st.columns(2)
        with map_col_base:
            st.markdown(f"**Base:** `{ep_base['policy']['label']}`")
            st.pydeck_chart(deck_base, key="deck_comp_base")
        with map_col_target:
            st.markdown(f"**Alvo:** `{ep_target['policy']['label']}`")
            st.pydeck_chart(deck_target, key="deck_comp_target")

    if st.session_state.playing:
        time.sleep(0.08)
        st.rerun(scope="fragment")

render_interactive_workspace()