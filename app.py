"""ARC Episode Viewer - Dashboard Operacional para Simulação de Cabos Submarinos."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
import altair as alt

from arc_student import BundleValidationError, load_bundle, load_bundle_bytes

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_bundles"

def bundle_label(bundle: dict) -> str:
    episode = bundle["episode"]
    scenario = episode["scenario"]
    policy = episode["policy"]
    return f"{scenario['id']} | {policy['label']} | seed {scenario['world_seed']}"

def metric(metrics: dict, name: str, default=0):
    value = metrics.get(name, default)
    return default if value is None else value

# --- CONFIGURAÇÃO DA PÁGINA E CSS ---
st.set_page_config(page_title="ARC Episode Viewer", layout="wide")
st.title("ARC Episode Viewer")
st.caption("Visualização de cenários ARC. Leitura de ficheiros JSON.")

st.markdown("""
    <style>
    .mapboxgl-ctrl-bottom-left, .mapboxgl-ctrl-bottom-right { display: none !important; }
    div[data-baseweb="select"] { height: auto !important; }
    div[data-baseweb="select"] * { white-space: normal !important; word-break: break-word !important; text-overflow: unset !important; }
    div[data-baseweb="popover"] ul[role="listbox"] li, 
    div[data-baseweb="popover"] ul[role="listbox"] li * { white-space: normal !important; word-break: break-word !important; text-overflow: unset !important; padding-top: 8px; padding-bottom: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DE DADOS ---
bundles: dict[str, dict] = {}
errors = []
for path in sorted(SAMPLE_DIR.glob("*.json")):
    try:
        bundle = load_bundle(path)
        bundles[bundle["episode"]["id"]] = bundle
    except BundleValidationError as error:
        errors.append(f"{path.name}: {error}")

uploaded_files = st.sidebar.file_uploader("Adicionar episode bundles", type=["json"], accept_multiple_files=True)
for uploaded in uploaded_files:
    try:
        bundle = load_bundle_bytes(uploaded.getvalue(), uploaded.name)
        bundles[bundle["episode"]["id"]] = bundle
    except BundleValidationError as error:
        errors.append(f"{uploaded.name}: {error}")

for error in errors: st.error(error)
if not bundles:
    st.warning("Não existem bundles válidos para apresentar.")
    st.stop()

episode_id = st.sidebar.selectbox("Episódio", options=list(bundles), format_func=lambda value: bundle_label(bundles[value]))
bundle = bundles[episode_id]
episode = bundle["episode"]
scenario = episode["scenario"]
policy = episode["policy"]
metrics = episode["metrics"]

# --- TÍTULO E MÉTRICAS GLOBAIS ---
st.header(f"Cenário: {scenario['id']}")
st.subheader(f"Política: {policy['label']}", divider="gray")

columns = st.columns(6)
columns[0].metric("Resolvidos", f"{int(metric(metrics, 'n_resolved'))}/{int(metric(metrics, 'n_incidents'))}")
columns[1].metric("Taxa de resolução", f"{100 * float(metric(metrics, 'resolution_rate')):.1f}%")
columns[2].metric("Primeira resposta", f"{float(metric(metrics, 'mean_first_response_time_h')):.1f} h")
columns[3].metric("Tempo de resolução", f"{float(metric(metrics, 'mean_time_to_resolution_h')):.1f} h")
columns[4].metric("Despachos duplicados", int(metric(metrics, "duplicate_dispatch_count")))
columns[5].metric("Severidade por resolver", f"{float(metric(metrics, 'unresolved_severity')):.2f}")
st.divider()

# --- ESTADO TEMPORAL E NAVEGAÇÃO ---
timeline = episode["timeline"]
max_frame = len(timeline) - 1

if "step_slider" not in st.session_state: st.session_state.step_slider = 0
if "playing" not in st.session_state: st.session_state.playing = False

if st.session_state.playing:
    if st.session_state.step_slider < max_frame: st.session_state.step_slider += 1
    else: st.session_state.playing = False

def click_play():
    if st.session_state.step_slider >= max_frame: st.session_state.step_slider = 0 
    st.session_state.playing = not st.session_state.playing

def click_reset():
    st.session_state.step_slider = 0
    st.session_state.playing = False 

st.slider("Navegação Temporal", min_value=0, max_value=max_frame, key="step_slider", label_visibility="collapsed")

col_play, col_reset, _ = st.columns([1.5, 1.5, 7])
with col_play: st.button("⏸ Pausa" if st.session_state.playing else "▶ Reproduzir", on_click=click_play, width="stretch")
with col_reset: st.button("⏮ Reset", on_click=click_reset, width="stretch")

frame = timeline[st.session_state.step_slider]

# --- ABA LATERAL: FOCUS MODE ---
st.sidebar.divider()
st.sidebar.caption("**🎯 FOCUS MODE**")

vessel_ids = [v["id"] for v in frame["vessels"]]
navio_focado = st.sidebar.selectbox("Selecionar Navio", options=["Visão Global"] + vessel_ids, label_visibility="collapsed", key="focus_sel")

# 1. CRIAR CAIXA DE PROTEÇÃO PARA O MENU LATERAL (Evita duplicações)
focus_ui = st.sidebar.empty()

# --- ABA LATERAL: DETALHES DO EPISÓDIO ---
# Imprimimos os detalhes já, para ficarem sempre no fundo de forma fixa
st.sidebar.divider()
st.sidebar.caption("**📋 DETALHES DO EPISÓDIO**")
st.sidebar.caption(f"**Duração:** {timeline[-1]['time_h']:.1f}h | **Passos:** {len(timeline)}")
st.sidebar.caption(f"**Seed:** {scenario['world_seed']} | **Split:** {scenario['split']}")
st.sidebar.caption(f"**Banco:** {scenario['bank_id']} | **Base:** {policy['kind']}")

# Variáveis base de controlo
dados_navio_focado = None
incidente_alvo_do_foco = None
isolar_entidades = False
mostrar_linhas = False
mostrar_rastos = False

# Injetar o menu no espaço reservado se estiver em Focus Mode
if navio_focado != "Visão Global":
    dados_navio_focado = next((v for v in frame["vessels"] if v["id"] == navio_focado), None)
    if dados_navio_focado:
        incidente_alvo_do_foco = dados_navio_focado.get("assigned_incident_id")
        raw_status = dados_navio_focado["status"]
        
        if raw_status == "transit": state_str = f"Em trânsito ➔ {incidente_alvo_do_foco}"
        elif raw_status == "on_site": state_str = f"No local ➔ {incidente_alvo_do_foco}"
        elif raw_status == "repairing": state_str = f"A reparar ➔ {incidente_alvo_do_foco}"
        elif raw_status == "returning": state_str = "A regressar ao Porto"
        elif raw_status == "idle": state_str = "A aguardar ordens"
        else: state_str = raw_status.capitalize()
        
        # O ".container()" dentro do ".empty()" garante que nada se duplica
        with focus_ui.container():
            st.markdown(f"**Estado:** {state_str}")
            st.markdown(f"**Posição:** {dados_navio_focado['lat']:.3f}, {dados_navio_focado['lon']:.3f}")
            st.divider()
            st.caption("**CONTROLOS DO NAVIO**")
            isolar_entidades = st.toggle("Ocultar restantes", value=True, key="tog_iso")
            mostrar_linhas = st.checkbox("Mostrar Trajetória", value=True, key="chk_lin_foc")
            mostrar_rastos = st.checkbox("Mostrar Rasto", value=False, key="chk_rst_foc")
else:
    # Se mudar para visão global, o Python esvazia os botões antigos automaticamente
    focus_ui.empty() 

st.divider()

# --- CONTROLOS DO MAPA (TOPO) ---
col_step, col_options = st.columns([2, 8])

with col_step:
    st.markdown(f"<div style='padding-top: 6px;'><b>Passo {frame['step']} ({frame['time_h']:.1f}h)</b></div>", unsafe_allow_html=True)

top_ui = col_options.empty()
# mostrar_heatmap = False

if navio_focado == "Visão Global":
    with top_ui.container():
        c1, c2, c3, _ = st.columns([2.2, 2.2, 2.5, 5.1])
        with c1: mostrar_linhas = st.checkbox("Trajetórias globais", value=True, key="chk_lin_glob")
        with c2: mostrar_rastos = st.checkbox("Rastos globais", value=False, key="chk_rst_glob")
        # with c3: mostrar_heatmap = st.checkbox("🔥 Heatmap Incidentes", value=False, key="chk_heat_glob")
else:
    top_ui.empty()

# --- AGRUPAMENTO DE ENTIDADES ---
location_groups = {}
def add_to_group(lat, lon, item_data):
    key = (round(lat, 4), round(lon, 4))
    if key not in location_groups: location_groups[key] = []
    location_groups[key].append(item_data)

# 1. Navios
for vessel in frame["vessels"]:
    v_type = vessel.get("type", "unknown")
    raw_status = vessel["status"]
    assigned_id = vessel.get("assigned_incident_id")
    
    # REVERTIDO: "on_site" volta a significar puramente "No local"
    if raw_status == "transit" and assigned_id is not None: status_detail = f"Em trânsito para o Incidente {assigned_id}"
    elif raw_status == "on_site" and assigned_id is not None: status_detail = f"No local do Incidente {assigned_id}"
    elif raw_status == "repairing" and assigned_id is not None: status_detail = f"A reparar o Incidente {assigned_id}"
    elif raw_status == "returning": status_detail = "A regressar ao Porto"
    elif raw_status == "idle": status_detail = "Inativo (A aguardar ordens)"
    else: status_detail = raw_status.capitalize() + (f" (Incidente {assigned_id})" if assigned_id else "")
            
    color = [0, 120, 255, 255] if v_type == "patrol" else [255, 140, 0, 255] if v_type == "repair" else [150, 150, 150, 255]
    
    ignorar = False
    if navio_focado != "Visão Global" and vessel["id"] != navio_focado:
        if isolar_entidades: ignorar = True 
        else: color = [120, 120, 120, 60]   
            
    if not ignorar:
        add_to_group(vessel["lat"], vessel["lon"], {"id": vessel["id"], "kind": f"Navio ({v_type.capitalize()})", "status": status_detail, "color": color, "priority": 2})

for incident in frame["incidents"]:
    raw_status = incident["status"]
    if raw_status == "open": color, status_detail = [255, 40, 40, 255], "Aberto (A aguardar)"
    elif raw_status in ["on_site", "repairing"]: color, status_detail = [0, 220, 220, 255], "Navio no local" if raw_status == "on_site" else "Em reparação"
    elif raw_status == "resolved": color, status_detail = [100, 100, 100, 120], "Resolvido"
    else: color, status_detail = [100, 100, 100, 120], raw_status.capitalize()

    ignorar = False
    if navio_focado != "Visão Global" and incident["id"] != incidente_alvo_do_foco:
        if isolar_entidades: ignorar = True
        else: color = [120, 120, 120, 60]

    if not ignorar: add_to_group(incident["lat"], incident["lon"], {"id": f"incident_{incident['id']}", "kind": "Incidente", "status": status_detail, "color": color, "priority": 1})

for port in episode["map"].get("ports", []):
    ignorar = False
    color = [255, 255, 255, 255]
    if navio_focado != "Visão Global":
        if isolar_entidades: ignorar = True
        else: color = [150, 150, 150, 40]
    if not ignorar: add_to_group(port["lat"], port["lon"], {"id": port["id"], "kind": "Porto", "status": "Static", "color": color, "priority": 3})
    
for point in episode["map"].get("landing_points", []):
    ignorar = False
    color = [155, 89, 182, 255]
    if navio_focado != "Visão Global":
        if isolar_entidades: ignorar = True
        else: color = [120, 80, 150, 40]
    if not ignorar: add_to_group(point["lat"], point["lon"], {"id": point["id"], "kind": "Landing Point", "status": "Static", "color": color, "priority": 4})

# --- PREPARAÇÃO DAS CAMADAS ---
map_rows, table_rows, line_rows, trail_rows = [], [], [], []

for (lat, lon), items in location_groups.items():
    items.sort(key=lambda x: x["priority"])
    tooltip_parts = [f"ID: {it['id']}\n{it['kind']}\nEstado: {it['status']}" for it in items]
    for it in items: table_rows.append({"lat": lat, "lon": lon, "id": it["id"], "kind": it["kind"], "status": it["status"]})
    map_rows.append({"lat": lat, "lon": lon, "count": str(len(items)) if len(items) > 1 else "", "color": items[0]["color"], "tooltip_text": "\n-------------------------\n".join(tooltip_parts)})

if mostrar_linhas:
    for vessel in frame["vessels"]:
        if navio_focado != "Visão Global" and vessel["id"] != navio_focado: continue
        assigned_id = vessel.get("assigned_incident_id")
        if vessel["status"] == "transit" and assigned_id is not None:
            alvo = next((inc for inc in frame["incidents"] if inc["id"] == assigned_id), None)
            if alvo:
                start_lon, start_lat = vessel["lon"], vessel["lat"]
                end_lon, end_lat = alvo["lon"], alvo["lat"]
                num_dashes = 12 
                for i in range(num_dashes):
                    f1, f2 = i / num_dashes, (i + 0.5) / num_dashes 
                    line_rows.append({
                        "start": [start_lon + (end_lon - start_lon) * f1, start_lat + (end_lat - start_lat) * f1],
                        "end": [start_lon + (end_lon - start_lon) * f2, start_lat + (end_lat - start_lat) * f2],
                        "color": [255, 235, 150, 70] 
                    })

if mostrar_rastos:
    trails_data = {}
    for t in timeline[:st.session_state.step_slider + 1]:
        for v in t["vessels"]:
            if navio_focado != "Visão Global" and v["id"] != navio_focado: continue
            vid = v["id"]
            if vid not in trails_data: trails_data[vid] = []
            trails_data[vid].append([v["lon"], v["lat"]])
    for vid, coords in trails_data.items():
        if len(coords) > 1:
            trail_rows.append({"path": coords, "color": [255, 255, 255, 80]})

df_map = pd.DataFrame(map_rows)
df_table = pd.DataFrame(table_rows)
df_lines = pd.DataFrame(line_rows)
df_trails = pd.DataFrame(trail_rows)

# --- RENDERIZAÇÃO: MAPA E TABELAS GERAIS ---
left, right = st.columns([2, 1])
with left:
    layers_list = [
        pdk.Layer("ScatterplotLayer", data=df_map, get_position="[lon, lat]", get_fill_color="color", get_radius=60000, radius_min_pixels=8, radius_max_pixels=18, pickable=True),
        pdk.Layer("TextLayer", data=df_map, get_position="[lon, lat]", get_text="count", get_size=13, get_color=[255, 255, 255, 255], get_text_anchor='"middle"', get_alignment_baseline='"center"', pickable=False)
    ]
    if not df_lines.empty: layers_list.insert(0, pdk.Layer("LineLayer", data=df_lines, get_source_position="start", get_target_position="end", get_color="color", get_width=3, pickable=False))
    if not df_trails.empty: layers_list.insert(0, pdk.Layer("PathLayer", data=df_trails, get_path="path", get_color="color", width_scale=20, width_min_pixels=2, get_width=2, pickable=False))

    #if mostrar_heatmap:
           # heat_data = [{"lat": inc["lat"], "lon": inc["lon"]} for inc in frame["incidents"]]
            #if heat_data:
                #layer_heat = pdk.Layer(
                    #"HeatmapLayer",
                    #data=pd.DataFrame(heat_data),
                    #get_position="[lon, lat]",
                    #get_weight=1,
                    #radius_pixels=60,
                    #opacity=0.8
               # )
               # layers_list.append(layer_heat)
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
            "overflow": "visible"
        }
    }
    
    center_lat, center_lon, map_zoom = 20, -40, 2
    if navio_focado != "Visão Global" and dados_navio_focado:
        center_lat, center_lon = dados_navio_focado["lat"], dados_navio_focado["lon"]
        map_zoom = 3 
            
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=map_zoom, pitch=0)
    st.pydeck_chart(pdk.Deck(layers=layers_list, initial_view_state=view_state, tooltip=tooltip, map_style="dark"))
    
    st.markdown("""
    <div style="font-size: 0.9em; color: #a0a0a0; padding-top: 15px; text-align: center;">
        <span style="color: rgb(0, 120, 255);">●</span> Patrulha &nbsp;&nbsp;<span style="color: rgb(255, 140, 0);">●</span> Reparação &nbsp;&nbsp;<span style="color: rgb(255, 40, 40);">●</span> Alerta Aberto &nbsp;&nbsp;<span style="color: rgb(0, 220, 220);">●</span> A Reparar &nbsp;&nbsp;<span style="color: rgb(255, 255, 255);">●</span> Porto &nbsp;&nbsp;<span style="color: rgb(155, 89, 182);">●</span> Landing Point
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("**Estado no instante atual**")
    
    df_navios = df_table[df_table["kind"].str.contains("Navio")].copy()
    df_incidentes = df_table[df_table["kind"] == "Incidente"].copy()
    df_infra = df_table[df_table["kind"].isin(["Porto", "Landing Point"])].copy()
    
    for df in [df_navios, df_incidentes, df_infra]:
        if not df.empty: df["Coords"] = df.apply(lambda row: f"{row['lat']:.3f}, {row['lon']:.3f}", axis=1)

    df_navios = df_navios[["id", "status", "kind", "Coords"]].rename(columns={"id": "Navio", "status": "Estado", "kind": "Tipo"}) if not df_navios.empty else pd.DataFrame(columns=["Navio", "Estado", "Tipo", "Coords"])
    if not df_navios.empty: df_navios["Tipo"] = df_navios["Tipo"].str.replace(r"Navio \(|\)", "", regex=True)
    
    df_incidentes = df_incidentes[["id", "status", "Coords"]].rename(columns={"id": "Incidente", "status": "Estado"}) if not df_incidentes.empty else pd.DataFrame(columns=["Incidente", "Estado", "Coords"])
    df_infra = df_infra[["id", "kind", "Coords"]].rename(columns={"id": "Nome", "kind": "Tipo"}) if not df_infra.empty else pd.DataFrame(columns=["Nome", "Tipo", "Coords"])
    
    t_nav, t_inc, t_inf = st.tabs([f"⛴️ Navios ({len(df_navios)})", f"🚨 Incidentes ({len(df_incidentes)})", f"🏢 Infra ({len(df_infra)})"])
    with t_nav: st.dataframe(df_navios, hide_index=True, width="stretch", height=400)
    with t_inc: st.dataframe(df_incidentes, hide_index=True, width="stretch", height=400)
    with t_inf: st.dataframe(df_infra, hide_index=True, width="stretch", height=400)

st.divider()

# --- GRÁFICOS DINÂMICOS  ---
chart_rows = [{"time_h": item["time_h"], "open": sum(inc["status"] != "resolved" for inc in item["incidents"]), "resolved": sum(inc["status"] == "resolved" for inc in item["incidents"]), "cumulative_reward": item["cumulative_reward"]} for item in timeline]
df_chart = pd.DataFrame(chart_rows)

def criar_grafico_com_regua(df, y_cols, title, color_range):
    base = alt.Chart(df).transform_fold(y_cols, as_=["Variável", "Valor"]).mark_line().encode(
        x=alt.X("time_h:Q", title="Tempo (h)"), y=alt.Y("Valor:Q", title=""), color=alt.Color("Variável:N", scale=alt.Scale(range=color_range))
    ).properties(title=title, height=250)
    regua = alt.Chart(pd.DataFrame({'x': [frame["time_h"]]})).mark_rule(color='red', strokeWidth=2).encode(x='x:Q')
    return base + regua

chart_left, chart_right = st.columns(2)
with chart_left: st.altair_chart(criar_grafico_com_regua(df_chart, ["open", "resolved"], "Incidentes ao longo do tempo", ["#ff2b2b", "#00dcdc"]), width="stretch")
with chart_right: st.altair_chart(criar_grafico_com_regua(df_chart, ["cumulative_reward"], "Recompensa Acumulada", ["#0078ff"]), width="stretch")

st.divider()

# --- DIAGNÓSTICO DO AGENTE (RECOMPENSAS) ---
st.markdown("**Diagnóstico de Decisão do Agente**")

# 1. Extrair valores absolutos das métricas para calcular percentagens
val_age = abs(metric(metrics, 'incident_age_penalty'))
val_transit = abs(metric(metrics, 'transit_penalty'))
val_adv = abs(metric(metrics, 'unassessed_adversarial_penalty'))
total_pen = val_age + val_transit + val_adv

# 2. Calcular percentagens em formato string para injetar no dashboard
pct_age = f"{(val_age/total_pen)*100:.1f}% do total" if total_pen > 0 else "0%"
pct_transit = f"{(val_transit/total_pen)*100:.1f}% do total" if total_pen > 0 else "0%"
pct_adv = f"{(val_adv/total_pen)*100:.1f}% do total" if total_pen > 0 else "0%"

# 3. Mini-painel de métricas no topo (Com os deltas a mostrar a percentagem)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Recompensa Total", f"{metric(metrics, 'total_reward'):.2f}")
m2.metric("Penalidade: Idade SLA", f"{metric(metrics, 'incident_age_penalty'):.2f}", pct_age, delta_color="off")
m3.metric("Penalidade: Trânsito", f"{metric(metrics, 'transit_penalty'):.2f}", pct_transit, delta_color="off")
m4.metric("Penalidade: Adversarial", f"{metric(metrics, 'unassessed_adversarial_penalty'):.2f}", pct_adv, delta_color="off")

# 4. Gráfico 1: Recompensa Instantânea (O "Pulso" do Agente)
step_rewards = [{"Tempo (h)": t["time_h"], "Recompensa no Passo": t["reward"]} for t in timeline]
df_step = pd.DataFrame(step_rewards)

base_step = alt.Chart(df_step).mark_bar(color="#ff4b4b", size=2.5).encode(
    x=alt.X("Tempo (h):Q", title="Tempo (h)"),
    y=alt.Y("Recompensa no Passo:Q", title="Penalização Instantânea"),
    tooltip=[alt.Tooltip("Tempo (h)"), alt.Tooltip("Recompensa no Passo", format=".4f")]
).properties(height=250, title="Sangramento de Pontos por Passo (Pulso)")

regua_step = alt.Chart(pd.DataFrame({'x': [frame["time_h"]]})).mark_rule(color='white', strokeWidth=2).encode(x='x:Q')

# 5. Gráfico 2: Donut de Distribuição
df_donut = pd.DataFrame({
    "Componente": ["Idade do Incidente", "Trânsito", "Risco Adversarial"],
    "Valor": [val_age, val_transit, val_adv]
})
df_donut = df_donut[df_donut["Valor"] > 0] # Remover zeros para não estragar o gráfico

donut_chart = alt.Chart(df_donut).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field="Valor", type="quantitative"),
    color=alt.Color(field="Componente", type="nominal", scale=alt.Scale(scheme="set2"), legend=alt.Legend(orient="bottom", title="")),
    tooltip=["Componente", alt.Tooltip("Valor", format=".2f")]
).properties(height=250, title="Peso Absoluto das Falhas")

# 6. Renderizar os dois gráficos lado a lado
c1, c2 = st.columns([2.5, 1.5])
with c1: st.altair_chart((base_step + regua_step), use_container_width=True)
with c2: st.altair_chart(donut_chart, use_container_width=True)

st.divider()

# --- GRÁFICO DE GANTT ---
st.markdown("Cronograma de Operações")

gantt_blocks = []
for v_id in vessel_ids:
    current_status = None
    start_time = 0
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
                        "Duração": f"{duracao:.1f}h"
                    })
                current_status = status
                start_time = t["time_h"]
    
    # Fechar o último bloco no fim do episódio
    if current_status is not None:
        duracao = timeline[-1]["time_h"] - start_time
        gantt_blocks.append({
            "Navio": v_id, 
            "Estado": current_status, 
            "Início": start_time, 
            "Fim": timeline[-1]["time_h"],
            "Duração": f"{duracao:.1f}h"
        })

if gantt_blocks:
    df_gantt = pd.DataFrame(gantt_blocks)
    
    domain_estados = ["idle", "transit", "on_site", "repairing", "returning"]
    range_cores = ["#555555", "#0078ff", "#00dcdc", "#ff8c00", "#9b59b6"]
    
    altura_dinamica = max(150, len(vessel_ids) * 50 + 60)
    
    gantt_base = alt.Chart(df_gantt).mark_bar(cornerRadius=4, size=24).encode(
        x=alt.X("Início:Q", title="Tempo (h)"),
        x2="Fim:Q",
        y=alt.Y("Navio:N", title="", sort=vessel_ids, axis=alt.Axis(grid=False, tickSize=0, domain=False, labelPadding=10, labelFontSize=12)),
        color=alt.Color("Estado:N", scale=alt.Scale(domain=domain_estados, range=range_cores), legend=alt.Legend(title="", orient='top', labelFontSize=12)),
        tooltip=[
            alt.Tooltip("Navio:N", title="Navio"),
            alt.Tooltip("Estado:N", title="Estado"),
            alt.Tooltip("Início:Q", title="Início (h)", format=".1f"),
            alt.Tooltip("Fim:Q", title="Fim (h)", format=".1f"),
            alt.Tooltip("Duração:N", title="Duração total")
        ]
    ).properties(height=altura_dinamica)
    
    regua_gantt = alt.Chart(pd.DataFrame({'x': [frame["time_h"]]})).mark_rule(color='red', strokeWidth=2).encode(x='x:Q')
    
    st.altair_chart((gantt_base + regua_gantt), use_container_width=True)

# --- TABELAS HISTÓRICAS DE ACONTECIMENTOS ---
hist_decisions, hist_events = [], []
for t in timeline[:st.session_state.step_slider + 1]:
    for d in t.get("decisions", []):
        if d.get("action_type") != "hold": hist_decisions.append({"Passo": t["step"], "Tempo": f"{t['time_h']:.1f}h", **d})
    for e in t.get("events", []): hist_events.append({"Passo": t["step"], "Tempo": f"{t['time_h']:.1f}h", **e})

df_dec = pd.DataFrame(hist_decisions)
if not df_dec.empty: df_dec = df_dec.iloc[::-1][["Passo", "Tempo", "agent_id", "action_type", "target_incident_id", "outcome"]]
df_ev = pd.DataFrame(hist_events)
if not df_ev.empty: df_ev = df_ev.iloc[::-1]

details_left, details_right = st.columns(2)
with details_left:
    st.markdown("**Últimas Decisões Relevantes** *(Ações de despacho)*")
    st.dataframe(df_dec, hide_index=True, width="stretch", height=250)
with details_right:
    st.markdown("**Registo de Acontecimentos** *(Surgimento de Alarmes)*")
    st.dataframe(df_ev, hide_index=True, width="stretch", height=250)

st.divider()

# --- COMPARAÇÃO DE POLÍTICAS ---
st.subheader("Comparar episódios")
compatible = {k: c for k, c in bundles.items() if c["episode"]["scenario"]["id"] == scenario["id"] and c["episode"]["scenario"]["world_seed"] == scenario["world_seed"]}
selected_for_comparison = st.multiselect("Episódios com o mesmo cenário e seed", options=list(compatible), default=list(compatible), format_func=lambda value: compatible[value]["episode"]["policy"]["label"])
comparison_rows = [{"policy": compatible[sel]["episode"]["policy"]["label"], "total_reward": metric(compatible[sel]["episode"]["metrics"], "total_reward", None), "resolution_rate": metric(compatible[sel]["episode"]["metrics"], "resolution_rate", None), "mean_first_response_time_h": metric(compatible[sel]["episode"]["metrics"], "mean_first_response_time_h", None), "mean_time_to_resolution_h": metric(compatible[sel]["episode"]["metrics"], "mean_time_to_resolution_h", None), "unresolved_severity": metric(compatible[sel]["episode"]["metrics"], "unresolved_severity", None), "duplicate_dispatch_count": metric(compatible[sel]["episode"]["metrics"], "duplicate_dispatch_count", None)} for sel in selected_for_comparison]
st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")

with st.expander("Metadados e métricas completas"): st.json({"generator": bundle["generator"], "scenario": scenario, "policy": policy, "metrics": metrics})

# --- GATILHO DA ANIMAÇÃO ---
if st.session_state.playing:
    time.sleep(0.1) 
    st.rerun()