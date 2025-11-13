import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import geopandas as gpd
import xgboost as xgb
from statsmodels.tsa.seasonal import seasonal_decompose
import unidecode
import re
import os

# --- 1. Inicializar la App de Dash ---
app = dash.Dash(__name__)
server = app.server
app.title = "Dashboard de Homicidios en Colombia"

# --- 2. RUTAS DE ARCHIVOS (Rutas relativas corregidas) ---
csv_file = "Dataset/HOMICIDIO_20251026.csv"
shapefile_path = "Dataset/MGN_ADM_DPTO_POLITICO.shp"
model_mean_file = "xgb_final_model_mean.json"
model_lower_file = "xgb_final_model_lower.json"
model_upper_file = "xgb_final_model_upper.json"


# --- 3. Función de Limpieza de Nombres (Validada) ---
def limpiar_nombre(nombre):
    nombre_limpio = unidecode.unidecode(str(nombre)).upper()
    nombre_limpio = re.sub(r'[.,]', '', nombre_limpio)
    if 'BOGOTA' in nombre_limpio: return 'BOGOTA'
    if 'VALLE DEL CAUCA' in nombre_limpio: return 'VALLE DEL CAUCA'
    if 'SAN ANDRES ISLAS' in nombre_limpio: return 'SAN ANDRES ISLAS'
    if 'SAN ANDRES PROVIDENCIA' in nombre_limpio: return 'SAN ANDRES ISLAS'
    return nombre_limpio.strip()

# --- 4. Carga y Preparación de Datos (¡SOLO LIGERO EN INICIO!) ---
print("Iniciando carga de datos LIGEROS...")
df_raw = pd.read_csv(csv_file)

# --- Limpieza y Estandarización de columnas ---
df_raw.columns = df_raw.columns.str.strip()
df_raw.columns = df_raw.columns.str.upper()

# --- Procesamiento inicial (ligero) ---
df_raw['FECHA HECHO'] = pd.to_datetime(df_raw['FECHA HECHO'], format='%d/%m/%Y')
df_raw['SEXO'] = df_raw['SEXO'].replace({'NO REPORTA': 'NO INFORMADO', 'SIN ESTABLECER': 'NO INFORMADO'})
df_agregado = df_raw.groupby('DEPARTAMENTO')['CANTIDAD'].sum().reset_index()
df_agregado['NOMBRE_LIMPIO'] = df_agregado['DEPARTAMENTO'].apply(limpiar_nombre)

# --- Preparación para la Serie de Tiempo (ML) ---
df_monthly = df_raw.set_index('FECHA HECHO').resample('MS')['CANTIDAD'].sum().to_frame()
df_ml = df_monthly.copy()
df_ml['mes'] = df_ml.index.month
df_ml['año'] = df_ml.index.year
df_ml['lag_1'] = df_ml['CANTIDAD'].shift(1)
df_ml['lag_12'] = df_ml['CANTIDAD'].shift(12)
df_ml['media_movil_3'] = df_ml['CANTIDAD'].shift(1).rolling(window=3).mean()
df_ml = df_ml.dropna()
y_full = df_ml['CANTIDAD']
X_full = df_ml.drop('CANTIDAD', axis=1)

# --- KPIs y Tablas descriptivas ---
total_homicidios = df_raw['CANTIDAD'].sum()
fecha_inicio = df_raw['FECHA HECHO'].min().strftime('%Y-%m-%d')
fecha_fin = df_raw['FECHA HECHO'].max().strftime('%Y-%m-%d')
kpi_depto_max_data = df_agregado.loc[df_agregado['CANTIDAD'].idxmax()]
kpi_depto_min_data = df_agregado.loc[df_agregado['CANTIDAD'].idxmin()]
kpi_depto_max_nombre = kpi_depto_max_data['DEPARTAMENTO']
kpi_depto_max_valor = f"{kpi_depto_max_data['CANTIDAD']:,.0f}"
kpi_depto_min_nombre = kpi_depto_min_data['DEPARTAMENTO']
kpi_depto_min_valor = f"{kpi_depto_min_data['CANTIDAD']:,.0f}"
df_null_counts = df_raw.isnull().sum().reset_index()
df_null_counts.columns = ['Variable', 'Conteo de Nulos']
df_head = df_raw.head(10)
df_desc_numeric_raw = df_raw.describe(include=[np.number]).T.reset_index().round(2)
df_desc_numeric = df_desc_numeric_raw.rename(columns={
    'index': 'Variable', 'count': 'Conteo', 'mean': 'Media', 'std': 'Desv. Est.',
    'min': 'Mínimo', '25%': 'Q1 (25%)', '50%': 'Mediana (50%)', '75%': 'Q3 (75%)', 'max': 'Máximo'
})
df_desc_categ_raw = df_raw.describe(include=['object']).T.reset_index()
df_desc_categ = df_desc_categ_raw.rename(columns={
    'index': 'Variable', 'count': 'Conteo', 'unique': 'Valores Únicos',
    'top': 'Moda (Valor Más Frec.)', 'freq': 'Frecuencia de la Moda'
})
data_metrics_xgb = {
    'Métrica': ['MAPE Promedio (Rolling)', 'MAE Promedio (Rolling)', 'RMSE Promedio (Rolling)'],
    'Valor': ['4.81 %', '53.49 Homicidios', '65.05 Homicidios']
}
df_metrics_xgb = pd.DataFrame(data_metrics_xgb)
print("¡Datos ligeros cargados y procesados!")

# --- 5. ESTILOS VISUALES UNIFICADOS Y PROFESIONALES ---
COLORS = {
    'background': '#F8FAFC', 'text': '#1E293B', 'primary': '#0F4C75',
    'secondary': '#3282B8', 'accent': '#BBE1FA', 'card_bg': '#FFFFFF',
    'border': '#E2E8F0',
    'sarima': '#EF553B',
    'xgboost': '#00CC96'
}
template_estilo = "plotly_white"
kpi_card_style = {
    'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '18px',
    'textAlign': 'center', 'margin': '10px', 'width': '30%', 'display': 'inline-block',
    'boxShadow': '0px 2px 8px rgba(0,0,0,0.08)', 'border': f'1px solid {COLORS["border"]}'
}
kpi_title_style = {'fontSize': '16px', 'color': COLORS['secondary'], 'fontWeight': '600'}
kpi_value_style = {'fontSize': '28px', 'fontWeight': 'bold', 'color': COLORS['primary']}
kpi_card_style_2_col = kpi_card_style.copy()
kpi_card_style_2_col['width'] = '45%'
kpi_card_style_2_col['padding'] = '10px'
kpi_title_style_small = kpi_title_style.copy()
kpi_title_style_small['fontSize'] = '14px'
kpi_value_style_small = kpi_value_style.copy()
kpi_value_style_small['fontSize'] = '22px'
kpi_row_style = {
    'display': 'flex', 'justifyContent': 'space-around',
    'flexWrap': 'nowrap', 'marginBottom': '20px'
}
table_style = {
    'overflowX': 'auto', 'marginTop': '20px',
    'border': f'1px solid {COLORS["border"]}',
    'backgroundColor': COLORS['card_bg']
}
cell_style = {
    'textAlign': 'left', 'padding': '8px',
    'border': f'1px solid {COLORS["border"]}',
    'fontSize': '14px', 'color': COLORS['text']
}
header_style = {
    'fontWeight': 'bold', 'backgroundColor': COLORS['accent'],
    'border': f'1px solid {COLORS["border"]}',
    'fontSize': '15px', 'color': COLORS['primary']
}
tabs_styles = {
    'background': COLORS['background'],
    'color': COLORS['text'], 'height': '60px'
}
tab_style = {
    'padding': '18px', 'fontWeight': 'bold',
    'border': f'1px solid {COLORS["border"]}',
    'backgroundColor': COLORS['card_bg'], 'color': COLORS['secondary'],
}
tab_selected_style = {
    'padding': '18px', 'borderTop': f'3px solid {COLORS["primary"]}',
    'borderBottom': 'none', 'borderLeft': f'1px solid {COLORS["border"]}',
    'borderRight': f'1px solid {COLORS["border"]}',
    'backgroundColor': COLORS['background'], 'color': COLORS['primary'],
    'fontWeight': 'bold'
}

# --- 6. Creación de Gráficos LIGEROS (EDA 1 y Metodología) ---
print("Generando figuras EDA LIGERAS...")

# --- 6a. Gráficos para EDA 1 ---
fig_top_deptos = px.bar(
    df_raw.groupby('DEPARTAMENTO')['CANTIDAD'].sum().nlargest(10).reset_index().sort_values(by='CANTIDAD', ascending=True),
    x='CANTIDAD', y='DEPARTAMENTO', orientation='h',
    title='Top 10 Departamentos por Homicidios', template=template_estilo
)
fig_zona = px.pie(
    df_raw['ZONA'].value_counts().reset_index(), values='count', names='ZONA',
    title='Distribución por Zona', template=template_estilo
)
fig_sexo = px.pie(
    df_raw['SEXO'].value_counts().reset_index(), values='count', names='SEXO',
    title='Distribución por Sexo', template=template_estilo
)
df_zona_sexo = df_raw.groupby(['ZONA', 'SEXO'])['CANTIDAD'].sum().reset_index()
fig_zona_sexo = px.bar(
    df_zona_sexo, x='ZONA', y='CANTIDAD', color='SEXO',
    title='Homicidios por Zona y Sexo', barmode='stack', template=template_estilo
)

# --- BLOQUE 1: GRÁFICOS DE DESCOMPOSICIÓN SEPARADOS ---
print("Generando figuras de Descomposición...")
decomposition = seasonal_decompose(df_monthly['CANTIDAD'], model='additive', period=12)

# Figura 1: Observado
fig_deco_obs = go.Figure()
fig_deco_obs.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observado', line=dict(color=COLORS['secondary'])))
fig_deco_obs.update_layout(title="Descomposición: Observado", height=250, margin=dict(t=50, b=30, l=30, r=30))

# Figura 2: Tendencia
fig_deco_trend = go.Figure()
fig_deco_trend.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Tendencia', line=dict(color='#D62728')))
fig_deco_trend.update_layout(title="Descomposición: Tendencia", height=250, margin=dict(t=50, b=30, l=30, r=30))

# Figura 3: Estacionalidad
fig_deco_seas = go.Figure()
fig_deco_seas.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Estacionalidad', line=dict(color='#2CA02C')))
fig_deco_seas.update_layout(title="Descomposición: Estacionalidad", height=250, margin=dict(t=50, b=30, l=30, r=30))

# Figura 4: Residuo
fig_deco_resid = go.Figure()
fig_deco_resid.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='markers', name='Residuo', marker=dict(color='#9467BD', size=3, opacity=0.7)))
fig_deco_resid.update_layout(title="Descomposición: Residuo", height=250, margin=dict(t=50, b=30, l=30, r=30))
# --- FIN BLOQUE 1 ---

# --- GRÁFICO DE COMPARACIÓN DE MODELOS (Basado en datos manuales) ---
print("Generando gráfico comparativo de modelos...")
models_data = {
    'Modelo': ['SARIMA', 'XGBoost', 'SARIMA', 'XGBoost', 'SARIMA', 'XGBoost'],
    'Métrica': ['MAE', 'MAE', 'RMSE', 'RMSE', 'MAPE (%)', 'MAPE (%)'],
    'Valor': [57.72, 53.49, 72.80, 65.05, 5.17, 4.81]
}
df_compare = pd.DataFrame(models_data)

fig_model_compare = px.bar(
    df_compare, x='Métrica', y='Valor', color='Modelo', barmode='group',
    title='Comparativa de Rendimiento: SARIMA vs XGBoost (Rolling Forecast)',
    text='Valor',
    color_discrete_map={'SARIMA': COLORS['sarima'], 'XGBoost': COLORS['xgboost']},
    template=template_estilo
)
fig_model_compare.update_traces(textposition='outside')
fig_model_compare.update_layout(legend_title_text='Modelo')
# --- FIN GRÁFICO COMPARATIVO ---

# --- 6d. Aplicar estilo a figuras LIGERAS ---
for fig in [fig_top_deptos, fig_zona, fig_sexo, fig_zona_sexo,
            fig_deco_obs, fig_deco_trend, fig_deco_seas, fig_deco_resid,
            fig_model_compare]:
    fig.update_layout(
        plot_bgcolor=COLORS['card_bg'],
        paper_bgcolor=COLORS['card_bg'],
        font=dict(color=COLORS['text'], family="Roboto"),
        title_font=dict(color=COLORS['primary'], size=18),
    )

print("¡Figuras ligeras listas! Definiendo layout...")

# =========================================================================
# === FUNCIONES PESADAS (Carga perezosa) ===
# =========================================================================

# -------------------------------------------------------------------------
# ---- FUNCIÓN PESADA 1: CARGA GEOGRÁFICA Y MAPA (PESTAÑA 7.b) -----------
# -------------------------------------------------------------------------
def generate_mapa_content():
    """Carga GeoPandas y genera la figura del mapa (consumo de RAM alto)."""
    print("-> INICIANDO CARGA PESADA: GEOPANDAS y MAPA...")

    # Carga del Shapefile (¡Mueve esto aquí!)
    gdf = gpd.read_file(shapefile_path)

    # Procesamiento para el mapa
    # df_agregado y limpiar_nombre se definieron globalmente para evitar recálculos
    gdf['NOMBRE_LIMPIO'] = gdf['dpto_cnmbr'].apply(limpiar_nombre)
    gdf_mapa = gdf.merge(df_agregado, on='NOMBRE_LIMPIO', how='left')

    # Generación de la figura del mapa
    fig_mapa = px.choropleth_mapbox(
        gdf_mapa,
        geojson=gdf_mapa.geometry,
        locations=gdf_mapa.index,
        color="CANTIDAD",
        color_continuous_scale="Plasma",
        mapbox_style="open-street-map",
        zoom=4,
        center={"lat": 4.5709, "lon": -74.2973},
        opacity=0.6,
        hover_name="dpto_cnmbr",
        title="Homicidios Totales por Departamento (Mapa Interactivo)"
    )
    fig_mapa.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        paper_bgcolor=COLORS['card_bg'],
        mapbox_style="open-street-map",
        font=dict(color=COLORS['text'], family="Roboto"),
        title_font=dict(color=COLORS['primary'], size=18),
        coloraxis_colorbar=dict(title='Homicidios', tickfont=dict(color=COLORS['text']))
    )
    print("-> CARGA PESADA: MAPA COMPLETADA.")
    return fig_mapa


# -------------------------------------------------------------------------
# ---- FUNCIÓN PESADA 2: CARGA DE MODELOS Y PREDICCIÓN (PESTAÑA 7.c) ----
# -------------------------------------------------------------------------
def generate_model_content(y_full, X_full, df_ml):
    """Carga modelos XGBoost y genera las figuras de predicción/residuales."""
    print("-> INICIANDO CARGA PESADA: MODELOS XGBOOST y PREDICCIÓN...")

    # Carga de Modelos (¡Mueve esto aquí!)
    model_mean = xgb.XGBRegressor()
    model_mean.load_model(model_mean_file)
    model_lower = xgb.XGBRegressor()
    model_lower.load_model(model_lower_file)
    model_upper = xgb.XGBRegressor()
    model_upper.load_model(model_upper_file)

    # Lógica de predicción
    n_steps_future = 3
    future_dates = pd.date_range(start=df_ml.index.max() + pd.DateOffset(months=1), periods=n_steps_future, freq='MS')
    predictions_mean = []
    predictions_lower = []
    predictions_upper = []
    data_recursive = df_ml.copy()

    # --- Tu código de forecasting recursivo COMPLETO ---
    for date in future_dates:
        last_data = data_recursive.iloc[-12:, :]
        lag_1 = data_recursive['CANTIDAD'].iloc[-1]
        lag_12 = data_recursive['CANTIDAD'].iloc[-12]
        media_movil_3 = data_recursive['CANTIDAD'].iloc[-3:].mean()
        features = {'mes': date.month, 'año': date.year, 'lag_1': lag_1, 'lag_12': lag_12, 'media_movil_3': media_movil_3}
        features_df = pd.DataFrame(features, index=[date])
        pred_mean = model_mean.predict(features_df)[0]
        pred_lower = model_lower.predict(features_df)[0]
        pred_upper = model_upper.predict(features_df)[0]
        predictions_mean.append(pred_mean)
        predictions_lower.append(pred_lower)
        predictions_upper.append(pred_upper)
        new_row = features_df.copy()
        new_row['CANTIDAD'] = pred_mean
        data_recursive = pd.concat([data_recursive, new_row[data_recursive.columns]])
    # --- FIN del código de forecasting recursivo ---

    forecast_series_mean = pd.Series(predictions_mean, index=future_dates)
    forecast_series_lower = pd.Series(predictions_lower, index=future_dates)
    forecast_series_upper = pd.Series(predictions_upper, index=future_dates)
    last_date_hist = y_full.index[-1]
    last_value_hist = y_full.iloc[-1]
    connected_forecast_mean = pd.concat([pd.Series([last_value_hist], index=[last_date_hist]), forecast_series_mean])
    connected_forecast_lower = pd.concat([pd.Series([last_value_hist], index=[last_date_hist]), forecast_series_lower])
    connected_forecast_upper = pd.concat([pd.Series([last_value_hist], index=[last_date_hist]), forecast_series_upper])

    # Generación de fig_prediccion
    fig_prediccion = go.Figure()
    fig_prediccion.add_trace(go.Scatter(x=y_full.index, y=y_full, mode='lines', name='Datos Históricos (Reales)'))
    fig_prediccion.add_trace(go.Scatter(x=connected_forecast_lower.index, y=connected_forecast_lower, fill=None, mode='lines', line=dict(color='limegreen', width=0), showlegend=False))
    fig_prediccion.add_trace(go.Scatter(x=connected_forecast_upper.index, y=connected_forecast_upper, fill='tonexty', mode='lines', line=dict(color='limegreen', width=0), name='Intervalo de Confianza 95%', showlegend=True, fillcolor='rgba(152, 251, 152, 0.3)'))
    fig_prediccion.add_trace(go.Scatter(x=connected_forecast_mean.index, y=connected_forecast_mean, mode='lines', name='Predicción XGBoost (Fin de 2025)', line=dict(color='limegreen', dash='dot', width=3)))
    fig_prediccion.update_layout(title='Predicción Final del Modelo Campeón (XGBoost)', xaxis_title='Año', yaxis_title='Cantidad de Homicidios', hovermode="x unified")
    zoom_start_date = y_full.index[-24]
    zoom_end_date = future_dates[-1] + pd.DateOffset(months=2)
    fig_prediccion.update_xaxes(range=[zoom_start_date, zoom_end_date])

    # Generación de fig_residuales
    train_predictions = model_mean.predict(X_full)
    residuals = y_full - train_predictions
    residuals_df = pd.DataFrame({'Fecha': y_full.index, 'Residuales': residuals})
    fig_residuales = px.scatter(
        residuals_df, x='Fecha', y='Residuales',
        title='Análisis de Residuales (Errores) del Modelo',
        template=template_estilo
    )
    fig_residuales.add_hline(y=0, line_dash="dash", line_color=COLORS['primary'])
    fig_residuales.update_layout(xaxis_title='Año', yaxis_title='Error (Residual)')

    print("-> CARGA PESADA: MODELOS COMPLETADA.")
    return fig_prediccion, fig_residuales

# --- 7. Definir el Layout (Estructura Académica) ---
app.layout = html.Div([
    html.H1(
        "Dashboard: Análisis y Predicción de Homicidios en Colombia",
        style={
            'textAlign': 'center',
            'color': COLORS['primary'],
            'fontWeight': 'bold',
            'marginBottom': '20px'
        }
    ),
    dcc.Tabs(id="tabs-principales", value='tab-7', style=tabs_styles, children=[

        # Pestañas 1-6
        dcc.Tab(label='1. Introducción', value='tab-1', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div(style={'padding': '20px'}, children=[html.H2('Introducción')])
        ]),
        dcc.Tab(label='2. Contexto', value='tab-2', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div(style={'padding': '20px'}, children=[html.H2('Contexto')])
        ]),
        dcc.Tab(label='3. Planteamiento del problema', value='tab-3', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div(style={'padding': '20px'}, children=[html.H2('Planteamiento del problema')])
        ]),
        dcc.Tab(label='4. Objetivos y justificación', value='tab-4', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div(style={'padding': '20px'}, children=[html.H2('Objetivos y justificación')])
        ]),
        dcc.Tab(label='6. Metodología', value='tab-6', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H2('Metodología'),
                dcc.Tabs(id="tabs-metodologia", value='sub-tab-6a', children=[

                    # --- Sub-Pestaña 6a: Definición del Problema ---
                    dcc.Tab(label='a. Definición del Problema', value='sub-tab-6a', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div(style={'padding': '20px'}, children=[
                            html.H3('Definición del Problema a Resolver'),
                            html.P('El problema se define como una tarea de pronóstico de series de tiempo (Time Series Forecasting).'),
                            html.P('Esto implica utilizar datos históricos para predecir valores futuros en una secuencia ordenada por tiempo. El objetivo no es explicar por qué ocurren (causalidad), sino *cuántos* se espera que ocurran (predicción).'),
                            html.Hr(style={'margin': '20px 0'}),
                            html.H4('Variable Objetivo y Alcance', style={'color': COLORS['secondary']}),
                            html.P('La variable objetivo o de interés es la **CANTIDAD** total de homicidios.'),
                            html.Ul([
                                html.Li('Granularidad: La serie de tiempo se agrega a nivel mensual para capturar patrones estacionales y tendencias a mediano plazo.'),
                                html.Li('Alcance: El modelo es de carácter nacional, prediciendo el número total de homicidios en Colombia.'),
                                html.Li('Horizonte de Predicción: El objetivo es pronosticar los **próximos 3 meses** (Octubre, Noviembre y Diciembre de 2025).')
                            ])
                        ])
                    ]),

                    # --- Sub-Pestaña 6b: Preparación de los Datos ---
                    dcc.Tab(label='b. Preparación de los Datos', value='sub-tab-6b', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div(style={'padding': '20px'}, children=[
                            html.H3('Preparación de los Datos'),
                            html.P('Para construir un modelo confiable, los datos crudos (331,026 registros) pasaron por un proceso riguroso de 4 etapas:'),
                            html.Hr(style={'margin': '20px 0'}),
                            html.H4('1. Limpieza de Datos Crudos', style={'color': COLORS['secondary']}),
                            html.H5('Verificación de Nulos', style={'marginTop': '15px'}),
                            html.P('Se confirmó que el dataset estaba completo. La siguiente tabla muestra que no hay valores nulos en ninguna columna:'),
                            dash_table.DataTable(
                                data=df_null_counts.to_dict('records'),
                                columns=[{'name': i, 'id': i} for i in df_null_counts.columns],
                                style_table={**table_style, 'maxWidth': '500px', 'margin': 'auto'},
                                style_cell=cell_style,
                                style_header=header_style
                            ),
                            html.H5('Conversión de Fechas', style={'marginTop': '15px'}),
                            html.P('La columna "FECHA HECHO" (tipo object) se transformó a formato datetime (evidencia del código y resultado):'),
                            html.Pre(html.Code(f"""
# Conversión de la columna de fecha
df_raw['FECHA HECHO'] = pd.to_datetime(df_raw['FECHA HECHO'], format='%d/%m/%Y')

# Resultado:
# Fecha mínima: {fecha_inicio}
# Fecha máxima: {fecha_fin}
""", style={'fontFamily': 'monospace'}), style={'backgroundColor': '#F8FAFC', 'padding': '10px', 'borderRadius': '5px', 'border': f'1px solid {COLORS["border"]}', 'overflowX': 'auto'}),
                            html.Hr(style={'margin': '20px 0'}),
                            html.H4('2. Transformación a Serie de Tiempo', style={'color': COLORS['secondary']}),
                            html.P('El dataset granular se transformó en una serie de tiempo nacional para el análisis de forecasting.'),
                            html.H5('Agregación Mensual (Resampling)', style={'marginTop': '15px'}),
                            html.Pre(html.Code(f"""
# Agrupar por Mes (MS: Month Start) y sumar
df_monthly = df_raw.set_index('FECHA HECHO') \
                    .resample('MS')['CANTIDAD'].sum().to_frame()

# Resultado: 273 meses (desde {fecha_inicio} hasta {fecha_fin})
""", style={'fontFamily': 'monospace'}), style={'backgroundColor': '#F8FAFC', 'padding': '10px', 'borderRadius': '5px', 'border': f'1px solid {COLORS["border"]}', 'overflowX': 'auto'}),
                            html.H5('Análisis de Componentes', style={'marginTop': '15px'}),
                            html.P('Se descompuso la serie para analizar visualmente sus componentes:'),
                            dcc.Graph(figure=fig_deco_obs),
                            dcc.Graph(figure=fig_deco_trend),
                            dcc.Graph(figure=fig_deco_seas),
                            dcc.Graph(figure=fig_deco_resid),
                            html.H5('Prueba de Estacionariedad (ADF)', style={'marginTop': '15px'}),
                            html.P('Se aplicó la prueba de Dickey-Fuller Aumentada. El resultado confirmó que la serie ES ESTACIONARIA:'),
                            html.Pre(html.Code("""
--- Prueba de Dickey-Fuller Aumentada (ADF Test) ---
Estadístico ADF: -3.8393
p-value: 0.0025

--- Interpretación ---
El p-value (0.0025) es menor o igual a 0.05.
Conclusión: La serie ES ESTACIONARIA.
""", style={'fontFamily': 'monospace'}), style={'backgroundColor': '#F8FAFC', 'padding': '10px', 'borderRadius': '5px', 'border': f'1px solid {COLORS["border"]}', 'overflowX': 'auto'}),
                            html.Hr(style={'margin': '20px 0'}),
                            html.H4('3. Ingeniería de Características (XGBoost)', style={'color': COLORS['secondary']}),
                            html.P('A diferencia de SARIMA, los modelos de Machine Learning como XGBoost no entienden la secuencia temporal. Debemos crear "características" (features) a partir de la fecha y los valores pasados:'),
                            html.Pre(html.Code(f"""
# --- Creación de Features ---
df_ml['mes'] = df_ml.index.month
df_ml['año'] = df_ml.index.year
df_ml['lag_1'] = df_ml['CANTIDAD'].shift(1)
df_ml['lag_12'] = df_ml['CANTIDAD'].shift(12)
df_ml['media_movil_3'] = df_ml['CANTIDAD'].shift(1).rolling(window=3).mean()

# Se eliminan los primeros 12 meses que contienen NaN por los lags
df_ml = df_ml.dropna()
""", style={'fontFamily': 'monospace'}), style={'backgroundColor': '#F8FAFC', 'padding': '10px', 'borderRadius': '5px', 'border': f'1px solid {COLORS["border"]}', 'overflowX': 'auto'}),
                            html.Hr(style={'margin': '20px 0'}),
                            html.H4('4. Estrategia de Validación', style={'color': COLORS['secondary']}),
                            html.P('Se optó por una **Validación Cruzada Rodante (Rolling Forecast)**. Esta técnica simula un escenario real:'),
                            html.Ul([
                                html.Li('Fold 1: Se entrena con datos hasta 2021 para predecir 2022.'),
                                html.Li('Fold 2: Se entrena con datos hasta 2022 para predecir 2023.'),
                                html.Li('Fold 3: Se entrena con datos hasta 2023 para predecir 2024.'),
                            ]),
                            html.P('El rendimiento final (ej. MAPE 4.81%) es el promedio de estos folds, lo que da una medida muy robusta de la estabilidad del modelo.')
                        ])
                    ]),

                    # --- Sub-Pestaña 6c: Selección del Modelo ---
                    dcc.Tab(label='c. Selección del Modelo', value='sub-tab-6c', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div(style={'padding': '20px'}, children=[
                            html.H3('Selección del Modelo o Algoritmo'),
                            html.P('Se evaluaron dos enfoques fundamentalmente distintos para abordar el problema de predicción: un modelo estadístico clásico y un modelo de Machine Learning moderno.'),
                            html.Hr(style={'margin': '20px 0'}),
                            html.H4('1. Modelo Estadístico: SARIMA', style={'color': COLORS['sarima']}),
                            html.P('Seasonal AutoRegressive Integrated Moving Average. Es el estándar de oro en estadística clásica para series con estacionalidad.'),
                            html.Div(children='SARIMA(p,d,q)(P,D,Q)m', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'margin': '15px', 'fontFamily': 'monospace'}),
                            html.P('Se realizó una búsqueda de hiperparámetros (Auto-ARIMA) encontrando la configuración óptima ARIMA(1,1,2)(2,1,0)[12].'),
                            html.P('Resultados de la Validación Cruzada (Rolling):'),
                            html.Ul([
                                html.Li([html.Strong('MAE Promedio: '), '57.72 homicidios']),
                                html.Li([html.Strong('RMSE Promedio: '), '72.80 homicidios']),
                                html.Li([html.Strong('MAPE Promedio: '), '5.17 %']),
                            ]),
                            html.Hr(style={'margin': '20px 0'}),
                            html.H4('2. Modelo de Machine Learning: XGBoost', style={'color': COLORS['xgboost']}),
                            html.P('Extreme Gradient Boosting. Un algoritmo de aprendizaje supervisado basado en árboles de decisión.'),
                            html.Div(children='Predicción = Σ (Resultados de Árboles de Decisión)', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'margin': '15px', 'fontFamily': 'monospace'}),
                            html.P('A diferencia de SARIMA, XGBoost no "ve" el tiempo, sino que aprende relaciones no lineales entre las variables creadas en la ingeniería de características (lags, mes, año).'),
                            html.P('Resultados de la Validación Cruzada (Rolling):'),
                            html.Ul([
                                html.Li([html.Strong('MAE Promedio: '), '53.49 homicidios']),
                                html.Li([html.Strong('RMSE Promedio: '), '65.05 homicidios']),
                                html.Li([html.Strong('MAPE Promedio: '), '4.81 %']),
                            ]),
                            html.Hr(style={'margin': '20px 0'}),
                            html.H4('3. Justificación y Ganador', style={'color': COLORS['secondary']}),
                            html.P('Para seleccionar el modelo final, se comparó el rendimiento promedio en los 3 folds de validación (2022, 2023, 2024).'),
                            dcc.Graph(figure=fig_model_compare),
                            html.H5('Conclusión:', style={'marginTop': '15px'}),
                            html.P([
                                'El modelo ', html.Strong('XGBoost (Verde)', style={'color': COLORS['xgboost']}),
                                ' superó al modelo SARIMA en todas las métricas de error evaluadas. Su capacidad para capturar patrones complejos y no lineales le permitió reducir el error porcentual (MAPE) al 4.81%, haciéndolo la opción más precisa y robusta para este proyecto.'
                            ])
                        ])
                    ]),

                    # --- Sub-Pestaña 6d: Evaluación del Modelo ---
                    dcc.Tab(label='d. Evaluación del Modelo', value='sub-tab-6d', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div(style={'padding': '20px'}, children=[
                            html.H3('Proceso de entrenamiento y Evaluación'),
                            html.P('Entrenamiento realizado sobre el set de entrenamiento.'),
                            html.P('Métricas de evaluación: MAE (Error Absoluto Medio) y MAPE (Error Porcentual Absoluto Medio) para interpretabilidad.'),
                            html.P('Validación utilizada: Rolling Forecast para determinar el modelo más robusto.')
                        ])
                    ]),
                ])
            ])
        ]),

        # Pestaña 7: Resultados y análisis final
        dcc.Tab(label='7. Resultados y análisis final', value='tab-7', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div(id='contenedor-tab-7', style={'padding': '20px'}, children=[
                html.H2('Resultados y Análisis Final'),
                dcc.Tabs(id="tabs-anidadas", value='sub-tab-d', children=[

                    # --- Sub-Pestaña a: EDA 1 (LIGERA) ---
                    dcc.Tab(label='a. EDA 1', value='sub-tab-a', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div(style={'padding': '20px'}, children=[
                            html.H3('EDA 1: Estadísticas Descriptivas y Distribuciones'),
                            html.P('Análisis inicial del conjunto de datos, estadísticas de resumen y distribuciones de variables clave.'),
                            html.Div([
                                html.Div([html.H4("Total Homicidios", style=kpi_title_style), html.P(f"{total_homicidios:,.0f}", style=kpi_value_style)], style=kpi_card_style),
                                html.Div([html.H4("Fecha Inicio", style=kpi_title_style), html.P(fecha_inicio, style=kpi_value_style)], style=kpi_card_style),
                                html.Div([html.H4("Fecha Fin", style=kpi_title_style), html.P(fecha_fin, style=kpi_value_style)], style=kpi_card_style),
                            ], style=kpi_row_style),
                            html.Hr(),
                            html.H4('Vista Previa del Dataset', style={'marginTop': '20px'}),
                            dash_table.DataTable(
                                data=df_head.to_dict('records'),
                                columns=[{'name': i, 'id': i} for i in df_head.columns],
                                style_table=table_style, style_cell=cell_style, style_header=header_style
                            ),
                            html.H4('Estadísticas Descriptivas (Variables Numéricas)', style={'marginTop': '30px'}),
                            dash_table.DataTable(
                                data=df_desc_numeric.to_dict('records'),
                                columns=[{'name': i, 'id': i} for i in df_desc_numeric.columns],
                                style_table=table_style, style_cell=cell_style, style_header=header_style
                            ),
                            html.H4('Estadísticas Descriptivas (Variables Categóricas)', style={'marginTop': '30px'}),
                            dash_table.DataTable(
                                data=df_desc_categ.to_dict('records'),
                                columns=[{'name': i, 'id': i} for i in df_desc_categ.columns],
                                style_table=table_style, style_cell=cell_style, style_header=header_style
                            ),
                            html.Hr(),
                            html.H4('Resumen Visual de Distribuciones', style={'textAlign': 'center', 'marginTop': '30px'}),
                            html.Div([
                                html.Div([dcc.Graph(figure=fig_zona)], style={'width': '49%', 'display': 'inline-block'}),
                                html.Div([dcc.Graph(figure=fig_sexo)], style={'width': '49%', 'display': 'inline-block'})
                            ]),
                            dcc.Graph(figure=fig_zona_sexo),
                            dcc.Graph(figure=fig_top_deptos),
                        ])
                    ]),

                    # --- Sub-Pestaña b: EDA 2 (MAPA - CARGA PEREZOSA) ---
                    dcc.Tab(label='b. EDA 2', value='sub-tab-b', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div(id='mapa-contenedor', children=[
                            html.H3('EDA 2: Análisis Geográfico y Temporal'),
                            html.Div('Cargando mapa interactivo... (Esto puede tardar unos segundos la primera vez)', id='mapa-placeholder', style={'padding': '50px', 'textAlign': 'center'})
                        ])
                    ]),

                    # --- Sub-Pestaña c: Visualización (MODELO - CARGA PEREZOSA) ---
                    dcc.Tab(label='c. Visualización del modelo', value='sub-tab-c', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div(id='modelo-contenedor', children=[
                            html.H3('Visualización de Resultados del Modelo (XGBoost)'),
                            html.Div('Cargando modelos y generando predicción...', id='modelo-placeholder', style={'padding': '50px', 'textAlign': 'center'})
                        ])
                    ]),

                    # --- Sub-Pestaña d: Indicadores (LIGERA) ---
                    dcc.Tab(label='d. Indicadores de Evaluación del Modelo', value='sub-tab-d', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div(style={'padding': '20px'}, children=[
                            html.H3('Indicadores de Evaluación del Modelo (XGBoost)'),
                            html.P('La siguiente tabla muestra el rendimiento promedio de nuestro modelo campeón (XGBoost), validado con la técnica de Rolling Forecast (2022-2024).'),
                            html.H4('Métricas de Robustez del Modelo Campeón', style={'textAlign': 'center', 'marginTop': '30px'}),
                            dash_table.DataTable(
                                data=df_metrics_xgb.to_dict('records'),
                                columns=[{'name': i, 'id': i} for i in df_metrics_xgb.columns],
                                style_table=table_style,
                                style_cell=cell_style,
                                style_header=header_style,
                                style_cell_conditional=[
                                    {'if': {'column_id': 'Métrica'}, 'textAlign': 'left'},
                                    {'if': {'column_id': 'Valor'}, 'textAlign': 'center'}
                                ]
                            ),
                            html.H4('Interpretación de los Indicadores', style={'marginTop': '30px'}),
                            html.P("Los indicadores de error nos dicen, en promedio, qué tan preciso es el modelo:"),
                            html.Ul([
                                html.Li(f"MAPE (4.81%): Esta es la métrica más importante. Significa que, en promedio, las predicciones del modelo tienen un error de solo 4.81% con respecto al valor real. Un error tan bajo indica una precisión muy alta."),
                                html.Li(f"MAE (53.49): En promedio, el modelo se equivoca por ~53 homicidios (hacia arriba o hacia abajo) cada mes. Dado que el promedio mensual es de más de 1000, este error absoluto es bastante bajo."),
                                html.Li(f"RMSE (65.05): Es similar al MAE, pero penaliza más los errores grandes. El hecho de que sea cercano al MAE sugiere que el modelo no comete errores esporádicos gigantescos.")
                            ]),
                            html.P("En conjunto, estas métricas demuestran que el modelo XGBoost es preciso, confiable y estable para el pronóstico.", style={'fontWeight': 'bold'})
                        ])
                    ]),

                    # --- Sub-Pestaña e: Limitaciones (LIGERA) ---
                    dcc.Tab(label='e. Limitaciones', value='sub-tab-e', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div(style={'padding': '20px'}, children=[
                            html.H3('Limitaciones y Consideraciones Finales'),
                            html.P('Todo modelo es una simplificación de la realidad. A pesar de su alta precisión (MAPE 4.81%), es fundamental comprender el alcance y las restricciones de este análisis para una correcta interpretación de los resultados.'),
                            html.Hr(style={'margin': '20px 0'}),
                            html.H4('1. Ausencia de Variables Exógenas (Causales)', style={'color': COLORS['primary']}),
                            html.P('El modelo actual es puramente **autorregresivo**, lo que significa que sus predicciones se basan únicamente en los valores y patrones históricos de los propios homicidios (lags, medias móviles, estacionalidad).'),
                            html.Ul([
                                html.Li('El modelo captura patrones (el "qué" y "cuándo"), pero no entiende las causas raíz (el "por qué").'),
                                html.Li('No considera factores externos cruciales que actúan como "drivers" del crimen, tales como:'),
                                html.Ul([
                                    html.Li('Indicadores socioeconómicos (desempleo, inflación, pobreza).'),
                                    html.Li('Cambios en estrategias de seguridad o políticas públicas.'),
                                    html.Li('Inversión social, conflictos armados o procesos de paz.')
                                ]),
                            ]),
                            html.Hr(style={'margin': '20px 0'}),
                            html.H4('2. Supuesto de Estabilidad de Patrones (Eventos "Cisne Negro")', style={'color': COLORS['primary']}),
                            html.P('Como todo modelo de forecasting, este asume que los patrones observados en el pasado (2003-2025) continuarán de forma similar en el futuro inmediato.'),
                            html.Ul([
                                html.Li('El modelo es excelente para predecir el comportamiento "normal" de la serie.'),
                                html.Li('Ejemplos de estos eventos podrían ser:'),
                                html.Ul([
                                    html.Li('El inicio de una pandemia (como la de 2020).'),
                                    html.Li('Un cese al fuego a nivel nacional inesperado.'),
                                    html.Li('Un desastre natural a gran escala o un evento de agitación social masiva.')
                                ]),
                                html.Li('Si ocurriera un evento de esta magnitud, las predicciones del modelo quedarían invalidadas, ya que no tiene datos históricos de un evento similar para aprender.')
                            ]),
                            html.Hr(style={'margin': '20px 0'}),
                            html.H3('Mejoras Futuras', style={'marginTop': '30px'}),
                            html.P('Para superar estas limitaciones, futuras iteraciones del proyecto podrían explorar:'),
                            html.Ul([
                                html.Li(html.Strong('Modelos con Variables Exógenas (SARIMAX o XGBoost-Exógeno):'),' Incluir datos económicos y demográficos para capturar las causas raíz.'),
                                html.Li(html.Strong('Modelos Geográficos (Panel de Datos):'),' Desarrollar modelos a nivel departamental o municipal para predecir no solo "cuántos" sino "dónde", permitiendo una mejor asignación de recursos.')
                            ])
                        ])
                    ]),
                ])
            ])
        ]),

        # Pestaña 8 (sin cambios)
        dcc.Tab(label='8. Conclusiones', value='tab-8', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div(style={'padding': '20px'}, children=[html.H2('Conclusiones')])
        ]),
    ])
], style={
    'fontFamily': '"Roboto", "Open Sans", "Segoe UI", sans-serif',
    'backgroundColor': COLORS['background'],
    'color': COLORS['text']
})


# =========================================================================
# === CALLBACKS DE CARGA PEREZOSA (Lazy Loading) ===
# =========================================================================

# ----------------- CALLBACK para EDA 2 (MAPA) -----------------
@app.callback(
    Output('mapa-contenedor', 'children'),
    [Input('tabs-anidadas', 'value')]
)
def render_mapa_content(tab_value):
    if tab_value == 'sub-tab-b':
        fig_mapa = generate_mapa_content()

        # Replicamos el contenido estático que originalmente iba en esta pestaña
        return html.Div(style={'padding': '20px'}, children=[
            html.H3('EDA 2: Análisis Geográfico y Temporal'),
            html.P('Análisis de la distribución geográfica de los casos.'),
            html.Div([
                html.Div([html.H4("Depto. con MÁS Homicidios", style=kpi_title_style_small), html.P(f"{kpi_depto_max_nombre} ({kpi_depto_max_valor})", style=kpi_value_style_small)], style=kpi_card_style_2_col),
                html.Div([html.H4("Depto. con MENOS Homicidios", style=kpi_title_style_small), html.P(f"{kpi_depto_min_nombre} ({kpi_depto_min_valor})", style=kpi_value_style_small)], style=kpi_card_style_2_col),
            ], style=kpi_row_style),
            html.Hr(),
            dcc.Graph(figure=fig_mapa)
        ])
    return html.Div()


# ----------------- CALLBACK para Modelo (PREDICCIÓN) -----------------
@app.callback(
    Output('modelo-contenedor', 'children'),
    [Input('tabs-anidadas', 'value')]
)
def render_modelo_content(tab_value):
    if tab_value == 'sub-tab-c':
        fig_prediccion, fig_residuales = generate_model_content(y_full, X_full, df_ml)

        # Replicamos el contenido estático que originalmente iba en esta pestaña
        return html.Div(style={'padding': '20px'}, children=[
            html.H3('Visualización de Resultados del Modelo (XGBoost)'),
            html.P('A continuación se muestra el rendimiento del modelo campeón (XGBoost) para pronosticar los últimos meses del set de datos.'),
            html.Div([
                html.Div([html.H4("Inicio del Pronóstico", style=kpi_title_style_small), html.P("Octubre 2025", style=kpi_value_style_small)], style=kpi_card_style_2_col),
                html.Div([html.H4("Fin del Pronóstico", style=kpi_title_style_small), html.P("Diciembre 2025", style=kpi_value_style_small)], style=kpi_card_style_2_col),
            ], style=kpi_row_style),
            html.Hr(),
            dcc.Graph(figure=fig_prediccion),
            html.Hr(),
            html.H3('Análisis de Residuales'),
            html.P("Los residuales (errores) del modelo deben ser aleatorios y no mostrar patrones. Un gráfico de residuales centrado en cero, como el que se muestra a continuación, indica que el modelo ha capturado con éxito la estructura de los datos."),
            dcc.Graph(figure=fig_residuales)
        ])
    return html.Div()


# --- 8. Correr la App ---
if __name__ == '__main__':
    app.run(debug=True)
