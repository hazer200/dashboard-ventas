import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import warnings
import sqlite3
import os

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dashboard de Ventas & Proyecciones", layout="wide", initial_sidebar_state="expanded")

# ==================== CONFIGURACIÓN & CACHÉ ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ventas.db")

# Buscar el Excel en la carpeta del script o en la carpeta padre
_excel_candidates = [
    os.path.join(BASE_DIR, "Base_datos_proy (1).xlsx"),
    os.path.join(BASE_DIR, "Base_datos_proy.xlsx"),
    os.path.join(os.path.dirname(BASE_DIR), "Base_datos_proy (1).xlsx"),
    os.path.join(os.path.dirname(BASE_DIR), "Base_datos_proy.xlsx"),
]
EXCEL_PATH = next((p for p in _excel_candidates if os.path.exists(p)), _excel_candidates[1])

@st.cache_data(ttl=3600)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM ventas", conn)
    conn.close()
    return df

@st.cache_data(ttl=3600)
def prepare_time_series(df):
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Costo Total'] = df['UN'] * df['costo']
    df['Precio Total'] = df['UN'] * df['Precio']
    df['Ganancia'] = df['Precio Total'] - df['Costo Total']
    df['Porcentaje de Ganancia'] = (df['Ganancia'] / df['Precio Total'] * 100).fillna(0)
    
    monthly = df.set_index('Fecha')['UN'].resample('MS').sum().to_frame('sales')
    monthly['year'] = monthly.index.year
    monthly['month'] = monthly.index.month
    monthly['quarter'] = monthly.index.quarter
    
    for i in range(1, 4):
        monthly[f'sales_lag_{i}'] = monthly['sales'].shift(i)
    monthly['rolling_mean_3m'] = monthly['sales'].shift(1).rolling(window=3).mean()
    monthly.dropna(inplace=True)
    return monthly

@st.cache_data(ttl=3600)
def train_models(monthly_df):
    X = monthly_df[['year', 'month', 'quarter', 'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'rolling_mean_3m']]
    y = monthly_df['sales']
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        'Regresión Polinómica': make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    }
    
    metrics = {}
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        metrics[name] = {
            'MAE': mean_absolute_error(y, preds),
            'MSE': mean_squared_error(y, preds),
            'RMSE': np.sqrt(mean_squared_error(y, preds)),
            'R2': r2_score(y, preds),
            'model': model
        }
    return metrics, X.columns.tolist()

@st.cache_data(ttl=3600)
def forecast_best_model(_model, X_cols, monthly_df, months_ahead=20):
    future_dates = pd.date_range(start=monthly_df.index.max() + pd.DateOffset(months=1), periods=months_ahead, freq='MS')
    extended = monthly_df['sales'].copy()
    predictions = []
    
    for date in future_dates:
        lag1 = extended.loc[date - pd.DateOffset(months=1)] if date - pd.DateOffset(months=1) in extended.index else np.nan
        lag2 = extended.loc[date - pd.DateOffset(months=2)] if date - pd.DateOffset(months=2) in extended.index else np.nan
        lag3 = extended.loc[date - pd.DateOffset(months=3)] if date - pd.DateOffset(months=3) in extended.index else np.nan
        roll = extended.loc[:date - pd.DateOffset(months=1)].tail(3).mean()
        
        feat = pd.DataFrame([[date.year, date.month, (date.month-1)//3+1, lag1, lag2, lag3, roll]], columns=X_cols)
        pred = _model.predict(feat)[0]
        predictions.append(pred)
        extended.loc[date] = pred
        
    return pd.Series(predictions, index=future_dates)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Configuración")
    run_etl_btn = st.button("🔄 Ejecutar ETL desde Excel", type="primary")
    
    if run_etl_btn:
        with st.spinner("Procesando datos iterativamente..."):
            from etl import run_etl
            success = run_etl(EXCEL_PATH)
            if success:
                st.success("✅ ETL completada. Datos actualizados en SQLite.")
                st.cache_data.clear()
    
    st.divider()
    st.info("💡 **Instrucciones:**\n1. Ejecuta la ETL la primera vez.\n2. Los modelos se entrenan automáticamente.\n3. Usa las pestañas para responder cada pregunta.")

# ==================== CARGA DE DATOS ====================
df = load_data()
if df.empty:
    st.error("🚫 No hay datos en la base de datos. Ejecuta la ETL desde el sidebar.")
    st.stop()

monthly_df = prepare_time_series(df)
metrics, X_cols = train_models(monthly_df)
best_model_name = max(metrics, key=lambda k: metrics[k]['R2'])
best_model = metrics[best_model_name]['model']
future_forecast = forecast_best_model(best_model, X_cols, monthly_df)

# ==================== UI PRINCIPAL ====================
st.title("📊 Dashboard de Análisis y Proyección de Ventas")
st.caption(f"Último dato histórico: {monthly_df.index.max().strftime('%Y-%m-%d')} | Modelo seleccionado: **{best_model_name}** (R²: {metrics[best_model_name]['R2']:.4f})")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ Sublíneas", "2️⃣ Mundos", "3️⃣ Estacionalidad", 
    "4️⃣ Modelos & Métricas", "5️⃣ Ganancia por Producto", "6️⃣ Costo Proyectado Nov-Dic 2027"
])

with tab1:
    st.header("📦 Ventas por Sublínea")
    sublineas = df.groupby("Mat:SubLinea")["UN"].sum().sort_values(ascending=False).reset_index()
    top10 = sublineas.head(10)
    
    fig1 = px.bar(top10, x="UN", y="Mat:SubLinea", orientation="h", 
                  title="Top 10 Sublíneas por Unidades Vendidas",
                  labels={"UN": "Unidades", "Mat:SubLinea": "Sublínea"},
                  color="UN", color_continuous_scale="Viridis")
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    st.metric(label="Total de Sublíneas", value=sublineas.shape[0])
    st.success(f"🏆 Mayor ventas: **{top10.iloc[0]['Mat:SubLinea']}** ({top10.iloc[0]['UN']:,} unidades)")

with tab2:
    st.header("🌍 Distribución por Mat:Mundo")
    mundos = df.groupby("Mat:Mundo")["UN"].sum().reset_index()
    
    fig2 = px.pie(mundos, values="UN", names="Mat:Mundo", 
                  title="Participación de Ventas por Mundo",
                  hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)
    
    top_mundo = mundos.sort_values("UN", ascending=False).iloc[0]
    st.success(f"🏆 Mundo líder: **{top_mundo['Mat:Mundo']}** ({top_mundo['UN']:,} unidades)")

with tab3:
    st.header("📈 Estacionalidad de Ventas")
    monthly_season = monthly_df.groupby("month")["sales"].agg(["mean", "std", "min", "max"]).reset_index()
    month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    monthly_season["month_name"] = monthly_season["month"].apply(lambda x: month_names[x-1])
    
    fig3 = px.line(monthly_season, x="month_name", y="mean", 
                   title="Promedio Mensual de Ventas (Estacionalidad)",
                   labels={"month_name": "Mes", "mean": "Promedio Unidades"},
                   markers=True, color_discrete_sequence=["#1f77b4"])
    fig3.update_yaxes(title="Unidades Vendidas (Promedio)")
    st.plotly_chart(fig3, use_container_width=True)
    
    peak = monthly_season.loc[monthly_season["mean"].idxmax()]
    low = monthly_season.loc[monthly_season["mean"].idxmin()]
    st.info(f"📊 **Estacionalidad detectada:**\n- 🟢 Mes Pico: **{peak['month_name']}** ({peak['mean']:.0f} uds)\n- 🔴 Mes Bajo: **{low['month_name']}** ({low['mean']:.0f} uds)\n- 📉 Variación: **{((peak['mean']/low['mean'])-1)*100:.1f}%**")

with tab4:
    st.header("🤖 Comparación de Modelos de Proyección")
    df_metrics = pd.DataFrame(metrics).T.drop(columns=["model"])
    df_metrics.index.name = "Modelo"
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_metrics.style.format("{:.2f}"), use_container_width=True)
        
    with col2:
        fig4 = px.bar(df_metrics.reset_index(), x="Modelo", y="R2", 
                      title="Precisión por Modelo (R²)",
                      labels={"R2": "R-Squared", "Modelo": "Modelo"},
                      color="R2", color_continuous_scale="RdYlGn")
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
        
    st.subheader("📉 Histórico vs Proyección")
    hist_plot = pd.DataFrame({"Fecha": monthly_df.index, "Histórico": monthly_df["sales"]})
    future_plot = pd.DataFrame({"Fecha": future_forecast.index, "Proyección": future_forecast})
    combined = pd.concat([hist_plot, future_plot])
    
    fig5 = px.line(combined, x="Fecha", y=["Histórico", "Proyección"],
                   title=f"Proyección de Ventas ({best_model_name})",
                   labels={"value": "Unidades", "Fecha": "Fecha"})
    fig5.update_traces(mode='lines+markers')
    st.plotly_chart(fig5, use_container_width=True)

with tab5:
    st.header("💰 Productos con Mayor Ganancia Proyectada")
    # Calcular proporciones históricas por producto y mes
    hist_prop = df.groupby([df["Fecha"].dt.month.rename("month"), "Mat:Tipo Articulo"])["UN"].sum()
    month_totals = hist_prop.groupby(level=0).sum()
    hist_prop = (hist_prop / month_totals).reset_index(name="proportion")
    
    # Proyectar ganancias para los próximos 6 meses
    future_months = pd.Series(pd.date_range(start=monthly_df.index.max() + pd.DateOffset(months=1), periods=6, freq='MS'))
    proj_data = []
    for f_date in future_months:
        m = f_date.month
        total_pred = future_forecast.get(f_date, 0)
        month_props = hist_prop[hist_prop["month"] == m]
        for _, row in month_props.iterrows():
            pred_un = total_pred * row["proportion"]
            prod_data = df[df["Mat:Tipo Articulo"] == row["Mat:Tipo Articulo"]]
            avg_cost = prod_data["costo"].mean()
            avg_price = prod_data["Precio"].mean()
            ganancia = pred_un * (avg_price - avg_cost)
            proj_data.append({"Fecha": f_date, "Producto": row["Mat:Tipo Articulo"], 
                              "UN_Proyectadas": pred_un, "Ganancia_Proyectada": ganancia})
            
    df_proj = pd.DataFrame(proj_data)
    top_ganancia = df_proj.groupby("Producto")["Ganancia_Proyectada"].sum().nlargest(10).reset_index()
    
    fig6 = px.bar(top_ganancia, x="Ganancia_Proyectada", y="Producto", orientation="h",
                  title="Top 10 Productos por Ganancia Proyectada (6 meses)",
                  labels={"Ganancia_Proyectada": "Ganancia Total Proyectada ($)", "Producto": "Tipo de Artículo"},
                  color="Ganancia_Proyectada", color_continuous_scale="Greens")
    fig6.update_layout(showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)

with tab6:
    st.header("📅 Costo de Producción Proyectado (Nov-Dic 2027)")
    target_dates = [pd.Timestamp("2027-11-01"), pd.Timestamp("2027-12-01")]
    costs_data = []
    
    for f_date in target_dates:
        m = f_date.month
        total_pred = future_forecast.get(f_date, 0)
        month_props = hist_prop[hist_prop["month"] == m]
        for _, row in month_props.iterrows():
            pred_un = total_pred * row["proportion"]
            avg_cost = df[df["Mat:Tipo Articulo"] == row["Mat:Tipo Articulo"]]["costo"].mean()
            costs_data.append({"Mes": f_date.strftime("%Y-%m"), "Producto": row["Mat:Tipo Articulo"], 
                               "UN": pred_un, "Costo_Total": pred_un * avg_cost})
               
    df_costs = pd.DataFrame(costs_data)
    top10_nov_dec = df_costs.groupby("Producto")["Costo_Total"].sum().nlargest(10).reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        nov_data = df_costs[df_costs["Mes"] == "2027-11"].groupby("Producto")["Costo_Total"].sum().nlargest(5).reset_index()
        fig7 = px.bar(nov_data, x="Costo_Total", y="Producto", orientation="h",
                      title="Costo Proyectado - Noviembre 2027 (Top 5)",
                      color="Costo_Total", color_continuous_scale="Oranges")
        fig7.update_layout(showlegend=False)
        st.plotly_chart(fig7, use_container_width=True)
        
    with col2:
        dec_data = df_costs[df_costs["Mes"] == "2027-12"].groupby("Producto")["Costo_Total"].sum().nlargest(5).reset_index()
        fig8 = px.bar(dec_data, x="Costo_Total", y="Producto", orientation="h",
                      title="Costo Proyectado - Diciembre 2027 (Top 5)",
                      color="Costo_Total", color_continuous_scale="Reds")
        fig8.update_layout(showlegend=False)
        st.plotly_chart(fig8, use_container_width=True)
        
    st.dataframe(df_costs[df_costs["Producto"].isin(top10_nov_dec["Producto"])].sort_values(["Mes", "Costo_Total"], ascending=[True, False]), 
                 use_container_width=True, height=300)
    st.caption("📊 *Los costos se calculan usando el costo unitario promedio histórico ponderado por la proyección de ventas del modelo.*")

# Footer
st.divider()
st.caption("Dashboard generado con Streamlit + Plotly + SQLite | Modelo iterativo de proyección temporal")