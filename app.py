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
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1️⃣ Sublíneas", "2️⃣ Mundos", "3️⃣ Estacionalidad", 
    "4️⃣ Modelos & Métricas", "5️⃣ Ganancia por Producto", "6️⃣ Costo Proyectado Nov-Dic 2027",
    "7️⃣ Rentabilidad"
])

with tab1:
    st.header("📦 Ventas por Sublínea")
    sublineas = df.groupby("Mat:SubLinea")["UN"].sum().sort_values(ascending=False).reset_index()
    top10 = sublineas.head(10).copy()
    total_un = sublineas["UN"].sum()
    top10["Participación (%)"] = (top10["UN"] / total_un * 100).round(2)
    top10["etiqueta"] = top10["Participación (%)"].apply(lambda x: f"{x:.1f}%")

    fig1 = px.bar(top10, x="UN", y="Mat:SubLinea", orientation="h",
                  title="Top 10 Sublíneas por Unidades Vendidas",
                  labels={"UN": "Unidades", "Mat:SubLinea": "Sublínea"},
                  color="UN", color_continuous_scale="Viridis",
                  text="etiqueta")
    fig1.update_traces(textposition="outside", textfont=dict(size=12))
    fig1.update_layout(showlegend=False, xaxis=dict(range=[0, top10["UN"].max() * 1.15]))
    st.plotly_chart(fig1, use_container_width=True)
    st.metric(label="Total de Sublíneas", value=sublineas.shape[0])
    st.success(f"🏆 Mayor ventas: **{top10.iloc[0]['Mat:SubLinea']}** ({top10.iloc[0]['UN']:,} unidades — {top10.iloc[0]['Participación (%)']:.1f}% del total)")

with tab2:
    st.header("🌍 Distribución por Mat:Mundo")
    mundos = df.groupby("Mat:Mundo")["UN"].sum().reset_index()
    
    fig2 = px.pie(mundos, values="UN", names="Mat:Mundo",
                  title="Participación de Ventas por Mundo",
                  hole=0.35, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig2.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(size=14),
        insidetextorientation="radial"
    )
    fig2.update_layout(
        showlegend=False,
        height=550,
        margin=dict(t=60, b=20, l=20, r=20),
        title_font=dict(size=18)
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    top_mundo = mundos.sort_values("UN", ascending=False).iloc[0]
    st.success(f"🏆 Mundo líder: **{top_mundo['Mat:Mundo']}** ({top_mundo['UN']:,} unidades)")

with tab3:
    st.header("📈 Estacionalidad de Ventas")
    month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    # Ventas mensuales por año
    monthly_by_year = monthly_df.reset_index().copy()
    monthly_by_year["month_name"] = monthly_by_year["month"].apply(lambda x: month_names[x-1])
    monthly_by_year["Año"] = monthly_by_year["year"].astype(str)
    monthly_by_year["etiqueta"] = monthly_by_year["sales"].apply(lambda x: f"{x:,.0f}")

    fig3 = px.line(
        monthly_by_year, x="month_name", y="sales", color="Año",
        title="Ventas Mensuales por Año",
        labels={"month_name": "Mes", "sales": "Unidades Vendidas", "Año": "Año"},
        markers=True, text="etiqueta",
        category_orders={"month_name": month_names},
        color_discrete_sequence=["#e63946", "#2a9d8f", "#e9c46a", "#457b9d", "#f4a261", "#6a0572", "#3a86ff"]
    )
    fig3.update_traces(textposition="top center", textfont=dict(size=10))

    fig3.update_layout(
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Unidades Vendidas",
        xaxis_title="Mes"
    )
    st.plotly_chart(fig3, use_container_width=True)

    monthly_season = monthly_df.groupby("month")["sales"].mean().reset_index()
    monthly_season["month_name"] = monthly_season["month"].apply(lambda x: month_names[x-1])

    peak = monthly_season.loc[monthly_season["sales"].idxmax()]
    low  = monthly_season.loc[monthly_season["sales"].idxmin()]
    st.info(f"📊 **Estacionalidad detectada:**\n- 🟢 Mes Pico: **{peak['month_name']}** ({peak['sales']:,.0f} uds promedio)\n- 🔴 Mes Bajo: **{low['month_name']}** ({low['sales']:,.0f} uds promedio)\n- 📉 Variación: **{((peak['sales']/low['sales'])-1)*100:.1f}%**")

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
                      color="R2", color_continuous_scale="RdYlGn",
                      text=df_metrics["R2"].round(4).astype(str))
        fig4.update_traces(textposition="outside", textfont=dict(size=12))
        fig4.update_layout(showlegend=False, yaxis=dict(range=[0, df_metrics["R2"].max() * 1.15]))
        st.plotly_chart(fig4, use_container_width=True)

    # Gráficas MAE, MSE, RMSE al estilo de R2
    col3, col4, col5 = st.columns(3)
    df_m = df_metrics.reset_index()

    with col3:
        fig_mae = px.bar(df_m, x="Modelo", y="MAE",
                         title="Error Absoluto Medio (MAE)",
                         labels={"MAE": "MAE", "Modelo": "Modelo"},
                         color="MAE", color_continuous_scale="RdYlGn_r",
                         text=df_m["MAE"].round(2).astype(str))
        fig_mae.update_traces(textposition="outside", textfont=dict(size=12))
        fig_mae.update_layout(showlegend=False, yaxis=dict(range=[0, df_m["MAE"].max() * 1.15]))
        st.plotly_chart(fig_mae, use_container_width=True)

    with col4:
        fig_mse = px.bar(df_m, x="Modelo", y="MSE",
                         title="Error Cuadrático Medio (MSE)",
                         labels={"MSE": "MSE", "Modelo": "Modelo"},
                         color="MSE", color_continuous_scale="RdYlGn_r",
                         text=df_m["MSE"].round(2).astype(str))
        fig_mse.update_traces(textposition="outside", textfont=dict(size=12))
        fig_mse.update_layout(showlegend=False, yaxis=dict(range=[0, df_m["MSE"].max() * 1.15]))
        st.plotly_chart(fig_mse, use_container_width=True)

    with col5:
        fig_rmse = px.bar(df_m, x="Modelo", y="RMSE",
                          title="Raíz del Error Cuadrático (RMSE)",
                          labels={"RMSE": "RMSE", "Modelo": "Modelo"},
                          color="RMSE", color_continuous_scale="RdYlGn_r",
                          text=df_m["RMSE"].round(2).astype(str))
        fig_rmse.update_traces(textposition="outside", textfont=dict(size=12))
        fig_rmse.update_layout(showlegend=False, yaxis=dict(range=[0, df_m["RMSE"].max() * 1.15]))
        st.plotly_chart(fig_rmse, use_container_width=True)
        
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
                  color="Ganancia_Proyectada", color_continuous_scale="Greens",
                  text=top_ganancia["Ganancia_Proyectada"].apply(lambda x: f"${x:,.2f}"))
    fig6.update_traces(textposition="outside", textfont=dict(size=11))
    fig6.update_layout(showlegend=False, xaxis=dict(range=[0, top_ganancia["Ganancia_Proyectada"].max() * 1.2]))
    st.plotly_chart(fig6, use_container_width=True)

    total_ganancia = df_proj["Ganancia_Proyectada"].sum()
    st.metric(label="💰 Total Ganancia Proyectada (todos los productos, 6 meses)", value=f"${total_ganancia:,.2f}")

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
                      color="Costo_Total", color_continuous_scale="Oranges",
                      text=nov_data["Costo_Total"].apply(lambda x: f"${x:,.2f}"))
        fig7.update_traces(textposition="outside", textfont=dict(size=11))
        fig7.update_layout(showlegend=False, xaxis=dict(range=[0, nov_data["Costo_Total"].max() * 1.2]))
        st.plotly_chart(fig7, use_container_width=True)
        total_nov = df_costs[df_costs["Mes"] == "2027-11"]["Costo_Total"].sum()
        st.metric(label="🟠 Total Costo Noviembre 2027", value=f"${total_nov:,.2f}")

    with col2:
        dec_data = df_costs[df_costs["Mes"] == "2027-12"].groupby("Producto")["Costo_Total"].sum().nlargest(5).reset_index()
        fig8 = px.bar(dec_data, x="Costo_Total", y="Producto", orientation="h",
                      title="Costo Proyectado - Diciembre 2027 (Top 5)",
                      color="Costo_Total", color_continuous_scale="Reds",
                      text=dec_data["Costo_Total"].apply(lambda x: f"${x:,.2f}"))
        fig8.update_traces(textposition="outside", textfont=dict(size=11))
        fig8.update_layout(showlegend=False, xaxis=dict(range=[0, dec_data["Costo_Total"].max() * 1.2]))
        st.plotly_chart(fig8, use_container_width=True)
        total_dec = df_costs[df_costs["Mes"] == "2027-12"]["Costo_Total"].sum()
        st.metric(label="🔴 Total Costo Diciembre 2027", value=f"${total_dec:,.2f}")
        
    df_costs_display = df_costs[df_costs["Producto"].isin(top10_nov_dec["Producto"])].sort_values(["Mes", "Costo_Total"], ascending=[True, False]).copy()
    df_costs_display["Costo_Total"] = df_costs_display["Costo_Total"].round(2)
    st.dataframe(
        df_costs_display.style.format({"Costo_Total": "${:,.2f}", "UN": "{:,.2f}"}),
        use_container_width=True, height=300
    )
    st.caption("📊 *Los costos se calculan usando el costo unitario promedio histórico ponderado por la proyección de ventas del modelo.*")

with tab7:
    st.header("📈 Rentabilidad por Tipo de Artículo")
    st.caption("Rentabilidad = (Ganancia / Inversión) × 100 | Inversión = Costo Total | Ganancia = Precio Total - Costo Total")

    # Calcular rentabilidad histórica por tipo de artículo
    rent_df = df.copy()
    rent_df["Costo_Total"]   = rent_df["UN"] * rent_df["costo"]
    rent_df["Precio_Total"]  = rent_df["UN"] * rent_df["Precio"]
    rent_df["Ganancia"]      = rent_df["Precio_Total"] - rent_df["Costo_Total"]

    rentabilidad = rent_df.groupby("Mat:Tipo Articulo").agg(
        Inversion=("Costo_Total", "sum"),
        Ganancia=("Ganancia", "sum")
    ).reset_index()
    rentabilidad = rentabilidad[rentabilidad["Inversion"] > 0]
    rentabilidad["Rentabilidad (%)"] = (rentabilidad["Ganancia"] / rentabilidad["Inversion"] * 100).round(2)
    rentabilidad = rentabilidad.sort_values("Rentabilidad (%)", ascending=False)

    # Métricas globales en la parte superior — datos proyectados (20 meses)
    # Reutiliza hist_prop calculado en tab5
    all_future_months = pd.date_range(
        start=monthly_df.index.max() + pd.DateOffset(months=1), periods=20, freq='MS'
    )
    proj_full = []
    for f_date in all_future_months:
        m = f_date.month
        total_pred = future_forecast.get(f_date, 0)
        month_props = hist_prop[hist_prop["month"] == m]
        for _, row in month_props.iterrows():
            pred_un    = total_pred * row["proportion"]
            prod_data  = df[df["Mat:Tipo Articulo"] == row["Mat:Tipo Articulo"]]
            avg_cost   = prod_data["costo"].mean()
            avg_price  = prod_data["Precio"].mean()
            costo_tot  = pred_un * avg_cost
            precio_tot = pred_un * avg_price
            ganancia   = precio_tot - costo_tot
            proj_full.append({"Costo_Total": costo_tot, "Precio_Total": precio_tot, "Ganancia": ganancia})

    df_proj_full = pd.DataFrame(proj_full)
    costo_total_proy    = df_proj_full["Costo_Total"].sum()
    ganancia_total_proy = df_proj_full["Ganancia"].sum()
    precio_total_proy   = df_proj_full["Precio_Total"].sum()
    margen_proy         = (ganancia_total_proy / precio_total_proy * 100) if precio_total_proy > 0 else 0

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("💵 Costo Total Proyectado (20 meses)", f"${costo_total_proy:,.2f}")
    with col_m2:
        st.metric("💰 Ganancia Total Proyectada (20 meses)", f"${ganancia_total_proy:,.2f}")
    with col_m3:
        st.metric("📊 Margen Proyectado", f"{margen_proy:.2f}%")

    st.divider()

    top15 = rentabilidad.head(15).copy()

    # Gráfica de barras horizontales
    fig_rent = px.bar(
        top15, x="Rentabilidad (%)", y="Mat:Tipo Articulo", orientation="h",
        title="Top 15 Tipos de Artículo por Rentabilidad (%)",
        labels={"Mat:Tipo Articulo": "Tipo de Artículo", "Rentabilidad (%)": "Rentabilidad (%)"},
        color="Rentabilidad (%)", color_continuous_scale="RdYlGn",
        text=top15["Rentabilidad (%)"].apply(lambda x: f"{x:.2f}%")
    )
    fig_rent.update_traces(textposition="outside", textfont=dict(size=11))
    fig_rent.update_layout(
        showlegend=False,
        height=520,
        xaxis=dict(range=[0, top15["Rentabilidad (%)"].max() * 1.2])
    )
    st.plotly_chart(fig_rent, use_container_width=True)

    # Métricas resumen
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Mayor Rentabilidad",
                  f"{rentabilidad.iloc[0]['Mat:Tipo Articulo']}",
                  f"{rentabilidad.iloc[0]['Rentabilidad (%)']:.2f}%")
    with col2:
        st.metric("📊 Rentabilidad Promedio",
                  f"{rentabilidad['Rentabilidad (%)'].mean():.2f}%")
    with col3:
        st.metric("📉 Menor Rentabilidad",
                  f"{rentabilidad.iloc[-1]['Mat:Tipo Articulo']}",
                  f"{rentabilidad.iloc[-1]['Rentabilidad (%)']:.2f}%")

    # Tabla completa
    st.subheader("📋 Detalle de Rentabilidad por Producto")
    st.dataframe(
        rentabilidad.rename(columns={"Mat:Tipo Articulo": "Tipo de Artículo"})
            .style.format({
                "Inversion": "${:,.2f}",
                "Ganancia": "${:,.2f}",
                "Rentabilidad (%)": "{:.2f}%"
            }),
        use_container_width=True,
        height=400
    )

# Footer
st.divider()
st.caption("Dashboard generado con Streamlit + Plotly + SQLite | Modelo iterativo de proyección temporal")