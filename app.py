import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard de Análisis", layout="wide")
st.title("📊 Dashboard de Exploración y Análisis de Anomalías")

BASE_URL = "http://localhost:8000"

@st.cache_data(ttl=10)
def fetch_json(endpoint):
    try:
        response = requests.get(f"{BASE_URL}{endpoint}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error {response.status_code} al consultar {endpoint}")
            return None
    except Exception as e:
        st.error(f"Error al conectar con la API: {e}")
        return None

# --- Sidebar ---
st.sidebar.header("Opciones")
seccion = st.sidebar.radio("Selecciona sección", [
    "📈 Tiempo Real", "📊 Distribución", "🔗 Correlación", "⚖️ Dispersión", 
    "📉 Histórico", "🚨 Anomalías Detalladas", "📅 Análisis Diario", "📆 Comparar Días"])

# Obtener datasets disponibles
current_data = fetch_json("/current-time-data")
dataset_names = list(current_data.get("datasets", {}).keys()) if current_data else []

# --- Sección: Tiempo Real ---
if seccion == "📈 Tiempo Real":
    st.subheader("📡 Datos en Tiempo Real (últimos 16 registros)")

    selected_ds = st.selectbox("Selecciona un Dataset", dataset_names, key="rt_ds")
    
    if selected_ds:
        rt_df = pd.DataFrame(current_data["datasets"][selected_ds]["data"])

        if rt_df.empty:
            st.warning("No hay datos disponibles para este dataset.")
        else:
            numeric_cols = rt_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_var = st.selectbox("Selecciona una Variable", numeric_cols, key="rt_var")

            if 'month' in rt_df.columns:
                rt_df["timestamp"] = pd.to_datetime({
                    "year": int(current_data["timestamp"][:4]),
                    "month": rt_df["month"],
                    "day": rt_df["day"],
                    "hour": rt_df["hour"],
                    "minute": rt_df["minute"]
                })
            else:
                rt_df["timestamp"] = pd.to_datetime(current_data["timestamp"])

            rt_df.sort_values("timestamp", inplace=True)

            st.plotly_chart(
                px.line(rt_df, x="timestamp", y=selected_var, markers=True,
                        title=f"{selected_var} en tiempo real"),
                use_container_width=True
            )

            with st.expander("🔎 Ver datos en tabla"):
                st.dataframe(rt_df[["timestamp", selected_var]])

    st.divider()
    st.subheader("⚠️ Resumen de Anomalías")
    anomaly_df = pd.DataFrame(current_data.get("anomaly_summary", {}).items(), columns=["Dataset", "Total"])
    if not anomaly_df.empty:
        st.plotly_chart(
            px.bar(anomaly_df, x="Dataset", y="Total", color="Dataset", title="Anomalías por Dataset"),
            use_container_width=True
        )
    else:
        st.info("No se detectaron anomalías.")

# --- Sección: Distribución ---
elif seccion == "📊 Distribución":
    ds = st.selectbox("Selecciona Dataset", dataset_names)
    if ds:
        df = pd.DataFrame(current_data["datasets"][ds]["data"])
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        variable = st.selectbox("Variable", numeric_cols)
        if variable:
            dist_data = fetch_json(f"/data-distribution/{ds}/{variable}")
            if dist_data:
                fig = go.Figure(data=[go.Bar(x=dist_data["histogram"]["bins"], y=dist_data["histogram"]["counts"])])
                fig.update_layout(title=f"Distribución de {variable}")
                st.plotly_chart(fig, use_container_width=True)

# --- Sección: Correlación ---
elif seccion == "🔗 Correlación":
    ds = st.selectbox("Dataset para correlación", dataset_names)
    if ds:
        corr = fetch_json(f"/correlation/{ds}")
        if corr:
            corr_df = pd.DataFrame(corr["correlation_matrix"])
            fig = px.imshow(corr_df, text_auto=True, aspect="auto", title="Matriz de Correlación", width=1400, height=1000)
            st.plotly_chart(fig, use_container_width=False)

# --- Sección: Dispersión ---
elif seccion == "⚖️ Dispersión":
    ds = st.selectbox("Dataset para dispersión", dataset_names)
    if ds:
        df = pd.DataFrame(current_data["datasets"][ds]["data"])
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        var_x = st.selectbox("Variable X", numeric_cols)
        var_y = st.selectbox("Variable Y", numeric_cols)
        if var_x and var_y:
            scatter = fetch_json(f"/scatter/{ds}/{var_x}/{var_y}")
            if scatter:
                sdf = pd.DataFrame(scatter["data"])
                fig = px.scatter(sdf, x=var_x, y=var_y, title=f"{var_x} vs {var_y}")
                st.plotly_chart(fig, use_container_width=True)

# --- Sección: Histórico ---
elif seccion == "📉 Histórico":
    ds = st.selectbox("Dataset histórico", dataset_names)
    if ds:
        df = pd.DataFrame(current_data["datasets"][ds]["data"])
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        variable = st.selectbox("Variable", numeric_cols)
        if variable:
            history = fetch_json(f"/historical-data/{ds}/{variable}?limit=200")
            if history:
                hist_df = pd.DataFrame(history["data"])
                fig = px.line(hist_df, x="measured_on", y=variable, title=f"Histórico de {variable}")
                st.plotly_chart(fig, use_container_width=True)

# --- Sección: Análisis Detallado ---
elif seccion == "🚨 Anomalías Detalladas":
    ds = st.selectbox("Dataset para ver anomalías", dataset_names)
    if ds:
        details = fetch_json(f"/detailed-anomalies/{ds}")
        if details:
            for var, rows in details["anomalies"].items():
                if rows:
                    st.markdown(f"### 🔴 Anomalías en `{var}`")
                    st.dataframe(pd.DataFrame(rows))

# --- Sección: Análisis Diario ---
elif seccion == "📅 Análisis Diario":
    st.subheader("Análisis del Día Anterior")
    analysis = fetch_json("/daily-analysis")
    if analysis:
        for ds, stats in analysis["daily_analysis"].items():
            st.markdown(f"### Dataset: `{ds}`")
            st.write("📊 Anomalías:", stats["anomalies_detected"])
            st.write("⚡ Cambios Repentinos:", stats["sudden_changes_detected"])

# --- Sección: Comparación ---
elif seccion == "📆 Comparar Días":
    st.subheader("Comparar Días Analizados")
    comp = fetch_json("/compare-analyses")
    if comp:
        for ds in dataset_names:
            fechas, anom, cambios = [], [], []
            for item in comp["comparison"]:
                if ds in item["datasets"]:
                    fechas.append(item["date"])
                    anom.append(item["datasets"][ds]["total_anomalies"])
                    cambios.append(item["datasets"][ds]["sudden_changes"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fechas, y=anom, mode='lines+markers', name='Anomalías'))
            fig.add_trace(go.Scatter(x=fechas, y=cambios, mode='lines+markers', name='Cambios Repentinos'))
            fig.update_layout(title=f"Evolución de {ds}", xaxis_title="Fecha", yaxis_title="Cantidad")
            st.plotly_chart(fig, use_container_width=True)
