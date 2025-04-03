from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de los archivos CSV
csv_files = {
    "electrical": "2107_electrical_data.csv",
    "environment": "2107_environment_data.csv",
    "irradiance": "2107_irradiance_data.csv",
}

# Función para cargar y limpiar los datasets (día y mes actual, primer año válido)
def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        return None

    df = pd.read_csv(file_path)

    if "measured_on" not in df.columns:
        print(f"⚠️ No se encontró la columna 'measured_on' en {file_path}")
        return None

    df["measured_on"] = pd.to_datetime(df["measured_on"])
    df.set_index("measured_on", inplace=True)

    today = datetime.now()
    day = today.day
    month = today.month

    # Buscar el primer año que tenga datos para este mes y día
    available_years = sorted(df.index.year.unique())
    filtered_df = None

    for year in available_years:
        subset = df[(df.index.year == year) & (df.index.month == month) & (df.index.day == day)]
        if not subset.empty:
            filtered_df = subset
            break

    if filtered_df is None:
        print(f"⚠️ No se encontraron registros para el {day}/{month} en ningún año.")
        return None

    # Limpiar valores faltantes
    filtered_df = filtered_df.copy()
    filtered_df.fillna(method='ffill', inplace=True)
    filtered_df.fillna(method='bfill', inplace=True)

    return filtered_df

# Cargar los datasets al iniciar
datasets = {name: load_and_clean_data(path) for name, path in csv_files.items()}

@app.get("/irradiance-vs-power")
async def get_irradiance_vs_power():
    """
    Relaciona la irradiancia con la potencia generada por los inversores.
    Devuelve puntos para graficar irradiancia vs potencia total.
    También detecta si hay desconexión de sensores por falta de datos recientes.
    """
    if "irradiance" not in datasets or "electrical" not in datasets:
        return {"error": "No se encontraron los datasets necesarios"}

    df_irr = datasets["irradiance"]
    df_elec = datasets["electrical"]

    if df_irr is None or df_elec is None:
        return {"error": "Datasets vacíos"}

    try:
        # Redondear la hora actual a múltiplos de 5 minutos
        now = datetime.now()
        rounded_now = now.replace(
            minute=(now.minute // 5) * 5,
            second=0,
            microsecond=0
        )

        expected_time_str = rounded_now.strftime("%Y-%m-%d %H:%M:%S")

        # Seleccionar columna de irradiancia
        irradiance_col = "poa_irradiance_o_149574"
        df_irr_sel = df_irr[[irradiance_col]].copy()

        # Seleccionar columnas de potencia AC en electrical
        power_cols = [col for col in df_elec.columns if "ac_power" in col]
        df_power_sel = df_elec[power_cols].copy()
        df_power_sel["total_power"] = df_power_sel.sum(axis=1)

        # Unir ambos datasets por el índice (tiempo)
        combined = pd.merge(
            df_irr_sel,
            df_power_sel[["total_power"]],
            left_index=True,
            right_index=True
        )
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

        # Verificar si hay datos para el tiempo actual redondeado
        has_current_data = rounded_now in combined.index
        alert = None
        if not has_current_data and 6 <= rounded_now.hour <= 18:
            alert = f"⚠️ No hay datos para el horario esperado ({expected_time_str}). Posible desconexión del sensor."

        # Formatear datos
        result = [
            {
                "timestamp": ts.isoformat(),
                "irradiance": float(row[irradiance_col]),
                "total_power": float(row["total_power"])
            }
            for ts, row in combined.iterrows()
        ]

        return {
            "graph_type": "scatter",
            "description": "Relación entre Irradiancia (W/m²) y Potencia Generada (W)",
            "data": result,
            "expected_time": expected_time_str,
            "alert": alert
        }

    except Exception as e:
        return {"error": f"Error al procesar los datos: {str(e)}"}

@app.get("/histogram/{measurement_type}")
async def get_voltage_current_histogram(measurement_type: str):
    valid_types = ["voltage", "current"]
    if measurement_type not in valid_types:
        return {"error": f"Tipo no válido. Usa uno de: {valid_types}"}

    df_elec = datasets.get("electrical")
    if df_elec is None:
        return {"error": "Dataset 'electrical' no disponible"}

    try:
        if measurement_type == "voltage":
            cols = [c for c in df_elec.columns if "ac_voltage" in c]
        else:
            cols = [c for c in df_elec.columns if "ac_current" in c]

        if not cols:
            return {"error": f"No se encontraron columnas para {measurement_type}"}

        df_filtered = df_elec[cols].copy()
        df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna()

        all_values = df_filtered.values.flatten()

        now = datetime.now()
        rounded_now = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        expected_time_str = rounded_now.strftime("%Y-%m-%d %H:%M:%S")
        # Buscar el dato más cercano hacia atrás dentro de un margen
        closest_record = df_filtered.loc[df_filtered.index <= rounded_now]
        has_data_now = not closest_record.empty
        alert = None

        if not has_data_now and 6 <= rounded_now.hour <= 18:
            alert = f"⚠️ No se detectaron datos recientes de {measurement_type} para el horario esperado ({expected_time_str})."

        hist, bin_edges = np.histogram(all_values, bins=50)

        bins_labels = [f"Bin {i+1} ({round(bin_edges[i], 2)} - {round(bin_edges[i+1], 2)})" for i in range(len(bin_edges)-1)]

        # Agrupación por franja horaria
        def categorize_hour(hour):
            if 6 <= hour < 12:
                return "morning"
            elif 12 <= hour < 18:
                return "afternoon"
            elif 18 <= hour < 24:
                return "evening"
            else:
                return "night"

        time_distribution = df_filtered.copy()
        time_distribution["hour_group"] = df_filtered.index.hour.map(categorize_hour)
        hour_counts = time_distribution["hour_group"].value_counts().to_dict()

        max_inverter = df_filtered.mean().idxmax()
        min_inverter = df_filtered.mean().idxmin()

        extremes = {
            "highest": {"inverter": max_inverter, "value": float(df_filtered[max_inverter].mean())},
            "lowest": {"inverter": min_inverter, "value": float(df_filtered[min_inverter].mean())}
        }

        return {
            "graph_type": "histogram",
            "measurement_type": measurement_type,
            "description": f"Distribución de {'Voltaje' if measurement_type == 'voltage' else 'Corriente'} AC (todos los inversores)",
            "bins": bins_labels,
            "counts": hist.tolist(),
            "expected_time": expected_time_str,
            "alert": alert,
            "time_distribution": hour_counts,
            "extremes": extremes
        }

    except Exception as e:
        return {"error": f"Error al procesar el histograma: {str(e)}"}

@app.get("/temperature-vs-power")
async def get_temperature_vs_power():
    df_temp = datasets.get("environment")
    df_power = datasets.get("electrical")

    if df_temp is None or df_power is None:
        return {"error": "Faltan datasets de environment o electrical"}

    try:
        temp_col = "ambient_temperature_o_149575"
        power_cols = [col for col in df_power.columns if "ac_power" in col]
        
        df_temp_sel = df_temp[[temp_col]].copy()
        df_power_sel = df_power[power_cols].copy()
        df_power_sel["total_power"] = df_power_sel.sum(axis=1)

        combined = pd.merge(df_temp_sel, df_power_sel[["total_power"]], left_index=True, right_index=True)
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

        now = datetime.now()
        rounded_now = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        expected_time_str = rounded_now.strftime("%Y-%m-%d %H:%M:%S")
        closest_record = combined.loc[combined.index <= rounded_now]
        has_data_now = not closest_record.empty
        alert = None

        if not has_data_now and 6 <= rounded_now.hour <= 18:
            alert = f"⚠️ No hay datos para la hora esperada ({expected_time_str}). Posible desconexión de sensores."

        combined["time_slot"] = combined.index.hour.map(
            lambda h: "morning" if 6 <= h < 12 else "afternoon" if 12 <= h < 18 else "evening" if 18 <= h < 24 else "night")
        time_slot_summary = combined.groupby("time_slot").size().to_dict()

        result = [
            {
                "timestamp": ts.isoformat(),
                "temperature": float(row[temp_col]),
                "total_power": float(row["total_power"])
            } for ts, row in combined.iterrows()
        ]

        return {
            "graph_type": "scatter",
            "description": "Relación entre Temperatura Ambiente (°C) y Potencia Generada (W)",
            "data": result,
            "time_slot_summary": time_slot_summary,
            "expected_time": expected_time_str,
            "alert": alert
        }

    except Exception as e:
        return {"error": f"Error al procesar los datos: {str(e)}"}

@app.get("/wind-vs-temperature")
async def get_wind_vs_temperature():
    """
    Relación entre la velocidad del viento y la temperatura ambiente.
    Devuelve puntos para graficar viento vs temperatura.
    """
    if "environment" not in datasets:
        return {"error": "Dataset 'environment' no disponible"}

    df_env = datasets["environment"]

    if df_env is None:
        return {"error": "Datos de entorno no disponibles"}

    try:
        temp_col = "ambient_temperature_o_149575"
        wind_col = "wind_speed_o_149576"

        df_selected = df_env[[temp_col, wind_col]].copy()
        df_selected = df_selected.replace([np.inf, -np.inf], np.nan).dropna()

        now = datetime.now()
        rounded_now = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        expected_time_str = rounded_now.strftime("%Y-%m-%d %H:%M:%S")

        closest_record = df_selected.loc[df_selected.index <= rounded_now]
        has_data_now = not closest_record.empty
        alert = None
        if not has_data_now:
            alert = f"⚠️ No hay datos de viento y temperatura para el horario esperado ({expected_time_str})."

        result = [
            {
                "timestamp": ts.isoformat(),
                "temperature": float(row[temp_col]),
                "wind_speed": float(row[wind_col])
            }
            for ts, row in df_selected.iterrows()
        ]

        return {
            "graph_type": "line",
            "description": "Relación entre Velocidad del Viento (m/s) y Temperatura Ambiente (°C)",
            "data": result,
            "expected_time": expected_time_str,
            "alert": alert
        }

    except Exception as e:
        return {"error": f"Error al procesar los datos: {str(e)}"}

@app.get("/power-anomalies")
async def detect_power_anomalies():
    df_elec = datasets.get("electrical")
    if df_elec is None:
        return {"error": "Dataset 'electrical' no disponible"}

    try:
        power_cols = [col for col in df_elec.columns if "ac_power" in col]
        if not power_cols:
            return {"error": "No se encontraron columnas de potencia"}

        df_power = df_elec[power_cols].copy()
        df_power = df_power.replace([np.inf, -np.inf], np.nan).dropna()

        anomalies = {}
        for col in power_cols:
            Q1 = df_power[col].quantile(0.25)
            Q3 = df_power[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df_power[(df_power[col] < lower) | (df_power[col] > upper)]
            anomalies[col] = outliers.index.strftime("%Y-%m-%d %H:%M:%S").tolist()

        total_anomalies = sum(len(v) for v in anomalies.values())

        return {
            "graph_type": "line",
            "description": "Detección de Anomalías en la Potencia de los Inversores",
            "total_anomalies": total_anomalies,
            "anomalies": anomalies
        }

    except Exception as e:
        return {"error": f"Error al detectar anomalías: {str(e)}"}

@app.get("/energy-by-hour")
async def energy_by_hour():
    df_elec = datasets.get("electrical")
    if df_elec is None:
        return {"error": "Dataset 'electrical' no disponible"}

    try:
        power_cols = [col for col in df_elec.columns if "ac_power" in col]
        if not power_cols:
            return {"error": "No se encontraron columnas de potencia"}

        df_power = df_elec[power_cols].copy()
        df_power["total_power"] = df_power.sum(axis=1)

        df_hourly = df_power.resample("H").sum()
        df_hourly = df_hourly.replace([np.inf, -np.inf], np.nan).dropna()

        result = [
            {
                "hour": ts.strftime("%H:00"),
                "energy_generated": float(row["total_power"])
            }
            for ts, row in df_hourly.iterrows()
        ]

        return {
            "graph_type": "bar",
            "description": "Comparación de Energía Generada en Diferentes Horas del Día",
            "data": result
        }

    except Exception as e:
        return {"error": f"Error al calcular la energía por hora: {str(e)}"}
