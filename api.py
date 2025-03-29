from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import glob
from fastapi import Query

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

# Variable global para almacenar los últimos datos válidos
last_sent_data = {"timestamp": None, "datasets": {}}

# Configuración de horario nocturno
NIGHT_START = 19  # 19:00 (7 PM)
NIGHT_END = 7     # 07:00 (7 AM)

# Función para cargar y limpiar los datasets
def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        return None
    
    df = pd.read_csv(file_path)

    # Convertir la columna de fecha a datetime
    if "measured_on" in df.columns:
        df["measured_on"] = pd.to_datetime(df["measured_on"])
        df.set_index("measured_on", inplace=True)

        # Filtrar solo registros del mismo día y mes actual
        today = datetime.now()
        filtered_df = df[(df.index.month == today.month) & (df.index.day == today.day)]
        
        if filtered_df.empty:
            print(f"⚠️ No hay registros en el mismo mes y día para {file_path}")
            return None

        # Obtener el primer año disponible con registros en el día y mes actual
        first_year = filtered_df.index.year.min()
        df = df[df.index.year == first_year]

        # Rellenar valores faltantes con forward fill y backward fill
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        return df
    else:
        print(f"⚠️ No se encontró la columna 'measured_on' en {file_path}")
        return None

# Cargar los datasets asegurando que solo se tome el primer año con registros en el mismo día y mes
datasets = {name: load_and_clean_data(path) for name, path in csv_files.items()}

# Función para detectar anomalías con el método IQR
def detectar_anomalias_iqr(df, col):
    Q1 = np.percentile(df[col].dropna(), 25)
    Q3 = np.percentile(df[col].dropna(), 75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return df[(df[col] < limite_inferior) | (df[col] > limite_superior)]

def get_last_16_records(df):
    df = df.copy()

    # Extraer fecha y hora del índice
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["year"] = df.index.year

    # Filtrar registros del mismo día y mes
    today = datetime.now()
    df_filtered = df[
        (df["month"] == today.month) &
        (df["day"] == today.day) &
        ((df.index.hour < today.hour) | 
         ((df.index.hour == today.hour) & (df.index.minute <= today.minute)))
    ]

    if df_filtered.empty:
        return None

    # Asegurar que los datos sean cada 5 minutos
    df_filtered = df_filtered[df_filtered["minute"] % 5 == 0]

    # Ordenar de forma descendente y tomar los últimos 16 registros
    past_records = df_filtered.sort_index(ascending=False).iloc[:16]

    return past_records if not past_records.empty else None

@app.get("/current-time-data")
async def get_real_time_data():
    """
    Obtiene los últimos 16 registros basados en el último dato encontrado.
    Si no hay datos exactos, busca los más cercanos anteriores con intervalos de 5 minutos.
    Verifica si los datos entre los datasets coinciden y genera un resumen de anomalías.
    """
    global last_sent_data
    current_time = datetime.now()
    current_hour = current_time.hour
    new_data = {
        "timestamp": current_time.isoformat(),
        "datasets": {},
        "is_anomalous": False,
        "anomaly_summary": {}
    }

    has_new_data = False
    dataset_timestamps = {}

    for dataset_name, df in datasets.items():
        if df is not None:
            nearest_data = get_last_16_records(df)
            if nearest_data is not None and not nearest_data.empty:
                nearest_data = nearest_data.replace([np.inf, -np.inf], np.nan).fillna(0)

                anomalies = {}
                total_anomalies = 0  # Para el resumen de anomalías

                for col in nearest_data.columns:
                    anomaly_data = detectar_anomalias_iqr(nearest_data, col)
                    anomalies[col] = len(anomaly_data)
                    total_anomalies += len(anomaly_data)

                # Guardar timestamp para verificar coherencia
                dataset_timestamps[dataset_name] = nearest_data.index[0]

                new_data["datasets"][dataset_name] = {
                    "data": nearest_data.to_dict(orient="records"),
                    "anomalies": anomalies
                }

                # Agregar al resumen de anomalías
                new_data["anomaly_summary"][dataset_name] = total_anomalies

                has_new_data = True

    # Verificar coherencia entre datasets
    unique_timestamps = set(dataset_timestamps.values())

    # Si los datasets tienen timestamps diferentes, revisar si es de noche
    if len(unique_timestamps) > 1:
        if current_hour >= NIGHT_START or current_hour < NIGHT_END:
            # Si es de noche, permitir que irradiance y electrical no tengan datos sin marcarlos como anómalos
            if "irradiance" in dataset_timestamps:
                del dataset_timestamps["irradiance"]
            if "electrical" in dataset_timestamps:
                del dataset_timestamps["electrical"]

            unique_timestamps = set(dataset_timestamps.values())

        # Si aún hay diferencias en los timestamps, marcar como anómalo
        if len(unique_timestamps) > 1:
            new_data["is_anomalous"] = True

    if has_new_data:
        last_sent_data = new_data
    else:
        new_data = last_sent_data

    return new_data

@app.get("/historical-data/{dataset_name}/{variable}")
async def get_historical_data(dataset_name: str, variable: str, limit: int = 5):
    """ Devuelve los últimos 'n' datos históricos de una variable en un dataset """
    if dataset_name not in datasets or datasets[dataset_name] is None:
        return {"error": "Dataset no encontrado"}
    
    df = datasets[dataset_name]
    
    if variable not in df.columns:
        return {"error": "Variable no encontrada en el dataset"}

    historical_data = df[[variable]].dropna().tail(limit).reset_index()
    
    return {"dataset": dataset_name, "variable": variable, "data": historical_data.to_dict(orient="records")}

@app.get("/correlation/{dataset_name}")
async def get_correlation_matrix(dataset_name: str):
    """ Devuelve la matriz de correlación entre variables numéricas en un dataset """
    if dataset_name not in datasets or datasets[dataset_name] is None:
        return {"error": "Dataset no encontrado"}

    df = datasets[dataset_name].select_dtypes(include=[np.number])  # Solo variables numéricas
    correlation_matrix = df.corr().fillna(0).to_dict()

    return {"dataset": dataset_name, "correlation_matrix": correlation_matrix}

@app.get("/scatter/{dataset_name}/{var_x}/{var_y}")
async def get_scatter_data(dataset_name: str, var_x: str, var_y: str, limit: int = 100):
    """ Devuelve datos de dos variables para hacer un scatter plot """
    if dataset_name not in datasets or datasets[dataset_name] is None:
        return {"error": "Dataset no encontrado"}

    df = datasets[dataset_name]
    
    if var_x not in df.columns or var_y not in df.columns:
        return {"error": "Una o ambas variables no existen en el dataset"}

    scatter_data = df[[var_x, var_y]].dropna().tail(limit).reset_index()

    return {"dataset": dataset_name, "variables": [var_x, var_y], "data": scatter_data.to_dict(orient="records")}



@app.get("/detailed-anomalies/{dataset_name}")
async def get_detailed_anomalies(dataset_name: str):
    """ Devuelve los valores de las anomalías detectadas en el dataset """
    if dataset_name not in datasets or datasets[dataset_name] is None:
        return {"error": "Dataset no encontrado"}

    df = datasets[dataset_name]
    anomalies = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        outliers = detectar_anomalias_iqr(df, col)
        anomalies[col] = outliers.to_dict(orient="records") if not outliers.empty else []

    return {"dataset": dataset_name, "anomalies": anomalies}

@app.get("/data-distribution/{dataset_name}/{variable}")
async def get_data_distribution(dataset_name: str, variable: str):
    """Distribución de datos con búsqueda de datos más cercanos"""
    if dataset_name not in datasets or datasets[dataset_name] is None:
        return {"error": "Dataset no encontrado"}

    df = datasets[dataset_name]
    current_time = datetime.now()
    
    nearest_data = get_nearest_time_data(df, current_time)
    
    if nearest_data is None or nearest_data.empty:
        return {"message": "No hay datos disponibles para la distribución en este tiempo"}

    nearest_data = nearest_data.replace([np.inf, -np.inf], np.nan).fillna(0)

    hist, bin_edges = np.histogram(nearest_data[variable].dropna(), bins=50)

    return {
        "variable": variable,
        "histogram": {
            "bins": bin_edges.tolist(),
            "counts": hist.tolist()
        },
        "statistics": {
            "mean": float(nearest_data[variable].mean()),
            "std": float(nearest_data[variable].std()),
            "min": float(nearest_data[variable].min()),
            "max": float(nearest_data[variable].max())
        }
    }


@app.get("/daily-analysis")
async def get_daily_analysis():
    """
    Analiza los datos del día anterior, guarda el análisis como archivo JSON
    y elimina el más antiguo si hay más de 5.
    """
    records_per_day = 288  # 5 minutos de intervalo durante 24h
    analysis_result = {}

    day_to_analyze = datetime.now() - timedelta(days=1)
    month = day_to_analyze.month
    day = day_to_analyze.day

    for dataset_name, df in datasets.items():
        if df is None or df.empty:
            continue

        df_clean = df.copy().replace([np.inf, -np.inf], np.nan).fillna(0)
        df_filtered = df_clean[(df_clean.index.month == month) & (df_clean.index.day == day)]
        df_filtered = df_filtered[df_filtered.index.minute % 5 == 0]
        df_filtered = df_filtered.sort_index(ascending=False).iloc[:records_per_day].sort_index()

        if df_filtered.empty:
            continue

        anomalies_summary = {}
        sudden_changes = {}

        for col in df_filtered.columns:
            iqr_anomalies = detectar_anomalias_iqr(df_filtered, col)
            anomalies_summary[col] = len(iqr_anomalies)

            diff = df_filtered[col].diff().abs()
            sudden_peaks = diff[diff > diff.mean() + 3 * diff.std()]
            sudden_changes[col] = {
                "count": int(sudden_peaks.count()),
                "max_jump": float(sudden_peaks.max()) if not sudden_peaks.empty else 0.0
            }

        analysis_result[dataset_name] = {
            "date_analyzed": day_to_analyze.strftime("%Y-%m-%d"),
            "total_records_analyzed": len(df_filtered),
            "anomalies_detected": anomalies_summary,
            "sudden_changes_detected": sudden_changes
        }

    # Guardar el archivo
    analysis_dir = "analyses"
    os.makedirs(analysis_dir, exist_ok=True)
    file_name = f"analysis_{day_to_analyze.strftime('%Y-%m-%d')}.json"
    file_path = os.path.join(analysis_dir, file_name)

    # Eliminar archivos antiguos si hay más de 5
    existing_files = sorted(glob.glob(os.path.join(analysis_dir, "analysis_*.json")))
    if len(existing_files) >= 5:
        os.remove(existing_files[0])  # Elimina el más antiguo

    with open(file_path, "w") as f:
        json.dump({"daily_analysis": analysis_result}, f, indent=4)

    return {"daily_analysis": analysis_result}


@app.get("/get-analysis")
async def get_saved_analysis(date: str = Query(..., description="Formato: YYYY-MM-DD")):
    """
    Devuelve el análisis guardado de una fecha específica.
    """
    file_path = f"analyses/analysis_{date}.json"
    if not os.path.exists(file_path):
        return {"error": f"No se encontró el análisis para la fecha {date}"}
    
    with open(file_path, "r") as f:
        analysis = json.load(f)

    return analysis

@app.get("/compare-analyses")
async def compare_saved_analyses():
    """
    Compara los análisis guardados (máx 5) y devuelve datos
    para graficar anomalías y cambios repentinos.
    """
    analysis_dir = "analyses"
    if not os.path.exists(analysis_dir):
        return {"error": "No se encontraron análisis guardados"}

    analysis_files = sorted(glob.glob(os.path.join(analysis_dir, "analysis_*.json")))
    if not analysis_files:
        return {"error": "No hay análisis disponibles"}

    comparison_result = []

    for file_path in analysis_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            day_data = {
                "date": file_path.split("analysis_")[-1].replace(".json", ""),
                "datasets": {}
            }

            for dataset_name, values in data["daily_analysis"].items():
                anomalies_total = sum(values["anomalies_detected"].values())
                sudden_total = sum(v["count"] for v in values["sudden_changes_detected"].values())

                day_data["datasets"][dataset_name] = {
                    "total_anomalies": anomalies_total,
                    "sudden_changes": sudden_total
                }

            comparison_result.append(day_data)

    return {"comparison": comparison_result}

