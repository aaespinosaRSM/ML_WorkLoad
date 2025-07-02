import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# --- Cargar modelo ---
@st.cache_resource
def load_model():
    return joblib.load("modelo_random_forest.pkl")

def get_model_features():
    """Obtener las características que el modelo espera"""
    try:
        model = load_model()
        # Intentar obtener feature_names_in_ si está disponible
        if hasattr(model, 'feature_names_in_'):
            return model.feature_names_in_.tolist()
        else:
            return None
    except:
        return None

model = load_model()

# --- Encabezado ---
st.title("📊 Predicción de Carga de Trabajo Semanal")
st.markdown("Sube tu archivo semanal para obtener predicciones de horas.")

# Mostrar características esperadas por el modelo
model_features = get_model_features()
if model_features:
    with st.expander("🔍 Características que espera el modelo"):
        st.write("El modelo fue entrenado con estas características:")
        for i, feature in enumerate(model_features, 1):
            st.write(f"{i}. {feature}")

# --- Subida de archivo ---
archivo = st.file_uploader("Sube el archivo CSV", type=["csv"])

if archivo is not None:
    try:
        # Intentar leer con utf-8, si falla usa latin1
        try:
            df = pd.read_csv(archivo, encoding="utf-8")
        except UnicodeDecodeError:
            archivo.seek(0)  # Reiniciar el cursor del archivo
            df = pd.read_csv(archivo, encoding="latin1")

        st.success("✅ Archivo cargado correctamente")
        st.write("Columnas del archivo:", df.columns.tolist())
        
        # Mostrar las primeras filas para verificar los datos
        st.write("Primeras 5 filas del archivo:")
        st.dataframe(df.head())

        # Preparar el DataFrame ordenándolo por fecha para crear los lags correctamente
        if 'Week_Start' in df.columns:
            # Convertir fechas con formato día/mes/año
            try:
                df['Week_Start'] = pd.to_datetime(df['Week_Start'], format='%d/%m/%Y', errors='coerce')
            except:
                # Si falla, intentar con dayfirst=True
                df['Week_Start'] = pd.to_datetime(df['Week_Start'], dayfirst=True, errors='coerce')
            
            # Verificar si hay fechas que no se pudieron convertir
            if df['Week_Start'].isnull().any():
                st.warning("⚠️ Algunas fechas no se pudieron convertir correctamente. Verifica el formato de Week_Start.")
            
            df = df.sort_values('Week_Start').reset_index(drop=True)

        # 1. Crear variables lag para HORAS (valores históricos)
        if 'HORAS' in df.columns:
            # Lags simples
            for i in range(1, 5):
                df[f'HORAS_lag{i}'] = df['HORAS'].shift(i)
            
            # Rolling features (media y desviación estándar móvil)
            df['HORAS_roll_mean4'] = df['HORAS'].rolling(window=4, min_periods=1).mean()
            df['HORAS_roll_std4'] = df['HORAS'].rolling(window=4, min_periods=1).std()
            
            # Rellenar NaN en lags con la media
            lag_cols = [f'HORAS_lag{i}' for i in range(1, 5)]
            for col in lag_cols:
                df[col] = df[col].fillna(df['HORAS'].mean())
            
            # Rellenar NaN en rolling std con 0
            df['HORAS_roll_std4'] = df['HORAS_roll_std4'].fillna(0)
            
        else:
            st.warning("⚠️ No se encontró la columna 'HORAS'. Se usarán valores por defecto.")
            # Crear lags y rolling features con valores por defecto
            for i in range(1, 5):
                df[f'HORAS_lag{i}'] = 400
            df['HORAS_roll_mean4'] = 400
            df['HORAS_roll_std4'] = 50

        # 2. Crear variables dummy para Client_Industry
        if 'Client_Industry' in df.columns:
            # Obtener valores únicos
            unique_industries = df['Client_Industry'].unique()
            st.write("Industrias encontradas:", unique_industries)
            
            # Crear dummies para todas las industrias
            industry_dummies = pd.get_dummies(df['Client_Industry'], prefix='Client_Industry')
            df = pd.concat([df, industry_dummies], axis=1)
        else:
            st.warning("⚠️ No se encontró la columna 'Client_Industry'.")

        # 3. Crear variables dummy para Project_Type
        if 'Project_Type' in df.columns:
            # Obtener valores únicos
            unique_projects = df['Project_Type'].unique()
            st.write("Tipos de proyecto encontrados:", unique_projects)
            
            # Crear dummies para todos los tipos de proyecto
            project_dummies = pd.get_dummies(df['Project_Type'], prefix='Project_Type')
            df = pd.concat([df, project_dummies], axis=1)
        else:
            st.warning("⚠️ No se encontró la columna 'Project_Type'.")

        # 4. Si tenemos las características esperadas por el modelo, usarlas
        if model_features:
            expected_features = model_features
        else:
            # Características básicas si no podemos obtenerlas del modelo
            expected_features = [
                'Budget_Hours', 'Planned_Hours', 'Economic_Index', 'Month', 'Quarter',
                'Holiday_Flag', 'HORAS_lag1', 'HORAS_lag2', 'HORAS_lag3', 'HORAS_lag4',
                'HORAS_roll_mean4', 'HORAS_roll_std4'
            ]
            
            # Añadir todas las columnas dummy que existan
            dummy_cols = [col for col in df.columns if col.startswith(('Client_Industry_', 'Project_Type_'))]
            expected_features.extend(dummy_cols)
        
        st.write("### Características que se intentarán usar:")
        st.write(expected_features)
        
        # 5. Crear columnas faltantes con valores por defecto
        missing_features = []
        for feature in expected_features:
            if feature not in df.columns:
                missing_features.append(feature)
                if feature.startswith('Client_Industry_'):
                    df[feature] = 0
                elif feature.startswith('Project_Type_'):
                    df[feature] = 0
                elif feature.startswith('HORAS_lag'):
                    df[feature] = 400
                elif feature == 'HORAS_roll_mean4':
                    df[feature] = 400
                elif feature == 'HORAS_roll_std4':
                    df[feature] = 50
                else:
                    # Para otras características, usar 0 como valor por defecto
                    df[feature] = 0
                    
        if missing_features:
            st.warning(f"⚠️ Se crearon {len(missing_features)} características faltantes con valores por defecto")
            with st.expander("Ver características creadas"):
                st.write(missing_features)
        
        # Verificar que todas las características están presentes
        available_features = [f for f in expected_features if f in df.columns]
        truly_missing = [f for f in expected_features if f not in df.columns]
        
        if truly_missing:
            st.error(f"❌ No se pudieron crear estas características: {truly_missing}")
        else:
            # Seleccionar solo las features que el modelo espera
            X = df[expected_features]
            
            # Verificar y convertir tipos de datos
            for col in X.columns:
                if X[col].dtype == 'object':
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        X[col] = X[col].fillna(0)
                    except:
                        pass
            
            st.write("### Características utilizadas para predicción:")
            st.dataframe(X.head())
            
            # Información sobre los datos
            st.write("**Shape de los datos:**", X.shape)
            st.write("**Valores nulos por columna:**")
            null_counts = X.isnull().sum()
            if null_counts.sum() > 0:
                st.write(null_counts[null_counts > 0])
            else:
                st.write("✅ No hay valores nulos")
            
            # Hacer predicciones
            try:
                predictions = model.predict(X)
                df["Pred_HORAS"] = predictions
                
                st.success("✅ Predicciones generadas correctamente")
                
                # Mostrar resultados
                result_columns = ['Week_Start', 'HORAS', 'Pred_HORAS']
                available_result_cols = [col for col in result_columns if col in df.columns]
                
                if available_result_cols:
                    st.write("### Resultados de Predicción:")
                    st.dataframe(df[available_result_cols])
                    
                    # Gráfico de comparación si tenemos datos reales
                    if 'HORAS' in df.columns and 'Week_Start' in df.columns:
                        fig = px.line(df, x="Week_Start", y=["HORAS", "Pred_HORAS"], 
                                     title="Predicción vs Real",
                                     labels={"value": "Horas", "Week_Start": "Semana"})
                        fig.update_layout(
                            xaxis_title="Semana",
                            yaxis_title="Horas",
                            legend_title="Serie"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calcular métricas de error si hay datos reales
                        mae = np.mean(np.abs(df['HORAS'] - df['Pred_HORAS']))
                        rmse = np.sqrt(np.mean((df['HORAS'] - df['Pred_HORAS'])**2))
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Horas Predichas Promedio", f"{predictions.mean():.1f}")
                        with col2:
                            st.metric("MAE (Error Absoluto Medio)", f"{mae:.1f}")
                        with col3:
                            st.metric("RMSE", f"{rmse:.1f}")
                        with col4:
                            st.metric("Total Registros", len(df))
                    else:
                        # Solo métricas de predicción
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Horas Predichas Promedio", f"{predictions.mean():.1f}")
                        with col2:
                            st.metric("Horas Predichas Máximo", f"{predictions.max():.1f}")
                        with col3:
                            st.metric("Horas Predichas Mínimo", f"{predictions.min():.1f}")
                    
                    # Descargar resultados
                    csv_pred = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Descargar predicciones completas", 
                        data=csv_pred, 
                        file_name="predicciones_completas.csv", 
                        mime="text/csv"
                    )
                else:
                    st.warning("No se pudieron mostrar algunas columnas de resultado")
                    st.dataframe(df[['Pred_HORAS']])
                
            except Exception as pred_error:
                st.error(f"❌ Error al hacer predicciones: {pred_error}")
                st.write("**Diagnóstico del error:**")
                st.write("- Forma de los datos X:", X.shape)
                st.write("- Características esperadas:", len(expected_features))
                st.write("- Características disponibles:", len(available_features))
                
                # Mostrar tipos de datos
                st.write("**Tipos de datos:**")
                st.dataframe(X.dtypes.to_frame('Tipo'))

    except Exception as e:
        st.error(f"⚠️ Ocurrió un error al procesar el archivo: {e}")
        st.write("Asegúrate de que el archivo CSV tenga las columnas correctas.")

# --- Información adicional ---
with st.expander("ℹ️ Información sobre el modelo"):
    st.write("""
    Este modelo de predicción utiliza múltiples tipos de características:
    
    **Características básicas:**
    - Budget_Hours, Planned_Hours, Economic_Index
    - Month, Quarter, Holiday_Flag
    
    **Series temporales:**
    - HORAS_lag1 a HORAS_lag4: Valores históricos
    - HORAS_roll_mean4: Media móvil de 4 periodos
    - HORAS_roll_std4: Desviación estándar móvil de 4 periodos
    
    **Variables categóricas (dummy):**
    - Client_Industry_*: Industria del cliente
    - Project_Type_*: Tipo de proyecto
    
    **Nota:** El modelo crea automáticamente las características que falten con valores por defecto.
    """)

with st.expander("🔧 Diagnóstico y resolución de problemas"):
    st.write("""
    **Para resolver errores de características:**
    
    1. **Revisa las características esperadas** en la sección expandible de arriba
    2. **Tipos de proyecto comunes:** Consultoría estratégica, Implementación, Mantenimiento
    3. **Industrias comunes:** Banca, Energía, Retail, Salud, Tecnología
    4. **Datos históricos:** Se crean automáticamente usando valores previos de HORAS
    5. **Valores por defecto:** Se asignan automáticamente para características faltantes
    
    **Si el error persiste:** Verifica que las columnas categóricas tengan exactamente los mismos valores que se usaron durante el entrenamiento.
    """)