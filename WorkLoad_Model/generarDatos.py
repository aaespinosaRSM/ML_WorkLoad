import csv
from datetime import datetime, timedelta
import random
import math
from collections import deque

# Configuración inicial
start_date = datetime(2019, 9, 9)
n_weeks = 1000
output_file = 'large_dataset.csv'

# Cabeceras del archivo
headers = [
    'Week_Start', 'Month', 'Quarter', 'Holiday_Flag', 'Planned_Hours', 'Budget_Hours', 'Economic_Index',
    'HORAS', 'HORAS_lag1', 'HORAS_lag2', 'HORAS_lag3', 'HORAS_lag4',
    'HORAS_roll_mean4', 'HORAS_roll_std4',
    'Project_Type_Consultoría estratégica', 'Project_Type_Implementación', 'Project_Type_Mantenimiento',
    'Client_Industry_Banca', 'Client_Industry_Energía', 'Client_Industry_Retail', 
    'Client_Industry_Salud', 'Client_Industry_Tecnología'
]

# Inicialización de datos para lags
horas_deque = deque([400] * 4, maxlen=4)  # Valores iniciales

# Crear archivo CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    
    for week in range(n_weeks):
        # Generar fecha
        current_date = start_date + timedelta(weeks=week)
        month = current_date.month
        quarter = (month - 1) // 3 + 1
        
        # Bandera de festivo (5% de probabilidad)
        holiday_flag = 1 if random.random() < 0.05 else 0
        
        # Índice económico (tendencia + estacionalidad + ruido)
        economic_index = (
            100 + 
            0.04 * week + 
            3 * math.sin(2 * math.pi * week / 52) + 
            2 * math.cos(2 * math.pi * week / 13) + 
            random.uniform(-1, 1)
        )
        
        # Generar horas planificadas y presupuestadas
        planned_hours = 450 + 0.3 * week + 50 * math.sin(2 * math.pi * week / 26)
        budget_hours = planned_hours * random.uniform(0.85, 1.15)
        
        # Generar HORAS reales con efecto de festivos
        horas = planned_hours * random.uniform(0.8, 1.2)
        if holiday_flag:
            horas *= random.uniform(0.6, 1.1)
        horas = int(round(horas))
        
        # Obtener lags del deque
        lags = list(horas_deque)
        while len(lags) < 4:
            lags.insert(0, 400)  # Rellenar si es necesario
        
        # Calcular estadísticas móviles
        window = [horas] + lags[:3]  # Ventana actual + últimos 3 períodos
        roll_mean = sum(window) / min(len(window), 4)
        
        if len(window) > 1:
            roll_std = math.sqrt(
                sum((x - roll_mean) ** 2 for x in window) / (len(window) - 1))
        else:
            roll_std = 0.0
        
        # Generar variables categóricas (one-hot)
        project_type = random.choices(
            ['Consultoría estratégica', 'Implementación', 'Mantenimiento'], 
            weights=[0.4, 0.4, 0.2]
        )[0]
        project_encoded = [
            1 if project_type == 'Consultoría estratégica' else 0,
            1 if project_type == 'Implementación' else 0,
            1 if project_type == 'Mantenimiento' else 0
        ]
        
        client_industry = random.choice([
            'Banca', 'Energía', 'Retail', 'Salud', 'Tecnología'
        ])
        client_encoded = [
            1 if client_industry == 'Banca' else 0,
            1 if client_industry == 'Energía' else 0,
            1 if client_industry == 'Retail' else 0,
            1 if client_industry == 'Salud' else 0,
            1 if client_industry == 'Tecnología' else 0
        ]
        
        # Escribir fila
        row = [
            current_date.strftime('%d/%m/%Y'),
            month,
            quarter,
            holiday_flag,
            round(planned_hours, 1),
            round(budget_hours, 1),
            round(economic_index, 2),
            horas,
            lags[0],  # lag1
            lags[1],  # lag2
            lags[2],  # lag3
            lags[3],  # lag4
            round(roll_mean, 2),
            round(roll_std, 2)
        ] + project_encoded + client_encoded
        
        writer.writerow(row)
        
        # Actualizar deque para próximos lags
        horas_deque.appendleft(horas)

print(f"Archivo generado: {output_file} con {n_weeks} filas")