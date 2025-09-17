# ==============================
# analisis descriptivo Bike Sharing Dataset
# ==============================
# librerias necesarias:#   pip install pandas matplotlib reportlab

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# cargamos los  datasets
# -------------------------------
day = pd.read_csv("day.csv", parse_dates=["dteday"])
hour = pd.read_csv("hour.csv", parse_dates=["dteday"])

# -------------------------------
# aplicamos un resumen estadistico
# -------------------------------
print("Resumen day.csv:")
print(day.describe(include="all").T)
print("\nResumen hour.csv:")
print(hour.describe(include="all").T)

# -------------------------------
# generamos los graficos relevantes
# -------------------------------
season_map = {1: "Primavera", 2: "Verano", 3: "Otoño", 4: "Invierno"}
we_map = {1: "Bueno", 2: "Niebla/Mist", 3: "Lluvia/Nieve ligera", 4: "Severo"}

# apartado de serie temporal cnt diario
plt.figure(figsize=(12,4))
plt.plot(day["dteday"], day["cnt"], marker=".", linewidth=0.6)
plt.title("Serie temporal: cnt diario (2011-2012)")
plt.xlabel("Fecha")
plt.ylabel("cnt (alquileres diarios)")
plt.grid(True)
plt.tight_layout()
plt.show()

# correlacion day.csv
corr_day = day.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(9,7))
im = plt.imshow(corr_day, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr_day.columns)), corr_day.columns, rotation=90)
plt.yticks(range(len(corr_day.columns)), corr_day.columns)
plt.title("Mapa de calor de correlaciones (day.csv)")
plt.tight_layout()
plt.show()

# dispersion cnt vs temp x horas
plt.figure(figsize=(8,5))
plt.scatter(hour["temp"], hour["cnt"], s=8, alpha=0.4)
plt.title("Dispersión: cnt vs temp (hour.csv)")
plt.xlabel("Temp (normalizada)")
plt.ylabel("cnt (alquileres por hora)")
plt.grid(True)
plt.tight_layout()
plt.show()

# boxplot cnt por estación
groups = [day.loc[day["season"]==s, "cnt"].values for s in sorted(day["season"].unique())]
plt.figure(figsize=(8,5))
plt.boxplot(groups, labels=[season_map[s] for s in sorted(day["season"].unique())])
plt.title("Boxplot: cnt por estación (day.csv)")
plt.ylabel("cnt (diario)")
plt.xlabel("Estación")
plt.tight_layout()
plt.show()

# histograma cnt horario
plt.figure(figsize=(8,4))
plt.hist(hour["cnt"], bins=40)
plt.title("Histograma: cnt horario")
plt.xlabel("cnt (alquileres por hora)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# promedio cnt por hora del dia
avg_by_hour = hour.groupby("hr")["cnt"].mean()
plt.figure(figsize=(10,4))
plt.bar(avg_by_hour.index, avg_by_hour.values)
plt.title("Promedio cnt por hora (hour.csv)")
plt.xlabel("Hora del día (0-23)")
plt.ylabel("Promedio de cnt")
plt.xticks(range(0,24))
plt.grid(axis="y", linestyle="--", linewidth=0.4)
plt.tight_layout()
plt.show()

# boxplot cnt por clima
groups_w = [hour.loc[hour["weathersit"]==w, "cnt"].values for w in sorted(hour["weathersit"].unique())]
plt.figure(figsize=(8,5))
plt.boxplot(groups_w, labels=[we_map[w] for w in sorted(hour["weathersit"].unique())])
plt.title("Boxplot: cnt por clima (hour.csv)")
plt.ylabel("cnt (horario)")
plt.xlabel("Situación climática")
plt.tight_layout()
plt.show()

# serie temporal casual vs registered
plt.figure(figsize=(12,4))
plt.plot(day["dteday"], day["casual"], label="Casual", linewidth=0.7)
plt.plot(day["dteday"], day["registered"], label="Registered", linewidth=0.7)
plt.plot(day["dteday"], day["cnt"], label="Total (cnt)", linewidth=0.9)
plt.title("Casual vs Registered vs Total (day.csv)")
plt.xlabel("Fecha")
plt.ylabel("Cantidad de alquileres")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# correlacion hour.csv 
num_cols_hour = ["temp","atemp","hum","windspeed","casual","registered","cnt","hr","yr","mnth"]
corr_hour = hour[num_cols_hour].corr()
plt.figure(figsize=(9,6))
im2 = plt.imshow(corr_hour, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr_hour.columns)), corr_hour.columns, rotation=90)
plt.yticks(range(len(corr_hour.columns)), corr_hour.columns)
plt.title("Mapa de calor de correlaciones (hour.csv)")
plt.tight_layout()
plt.show()