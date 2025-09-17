# ============================================
# Análisis de Admisión con Árboles de Clasificación
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ==============================
# 1. Cargar los datos
# ==============================
# Usamos los dos archivos proporcionados
df1 = pd.read_csv("Admission_Predict.csv")
df2 = pd.read_csv("Admission_Predict_Ver1.1.csv")

# Unificamos en un solo DataFrame (ambos tienen mismas columnas)
df = pd.concat([df1, df2], ignore_index=True)

# ==============================
# 2. Preprocesamiento
# ==============================
# Eliminamos columna Serial No si existe
if "Serial No." in df.columns:
    df = df.drop(columns=["Serial No."])

# Variable binaria: yes si Chance >= 0.6, no si < 0.6
df["Admit_Binary"] = np.where(df["Chance of Admit "] >= 0.6, "yes", "no")

# ==============================
# 3. Análisis descriptivo
# ==============================
print("Valores faltantes:\n", df.isnull().sum())
print("\nDistribución de la variable respuesta:\n", df["Admit_Binary"].value_counts(normalize=True))

# ==============================
# 4. División en train / test
# ==============================
X = df.drop(columns=["Chance of Admit ", "Admit_Binary"])
y = df["Admit_Binary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==============================
# 5. Entrenamiento modelos
# ==============================
# Árbol de decisión
dt_model = DecisionTreeClassifier(random_state=42, max_depth=4)  # limitamos profundidad para evitar sobreajuste
dt_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=200)
rf_model.fit(X_train, y_train)

# ==============================
# 6. Evaluación
# ==============================
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Generar reportes de clasificación
report_dt = classification_report(y_test, y_pred_dt, output_dict=True)
report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

print("\n=== Árbol de Decisión ===")
print(classification_report(y_test, y_pred_dt))

print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))

# ==============================
# 7. Datos tabulares del análisis
# ============================== 
with pd.ExcelWriter("Resultados_Admisiones.xlsx") as writer:
    df.to_excel(writer, sheet_name="raw_data", index=False)
    df[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research','Chance of Admit ','Admit_Binary']].to_excel(writer, sheet_name="processed_data", index=False)
    # Predicciones
    test_df = X_test.copy()
    test_df['y_true'] = y_test.values
    test_df['pred_dt'] = y_pred_dt
    test_df['pred_rf'] = y_pred_rf
    test_df.to_excel(writer, sheet_name="predictions_test", index=False)
    # Matrices e importancias
    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=["yes","no"])
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=["yes","no"])
    pd.DataFrame(cm_dt, index=["yes","no"], columns=["yes","no"]).to_excel(writer, sheet_name="confusion_dt")
    pd.DataFrame(cm_rf, index=["yes","no"], columns=["yes","no"]).to_excel(writer, sheet_name="confusion_rf")
    importances = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": rf_model.feature_importances_
    }).sort_values(by="Importancia", ascending=False)
    importances.to_excel(writer, sheet_name="feature_importance", index=False)
    # Reportes
    pd.DataFrame(report_dt).to_excel(writer, sheet_name="report_dt")
    pd.DataFrame(report_rf).to_excel(writer, sheet_name="report_rf")

print("\nArchivo Excel 'Resultados_Admisiones.xlsx' creado exitosamente!")

# ==============================
# 8. Gráficas
# ==============================

# ---- Distribución de la variable respuesta
plt.figure(figsize=(5,4))
sns.countplot(data=df, x="Admit_Binary", palette="coolwarm")
plt.title("Distribución de Admit_Binary (yes/no)")
plt.show()

# ---- Matriz de confusión Árbol de Decisión
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=["yes","no"])
disp_dt.plot(cmap="Blues")
plt.title("Matriz de Confusión - Árbol de Decisión")
plt.show()

# ---- Matriz de confusión Random Forest
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["yes","no"])
disp_rf.plot(cmap="Greens")
plt.title("Matriz de Confusión - Random Forest")
plt.show()

# ---- Importancia de variables (Random Forest)
plt.figure(figsize=(8,5))
sns.barplot(data=importances, x="Importancia", y="Variable", palette="viridis")
plt.title("Importancia de variables - Random Forest")
plt.show()

# ---- Visualización del Árbol de Decisión
plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=X.columns, class_names=["yes","no"], filled=True, rounded=True, fontsize=10)
plt.title("Árbol de Decisión (profundidad limitada)")
plt.show()
