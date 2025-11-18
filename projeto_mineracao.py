# ============================================================
# PROJETO INDIVIDUAL – MINERAÇÃO DE DADOS
# Classificação (Random Forest) + Regressão (Regressão Linear)
# Problema: Previsão de churn e tempo de uso semanal
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# 1. DESCRIÇÃO DO PROBLEMA
# ------------------------------------------------------------
# Dataset fictício de clientes de um serviço digital.
# Objetivos:
# 1) Prever se o cliente irá cancelar contrato (churn) -> Classificação
# 2) Prever o tempo de uso semanal (horas) -> Regressão
# ============================================================

# ============================================================
# 2. ETL – CRIAÇÃO, LIMPEZA E PREPARAÇÃO DOS DADOS
# ============================================================

np.random.seed(42)
n = 800

# Criando dados fictícios
data = pd.DataFrame({
    "idade": np.random.randint(18, 65, n),
    "tempo_de_plano": np.random.randint(1, 48, n),  # meses
    "uso_semanal_horas": np.random.uniform(1, 20, n),
    "tickets_abertos": np.random.poisson(1, n),
    "satisfacao": np.random.randint(1, 6, n)  # 1 a 5
})

# Criando variável churn com lógica simples + ruído
data["churn"] = (
    (data["satisfacao"] <= 2).astype(int) |
    (data["tickets_abertos"] >= 3).astype(int)
)

# -----------------------------
# LIMPEZA
# -----------------------------
# Sem valores faltantes neste dataset, mas vamos simular limpeza
data = data.drop_duplicates()

print("\nVisualização inicial:")
print(data.head())

# ============================================================
# 3. EXPLORAÇÃO E VISUALIZAÇÕES
# ============================================================

plt.figure(figsize=(6, 4))
sns.countplot(data=data, x="churn")
plt.title("Distribuição de Churn")
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(data.corr(), annot=True, cmap="Blues")
plt.title("Correlação entre Variáveis")
plt.show()

sns.pairplot(data, hue="churn")
plt.show()

# ============================================================
# 4. MODELO 1 – RANDOM FOREST (CLASSIFICAÇÃO)
# ============================================================

X_class = data.drop("churn", axis=1)
y_class = data["churn"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.25, random_state=42
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_c, y_train_c)

pred_rf = rf.predict(X_test_c)

print("\n======= RESULTADOS RANDOM FOREST =======")
print("Acurácia:", accuracy_score(y_test_c, pred_rf))
print("\nRelatório de Classificação:")
print(classification_report(y_test_c, pred_rf))

# Feature importance
importances = pd.DataFrame({
    "feature": X_class.columns,
    "importancia": rf.feature_importances_
}).sort_values("importancia", ascending=False)

plt.figure(figsize=(6, 4))
sns.barplot(data=importances, x="importancia", y="feature")
plt.title("Importância das Features – Random Forest")
plt.show()

# ============================================================
# 5. MODELO 2 – REGRESSÃO LINEAR (REGRESSÃO)
# ============================================================

X_reg = data.drop("uso_semanal_horas", axis=1)
y_reg = data["uso_semanal_horas"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_r_scaled = scaler.fit_transform(X_train_r)
X_test_r_scaled = scaler.transform(X_test_r)

lr = LinearRegression()
lr.fit(X_train_r_scaled, y_train_r)

pred_lr = lr.predict(X_test_r_scaled)

print("\n======= RESULTADOS REGRESSÃO LINEAR =======")
print("MSE:", mean_squared_error(y_test_r, pred_lr))
print("R²:", r2_score(y_test_r, pred_lr))

plt.figure(figsize=(6, 4))
plt.scatter(y_test_r, pred_lr)
plt.xlabel("Valores reais")
plt.ylabel("Valores previstos")
plt.title("Regressão Linear – Real x Previsto")
plt.show()