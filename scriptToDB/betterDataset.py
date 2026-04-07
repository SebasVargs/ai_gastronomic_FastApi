import pandas as pd
import numpy as np

# === 1. Cargar dataset ===
df = pd.read_csv("../uploads/complementary_data.csv")

np.random.seed(42)  # reproducibilidad

# === 1. Crear máscara 60% True ===
mask = np.random.rand(len(df)) < 0.60

# === 2. Generar teléfonos móviles tipo Colombia (3XXXXXXXXX)
telefonos = (
    "3" +
    np.random.randint(100000000, 999999999, size=len(df)).astype(str)
)

# === 3. Crear columna vacía por defecto ===
df["telefono"] = ""

# === 4. Asignar teléfonos solo al 60% ===
df.loc[mask, "telefono"] = telefonos[mask]

# === 4. Guardar nuevo dataset ===
df.to_csv("complementary_data1.csv", index=False)

print("Columna telefono creada correctamente.")
print("Porcentaje con teléfono:", (df["telefono"] != "").mean())

