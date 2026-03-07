import pickle
from pathlib import Path

model_path = Path("../MLFLOW_BACKUP/SF-Crimes-MultiModel-FE/models/Xgboost/artifacts/model.pkl").resolve()

with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Tipo de objeto:", type(model))
print("Tiene predict:", hasattr(model, "predict"))
print("Tiene predict_proba:", hasattr(model, "predict_proba"))

if hasattr(model, "n_features_in_"):
    print("n_features_in_:", model.n_features_in_)
else:
    print("No tiene n_features_in_")

if hasattr(model, "classes_"):
    print("Número de clases:", len(model.classes_))
    print("Clases:", model.classes_)
else:
    print("No tiene classes_")

try:
    booster = model.get_booster()
    print("Booster cargado correctamente.")
    print("Feature names del booster:", booster.feature_names)
except Exception as e:
    print("No se pudieron leer feature_names del booster:", e)

print("\nParámetros del modelo:")
try:
    print(model.get_params())
except Exception as e:
    print("No se pudieron leer params:", e)