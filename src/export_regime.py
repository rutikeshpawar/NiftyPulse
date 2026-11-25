import json

import joblib

from src.config import BASE_DIR

model = joblib.load(BASE_DIR / "models" / "regime_model.joblib")
payload = {
    "features": model["features"],
    "mean": model["scaler"].mean_.tolist(),
    "scale": model["scaler"].scale_.tolist(),
    "centers": model["kmeans"].cluster_centers_.tolist(),
    "rank": {str(k): int(v) for k, v in model["rank"].items()},
}
with open(BASE_DIR / "models" / "regime_params.json", "w") as f:
    json.dump(payload, f, indent=2)
print("Saved models/regime_params.json")