"""
CsvDataService - Loads plate and restaurant data from complementary_data.csv
for use in the AI recommendation pipeline.

This does NOT replace MongoDB — it provides an alternative data source
specifically for the recommendation engine.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional


class CsvDataService:
    """Singleton-like service that loads CSV data once and serves it for recommendations."""

    def __init__(self, csv_path: Optional[str] = None):
        if csv_path is None:
            # Default path relative to project root
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(base_dir, "..", "..", "..")
            csv_path = os.path.join(project_root, "uploads", "complementary_data.csv")

        self.csv_path = os.path.normpath(csv_path)
        self.df: Optional[pd.DataFrame] = None
        self._load()

    def _load(self):
        """Load the CSV into a DataFrame."""
        if not os.path.exists(self.csv_path):
            print(f"⚠️  CSV no encontrado en: {self.csv_path}")
            self.df = pd.DataFrame()
            return

        self.df = pd.read_csv(self.csv_path)
        print(f"✅ CsvDataService: Cargados {len(self.df)} registros desde {self.csv_path}")

    def reload(self):
        """Reload the CSV (useful if the file changes)."""
        self._load()

    # ---- Plates ----

    def get_all_plates(self) -> List[Dict[str, Any]]:
        """Return all plates as a list of dicts (ready for ML pipeline)."""
        if self.df is None or self.df.empty:
            return []
        return self.df.to_dict(orient="records")

    def get_popular_plates(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return top plates sorted by popularity."""
        if self.df is None or self.df.empty:
            return []
        sorted_df = self.df.sort_values("popularidad", ascending=False).head(limit)
        return sorted_df.to_dict(orient="records")

    def get_plates_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Filter plates by categoria_plato (case-insensitive partial match)."""
        if self.df is None or self.df.empty:
            return []
        mask = self.df["categoria_plato"].str.contains(category, case=False, na=False)
        return self.df[mask].to_dict(orient="records")

    # ---- Restaurants ----

    def get_all_restaurants(self) -> List[Dict[str, Any]]:
        """Return unique restaurants extracted from the CSV."""
        if self.df is None or self.df.empty:
            return []

        restaurant_cols = [
            "restaurante_id", "nombre_restaurante", "categoria_rest",
            "latitud", "longitud",
            "lunes_apertura_min", "lunes_cierre_min",
            "martes_apertura_min", "martes_cierre_min",
            "miércoles_apertura_min", "miércoles_cierre_min",
            "jueves_apertura_min", "jueves_cierre_min",
            "viernes_apertura_min", "viernes_cierre_min",
            "sábado_apertura_min", "sábado_cierre_min",
            "domingo_apertura_min", "domingo_cierre_min",
        ]
        # Only keep columns that actually exist
        existing_cols = [c for c in restaurant_cols if c in self.df.columns]
        restaurants_df = self.df[existing_cols].drop_duplicates(subset=["restaurante_id"])
        return restaurants_df.to_dict(orient="records")

    # ---- Stats ----

    def get_stats(self) -> Dict[str, Any]:
        """Return basic stats about the loaded data."""
        if self.df is None or self.df.empty:
            return {"loaded": False, "total_records": 0}

        return {
            "loaded": True,
            "total_records": len(self.df),
            "unique_plates": self.df["plato_id"].nunique() if "plato_id" in self.df.columns else 0,
            "unique_restaurants": self.df["restaurante_id"].nunique() if "restaurante_id" in self.df.columns else 0,
            "categories": sorted(self.df["categoria_plato"].dropna().unique().tolist()) if "categoria_plato" in self.df.columns else [],
            "csv_path": self.csv_path,
        }
