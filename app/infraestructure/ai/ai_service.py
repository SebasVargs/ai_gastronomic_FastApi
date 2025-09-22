import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import joblib
import os
from typing import Dict, List, Any, Tuple

class AIRecommendationService:
    
    def __init__(self):
        self.restaurant_model = None
        self.plate_model = None
        self.scaler = StandardScaler()
        self.label_enconders = {}
        self.kmeans = None
        self.model_trained = False

        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok = True)


    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()

        categorical_columns = ["categoria", "categoria"]
        for col in categorical_columns:
            if col not in self.label_enconders:
                self.label_enconders[col] = LabelEncoder()
                df_processed[f'{col}_encoded'] = self.label_enconders[col].fit_transform(df_processed[col].astype(str))
            else:
                try:
                    df_processed[f'{col}_encoded'] = self.label_enconders[col].transform(df_processed[col].astype(str))
                except ValueError:
                    df_processed[f'{col}_encoded'] = 0
        
        df_processed['is_weekend'] = (df_processed['weekday'] >= 5).astype(int)
        df_processed['hour_from_minutes'] = (
            (df_processed['lunes_apertura_min'] + df_processed['lunes_cierre_min']) / 2 / 60
        ).fillna(12)  # Hora promedio, default 12:00
        
        # Normalizar precios (log para reducir outliers)
        df_processed['precio_log'] = np.log1p(df_processed['precio'])
        
        # Popularidad normalizada
        df_processed['popularidad_norm'] = (df_processed['popularidad'] - df_processed['popularidad'].min()) / \
                                          (df_processed['popularidad'].max() - df_processed['popularidad'].min() + 1e-6)
        
        return df_processed
    

    def train_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        df_processed = self.prepare_features(df)

        feature_columns = [
            'latitud', 'longitud', 'categoria_rest_encoded', 'categoria_plato_encoded',
            'precio_log', 'popularidad_norm', 'year', 'month', 'day', 'weekday',
            'is_weekend', 'hour_from_minutes',
            'lunes_apertura_min', 'lunes_cierre_min', 'martes_apertura_min', 'martes_cierre_min'
        ]

        available_features = [col for col in feature_columns if col in df_processed.columns]
        X = df_processed[available_features].fillna(0)
        y = df_processed['rating']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.restaurant_model = RandomForestRegressor(
            n_estimators = 100,
            max_depth = 10,
            random_state = 42,
            n_jobs = -1
        )

        self.restaurant_model.fit(X_train_scaled, y_train)

        y_pred = self.restaurant_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)

        self.kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.kmeans.fit(X_train_scaled)

        self.model_trained = True
        self.save_models()

        results = {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "features_used": available_features,
            "samples_trained": len(X_train),
            "model_saved": True
        }

        print(f"Modelo entrenado exitosamente. RMSE: {results['rmse']:.4f}")
        return results
    

    """AQUI ES PARA PREDECIR (NUMERO DE PREDICIONES POR CONSULTAR)"""
    def predict_rating(self, restaurant_data: Dict[str, Any]) -> float:
        if not self.model_trained:
            self.load_models()
            if not self.model_trained:
                return 3.5
        
        df_input = pd.DataFrame([restaurant_data])
        df_processed = self.prepare_features(df_input)

        feature_columns = [
            'latitud', 'longitud', 'categoria_rest_encoded', 'categoria_plato_encoded',
            'precio_log', 'popularidad_norm', 'year', 'month', 'day', 'weekday',
            'is_weekend', 'hour_from_minutes',
            'lunes_apertura_min', 'lunes_cierre_min', 'martes_apertura_min', 'martes_cierre_min'
        ]

        available_features = [col for col in feature_columns if col in df_processed.columns]
        X = df_processed[available_features].fillna(0)

        X_scaled = self.scaler.transform(X)
        prediction = self.restaurant_model.predict(X_scaled)[0]

        return max(1.0, min(5.0, float(prediction)))
    

    def get_cluster_recommendations(self, user_preferences: List[str], restaurants_data: List[Dict]) -> List[Dict]:
        if not self.model_trained or not self.kmeans:
            return restaurants_data[:5]
        
        restaurants_df = pd.DataFrame(restaurants_data)
        if restaurants_df.empty:
            return []
        
        restaurants_processed = self.prepare_features(restaurants_df)
        feature_columns = [
            'latitud', 'longitud', 'categoria_rest_encoded', 'categoria_plato_encoded',
            'precio_log', 'popularidad_norm', 'is_weekend', 'hour_from_minutes'
        ]

        available_features = [col for col in feature_columns if col in restaurants_processed.columns]
        X = restaurants_processed[available_features].fillna(0)

        if X.empty:
            return restaurants_data[:5]
        
        X_scaled = self.scaler.transform(X)
        clusters = self.kmeans.predict(X_scaled)

        recommendations = []
        for i, restaurant in enumerate(restaurants_data):
            try:
                predicted_rating = self.predict_rating(restaurant)
                restaurant_with_prediction = restaurant.copy()
                restaurant_with_prediction['predicted_rating'] = predicted_rating
                restaurant_with_prediction['cluster'] = int(clusters[i])
                recommendations.append(restaurant_with_prediction)
            except Exception as e:
                print(f"Error procesando restaurante {i}: {e}")
                continue

        recommendations.sort(key=lambda x: x.get('predicted_rating', 3.0), reverse = True)
        return recommendations[:10]
    

    def save_models(self):
        try:
            if self.restaurant_model:
                joblib.dump(self.restaurant_model, f"{self.models_dir}/restaurant_model.pkl")
            if self.scaler:
                joblib.dump(self.scaler, f"{self.models_dir}/scaler.pkl")
            if self.label_encoders:
                joblib.dump(self.label_encoders, f"{self.models_dir}/label_encoders.pkl")
            if self.kmeans:
                joblib.dump(self.kmeans, f"{self.models_dir}/kmeans_model.pkl")
            print("Modelos guardados exitosamente")
        except Exception as e:
            print(f"Error guardando modelos: {e}")
            
    
    def load_models(self):
        """Carga los modelos pre-entrenados"""
        try:
            if os.path.exists(f"{self.models_dir}/restaurant_model.pkl"):
                self.restaurant_model = joblib.load(f"{self.models_dir}/restaurant_model.pkl")
            if os.path.exists(f"{self.models_dir}/scaler.pkl"):
                self.scaler = joblib.load(f"{self.models_dir}/scaler.pkl")
            if os.path.exists(f"{self.models_dir}/label_encoders.pkl"):
                self.label_encoders = joblib.load(f"{self.models_dir}/label_encoders.pkl")
            if os.path.exists(f"{self.models_dir}/kmeans_model.pkl"):
                self.kmeans = joblib.load(f"{self.models_dir}/kmeans_model.pkl")
            
            if all([self.restaurant_model, self.scaler, self.label_encoders]):
                self.model_trained = True
                print("Modelos cargados exitosamente")
            else:
                print("No se encontraron todos los archivos de modelo")
        except Exception as e:
            print(f"Error cargando modelos: {e}")
            self.model_trained = False


    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información sobre el estado del modelo"""
        return {
            "model_trained": self.model_trained,
            "has_restaurant_model": self.restaurant_model is not None,
            "has_scaler": self.scaler is not None,
            "has_kmeans": self.kmeans is not None,
            "label_encoders_count": len(self.label_encoders),
            "models_directory": self.models_dir
        }