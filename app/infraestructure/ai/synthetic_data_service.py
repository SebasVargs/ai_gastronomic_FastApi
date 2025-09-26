import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime

class SyntheticDataService:
    def __init__(self):
        self.df_synthetic = None
        self.model_rf = None
        self.model_kmeans = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.models_dir = "models"
        self.reports_dir = "reports" 
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def load_synthetic_data(self, csv_path: str) -> Dict[str, Any]:
        """
        Carga el dataset sintético de 80,000 registros
        """
        try:
            print(f"📁 Cargando datos sintéticos desde: {csv_path}")
            self.df_synthetic = pd.read_csv(csv_path)
            
            # Validar que tiene las columnas esperadas
            expected_columns = [
                'restaurante_id', 'latitud', 'longitud', 'categoria_rest',
                'lunes_apertura_min', 'lunes_cierre_min', 'martes_apertura_min', 'martes_cierre_min',
                'miércoles_apertura_min', 'miércoles_cierre_min', 'jueves_apertura_min', 'jueves_cierre_min',
                'viernes_apertura_min', 'viernes_cierre_min', 'sábado_apertura_min', 'sábado_cierre_min',
                'domingo_apertura_min', 'domingo_cierre_min', 'plato_id', 'nombre_plato', 
                'categoria_plato', 'precio', 'popularidad', 'year', 'month', 'day', 
                'weekday', 'rating', 'es_favorito'
            ]
            
            missing_columns = [col for col in expected_columns if col not in self.df_synthetic.columns]
            if missing_columns:
                print(f"⚠️  Columnas faltantes: {missing_columns}")
            
            # Información básica del dataset
            info = {
                "total_records": len(self.df_synthetic),
                "columns": list(self.df_synthetic.columns),
                "missing_columns": missing_columns,
                "data_types": self.df_synthetic.dtypes.astype(str).to_dict(),
                "null_values": self.df_synthetic.isnull().sum().to_dict(),
                "unique_restaurants": self.df_synthetic['restaurante_id'].nunique() if 'restaurante_id' in self.df_synthetic.columns else 0,
                "unique_plates": self.df_synthetic['plato_id'].nunique() if 'plato_id' in self.df_synthetic.columns else 0,
                "rating_distribution": self.df_synthetic['rating'].describe().to_dict() if 'rating' in self.df_synthetic.columns else {},
                "sample_data": self.df_synthetic.head(3).to_dict()
            }
            
            print(f"✅ Datos cargados exitosamente:")
            print(f"   📊 {info['total_records']:,} registros")
            print(f"   🏪 {info['unique_restaurants']:,} restaurantes únicos")
            print(f"   🍽️ {info['unique_plates']:,} platos únicos")
            print(f"   ⭐ Rating promedio: {self.df_synthetic['rating'].mean():.2f}" if 'rating' in self.df_synthetic.columns else "")
            
            return info
            
        except Exception as e:
            print(f"❌ Error cargando datos sintéticos: {str(e)}")
            raise e


    def prepare_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepara las características para el entrenamiento del modelo
        """
        if df is None:
            df = self.df_synthetic.copy()
        else:
            df = df.copy()
        
        if df is None or df.empty:
            raise ValueError("No hay datos para procesar")
        
        print("🔧 Preparando características para el modelo...")
        
        # 1. Encoding de variables categóricas
        categorical_columns = ['categoria_rest', 'categoria_plato']
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Para predicciones futuras
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # Manejar categorías nuevas
                        df[f'{col}_encoded'] = 0
        
        # 2. Features temporales avanzadas
        df['is_weekend'] = (df['weekday'] >= 5).astype(int) if 'weekday' in df.columns else 0
        df['is_holiday_season'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int) if 'month' in df.columns else 0
        df['quarter'] = ((df['month'] - 1) // 3 + 1) if 'month' in df.columns else 1
        
        # 3. Features de horarios
        df['avg_opening_weekday'] = df[['lunes_apertura_min', 'martes_apertura_min', 'miércoles_apertura_min', 
                                       'jueves_apertura_min', 'viernes_apertura_min']].mean(axis=1)
        df['avg_opening_weekend'] = df[['sábado_apertura_min', 'domingo_apertura_min']].mean(axis=1)
        df['total_operating_hours'] = (df[['lunes_cierre_min', 'martes_cierre_min', 'miércoles_cierre_min',
                                         'jueves_cierre_min', 'viernes_cierre_min', 'sábado_cierre_min', 
                                         'domingo_cierre_min']].mean(axis=1) - 
                                     df[['lunes_apertura_min', 'martes_apertura_min', 'miércoles_apertura_min',
                                         'jueves_apertura_min', 'viernes_apertura_min', 'sábado_apertura_min',
                                         'domingo_apertura_min']].mean(axis=1))
        
        # 4. Features de precios
        df['precio_log'] = np.log1p(df['precio']) if 'precio' in df.columns else 0
        df['precio_per_popularity'] = df['precio'] / (df['popularidad'] + 1) if 'precio' in df.columns and 'popularidad' in df.columns else 0
        
        # 5. Features de popularidad
        df['popularidad_norm'] = (df['popularidad'] / 100) if 'popularidad' in df.columns else 0
        df['is_highly_popular'] = (df['popularidad'] > 80).astype(int) if 'popularidad' in df.columns else 0
        
        # 6. Features geográficas (si tienes múltiples ubicaciones)
        if 'latitud' in df.columns and 'longitud' in df.columns:
            # Centro aproximado de Cali
            cali_center_lat, cali_center_lon = 3.4516, -76.5320
            df['distance_from_center'] = np.sqrt(
                (df['latitud'] - cali_center_lat)**2 + (df['longitud'] - cali_center_lon)**2
            )
        
        # 7. Seleccionar características finales
        self.feature_columns = [
            'latitud', 'longitud', 'categoria_rest_encoded', 'categoria_plato_encoded',
            'precio_log', 'popularidad_norm', 'year', 'month', 'day', 'weekday',
            'is_weekend', 'is_holiday_season', 'quarter', 'avg_opening_weekday', 
            'avg_opening_weekend', 'total_operating_hours', 'precio_per_popularity',
            'is_highly_popular', 'distance_from_center'
        ]
        
        # Filtrar solo las columnas que existen
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        # Llenar valores nulos
        df[self.feature_columns] = df[self.feature_columns].fillna(0)
        
        print(f"✅ Características preparadas: {len(self.feature_columns)} features")
        print(f"   📋 Features: {self.feature_columns[:5]}... (mostrando primeras 5)")
        
        return df
    

    def train_comprehensive_model(self) -> Dict[str, Any]:
        """
        Entrena múltiples modelos y genera métricas completas
        """
        if self.df_synthetic is None or self.df_synthetic.empty:
            raise ValueError("Primero debe cargar los datos sintéticos con load_synthetic_data()")
        
        print("🤖 Iniciando entrenamiento comprehensivo del modelo...")
        start_time = datetime.now()
        
        # Preparar datos
        df_processed = self.prepare_features()
        X = df_processed[self.feature_columns]
        y = df_processed['rating']
        
        print(f"📊 Datos de entrenamiento: {len(X):,} muestras, {len(self.feature_columns)} características")
        
        # División estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
        )
        
        # Escalado
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # === 1. RANDOM FOREST PARA PREDICCIÓN DE RATING ===
        print("\n🌳 Entrenando Random Forest...")
        self.model_rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model_rf.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred_train = self.model_rf.predict(X_train_scaled)
        y_pred_test = self.model_rf.predict(X_test_scaled)
        
        # Métricas Random Forest
        rf_metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'feature_importance': dict(zip(self.feature_columns, self.model_rf.feature_importances_))
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model_rf, X_train_scaled, y_train, cv=5, scoring='r2')
        rf_metrics['cv_r2_mean'] = cv_scores.mean()
        rf_metrics['cv_r2_std'] = cv_scores.std()
        
        # === 2. K-MEANS PARA CLUSTERING ===
        print("\n🎯 Entrenando K-Means Clustering...")
        
        # Determinar número óptimo de clusters (método del codo)
        inertias = []
        k_range = range(3, 11)
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(X_train_scaled)
            inertias.append(kmeans_temp.inertia_)
        
        # Seleccionar K óptimo (simplificado)
        optimal_k = 5  # Puedes implementar método del codo más sofisticado
        
        self.model_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters_train = self.model_kmeans.fit_predict(X_train_scaled)
        clusters_test = self.model_kmeans.predict(X_test_scaled)
        
        # Análisis de clusters
        cluster_analysis = {}
        for cluster in range(optimal_k):
            cluster_mask = clusters_train == cluster
            cluster_analysis[f'cluster_{cluster}'] = {
                'size': np.sum(cluster_mask),
                'avg_rating': y_train[cluster_mask].mean(),
                'avg_price': df_processed.iloc[X_train.index[cluster_mask]]['precio'].mean() if 'precio' in df_processed.columns else 0,
                'avg_popularity': df_processed.iloc[X_train.index[cluster_mask]]['popularidad'].mean() if 'popularidad' in df_processed.columns else 0
            }
        
        # === 3. MÉTRICAS AVANZADAS ===
        print("\n📈 Calculando métricas avanzadas...")
        
        # Precisión por rangos de rating
        rating_ranges = {
            'bajo (1-2)': (y_test >= 1) & (y_test < 2),
            'medio (2-3)': (y_test >= 2) & (y_test < 3), 
            'bueno (3-4)': (y_test >= 3) & (y_test < 4),
            'excelente (4-5)': (y_test >= 4) & (y_test <= 5)
        }
        
        precision_by_range = {}
        for range_name, mask in rating_ranges.items():
            if np.sum(mask) > 0:
                range_mse = mean_squared_error(y_test[mask], y_pred_test[mask])
                precision_by_range[range_name] = {
                    'count': np.sum(mask),
                    'mse': range_mse,
                    'rmse': np.sqrt(range_mse)
                }
        
        # Guardar modelos
        self.save_models()
        
        # === RESULTADOS FINALES ===
        training_time = (datetime.now() - start_time).total_seconds()
        
        results = {
            'success': True,
            'training_time_seconds': training_time,
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(self.feature_columns),
            'random_forest_metrics': rf_metrics,
            'clustering_analysis': {
                'optimal_k': optimal_k,
                'cluster_details': cluster_analysis,
                'silhouette_score': 'No calculado'  # Podrías agregarlo
            },
            'precision_by_rating_range': precision_by_range,
            'model_quality_assessment': self._assess_model_quality(rf_metrics)
        }
        
        # Generar reporte
        self._generate_training_report(results)
        
        print(f"\n🎉 Entrenamiento completado en {training_time:.2f} segundos")
        print(f"📊 RMSE Test: {rf_metrics['test_rmse']:.4f}")
        print(f"📊 R² Test: {rf_metrics['test_r2']:.4f}")
        print(f"📊 R² CV: {rf_metrics['cv_r2_mean']:.4f} (±{rf_metrics['cv_r2_std']:.4f})")
        
        return results
    

    def _assess_model_quality(self, rf_metrics: Dict) -> Dict[str, str]:
        """Evalúa la calidad del modelo entrenado"""
        assessment = {}
        
        # RMSE Assessment
        rmse = rf_metrics['test_rmse']
        if rmse < 0.3:
            assessment['rmse_quality'] = "EXCELENTE"
        elif rmse < 0.5:
            assessment['rmse_quality'] = "BUENO" 
        elif rmse < 0.8:
            assessment['rmse_quality'] = "ACEPTABLE"
        else:
            assessment['rmse_quality'] = "NECESITA MEJORA"
        
        # R² Assessment  
        r2 = rf_metrics['test_r2']
        if r2 > 0.8:
            assessment['r2_quality'] = "EXCELENTE"
        elif r2 > 0.6:
            assessment['r2_quality'] = "BUENO"
        elif r2 > 0.4:
            assessment['r2_quality'] = "ACEPTABLE" 
        else:
            assessment['r2_quality'] = "NECESITA MEJORA"
        
        # Overfitting check
        train_r2 = rf_metrics['train_r2']
        if abs(train_r2 - r2) < 0.1:
            assessment['overfitting'] = "BAJO"
        elif abs(train_r2 - r2) < 0.2:
            assessment['overfitting'] = "MODERADO"
        else:
            assessment['overfitting'] = "ALTO"
        
        return assessment


    def _generate_training_report(self, results: Dict):
        """Genera un reporte detallado del entrenamiento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.reports_dir}/training_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE ENTRENAMIENTO - SISTEMA DE RECOMENDACIONES\n")
            f.write("=" * 80 + "\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tiempo de entrenamiento: {results['training_time_seconds']:.2f} segundos\n\n")
            
            f.write("DATOS DE ENTRENAMIENTO:\n")
            f.write(f"- Total de muestras: {results['total_samples']:,}\n")
            f.write(f"- Muestras de entrenamiento: {results['train_samples']:,}\n")
            f.write(f"- Muestras de prueba: {results['test_samples']:,}\n")
            f.write(f"- Características utilizadas: {results['features_used']}\n\n")
            
            f.write("MÉTRICAS RANDOM FOREST:\n")
            rf = results['random_forest_metrics']
            f.write(f"- RMSE (Train): {rf['train_rmse']:.4f}\n")
            f.write(f"- RMSE (Test): {rf['test_rmse']:.4f}\n")
            f.write(f"- R² (Train): {rf['train_r2']:.4f}\n")
            f.write(f"- R² (Test): {rf['test_r2']:.4f}\n")
            f.write(f"- R² (Cross-Validation): {rf['cv_r2_mean']:.4f} ± {rf['cv_r2_std']:.4f}\n\n")
            
            f.write("EVALUACIÓN DE CALIDAD:\n")
            quality = results['model_quality_assessment']
            f.write(f"- Calidad RMSE: {quality['rmse_quality']}\n")
            f.write(f"- Calidad R²: {quality['r2_quality']}\n")
            f.write(f"- Nivel de Overfitting: {quality['overfitting']}\n\n")
            
            f.write("TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:\n")
            importance_sorted = sorted(rf['feature_importance'].items(), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(importance_sorted[:10], 1):
                f.write(f"{i:2d}. {feature:<25}: {importance:.4f}\n")
        
        print(f"📄 Reporte guardado en: {report_path}")


    def save_models(self):
        """Guarda todos los modelos entrenados"""
        try:
            if self.model_rf:
                joblib.dump(self.model_rf, f"{self.models_dir}/random_forest_model.pkl")
            if self.model_kmeans:
                joblib.dump(self.model_kmeans, f"{self.models_dir}/kmeans_model.pkl")
            if self.scaler:
                joblib.dump(self.scaler, f"{self.models_dir}/scaler.pkl")
            if self.label_encoders:
                joblib.dump(self.label_encoders, f"{self.models_dir}/label_encoders.pkl")
            if self.feature_columns:
                joblib.dump(self.feature_columns, f"{self.models_dir}/feature_columns.pkl")
            
            print("💾 Modelos guardados exitosamente")
        except Exception as e:
            print(f"❌ Error guardando modelos: {e}")


    def load_models(self):
        """Carga modelos pre-entrenados"""
        try:
            model_files = {
                'random_forest_model.pkl': 'model_rf',
                'kmeans_model.pkl': 'model_kmeans', 
                'scaler.pkl': 'scaler',
                'label_encoders.pkl': 'label_encoders',
                'feature_columns.pkl': 'feature_columns'
            }
            
            for filename, attr_name in model_files.items():
                filepath = f"{self.models_dir}/{filename}"
                if os.path.exists(filepath):
                    setattr(self, attr_name, joblib.load(filepath))
            
            print("📥 Modelos cargados exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            return False


    def predict_rating(self, restaurant_data: Dict) -> float:
        """Predice rating para nuevos datos"""
        if not self.model_rf:
            if not self.load_models():
                return 3.5  # Rating por defecto
        
        # Convertir a DataFrame y preparar features
        df_input = pd.DataFrame([restaurant_data])
        df_processed = self.prepare_features(df_input)
        
        # Seleccionar características
        X = df_processed[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predecir
        prediction = self.model_rf.predict(X_scaled)[0]
        return max(1.0, min(5.0, float(prediction)))


    def get_model_status(self) -> Dict[str, Any]:
        """Retorna el estado actual de los modelos"""
        return {
            'synthetic_data_loaded': self.df_synthetic is not None,
            'synthetic_data_records': len(self.df_synthetic) if self.df_synthetic is not None else 0,
            'random_forest_trained': self.model_rf is not None,
            'kmeans_trained': self.model_kmeans is not None,
            'scaler_fitted': self.scaler is not None,
            'label_encoders_count': len(self.label_encoders),
            'feature_columns_count': len(self.feature_columns),
            'models_directory': self.models_dir,
            'reports_directory': self.reports_dir
        }