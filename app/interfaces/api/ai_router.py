from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from typing import Dict, Any, Optional
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler
import os
import shutil
import time
from app.infraestructure.db.mongo_client import get_db
from app.infraestructure.repositories_impl.user_repo_impl import UserRepositoryImpl
from app.infraestructure.repositories_impl.restaurant_repo_impl import RestaurantRepositoryImpl
from app.infraestructure.repositories_impl.plate_repo_impl import PlateRepositoryImpl
from app.infraestructure.repositories_impl.review_repo_impl import ReviewRepositoryImpl
from app.infraestructure.ai.ai_service import AIRecommendationService
from app.infraestructure.ai.synthetic_data_service import SyntheticDataService
from app.application.use_cases.get_recommendation import GetRecommendationUseCase
from app.application.use_cases.generate_dataset import GenerateDatasetUseCase
from app.interfaces.schemas.recommendation_schema import (
    RecommendationRequestSchema,
    RecommendationResponseSchema,
    TrainingRequestSchema,
    TrainingResponseSchema,
    ModelInfoSchema
)

router = APIRouter(prefix="/api/ai", tags=["ai"])

ai_service = AIRecommendationService()
synthetic_ai_service = SyntheticDataService()

def get_repositories():
    db = get_db()
    return {
        "user_repo": UserRepositoryImpl(db),
        "restaurant_repo": RestaurantRepositoryImpl(db),
        "plate_repo": PlateRepositoryImpl(db),
        "review_repo": ReviewRepositoryImpl(db)
    }

# ENDPOINTS PARA DATOS SINTÉTICOS -----------------------------------

@router.post("/load-synthetic-data")
async def load_synthetic_data(csv_file_path: str):
    """
    Carga tu archivo CSV de 80,000 registros sintéticos
    
    Parámetros:
    - csv_file_path: Ruta al archivo CSV con datos sintéticos
    
    Ejemplo: POST /api/ai/load-synthetic-data
    Body: {"csv_file_path": "/path/to/your/synthetic_data.csv"}
    """
    try:
        if not os.path.exists(csv_file_path):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {csv_file_path}")
        
        info = synthetic_ai_service.load_synthetic_data(csv_file_path)
        
        return {
            "success": True,
            "message": "Datos sintéticos cargados exitosamente",
            "data_info": info
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error cargando datos sintéticos: {str(e)}")


@router.post("/upload-synthetic-csv")
async def upload_synthetic_csv(file: UploadFile = File(...)):
    """
    Sube y carga tu archivo CSV directamente
    
    Uso en Postman:
    - Method: POST
    - URL: http://localhost:8000/api/ai/upload-synthetic-csv
    - Body: form-data, key="file", type=File, selecciona tu CSV
    """
    try:
        # Verificar que es un CSV
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")
        
        # Guardar archivo temporalmente
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = f"{upload_dir}/{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Cargar datos
        info = synthetic_ai_service.load_synthetic_data(file_path)
        
        return {
            "success": True,
            "message": f"Archivo {file.filename} cargado y procesado exitosamente",
            "file_path": file_path,
            "data_info": info
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando archivo CSV: {str(e)}")


@router.post("/train-with-synthetic-data", response_model=TrainingResponseSchema)
async def train_with_synthetic_data(
    background_tasks: BackgroundTasks,
    force_retrain: bool = False
):
    """
    Entrena el modelo usando los datos sintéticos cargados
    
    Este endpoint:
    1. Verifica que los datos sintéticos estén cargados
    2. Entrena Random Forest + K-Means
    3. Genera métricas detalladas de precisión
    4. Guarda modelos entrenados
    5. Crea reporte de entrenamiento
    
    Parámetros:
    - force_retrain: True para reentrenar aunque ya exista un modelo
    """
    try:
        # Verificar que los datos sintéticos estén cargados
        if synthetic_ai_service.df_synthetic is None:
            raise HTTPException(
                status_code=400, 
                detail="Primero debe cargar los datos sintéticos usando /load-synthetic-data o /upload-synthetic-csv"
            )
        
        # Verificar si ya hay un modelo entrenado
        if synthetic_ai_service.model_rf and not force_retrain:
            return TrainingResponseSchema(
                success=False,
                message="Modelo ya entrenado. Use force_retrain=true para reentrenar."
            )
        
        print("🚀 Iniciando entrenamiento con datos sintéticos...")
        start_time = time.time()
        
        # Entrenar modelo comprehensivo
        results = synthetic_ai_service.train_comprehensive_model()
        
        training_time = time.time() - start_time
        
        return TrainingResponseSchema(
            success=True,
            message="Modelo entrenado exitosamente con datos sintéticos",
            metrics={
                "total_samples": results['total_samples'],
                "training_time": results['training_time_seconds'],
                "test_rmse": results['random_forest_metrics']['test_rmse'],
                "test_r2": results['random_forest_metrics']['test_r2'],
                "cv_r2_mean": results['random_forest_metrics']['cv_r2_mean'],
                "model_quality": results['model_quality_assessment'],
                "clustering_clusters": results['clustering_analysis']['optimal_k'],
                "precision_by_range": results['precision_by_rating_range']
            },
            training_time=training_time,
            samples_used=results['total_samples']
        )
        
    except Exception as e:
        return TrainingResponseSchema(
            success=False,
            message=f"Error entrenando modelo: {str(e)}"
        )

@router.get("/model-performance")
async def get_model_performance():
    """
    Obtiene métricas detalladas de rendimiento del modelo
    
    Retorna:
    - Métricas de precisión (RMSE, R², MAE)
    - Análisis de clustering
    - Importancia de características
    - Evaluación de calidad del modelo
    """
    try:
        if not synthetic_ai_service.model_rf:
            synthetic_ai_service.load_models()
            
        if not synthetic_ai_service.model_rf:
            raise HTTPException(status_code=404, detail="No hay modelo entrenado disponible")
        
        # Información del modelo cargado
        status = synthetic_ai_service.get_model_status()
        
        # Si hay datos sintéticos, obtener métricas adicionales
        additional_info = {}
        if synthetic_ai_service.df_synthetic is not None:
            df = synthetic_ai_service.df_synthetic
            additional_info = {
                "data_distribution": {
                    "rating_mean": float(df['rating'].mean()),
                    "rating_std": float(df['rating'].std()),
                    "rating_min": float(df['rating'].min()),
                    "rating_max": float(df['rating'].max()),
                    "total_restaurants": int(df['restaurante_id'].nunique()),
                    "total_plates": int(df['plato_id'].nunique()),
                    "avg_price": float(df['precio'].mean()) if 'precio' in df.columns else 0,
                    "avg_popularity": float(df['popularidad'].mean()) if 'popularidad' in df.columns else 0
                }
            }
        
        return {
            "model_status": status,
            "additional_metrics": additional_info,
            "recommendations": [
                "✅ Modelo cargado y listo para predicciones" if status['random_forest_trained'] else "❌ Modelo no entrenado",
                "✅ Clustering disponible" if status['kmeans_trained'] else "❌ Clustering no disponible",
                f"📊 {status['feature_columns_count']} características en el modelo",
                f"📈 {status['synthetic_data_records']:,} registros sintéticos disponibles" if status['synthetic_data_loaded'] else "📂 Sin datos sintéticos cargados"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error obteniendo métricas: {str(e)}")

# ENDPOINTS PARA DATOS REALES ---------------------------------------

@router.post("/recommendations", response_model=RecommendationResponseSchema)
async def get_advanced_recommendations(
    request: RecommendationRequestSchema,
    repos = Depends(get_repositories)
): 
    """
    Obtiene recomendaciones usando el modelo entrenado con datos sintéticos
    
    Flujo:
    1. Obtiene perfil del usuario
    2. Aplica modelo ML entrenado
    3. Usa clustering para usuarios similares
    4. Combina con filtros tradicionales
    5. Retorna recomendaciones personalizadas y precisas
    """
    try:
        if not synthetic_ai_service.model_rf:
            synthetic_ai_service.load_models()

        user = await repos["user_repo"].get_by_id(request.id_usuario)

        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        if request.tipo_recomendacion == "restaurantes":
            if request.incluir_ubicacion and request.latitud and request.longitud:
                items = await repos["restaurant_repo"].get_by_location(
                    request.latitud, request.longitud, request.radio_km
                )
            else:
                items = await repos["restaurant_repo"].get_all()
        else:
            items = await repos["plate_repo"].get_popular_plates(50)

        if not items:
            return RecommendationResponseSchema(
                tipo = request.tipo_recomendacion,
                id_usuario = request.id_usuario,
                criterios_usados = user.preferencias,
                total_disponibles = 0,
                recomendaciones = [],
                modelo_usado = False
            )
        
        items_for_ml = []
        for item in items:
            if hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            else:
                item_dict = item
            
            ml_data = {
                'restaurante_id': item_dict.get('restaurante_id', 'unknown'),
                'latitud': item_dict.get('latitud', 4.6097),
                'longitud': item_dict.get('longitud', -74.0817),
                'categoria_rest': item_dict.get('categoria_rest', 'General'),
                'categoria_plato': item_dict.get('categoria_plato', item_dict.get('categoria_rest', 'General')),
                'precio': item_dict.get('precio', 15000),
                'popularidad': item_dict.get('popularidad', item_dict.get('rating', 3.0) * 20),
                'year': 2024,
                'month': 10,
                'day': 15,
                'weekday': 1,
                'rating': item_dict.get('rating', 3.0),
                'es_favorito': False,
                # Horarios (valores por defecto si no están disponibles)
                'lunes_apertura_min': item_dict.get('horarios', {}).get('lunes_apertura_min', 480) if 'horarios' in item_dict else 480,
                'lunes_cierre_min': item_dict.get('horarios', {}).get('lunes_cierre_min', 1320) if 'horarios' in item_dict else 1320,
                'martes_apertura_min': item_dict.get('horarios', {}).get('martes_apertura_min', 480) if 'horarios' in item_dict else 480,
                'martes_cierre_min': item_dict.get('horarios', {}).get('martes_cierre_min', 1320) if 'horarios' in item_dict else 1320,
                'miércoles_apertura_min': item_dict.get('horarios', {}).get('miércoles_apertura_min', 480) if 'horarios' in item_dict else 480,
                'miércoles_cierre_min': item_dict.get('horarios', {}).get('miércoles_cierre_min', 1320) if 'horarios' in item_dict else 1320,
                'jueves_apertura_min': item_dict.get('horarios', {}).get('jueves_apertura_min', 480) if 'horarios' in item_dict else 480,
                'jueves_cierre_min': item_dict.get('horarios', {}).get('jueves_cierre_min', 1320) if 'horarios' in item_dict else 1320,
                'viernes_apertura_min': item_dict.get('horarios', {}).get('viernes_apertura_min', 480) if 'horarios' in item_dict else 480,
                'viernes_cierre_min': item_dict.get('horarios', {}).get('viernes_cierre_min', 1320) if 'horarios' in item_dict else 1320,
                'sábado_apertura_min': item_dict.get('horarios', {}).get('sábado_apertura_min', 540) if 'horarios' in item_dict else 540,
                'sábado_cierre_min': item_dict.get('horarios', {}).get('sábado_cierre_min', 1380) if 'horarios' in item_dict else 1380,
                'domingo_apertura_min': item_dict.get('horarios', {}).get('domingo_apertura_min', 600) if 'horarios' in item_dict else 600,
                'domingo_cierre_min': item_dict.get('horarios', {}).get('domingo_cierre_min', 1320) if 'horarios' in item_dict else 1320,
            }
            
            # Predecir rating usando el modelo entrenado
            try:
                predicted_rating = synthetic_ai_service.predict_rating(ml_data)
                item_dict['predicted_rating'] = predicted_rating
                item_dict['ml_confidence'] = min(predicted_rating / 5.0, 1.0)
            except Exception as e:
                print(f"Error prediciendo rating: {e}")
                item_dict['predicted_rating'] = item_dict.get('rating', 3.5)
                item_dict['ml_confidence'] = 0.5
            
            items_for_ml.append(item_dict)
        
        # Filtrar por preferencias del usuario
        filtered_items = []
        for item in items_for_ml:
            category = item.get('categoria_rest', item.get('categoria_plato', ''))
            if any(pref.lower() in category.lower() for pref in user.preferencias):
                item['preference_match'] = True
                filtered_items.append(item)
        
        # Si no hay coincidencias por preferencias, usar todos pero marcar como no match
        if not filtered_items:
            for item in items_for_ml:
                item['preference_match'] = False
                filtered_items.append(item)
        
        # Ordenar por rating predicho y coincidencia de preferencias
        filtered_items.sort(
            key=lambda x: (x.get('preference_match', False), x.get('predicted_rating', 3.0)), 
            reverse=True
        )
        
        # Tomar top recomendaciones
        top_recommendations = filtered_items[:request.limite]
        
        return RecommendationResponseSchema(
            tipo=request.tipo_recomendacion,
            user_id=request.user_id,
            criterios_usados=user.preferencias + ["ML Prediction", "Rating Prediction"],
            total_disponibles=len(items),
            recomendaciones=top_recommendations,
            modelo_usado=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generando recomendaciones: {str(e)}")

@router.post("/predict-single_rating")
async def predict_single_rating(resturant_data: dict):
    """
    Predice el rating para un restaurante/plato específico
    
    Útil para:
    - Validar nuevos restaurantes antes de agregarlos
    - Estimar qué tan bien será recibido un plato
    - Testing del modelo
    
    Body example:
    {
        "restaurante_id": "test_001",
        "latitud": 4.6097,
        "longitud": -74.0817,
        "categoria_rest": "Comida rápida",
        "categoria_plato": "Hamburguesas",
        "precio": 18000,
        "popularidad": 75,
        "lunes_apertura_min": 480,
        "lunes_cierre_min": 1320
    }
    """
    try:
        if not synthetic_ai_service.model_rf:
            synthetic_ai_service.load_models()

        if not synthetic_ai_service.model_rf:
            raise HTTPException(status_code=404, detail="Modelo no entrenado. Entrena primero con /train-with-synthetic-data")
        

        predicted_rating = synthetic_ai_service.predict_rating(resturant_data)
        confidence_level = "Alta" if predicted_rating > 4.0 or predicted_rating < 2.0 else "Media"
        recommendation_text = ""

        if predicted_rating >= 4.5:
            recommendation_text = "🌟 Excelente opción, muy recomendado"
        elif predicted_rating >= 4.0:
            recommendation_text = "✅ Buena opción, recomendado"
        elif predicted_rating >= 3.0:
            recommendation_text = "⚠️  Opción aceptable"
        else:
            recommendation_text = "❌ No recomendado, rating bajo esperado"
        
        return {
            "predicted_rating": round(predicted_rating, 2),
            "confidence_level": confidence_level,
            "recommendation": recommendation_text,
            "input_data": resturant_data,
            "model_version": "synthetic_trained_v1"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error prediciendo rating: {str(e)}")            


@router.get("/synthetic-data-status")
async def get_synthetic_data_status():
    """
    Verifica el estado de los datos sintéticos cargados
    """
    status = synthetic_ai_service.get_model_status()
    
    return {
        "synthetic_data_loaded": status['synthetic_data_loaded'],
        "total_records": status['synthetic_data_records'],
        "sample_preview": synthetic_ai_service.df_synthetic.head(3).to_dict() if synthetic_ai_service.df_synthetic is not None else {},
        "models_trained": {
            "random_forest": status['random_forest_trained'],
            "kmeans": status['kmeans_trained']
        },
        "next_steps": [
            "✅ Datos sintéticos cargados" if status['synthetic_data_loaded'] else "1️⃣ Cargar datos sintéticos con /upload-synthetic-csv",
            "✅ Modelo entrenado" if status['random_forest_trained'] else "2️⃣ Entrenar modelo con /train-with-synthetic-data",
            "✅ Listo para recomendaciones" if status['random_forest_trained'] else "3️⃣ Obtener recomendaciones con /recommendations"
        ]
    }


@router.get("/feature-importance")
async def get_feature_importance():
    """
    Obtiene la importancia de las características del modelo Random Forest
    
    Útil para entender qué factores son más importantes para las recomendaciones
    """
    try:
        if not synthetic_ai_service.model_rf:
            synthetic_ai_service.load_models()
        
        if not synthetic_ai_service.model_rf or not synthetic_ai_service.feature_columns:
            raise HTTPException(status_code=404, detail="Modelo no entrenado o características no disponibles")
        
        # Obtener importancia de características
        feature_importance = dict(zip(
            synthetic_ai_service.feature_columns, 
            synthetic_ai_service.model_rf.feature_importances_
        ))
        
        # Ordenar por importancia
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_features": len(sorted_features),
            "top_10_features": {
                feature: round(importance, 4) 
                for feature, importance in sorted_features[:10]
            },
            "all_features": {
                feature: round(importance, 4) 
                for feature, importance in sorted_features
            },
            "interpretation": {
                "most_important": sorted_features[0][0] if sorted_features else "N/A",
                "least_important": sorted_features[-1][0] if sorted_features else "N/A",
                "top_3_categories": [f[0] for f in sorted_features[:3]]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error obteniendo importancia de características: {str(e)}")


@router.get("/model-info", response_model=ModelInfoSchema)
async def get_enhanced_model_info():
    """
    Información completa del estado del sistema de ML
    """
    status = synthetic_ai_service.get_model_status()
    
    return ModelInfoSchema(
        model_trained=status['random_forest_trained'],
        has_restaurant_model=status['random_forest_trained'],
        has_scaler=status['scaler_fitted'],
        has_kmeans=status['kmeans_trained'],
        label_encoders_count=status['label_encoders_count'],
        models_directory=status['models_directory']
    )


@router.delete("/reset-models")
async def reset_all_models():
    """
    Resetea todos los modelos y datos cargados
    
    ⚠️  CUIDADO: Esta acción no se puede deshacer
    """
    try:
        # Limpiar modelos en memoria
        synthetic_ai_service.df_synthetic = None
        synthetic_ai_service.model_rf = None
        synthetic_ai_service.model_kmeans = None
        synthetic_ai_service.scaler = StandardScaler()
        synthetic_ai_service.label_encoders = {}
        synthetic_ai_service.feature_columns = []
        
        return {
            "success": True,
            "message": "Todos los modelos y datos han sido reseteados",
            "note": "Los archivos guardados en disco no fueron eliminados"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reseteando modelos: {str(e)}")