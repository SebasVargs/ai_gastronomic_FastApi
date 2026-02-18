from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from typing import Dict, Any, Optional
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
import os
import shutil
import time
from datetime import datetime
from app.infraestructure.db.mongo_client import get_db
from app.infraestructure.repositories_impl.user_repo_impl import UserRepositoryImpl
from app.infraestructure.repositories_impl.restaurant_repo_impl import RestaurantRepositoryImpl
from app.infraestructure.repositories_impl.plate_repo_impl import PlateRepositoryImpl
from app.infraestructure.repositories_impl.review_repo_impl import ReviewRepositoryImpl
from app.infraestructure.ai.ai_service import AIRecommendationService
from app.infraestructure.ai.synthetic_data_service import SyntheticDataService
from app.infraestructure.data.csv_data_service import CsvDataService
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
csv_data_service = CsvDataService()  # Loads complementary_data.csv once


@router.get("/categories")
async def get_categories():
    """Returns the list of unique food categories from the CSV data."""
    stats = csv_data_service.get_stats()
    return stats.get("categories", [])


def get_repositories():
    db = get_db()
    return {
        "user_repo": UserRepositoryImpl(db),
        "restaurant_repo": RestaurantRepositoryImpl(db),
        "plate_repo": PlateRepositoryImpl(db),
        "review_repo": ReviewRepositoryImpl(db)
    }

# ENDPOINTS PARA DATOS SINTÉTICOS -----------------------------------

class CsvRequest(BaseModel):
    csv_file_path: str

@router.post("/load-synthetic-data")
async def load_synthetic_data(request: CsvRequest):
    """
    Carga tu archivo CSV de 80,000 registros sintéticos
    Body: {"csv_file_path": "/path/to/your/synthetic_data.csv"}
    """
    csv_file_path = request.csv_file_path
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
    """
    try:
        if not synthetic_ai_service.model_rf:
            synthetic_ai_service.load_models()

        user = await repos["user_repo"].get_by_id(request.user_id)

        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        if request.tipo_recomendacion == "restaurantes":
            items = csv_data_service.get_all_restaurants()
        else:
            items = csv_data_service.get_all_plates()

        print(f"📊 Total items obtenidos del CSV: {len(items)}")
        
        if not items:
            return RecommendationResponseSchema(
                tipo = request.tipo_recomendacion,
                user_id = request.user_id,
                criterios_usados = user.preferencias,
                total_disponibles = 0,
                recomendaciones = [],
                modelo_usado = False
            )
        
        items_for_processing = []
        for item in items:
            if hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            else:
                item_dict = item
            
            horarios = item_dict.get('horarios', {})
            if hasattr(horarios, 'model_dump'):
                horarios = horarios.model_dump()
            elif not isinstance(horarios, dict):
                horarios = {}

            ml_data = {
                'restaurante_id': item_dict.get('restaurante_id') or item_dict.get('id') or 'unknown',
                'latitud': item_dict.get('latitud', 4.6097),
                'longitud': item_dict.get('longitud', -74.0817),
                'categoria_rest': item_dict.get('categoria_rest') or item_dict.get('categoria') or 'General',
                'categoria_plato': item_dict.get('categoria_plato') or item_dict.get('categoria') or 'General',
                'precio': item_dict.get('precio', 15000),
                'popularidad': item_dict.get('popularidad') or ((item_dict.get('rating') or 3.0) * 20),
                'year': datetime.now().year,
                'month': datetime.now().month,
                'day': datetime.now().day,
                'weekday': datetime.now().weekday(),
                'rating': item_dict.get('rating') or ((item_dict.get('popularidad') or 60) / 20.0),
                'es_favorito': False,
                'lunes_apertura_min': item_dict.get('lunes_apertura_min') or horarios.get('lunes_apertura_min', 480),
                'lunes_cierre_min': item_dict.get('lunes_cierre_min') or horarios.get('lunes_cierre_min', 1320),
                'martes_apertura_min': item_dict.get('martes_apertura_min') or horarios.get('martes_apertura_min', 480),
                'martes_cierre_min': item_dict.get('martes_cierre_min') or horarios.get('martes_cierre_min', 1320),
                'miércoles_apertura_min': item_dict.get('miércoles_apertura_min') or horarios.get('miércoles_apertura_min', 480),
                'miércoles_cierre_min': item_dict.get('miércoles_cierre_min') or horarios.get('miércoles_cierre_min', 1320),
                'jueves_apertura_min': item_dict.get('jueves_apertura_min') or horarios.get('jueves_apertura_min', 480),
                'jueves_cierre_min': item_dict.get('jueves_cierre_min') or horarios.get('jueves_cierre_min', 1320),
                'viernes_apertura_min': item_dict.get('viernes_apertura_min') or horarios.get('viernes_apertura_min', 480),
                'viernes_cierre_min': item_dict.get('viernes_cierre_min') or horarios.get('viernes_cierre_min', 1320),
                'sábado_apertura_min': item_dict.get('sábado_apertura_min') or horarios.get('sábado_apertura_min', 540),
                'sábado_cierre_min': item_dict.get('sábado_cierre_min') or horarios.get('sábado_cierre_min', 1380),
                'domingo_apertura_min': item_dict.get('domingo_apertura_min') or horarios.get('domingo_apertura_min', 600),
                'domingo_cierre_min': item_dict.get('domingo_cierre_min') or horarios.get('domingo_cierre_min', 1320),
            }
            
            item_dict['ml_data'] = ml_data
            items_for_processing.append(item_dict)

        filtered_items = items_for_processing
        categorias_seleccionadas = []

        if request.precio_min is not None or request.precio_max is not None:
            precio_min = request.precio_min if request.precio_min is not None else 0
            precio_max = request.precio_max if request.precio_max is not None else float('inf')
            filtered_items = [item for item in filtered_items 
                            if precio_min <= item['ml_data'].get('precio', 0) <= precio_max]
            print(f"💰 Después de filtro precio ({precio_min}-{precio_max}): {len(filtered_items)} items")
        
        if request.categoria:
            categorias_seleccionadas = [c.strip().lower() for c in request.categoria.split(',') if c.strip()]
            filtered_items = [item for item in filtered_items 
                            if any(cat in (item['ml_data'].get('categoria_rest') or item['ml_data'].get('categoria_plato') or item.get('categoria') or '').lower() 
                                   for cat in categorias_seleccionadas)]
            print(f"🍕 Después de filtro categoría '{request.categoria}': {len(filtered_items)} items")
        
        if request.abierto_ahora and request.tipo_recomendacion == "restaurantes":
            now = datetime.now()
            dia_actual = request.dia_semana if request.dia_semana is not None else now.weekday()
            hora_actual = request.hora if request.hora is not None else now.hour * 60 + now.minute
            
            dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
            dia_nombre = dias_semana[dia_actual]
            
            items_abiertos = []
            for item in filtered_items:
                ml = item['ml_data']
                apertura_key = f'{dia_nombre}_apertura_min'
                cierre_key = f'{dia_nombre}_cierre_min'
                apertura = ml.get(apertura_key, 0)
                cierre = ml.get(cierre_key, 1440)
                
                if apertura <= hora_actual <= cierre:
                    items_abiertos.append(item)
            filtered_items = items_abiertos
            print(f"🕐 Después de filtro horario (abierto {dia_nombre} a las {hora_actual} min): {len(filtered_items)} items")

        if request.popularidad_minima is not None:
            filtered_items = [item for item in filtered_items 
                            if item['ml_data'].get('popularidad', 0) >= request.popularidad_minima]
            print(f"📈 Después de filtro popularidad mínima ({request.popularidad_minima}): {len(filtered_items)} items")

        if request.excluir_categorias:
            excluir_lower = [cat.lower() for cat in request.excluir_categorias]
            filtered_items = [item for item in filtered_items 
                            if not any(exc in (item['ml_data'].get('categoria_rest') or item['ml_data'].get('categoria_plato') or item.get('categoria') or '').lower() 
                                      for exc in excluir_lower)]
            print(f"❌ Después de filtro exclusión {request.excluir_categorias}: {len(filtered_items)} items")

        print(f"🤖 Ejecutando predicción ML batch en {len(filtered_items)} items...")

        # Extraer ml_data de todos los items filtrados de una vez
        ml_data_batch = [item['ml_data'] for item in filtered_items]

        # UNA sola llamada al modelo → vectorizado con pandas/sklearn
        predicted_ratings = synthetic_ai_service.predict_batch(ml_data_batch)

        # Asignar resultados de vuelta a cada item
        for item, predicted_rating in zip(filtered_items, predicted_ratings):
            item['predicted_rating'] = predicted_rating
            item['ml_confidence'] = min(predicted_rating / 5.0, 1.0)

        if request.rating_minimo is not None:
            filtered_items = [item for item in filtered_items 
                            if item.get('predicted_rating', 0) >= request.rating_minimo]
            print(f"⭐ Después de filtro rating mínimo ({request.rating_minimo}): {len(filtered_items)} items")
            
        if request.confianza_minima is not None:
            filtered_items = [item for item in filtered_items 
                            if item.get('ml_confidence', 0) >= request.confianza_minima]
            print(f"🎯 Después de filtro confianza ML mínima ({request.confianza_minima}): {len(filtered_items)} items")
        
        for item in filtered_items:
            category = item['ml_data'].get('categoria_rest') or item['ml_data'].get('categoria_plato') or ''
            if any(pref.lower() in category.lower() for pref in user.preferencias):
                item['preference_match'] = True
            else:
                item['preference_match'] = False
        
        print(f"✅ Items después de todos los filtros: {len(filtered_items)}")
        
        # DISTRIBUCIÓN PROPORCIONAL POR CATEGORÍA
        top_recommendations = []
        
        if len(categorias_seleccionadas) > 1:
            # Múltiples categorías: distribuir proporcionalmente
            items_por_categoria = request.limite // len(categorias_seleccionadas)
            items_extra = request.limite % len(categorias_seleccionadas)
            
            print(f"🎯 Distribuyendo {request.limite} items entre {len(categorias_seleccionadas)} categorías: {items_por_categoria} c/u + {items_extra} extra")
            
            for idx, cat in enumerate(categorias_seleccionadas):
                items_cat = [item for item in filtered_items 
                            if cat in (item['ml_data'].get('categoria_rest') or item['ml_data'].get('categoria_plato') or item.get('categoria') or '').lower()]
                
                items_cat.sort(
                    key=lambda x: (x.get('preference_match', False), x.get('predicted_rating', 3.0)), 
                    reverse=True
                )
                
                cantidad = items_por_categoria + (1 if idx < items_extra else 0)
                top_recommendations.extend(items_cat[:cantidad])
                print(f"  📋 {cat}: {len(items_cat)} disponibles, tomando {cantidad}")
            
            # Mezclar resultados para variedad
            import random
            random.shuffle(top_recommendations)
        else:
            # Una sola categoría o sin categoría: ordenar normalmente
            filtered_items.sort(
                key=lambda x: (x.get('preference_match', False), x.get('predicted_rating', 3.0)), 
                reverse=True
            )
            top_recommendations = filtered_items[:request.limite]
        
        print(f"🎁 Retornando {len(top_recommendations)} recomendaciones")
        
        criterios = user.preferencias + ["ML Prediction", "Rating Prediction"]
        if request.precio_min or request.precio_max:
            criterios.append(f"Precio: ${request.precio_min or 0:,.0f} - ${request.precio_max or 999999:,.0f}")
        if request.categoria:
            criterios.append(f"Categoría: {request.categoria}")
        if request.rating_minimo:
            criterios.append(f"Rating mínimo: {request.rating_minimo}")
        if request.abierto_ahora:
            criterios.append("Abierto ahora")
        if request.popularidad_minima:
            criterios.append(f"Popularidad mínima: {request.popularidad_minima}")
        if request.confianza_minima:
            criterios.append(f"Confianza ML mínima: {request.confianza_minima*100:.0f}%")
        if request.excluir_categorias:
            criterios.append(f"Excluir: {', '.join(request.excluir_categorias)}")
        
        return RecommendationResponseSchema(
            tipo=request.tipo_recomendacion,
            user_id=request.user_id,
            criterios_usados=criterios,
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