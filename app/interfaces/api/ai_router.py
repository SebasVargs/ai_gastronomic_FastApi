from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any
import time
from app.infraestructure.db.mongo_client import get_db
from app.infraestructure.repositories_impl.user_repo_impl import UserRepositoryImpl
from app.infraestructure.repositories_impl.restaurant_repo_impl import RestaurantRepositoryImpl
from app.infraestructure.repositories_impl.plate_repo_impl import PlateRepositoryImpl
from app.infraestructure.repositories_impl.review_repo_impl import ReviewRepositoryImpl
from app.infraestructure.ai.ai_service import AIRecommendationService
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

def get_repositories():
    db = get_db()
    return {
        "user_repo": UserRepositoryImpl(db),
        "restaurant_repo": RestaurantRepositoryImpl(db),
        "plate_repo": PlateRepositoryImpl(db),
        "review_repo": ReviewRepositoryImpl(db)
    }


@router.post("/recommendations", response_model=RecommendationResponseSchema)
async def get_recommendations(
    request: RecommendationRequestSchema,
    repos = Depends(get_repositories)
): 
    try:
        recommendation_use_case = GetRecommendationUseCase(
            repos["user_repo"],
            repos("restaurant_repo"),
            repos["plate_repo"],
            repos["review_repo"]
        )

        basic_recommendatios = await recommendation_use_case.execute(
            id_usuario = request.id_usuario,
            tipo_recomendacion = request.tipo_recomendacion
        )

        if ai_service.model_trained and basic_recommendatios["recomendaciones"]:
            user = await repos["user_repo"].get_by_id(request.id_usuario)
            if user:
                restaurants_data = []
                for item in basic_recommendatios["recomendaciones"]:
                    if hasattr(item, "model_dump"):
                        restaurant_dict = item.model_dump()
                    else:
                        restaurant_dict = item

                    restaurant_dict.update({
                        "categoria": restaurant_dict.get("categoria_rest", "General"),
                        "precio": 15000,
                        "popularidad": restaurant_dict.get("rating", 3,0) * 20,
                        "year": 2024,
                        "month": 10,
                        "day": 15,
                        "weekday": 1,
                        "rating": restaurant_dict.get("rating", 3.0),
                        "es_favorito": False
                    })
                    restaurants_data.append(restaurant_dict)
            
                ai_recommendatios = ai_service.get_cluster_recommendations(
                    user_preferences = user.preferencias,
                    restaurants_data = restaurants_data
                )

                if ai_recommendatios:
                    basic_recommendatios["recomendaciones"] = ai_recommendatios[:request.limite]
                    basic_recommendatios["modelo_usado"] = True
        
        return RecommendationResponseSchema(
            tipo = basic_recommendatios["tipo"],
            id_usuario = request.id_usuario,
            criterios_usados = basic_recommendatios["criterios"],
            total_disponibles = basic_recommendatios["total_disponibles"],
            recomendaciones = basic_recommendatios["recomendaciones"],
            modelo_usado = basic_recommendatios.get("modelo_usado", False)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generado recomendaciones: {str(e)}")
    

@router.post("/train", response_model=TrainingResponseSchema)
async def train_model(
    request: TrainingRequestSchema,
    background_tasks: BackgroundTasks,
    repos = Depends(get_repositories)
):
    try:
        if ai_service.model_trained and not request.force_retrain:
            return TrainingResponseSchema(
                success=False,
                message="Modelo ya está entrenado. Use force_retrain=true para reentrenar."
            )
        
        start_time = time.time()
        dataset_use_case = GenerateDatasetUseCase(
            repos["user_repo"],
            repos["restaurant_repo"],
            repos["plato_repo"],
            repos["review_repo"]
        )

        df = await dataset_use_case.execute()

        if df.empty:
            return TrainingResponseSchema(
                success = False,
                message = "No hay datos suficientes para entrenar el modelo"
            )
        
        training_results = ai_service.train_model(df)
        training_time = time.time() - start_time

        return TrainingResponseSchema(
            success = True,
            messsage = "Modelo entrenado exitosamente",
            metrics = training_results,
            training_time = training_time,
            samples_used = len(df)
        )
    
    except Exception as e:
        return TrainingResponseSchema(
            success = False,
            message = f"Error entrenando modelo: {str(e)}"
        )
    

@router.get("/model-info", response_model=ModelInfoSchema)
async def get_model_info():
    info = ai_service.get_model_info()
    return ModelInfoSchema(**info)


@router.post("/generate-dataset")
async def generate_dataset_endpoint(repos = Depends(get_repositories)):
    try:
        dataset_use_case = GenerateDatasetUseCase(
            repos["user_repo"],
            repos["restaurant_repo"],
            repos["plate_repo"],
            repos["review_repo"]
        )

        df = await dataset_use_case.execute()
        return {
            "message": "Dataset generado exitosamente",
            "total_records": len(df),
            "columns": list(df.columns),
            "sample_data": df.head().to_dict() if not df.empty else {}
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail= f"Error generado dataset: {str(e)}")