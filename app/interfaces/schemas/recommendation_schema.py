from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

class RecommendationRequestSchema(BaseModel):
    user_id: str = Field(..., min_length=1)
    tipo_recomendacion: str = Field(..., pattern="^(restaurantes|platos)$")
    limite: Optional[int] = Field(5, ge=1, le=20)
    incluir_ubicacion: Optional[bool] = False
    latitud: Optional[float] = Field(None, ge=-90, le=90)
    longitud: Optional[float] = Field(None, ge=-180, le=180)
    radio_km: Optional[float] = Field(10.0, gt=0, le=100)

class RecommendationResponseSchema(BaseModel):
    tipo: str
    user_id: str
    criterios_usados: List[str]
    total_disponibles: int
    recomendaciones: List[Union[Dict[str, Any], Any]]
    modelo_usado: bool = False
    timestamp: Optional[datetime] = None

class TrainingRequestSchema(BaseModel):
    force_retrain: Optional[bool] = False
    save_model: Optional[bool] = True

class TrainingResponseSchema(BaseModel):
    success: bool
    message: str
    metrics: Optional[Dict[str, Any]] = None
    training_time: Optional[float] = None
    samples_used: Optional[int] = None

class ModelInfoSchema(BaseModel):
    model_trained: bool
    has_restaurant_model: bool
    has_scaler: bool
    has_kmeans: bool
    label_encoders_count: int
    models_directory: str