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
    # Nuevos filtros
    precio_min: Optional[float] = Field(None, ge=0, description="Precio mínimo en COP")
    precio_max: Optional[float] = Field(None, ge=0, description="Precio máximo en COP")
    categoria: Optional[str] = Field(None, description="Categoría específica a filtrar")
    rating_minimo: Optional[float] = Field(None, ge=0, le=5, description="Rating mínimo predicho")
    abierto_ahora: Optional[bool] = Field(False, description="Solo restaurantes abiertos")
    dia_semana: Optional[int] = Field(None, ge=0, le=6, description="Día de la semana (0=Lunes)")
    hora: Optional[int] = Field(None, ge=0, le=1439, description="Hora en minutos desde medianoche")
    popularidad_minima: Optional[float] = Field(None, ge=0, le=100, description="Popularidad mínima")
    confianza_minima: Optional[float] = Field(None, ge=0, le=1, description="Confianza ML mínima (0-1)")
    excluir_categorias: Optional[List[str]] = Field(None, description="Categorías a excluir")

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