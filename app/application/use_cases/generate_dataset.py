from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from app.domain.repositories.user_repository import UserRepository
from app.domain.repositories.restaurant_repository import RestaurantRepository
from app.domain.repositories.plate_repository import PlateRepository
from app.domain.repositories.review_repository import ReviewRepository

class GenerateDatasetUseCase:
    def __init__(self, 
                 user_repository: UserRepository,
                 restaurant_repository: RestaurantRepository,
                 plate_repository: PlateRepository,
                 review_repository: ReviewRepository):
        self.user_repository = user_repository
        self.restaurant_repository = restaurant_repository
        self.plate_repository = plate_repository
        self.review_repository = review_repository

    async def execute(self) -> pd.DataFrame:
        """
        Genera un dataset con la estructura requerida para el modelo ML:
        restaurante_id, latitud, longitud, categoria_rest, horarios..., 
        plato_id, nombre_plato, categoria_plato, precio, popularidad, 
        year, month, day, weekday, rating, es_favorito
        """
        # Obtener todos los datos
        restaurants = await self.restaurant_repository.get_all()
        plates = await self.plate_repository.get_all()
        reviews = await self.review_repository.get_all()
        
        dataset_records = []
        
        for restaurant in restaurants:
            # Obtener platos del restaurante
            restaurant_plates = [p for p in plates if p.restaurante_id == restaurant.restaurante_id]
            
            if not restaurant_plates:
                continue
                
            for plate in restaurant_plates:
                # Obtener reviews del restaurante
                restaurant_reviews = [r for r in reviews if r.restaurante_id == restaurant.restaurante_id]
                
                if restaurant_reviews:
                    for review in restaurant_reviews:
                        record = {
                            'restaurante_id': restaurant.restaurante_id,
                            'latitud': restaurant.latitud,
                            'longitud': restaurant.longitud,
                            'categoria_rest': restaurant.categoria_rest,
                            'lunes_apertura_min': restaurant.horarios.lunes_apertura_min or 0,
                            'lunes_cierre_min': restaurant.horarios.lunes_cierre_min or 1440,
                            'martes_apertura_min': restaurant.horarios.martes_apertura_min or 0,
                            'martes_cierre_min': restaurant.horarios.martes_cierre_min or 1440,
                            'miércoles_apertura_min': restaurant.horarios.miércoles_apertura_min or 0,
                            'miércoles_cierre_min': restaurant.horarios.miércoles_cierre_min or 1440,
                            'jueves_apertura_min': restaurant.horarios.jueves_apertura_min or 0,
                            'jueves_cierre_min': restaurant.horarios.jueves_cierre_min or 1440,
                            'viernes_apertura_min': restaurant.horarios.viernes_apertura_min or 0,
                            'viernes_cierre_min': restaurant.horarios.viernes_cierre_min or 1440,
                            'sábado_apertura_min': restaurant.horarios.sábado_apertura_min or 0,
                            'sábado_cierre_min': restaurant.horarios.sábado_cierre_min or 1440,
                            'domingo_apertura_min': restaurant.horarios.domingo_apertura_min or 0,
                            'domingo_cierre_min': restaurant.horarios.domingo_cierre_min or 1440,
                            'plato_id': plate.plato_id,
                            'nombre_plato': plate.nombre_plato,
                            'categoria_plato': plate.categoria_plato,
                            'precio': plate.precio,
                            'popularidad': plate.popularidad,
                            'year': review.year,
                            'month': review.month,
                            'day': review.day,
                            'weekday': review.weekday,
                            'rating': review.rating,
                            'es_favorito': review.es_favorito
                        }
                        dataset_records.append(record)
                else:
                    # Si no hay reviews, crear registro con valores por defecto
                    now = datetime.utcnow()
                    record = {
                        'restaurante_id': restaurant.restaurante_id,
                        'latitud': restaurant.latitud,
                        'longitud': restaurant.longitud,
                        'categoria_rest': restaurant.categoria_rest,
                        'lunes_apertura_min': restaurant.horarios.lunes_apertura_min or 0,
                        'lunes_cierre_min': restaurant.horarios.lunes_cierre_min or 1440,
                        'martes_apertura_min': restaurant.horarios.martes_apertura_min or 0,
                        'martes_cierre_min': restaurant.horarios.martes_cierre_min or 1440,
                        'miércoles_apertura_min': restaurant.horarios.miércoles_apertura_min or 0,
                        'miércoles_cierre_min': restaurant.horarios.miércoles_cierre_min or 1440,
                        'jueves_apertura_min': restaurant.horarios.jueves_apertura_min or 0,
                        'jueves_cierre_min': restaurant.horarios.jueves_cierre_min or 1440,
                        'viernes_apertura_min': restaurant.horarios.viernes_apertura_min or 0,
                        'viernes_cierre_min': restaurant.horarios.viernes_cierre_min or 1440,
                        'sábado_apertura_min': restaurant.horarios.sábado_apertura_min or 0,
                        'sábado_cierre_min': restaurant.horarios.sábado_cierre_min or 1440,
                        'domingo_apertura_min': restaurant.horarios.domingo_apertura_min or 0,
                        'domingo_cierre_min': restaurant.horarios.domingo_cierre_min or 1440,
                        'plato_id': plate.plato_id,
                        'nombre_plato': plate.nombre_plato,
                        'categoria_plato': plate.categoria_plato,
                        'precio': plate.precio,
                        'popularidad': plate.popularidad,
                        'year': now.year,
                        'month': now.month,
                        'day': now.day,
                        'weekday': now.weekday(),
                        'rating': 3.0,  # Rating por defecto
                        'es_favorito': False
                    }
                    dataset_records.append(record)
        
        return pd.DataFrame(dataset_records)