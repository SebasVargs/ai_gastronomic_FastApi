from typing import List, Dict, Any
from app.domain.repositories.user_repository import UserRepository
from app.domain.repositories.restaurant_repository import RestaurantRepository
from app.domain.repositories.plate_repository import PlateRepository
from app.domain.repositories.review_repository import ReviewRepository

class GetRecommendationUseCase:
    def __init__(self, 
                 user_repository: UserRepository,
                 restaurant_repository: RestaurantRepository,
                 plate_repository: PlateRepository,
                 review_repository: ReviewRepository):
        self.user_repository = user_repository
        self.restaurant_repository = restaurant_repository
        self.plate_repository = plate_repository
        self.review_repository = review_repository

    async def execute(self, user_id: str, tipo_recomendacion: str = "restaurantes") -> Dict[str, Any]:
        # Obtener datos del usuario
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError("Usuario no encontrado")

        # Obtener historial de reviews del usuario
        user_reviews = await self.review_repository.get_by_user(user_id)
        
        # Obtener favoritos
        favorites = await self.review_repository.get_favorites_by_user(user_id)
        
        # Preparar datos para el modelo de ML
        user_data = {
            "id_user": user.id,
            "edad": user.edad,
            "origen": user.origen,
            "preferencias": user.preferencias,
            "reviews_count": len(user_reviews),
            "favorites_count": len(favorites),
            "avg_rating": sum([r.rating for r in user_reviews]) / len(user_reviews) if user_reviews else 0
        }

        if tipo_recomendacion == "restaurantes":
            return await self._get_restaurant_recommendations(user_data)
        elif tipo_recomendacion == "platos":
            return await self._get_plate_recommendations(user_data)
        else:
            raise ValueError("Tipo de recomendación no válido")

    async def _get_restaurant_recommendations(self, user_data: Dict) -> Dict[str, Any]:
        # Obtener todos los restaurantes
        restaurants = await self.restaurant_repository.get_all()
        
        # Filtrar por preferencias del usuario
        filtered_restaurants = []
        for restaurant in restaurants:
            if any(pref.lower() in restaurant.categoria_rest.lower() for pref in user_data["preferencias"]):
                filtered_restaurants.append(restaurant)
        
        # Si no hay coincidencias por preferencias, devolver los mejor calificados
        if not filtered_restaurants:
            filtered_restaurants = sorted(restaurants, key=lambda x: x.rating, reverse=True)[:10]
        
        return {
            "tipo": "restaurantes",
            "recomendaciones": filtered_restaurants[:5],
            "criterios": user_data["preferencias"],
            "total_disponibles": len(restaurants)
        }

    async def _get_plate_recommendations(self, user_data: Dict) -> Dict[str, Any]:
        # Obtener platos populares
        popular_plates = await self.plate_repository.get_popular_plates(20)
        
        # Filtrar por preferencias
        filtered_plates = []
        for plate in popular_plates:
            if any(pref.lower() in plate.categoria_plato.lower() for pref in user_data["preferencias"]):
                filtered_plates.append(plate)
        
        if not filtered_plates:
            filtered_plates = popular_plates[:10]
        
        return {
            "tipo": "platos",
            "recomendaciones": filtered_plates[:5],
            "criterios": user_data["preferencias"],
            "total_disponibles": len(popular_plates)
        }