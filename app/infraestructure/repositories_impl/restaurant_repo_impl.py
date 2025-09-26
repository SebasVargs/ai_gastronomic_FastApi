from typing import List, Optional
from bson import ObjectId
from pymongo.database import Database
import math
from app.domain.entities.restaurant import Restaurant
from app.domain.repositories.restaurant_repository import RestaurantRepository

class RestaurantRepositoryImpl(RestaurantRepository):
    def __init__(self, db: Database):
        self.db = db
        self.collection = db["restaurantes"]

    async def create(self, restaurant: Restaurant) -> Restaurant:
        restaurant_dict = restaurant.model_dump(exclude = {"_id"})
        result = self.collection.insert_one(restaurant_dict)
        restaurant_dict["_id"] = str(result.inserted_id)
        return Restaurant(**restaurant_dict)
    
    async def get_by_id(self, id_restaurante: str) -> Optional[Restaurant]:
        try:
            query = {"$or": [
                {"restaurante_id": id_restaurante},
                {"id": id_restaurante},
                {"_id": ObjectId(id_restaurante)}
            ]}
            doc = self.collection.find_one(query)
            if doc:
                doc["_id"] = str(doc["_id"])
                return Restaurant(**doc)
        except Exception:
            doc = self.collection.find_one({"restaurante_id": id_restaurante})
            if doc:
                doc["_id"] = str(doc["_id"])
                return Restaurant(**doc)
        return None
    
    async def get_all(self) -> List[Restaurant]:
        docs = list(self.collection.find({"activo": True}))
        restaurants = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            restaurants.append(Restaurant(**doc))
        return restaurants
    
    async def get_by_category(self, categoria: str) -> List[Restaurant]:
        query = {
            "categoria_rest": {"$regex": categoria, "$options": "i"},
            "activo": True
        }
        docs = list(self.collection.find(query))
        restaurants = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            restaurants.append(Restaurant(**doc))
        return restaurants
    
    async def get_by_location(self, latitud: float, longitud: float, radio_km: float) -> List[Restaurant]:
        # Convertir km a grados aproximadamente (1 grado ≈ 111 km)
        radio_grados = radio_km / 111.0
        
        query = {
            "latitud": {
                "$gte": latitud - radio_grados,
                "$lte": latitud + radio_grados
            },
            "longitud": {
                "$gte": longitud - radio_grados,
                "$lte": longitud + radio_grados
            },
            "activo": True
        }
        
        docs = list(self.collection.find(query))
        restaurants = []
        
        for doc in docs:
            # Cálculo más preciso de distancia usando fórmula haversine
            dist = self._calculate_distance(latitud, longitud, doc["latitud"], doc["longitud"])
            if dist <= radio_km:
                doc["_id"] = str(doc["_id"])
                doc["distancia_km"] = round(dist, 2)
                restaurants.append(Restaurant(**doc))
        
        return sorted(restaurants, key=lambda x: x.distancia_km if hasattr(x, 'distancia_km') else 0)
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula distancia entre dos puntos usando fórmula haversine"""
        R = 6371  # Radio de la Tierra en km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

    async def update(self, id_restaurante: str, restaurant: Restaurant) -> bool:
        restaurant_dict = restaurant.model_dump(exclude={"_id", "id", "restaurante_id"})
        query = {"restaurante_id": id_restaurante}
        result = self.collection.update_one(query, {"$set": restaurant_dict})
        return result.modified_count > 0

    async def delete(self, id_restaurante: str) -> bool:
        query = {"restaurante_id": id_restaurante}
        result = self.collection.update_one(query, {"$set": {"activo": False}})
        return result.modified_count > 0