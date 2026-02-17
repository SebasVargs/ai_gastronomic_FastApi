from typing import List, Optional
from bson import ObjectId
from pymongo.database import Database
from app.domain.entities.plate import Plate
from app.domain.repositories.plate_repository import PlateRepository

class PlateRepositoryImpl(PlateRepository):
    def __init__(self, db: Database):
        self.db = db
        self.collection = db["platos"]


    async def create(self, plate: Plate) -> Plate:
        plate_dict = plate.model_dump(exclude={"_id"})
        result = self.collection.insert_one(plate_dict)
        plate_dict["_id"] = str(result.inserted_id)
        return Plate(**plate_dict)
    
        
    async def get_all(self) -> List[Plate]:
        try:
            docs = list(self.collection.find())
            plates = []
            for doc in docs:
                doc["_id"] = str(doc["_id"])
                plates.append(Plate(**doc))
            return plates
        except Exception as e:
            print(f"Error en get_all: {e}")
            return []
        

    async def get_by_restaurant(self, restaurant_id: str) -> List[Plate]:
        query = {"restaurante_id": restaurant_id}
        docs = list(self.collection.find(query))
        plates = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            plates.append(Plate(**doc))
        return plates


    async def get_by_category(self, category: str) -> List[Plate]:
        try:
            docs = list(self.collection.find({
                "categoria": category
            }))
            plates = []
            for doc in docs:
                doc["_id"] = str(doc["_id"])
                plates.append(Plate(**doc))
            return plates
        except Exception as e:
            print(f"Error en get_by_category: {e}")
            return []


    async def get_by_id(self, id_plato: str) -> Optional[Plate]:
        try:
            query = {"$or": [
                {"plato_id": id_plato},
                {"_id": ObjectId(id_plato)}
            ]}
            doc = self.collection.find_one(query)
            if doc:
                doc["_id"] = str(doc["_id"])
                return Plate(**doc)
        except Exception:
            """Aqui hubo un cambio con plato_id"""
            doc = self.collection.find_one({"id": id_plato})
            if doc:
                doc["_id"] = str(doc["_id"])
                return Plate(**doc)
        return None
    

    async def get_popular_plates(self, limit = 10) -> List[Plate]:
        docs = list(self.collection.find().sort("popularidad", -1).limit(limit))
        plates = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            plates.append(Plate(**doc))
        return plates
    
    
    async def update(self, id_plate: str, plate: Plate) -> bool:
        plate_dict = plate.model_dump(exclude={"_id", "id"})
        query = {"id": id_plate}
        result = self.collection.update_one(query, {"$set": plate_dict})
        return result.modified_count > 0
    

    async def delete(self, id_plate: str) -> bool:
        query = {"id": id_plate}
        result = self.collection.delete_one(query)
        return result.deleted_count > 0