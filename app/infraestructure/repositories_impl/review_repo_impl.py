from typing import List, Optional
from bson import ObjectId
from pymongo.database import Database
from ...domain.entities.review import Review
from app.domain.entities.plate import Plate
from ...domain.repositories.review_repository import ReviewRepository

class ReviewRepositoryImpl(ReviewRepository):
    def __init__(self, db: Database):
        self.db = db
        self.collection = db["reviews"]

    async def create(self, review: Review) -> Review:
        review_dict = review.model_dump(exclude={"_id"})
        result = self.collection.insert_one(review_dict)
        review_dict["_id"] = str(result.inserted_id)
        return Review(**review_dict)

    async def get_by_user(self, user_id: str) -> List[Review]:
        query = {"id_usuario": user_id}
        docs = list(self.collection.find(query).sort("fecha", -1))
        reviews = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            reviews.append(Review(**doc))
        return reviews

    async def get_by_restaurant(self, restaurant_id: str) -> List[Review]:
        query = {"id_restaurante": restaurant_id}
        docs = list(self.collection.find(query).sort("fecha", -1))
        reviews = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            reviews.append(Review(**doc))
        return reviews

    async def get_favorites_by_user(self, user_id: str) -> List[Review]:
        query = {"id_usuario": user_id, "es_favorito": True}
        docs = list(self.collection.find(query))
        reviews = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            reviews.append(Review(**doc))
        return reviews

    async def get_by_id(self, review_id: str) -> Optional[Review]:
        try:
            query = {"$or": [{"id": review_id}, {"_id": ObjectId(review_id)}]}
            doc = self.collection.find_one(query)
            if doc:
                doc["_id"] = str(doc["_id"])
                return Review(**doc)
        except Exception:
            doc = self.collection.find_one({"id": review_id})
            if doc:
                doc["_id"] = str(doc["_id"])
                return Review(**doc)
        return None

    async def update(self, review_id: str, review: Review) -> bool:
        review_dict = review.model_dump(exclude={"_id", "id"})
        query = {"$or": [{"id": review_id}, {"_id": ObjectId(review_id)}]}
        result = self.collection.update_one(query, {"$set": review_dict})
        return result.modified_count > 0

    async def delete(self, review_id: str) -> bool:
        query = {"$or": [{"id": review_id}, {"_id": ObjectId(review_id)}]}
        result = self.collection.delete_one(query)
        return result

    async def get_by_category(self, categoria: str) -> List[Plate]:
        query = {
            "categoria": {"$regex": categoria, "$options": "i"},
            "activo": True
        }
        docs = list(self.collection.find(query))
        plates = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            plates.append(Plate(**doc))
        return plates