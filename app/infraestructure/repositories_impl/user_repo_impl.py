from typing import List, Optional
from bson import ObjectId
from pymongo.database import Database
from app.domain.entities.user import User
from app.domain.repositories.user_repository import UserRepository

class UserRepositoryImpl(UserRepository):
    def __init__(self, db: Database):
        self.db = db
        self.collection = db["usuarios"]

    async def create(self, user: User) -> User:
        user_dict = user.model_dump(exclude={"_id"})
        result = self.collection.insert_one(user_dict)
        user_dict["_id"] = str(result.inserted_id)
        return User(**user_dict)
    
    async def get_by_id(self, user_id: str) -> Optional[User]:
        try:
            query = {"$or": [{"id": user_id}, {"_id": ObjectId(user_id)}]}
            doc = self.collection.find_one(query)
            if doc:
                doc["_id"] = str(doc["_id"])
                return User(**doc)
        except Exception:
            doc = self.collection.find_one({"id": user_id})
            if doc:
                doc["_id"] = str(doc["_id"])
                return User(**doc)
        return None
    
    async def get_all(self) -> List[User]:
        docs = list(self.collection.find({"activo": True}))
        users = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            users.append(User(**doc))
        return users
    
    async def update(self, user_id: str, user: User) -> bool:
        user_dict = user.model_dump(exclude = {"_id", "id"})
        query = {"$or": [{"id": user_id}, {"_id": ObjectId(user_id)}]}
        result = self.collection.update_one(query, {"$set": {"activo": False}})
        return result.modified_count > 0
    
    async def delete(self, user_id: str) -> bool:
        query = {"$or": [{"id": user_id}, {"_id": ObjectId(user_id)}]}
        result = self.collection.update_one(query, {"$set": {"activo": False}})
        return result.modified_count > 0
    
    async def get_by_preferences(self, preferencias: List[str]) -> List[User]:
        query = {"preferencias": {"$in": preferencias}, "activo": True}
        docs = list(self.collection.find(query))
        users = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            users.append(User(**doc))
        return users