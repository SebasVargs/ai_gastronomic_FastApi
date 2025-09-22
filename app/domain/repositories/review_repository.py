from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities.review import Review

class ReviewRepository(ABC):
    @abstractmethod
    async def create(self, review: Review) -> Review:
        pass

    @abstractmethod
    async def get_by_user(self, user_id: str) -> List[Review]:
        pass

    @abstractmethod
    async def get_by_restaurant(self, restaurant_id: str) -> List[Review]:
        pass

    @abstractmethod
    async def get_favorites_by_user(self, user_id: str) -> List[Review]:
        pass

    @abstractmethod
    async def get_by_id(self, review_id: str) -> Optional[Review]:
        pass

    @abstractmethod
    async def update(self, review_id: str, review: Review) -> bool:
        pass

    @abstractmethod
    async def delete(self, review_id: str) -> bool:
        pass