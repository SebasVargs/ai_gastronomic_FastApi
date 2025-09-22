from typing import List, Optional
from abc import ABC, abstractmethod
from app.domain.entities.user import User

class UserRepository(ABC):

    @abstractmethod
    async def create(self, user: User) -> User:
        pass

    @abstractmethod
    async def get_by_id(self, user_id: str) -> Optional[User]:
        pass

    @abstractmethod
    async def get_all(self) -> List[User]:
        pass

    @abstractmethod
    async def update(self, user_id: str, user: User) -> bool:
        pass

    @abstractmethod
    async def delete(self, user_id: str) -> bool:
        pass

    @abstractmethod
    async def get_by_preferences(self, preferencias: List[str]) -> List[User]:
        pass