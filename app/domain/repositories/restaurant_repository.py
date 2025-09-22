from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities.restaurant import Restaurant

class RestaurantRepository(ABC):
    @abstractmethod
    async def create(self, restaurant: Restaurant) -> Restaurant:
        pass

    @abstractmethod
    async def get_by_id(self, restaurant_id: str) -> Optional[Restaurant]:
        pass

    @abstractmethod
    async def get_all(self) -> List[Restaurant]:
        pass

    @abstractmethod
    async def get_by_category(self, categoria: str) -> List[Restaurant]:
        pass

    @abstractmethod
    async def get_by_location(self, latitud: float, longitud: float, radio_km: float) -> List[Restaurant]:
        pass

    @abstractmethod
    async def update(self, restaurant_id: str, restaurant: Restaurant) -> bool:
        pass

    @abstractmethod
    async def delete(self, restaurant_id: str) -> bool:
        pass