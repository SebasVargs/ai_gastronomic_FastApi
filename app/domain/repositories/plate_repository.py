from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities.plate import Plate

class PlateRepository(ABC):
    @abstractmethod
    async def create(self, plate: Plate) -> Plate:
        pass

    @abstractmethod
    async def get_by_id(self, plate_id: str) -> Optional[Plate]:
        pass

    @abstractmethod
    async def get_by_category(self, categoria: str) -> List[Plate]:
        pass

    @abstractmethod
    async def get_popular_plates(self, limit: int = 10) -> List[Plate]:
        pass

    @abstractmethod
    async def update(self, plate_id: str, plate: Plate) -> bool:
        pass

    @abstractmethod
    async def delete(self, plate_id: str) -> bool:
        pass