from typing import Optional
from datetime import datetime
import uuid
from app.domain.entities.user import User
from app.domain.repositories.user_repository import UserRepository

class CreateUserUseCase:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository
    
    async def execute(self, nombre: str, edad: int, origen: str, preferencias: list) -> User:
        user = User(
            id = str(uuid.uuid4()),
            nombre = nombre,
            edad = edad,
            origen = origen,
            preferencias = preferencias,
            fecha_registro = datetime.utcnow(),
            activo = True
        )
    
        created_user = await self.user_repository.create(user)
        return created_user