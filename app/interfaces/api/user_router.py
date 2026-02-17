from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.infraestructure.db.mongo_client import get_db
from app.infraestructure.repositories_impl.user_repo_impl import UserRepositoryImpl
from app.application.use_cases.create_user import CreateUserUseCase
from app.interfaces.schemas.user_schema import UserCreateSchema, UserResponseSchema, UserUpdateSchema

router = APIRouter(prefix="/api/users", tags=["users"])

def get_user_repository():
    db = get_db()
    return UserRepositoryImpl(db)


@router.post("/", response_model=UserResponseSchema)
async def create_user(user_data: UserCreateSchema, user_repo = Depends(get_user_repository)):
    try:
        create_user_use_case = CreateUserUseCase(user_repo)
        user = await create_user_use_case.execute(
            nombre = user_data.nombre,
            edad = user_data.edad,
            origen = user_data.origen,
            preferencias = user_data.preferencias
        )
        return UserResponseSchema(**user.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@router.get("/{user_id}", response_model=UserResponseSchema)
async def get_user(user_id: str, user_repo = Depends(get_user_repository)):
    user = await user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail = "Usuario no encontrado")
    return UserResponseSchema(**user.model_dump())


@router.get("/", response_model=List[UserResponseSchema])
async def get_all_users(user_repo = Depends(get_user_repository)):
    users = await user_repo.get_all()
    return [UserResponseSchema(**user.model_dump()) for user in users]


@router.put("/{user_id}", response_model=bool)
async def update_user(user_id: str, user_update: UserUpdateSchema, user_repo = Depends(get_user_repository)):
    existing_user = await user_repo.get_by_id(user_id)
    if not existing_user:
        raise HTTPException(status_code=404, detail = "Usuario no encontrado")
    update_data = user_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(existing_user, field, value)
    
    success = await user_repo.update(user_id, existing_user)
    if not success:
        raise HTTPException(status_code=400, detail = "Error actualizando usuario")
    return success


@router.delete("/{user_id}")
async def delete_user(user_id: str, user_repo = Depends(get_user_repository)):
    success = await user_repo.delete(user_id)
    if not success:
        raise HTTPException(status_code=404, detail = "Usuario no encontrado")
    return {"message": "Usuario eliminado exitosamente"}