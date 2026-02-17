from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from app.infraestructure.db.mongo_client import get_db
from app.infraestructure.repositories_impl.plate_repo_impl import PlateRepositoryImpl
from app.interfaces.schemas.plate_schema import PlateCreateSchema, PlateResponseSchema
from app.domain.entities.plate import Plate

router = APIRouter(prefix="/api/plates", tags=["plates"])

def get_plate_repository():
    db = get_db()
    return PlateRepositoryImpl(db)


@router.post("/", response_model=PlateResponseSchema)
async def create_plate(plate_data: PlateCreateSchema, plate_repo = Depends(get_plate_repository)):
    try:
        plate = Plate(**plate_data.model_dump())
        create_plate = await plate_repo.create(plate)
        return PlateResponseSchema(**create_plate.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@router.get("/{plate_id}", response_model=PlateResponseSchema)
async def get_plate(id_plate: str, plate_repo = Depends(get_plate_repository)):
    plate = await plate_repo.get_by_id(id_plate)
    if not plate:
        raise HTTPException(status_code=404, detail="Plato no encontrado")
    return PlateResponseSchema(**plate.model_dump())


@router.get("/", response_model=List[PlateResponseSchema])
async def get_plates(
    categoria: Optional[str] = Query(None, description="Filtrar por categoría"),
    popularidad: Optional[bool] = Query(False, description="Obtener platos populares"),
    limit: Optional[int] = Query(None, description="Límite de resultados (opcional)"),
    plate_repo = Depends(get_plate_repository)
):
    if categoria:
        plates = await plate_repo.get_by_category(categoria)
    elif popularidad:
        # Si es por popularidad, siempre aplicamos un límite (por defecto 10 si no se especifica)
        limit = limit or 10
        plates = await plate_repo.get_popular_plates(limit)
    else:
        plates = await plate_repo.get_all()
    
    # Solo aplicamos el límite si se especifica y no es una búsqueda por popularidad
    if limit and not popularidad:
        plates = plates[:limit]
    
    return [PlateResponseSchema(**plate.model_dump()) for plate in plates]

