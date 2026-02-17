from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from app.infraestructure.db.mongo_client import get_db
from app.infraestructure.repositories_impl.restaurant_repo_impl import RestaurantRepositoryImpl
from app.interfaces.schemas.restaurant_schema import RestaurantCreateSchema, RestaurantResponseSchema

router = APIRouter(prefix="/api/restaurants", tags=["restaurants"])

def get_restaurant_repository():
    db = get_db()
    return RestaurantRepositoryImpl(db)


@router.post("/", response_model=RestaurantResponseSchema)
async def create_restaurant(restaurant_data: RestaurantCreateSchema, restaurant_repo = Depends(get_restaurant_repository)):
    try:
        from ...domain.entities.restaurant import Restaurant
        restaurant = Restaurant(**restaurant_data.model_dump())
        created_restaurant = await restaurant_repo.create(restaurant)
        return RestaurantResponseSchema(**created_restaurant.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@router.get("/{restaurant_id}", response_model=RestaurantResponseSchema)
async def get_restaurant(restaurant_id: str, restaurant_repo = Depends(get_restaurant_repository)):
    restaurant = await restaurant_repo.get_by_id(restaurant_id)
    if not restaurant:
        raise HTTPException(status_code=404, detail="Restaurante no encontrado")
    return RestaurantResponseSchema(
        id_restaurante=restaurant.id or restaurant._id,
        **{k: v for k, v in restaurant.model_dump().items() if k not in ['id', '_id']}
    )

@router.get("/", response_model=List[RestaurantResponseSchema])
async def get_restaurants(
    categoria: Optional[str] = Query(None, description="Filtrar por categoría"),
    latitud: Optional[float] = Query(None, description="Latitud para búsqueda por ubicación"),
    longitud: Optional[float] = Query(None, description="Longitud para búsqueda por ubicación"),
    radio_km: Optional[float] = Query(10.0, description="Radio en kilómetros"),
    restaurant_repo = Depends(get_restaurant_repository)
):
    if categoria:
        restaurants = await restaurant_repo.get_by_category(categoria)
    elif latitud is not None and longitud is not None:
        restaurants = await restaurant_repo.get_by_location(latitud, longitud, radio_km)
    else:
        restaurants = await restaurant_repo.get_all()
    
    return [RestaurantResponseSchema(
        id_restaurante=restaurant.id or restaurant._id,
        **{k: v for k, v in restaurant.model_dump().items() if k not in ['id', '_id']}
    ) for restaurant in restaurants]