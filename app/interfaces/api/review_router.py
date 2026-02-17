from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from app.infraestructure.db.mongo_client import get_db
from app.infraestructure.repositories_impl.review_repo_impl import ReviewRepositoryImpl
from app.application.use_cases.create_review import CreateReviewUseCase
from app.interfaces.schemas.review_schema import ReviewCreateSchema, ReviewResponseSchema

router = APIRouter(prefix="/api/reviews", tags=["reviews"])

def get_review_repository():
    db = get_db()
    return ReviewRepositoryImpl(db)


@router.post("/", response_model=ReviewResponseSchema)
async def create_review(review_data: ReviewCreateSchema, review_repo = Depends(get_review_repository)):
    try:
        create_review_use_case = CreateReviewUseCase(review_repo)
        review = await create_review_use_case.execute(
            user_id = review_data.id_usuario,
            restaurante_id = review_data.id_restaurante,
            rating = review_data.rating,
            comentario = review_data.comentario,
            es_favorito = review_data.es_favorito,
            fecha = review_data.fecha
        )
        return ReviewResponseSchema(**review.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@router.get("/{review_id}", response_model=ReviewResponseSchema)
async def get_review(review_id: str, review_repo = Depends(get_review_repository)):
    review = await review_repo.get_by_id(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review no encontrada")
    return ReviewResponseSchema(**review.model_dump())


@router.get("/", response_model=List[ReviewResponseSchema])
async def get_reviews(
    user_id: Optional[str] = Query(None, description="Filtrar por usuario"),
    restaurante_id: Optional[str] = Query(None, description="Filtrar por restaurante"),
    favorites_only: Optional[bool] = Query(False, description="Solo favoritos"),
    review_repo = Depends(get_review_repository)
):
    if user_id and favorites_only:
        reviews = await review_repo.get_favorites_by_user(user_id)
    elif user_id:
        reviews = await review_repo.get_by_user(user_id)
    elif restaurante_id:
        reviews = await review_repo.get_by_restaurant(restaurante_id)
    else:
        # No implementamos get_all() para reviews por privacidad
        raise HTTPException(status_code=400, detail="Debe especificar user_id o restaurant_id")
    
    return [ReviewResponseSchema(**review.model_dump()) for review in reviews]