from datetime import datetime
import uuid
from app.domain.entities.review import Review
from app.domain.repositories.review_repository import ReviewRepository

class CreateReviewUseCase:
    def __init__(self, review_repository: ReviewRepository):
        self.review_repository = review_repository
    
    async def execute(self, user_id: str, restaurante_id: str, rating: float, comentario: str = None, es_favorito: bool = False) -> Review:
        now = datetime.utcnow()

        review = Review(
            id = str(uuid.uuid4()),
            user_id = user_id,
            restaurante_id = restaurante_id,
            year = now.year,
            month = now.month,
            day = now.day,
            weekday = now.weekday(),
            fecha = now,
            rating = rating,
            comentario = comentario,
            es_favorito = es_favorito
        )
        
        created_review = await self.review_repository.create(review)
        return created_review