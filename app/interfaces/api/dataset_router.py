from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import io
import pandas as pd
from app.infraestructure.db.mongo_client import get_db
from app.infraestructure.repositories_impl.user_repo_impl import UserRepositoryImpl
from app.infraestructure.repositories_impl.restaurant_repo_impl import RestaurantRepositoryImpl
from app.infraestructure.repositories_impl.plate_repo_impl import PlateRepositoryImpl
from app.infraestructure.repositories_impl.review_repo_impl import ReviewRepositoryImpl
from app.application.use_cases.generate_dataset import GenerateDatasetUseCase

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

def get_repositories():
    db = get_db()
    return {
        "user_repo": UserRepositoryImpl(db),
        "restaurant_repo": RestaurantRepositoryImpl(db),
        "plate_repo": PlateRepositoryImpl(db),
        "review_repo": ReviewRepositoryImpl(db)
    }


@router.get("/download-csv")
async def download_dataset_csv(repos = Depends(get_repositories)):
    try:
        dataset_use_case = GenerateDatasetUseCase(
            repos["user_repo"],
            repos["restaurant_repo"],
            repos["plate_repo"],
            repos["review_repo"]
        )
        
        df = await dataset_use_case.execute()

        if df.empty:
            raise HTTPException(status_code=404, detail="No hay datos para generar el dataset")
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(csv_buffer.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=restaurant_dataset.csv"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generando CSV: {str(e)}")
    

@router.get("/preview")
async def preview_dataset(limit: int = 10, repos = Depends(get_repositories)):
    try:
        dataset_use_case = GenerateDatasetUseCase(
            repos["user_repo"],
            repos["restaurant_repo"],
            repos["plate_repo"],
            repos["review_repo"]
        )

        df = await dataset_use_case.execute()
        if df.empty:
            return {"message": "No hay datos disponibles", "data": [], "total_records": 0}
        
        return {
            "total_records": len(df),
            "columns": list(df.columns),
            "sample_data": df.head(limit).to_dict(orient="records")
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error obteniendo preview: {str(e)}")