from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

# Importar todos los routers
from app.interfaces.api.user_router import router as user_router
from app.interfaces.api.restaurant_router import router as restaurant_router
from app.interfaces.api.plate_router import router as plate_router
from app.interfaces.api.review_router import router as review_router
from app.interfaces.api.ai_router import router as ai_router
from app.interfaces.api.dataset_router import router as dataset_router

# Importar conexión DB
from app.infraestructure.db.mongo_client import get_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando aplicación de recomendación gastronomicas")
    try:
        db = get_db()
        db.command("ping")
        print("Conexión exitosa a MongoDB!")
    except Exception as e:
        print(f"Error conectando a MongoDB: {e}")

    yield
    print("Cerrando Aplicación")


app = FastAPI(
    title="🍽️ Sistema de Recomendaciones Gastronómicas",
    description="""
    API para recomendaciones de restaurantes y platos usando Machine Learning.
    
    **Características:**
    - ✨ Gestión de usuarios, restaurantes, platos y reviews
    - 🤖 Recomendaciones inteligentes con ML
    - 📊 Generación de datasets para entrenamiento
    - 🏗️ Clean Architecture + MVVM
    - 🔄 MongoDB como base de datos
    
    **Tecnologías:**
    - FastAPI + Python 3.11
    - MongoDB Atlas
    - scikit-learn para ML
    - Pandas para procesamiento de datos
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar todos los routers
app.include_router(user_router)
app.include_router(restaurant_router)
app.include_router(plate_router)
app.include_router(review_router)
app.include_router(ai_router)
app.include_router(dataset_router)

# Endpoint de salud
@app.get("/health")
async def health_check():
    try:
        db = get_db()
        db.command('ping')
        return {
            "status": "healthy",
            "message": "API funcionando correctamente",
            "database": "connected",
            "version": "1.0.0"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "Error de conexión a la base de datos",
                "error": str(e)
            }
        )

# Endpoint raíz
@app.get("/")
async def root():
    return {
        "message": "🍽️ Bienvenido al Sistema de Recomendaciones Gastronómicas",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "users": "/api/users",
            "restaurants": "/api/restaurants", 
            "plates": "/api/plates",
            "reviews": "/api/reviews",
            "ai": "/api/ai",
            "dataset": "/api/dataset"
        }
    }

# Manejador de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Error interno del servidor",
            "detail": str(exc),
            "path": str(request.url)
        }
    )

# Ejecutar aplicación
if __name__ == "__main__":
    print("🌟 Iniciando servidor FastAPI...")
    print("📖 Documentación disponible en: http://localhost:8000/docs")
    print("🔄 Redoc disponible en: http://localhost:8000/redoc")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )