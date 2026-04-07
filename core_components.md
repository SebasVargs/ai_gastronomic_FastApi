# Arquitectura y Componentes Core del Sistema

El proyecto está estructurado utilizando los principios de **Clean Architecture**, dividiendo las responsabilidades en capas para que el código sea testeable, mantenible e independiente de frameworks externos.

A continuación, se describen los componentes estructurales y funcionales del backend:

## 1. Capa de Interfaces (Controllers / Presenters)
Maneja la entrada/salida de datos de cara al cliente de la API.
- **Routers (`app/interfaces/api/`)**: Actúan como controladores de FastAPI, definiendo los endpoints (rutas) de usuarios, platos, restaurantes, inteligencia artificial y generación de datasets. Delegan la lógica de negocio a los *Use Cases*.
- **Schemas (`app/interfaces/schemas/`)**: Modelos de Pydantic (`UserCreateSchema`, `RecommendationResponseSchema`, etc.). Su función es la **validación de datos** tanto para el body de las peticiones (Request) como para las respuestas (Response).

## 2. Capa de Aplicación (Use Cases / Lógica de Negocio)
Contiene la lógica particular de la aplicación.
- **Use Cases (`app/application/use_cases/`)**: Orquestan el flujo de datos. Ejemplos de estos componentes:
  - `CreateUserUseCase`: Valida reglas de negocio para crear usuarios.
  - `GenerateDatasetUseCase`: Cruza datos de repositorios para armar el Dataset tabular de entrenamiento.
  - `GetRecommendationUseCase`: Coordina el modelo de Machine Learning con los gustos del usuario y los repositorios de platos/restaurantes.
  - `CreateReviewUseCase`: Lógica de negocio asociada a calificar algo.

## 3. Capa de Dominio (Entities)
El corazón del negocio; no depende de ninguna otra capa.
- **Entities (`app/domain/entities/`)**: Clases de Python puras que representan los objetos de negocio en su forma más básica. 
  - `User`, `Restaurant`, `Plate`, `Review`. Contienen reglas de negocio inherentes a sí mismos y no saben nada sobre la base de datos o el framework web.

## 4. Capa de Infraestructura (Servicios Externos, Base de Datos, ML)
Implementa los detalles técnicos, APIs externas, bases de datos y algoritmos.
- **Base de Datos (`app/infraestructure/db/mongo_client.py`)**: Cliente de conexión Singleton para MongoDB.
- **Implementación de Repositorios (`app/infraestructure/repositories_impl/`)**: Implementan las interfaces (contratos de acceso a datos) para hablar con la base de datos real. Contiene: `user_repo_impl`, `restaurant_repo_impl`, `plate_repo_impl`, y `review_repo_impl`.
- **Servicio de Inteligencia Artificial (`app/infraestructure/ai/`)**:
  - `AIRecommendationService`: Encargado de cargar el modelo Random Forest y K-Means, y de ofrecer métodos funcionales para predicción de rating en base a características (features).
  - `SyntheticDataService`: Genera, carga y gestiona el flujo de los 80,000 registros sintéticos en memoria para entrenar o reentrenar los modelos de ML.
- **Data Services (`app/infraestructure/data/`)**:
  - `CsvDataService`: Se encarga de levantar archivos CSV complementarios para leer información pesada que no requiere viajes a la base de datos de manera constante, optimizando así los tiempos de la API.
