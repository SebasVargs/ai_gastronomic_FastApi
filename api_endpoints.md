# Sistema de Recomendaciones Gastronómicas - API Endpoints

A continuación se listan todos los endpoints disponibles en el proyecto FastAPI.

## 🟢 Endpoints Generales (main.py)
| Endpoint | Método | Descripción | Parámetros Relevantes |
|---|---|---|---|
| `/health` | `GET` | Endpoint de diagnóstico de salud. | Comprueba que la API está viva y que MongoDB está conectado. |
| `/` | `GET` | Endpoint raíz del proyecto. | Devuelve un mensaje de bienvenida y el mapa base de rutas disponibles. |

## 📊 Dataset (`/api/dataset`)
| Endpoint | Método | Descripción | Parámetros Relevantes |
|---|---|---|---|
| `/api/dataset/download-csv` | `GET` | Genera y descarga el dataset actual como archivo CSV. | Ninguno |
| `/api/dataset/preview` | `GET` | Devuelve una muestra previa de los datos disponibles en la BD. | `limit` (opcional, int, por defecto: 10). |

## 👥 Usuarios (`/api/users`)
| Endpoint | Método | Descripción | Parámetros Relevantes |
|---|---|---|---|
| `/api/users/` | `POST` | Crea un usuario nuevo. | Body: Objeto `UserCreateSchema`. |
| `/api/users/` | `GET` | Lista todos los usuarios guardados en DB. | Ninguno |
| `/api/users/{user_id}` | `GET` | Obtiene la información de un usuario específico. | `user_id` (en URL). |
| `/api/users/{user_id}` | `PUT` | Actualiza la información de un usuario, p.ej. preferencias y edad. | `user_id` (en URL), Body: `UserUpdateSchema`. |
| `/api/users/{user_id}` | `DELETE`| Elimina a un usuario. | `user_id` (en URL). |

## 🍽️ Restaurantes (`/api/restaurants`)
| Endpoint | Método | Descripción | Parámetros Relevantes |
|---|---|---|---|
| `/api/restaurants/` | `POST` | Crea un restaurante nuevo. | Body: Objeto `RestaurantCreateSchema`. |
| `/api/restaurants/` | `GET` | Obtiene lista de restaurantes. | Query params: `categoria`, `latitud`, `longitud`, `radio_km`. Útil para búsquedas geoespaciales. |
| `/api/restaurants/{restaurant_id}` | `GET` | Obtiene un restaurante por su ID. | `restaurant_id` (en URL). |
| `/api/restaurants/{restaurant_id}/plates` | `GET` | Devuelve todos los platos de un restaurante específico directamente de memoria. | `restaurant_id` (en URL). Sin ML. |

## 🍝 Platos (`/api/plates`)
| Endpoint | Método | Descripción | Parámetros Relevantes |
|---|---|---|---|
| `/api/plates/` | `POST` | Crea un nuevo plato en DB. | Body: Objeto `PlateCreateSchema`. |
| `/api/plates/` | `GET` | Lista todos los platos o aplica filtros para buscar por relevancia. | Query params: `categoria`, `popularidad` (bool), `limit` (int). |
| `/api/plates/{plate_id}` | `GET` | Obtiene un plato por su ID. | `plate_id` (en URL). |

## ⭐ Reviews (`/api/reviews`)
| Endpoint | Método | Descripción | Parámetros Relevantes |
|---|---|---|---|
| `/api/reviews/` | `POST` | Crea una nueva review para un restaurante hecha por un usuario. | Body: `ReviewCreateSchema`. |
| `/api/reviews/` | `GET` | Filtra reviews publicadas, obligatorio pasar algún criterio de búsqueda. | Query params: `user_id`, `restaurante_id`, `favorites_only` (bool). |
| `/api/reviews/{review_id}` | `GET` | Obtiene una review específica por su ID. | `review_id` (en URL). |

## 🤖 Inteligencia Artificial (`/api/ai`)
| Endpoint | Método | Descripción | Parámetros Relevantes |
|---|---|---|---|
| `/api/ai/categories` | `GET` | Lista de categorías de comida únicas desde CSV. | Ninguno |
| `/api/ai/load-synthetic-data` | `POST` | Carga un archivo CSV de registros sintéticos local dando su ruta. | Body param: `csv_file_path`. |
| `/api/ai/upload-synthetic-csv` | `POST` | Sube y procesa un archivo CSV mediante FormData. | Requiere un archivo subido usando tipo `File` (`multipart/form-data`). |
| `/api/ai/train-with-synthetic-data`| `POST` | Fuerza y detona el entrenamiento cruzado (Random Forest + K-Means). | Query param: `force_retrain` (bool). |
| `/api/ai/model-performance` | `GET` | Consulta las métricas del modelo entrenado (R2, clustering, RMSE, etc). | Ninguno |
| `/api/ai/recommendations` | `POST` | **Endpoint principal de negocio:** Genera las recomendaciones reales de platos/restaurantes vía ML. | Body param: Schema complejo especificando usuario, tipo de recomendación y distintos filtros opcionales (precio, rating, abiertos, etc). |
| `/api/ai/predict-single_rating` | `POST` | Predice el rating para un restaurante o plato específico y determina la confianza. | Body param: Diccionario/Schema de características del restaurante. |
| `/api/ai/synthetic-data-status` | `GET` | Devuelve un panel de estado sobre métricas de la RAM, registros sintéticos e infos. | Ninguno |
| `/api/ai/feature-importance` | `GET` | Trae la importancia de variables (Feature Imp.) calculadas del Random Forest. | Indica en orden descendente las variables que afectan más el rating. |
| `/api/ai/model-info` | `GET` | Informa sobre de K-Means, RandomForest, Encoders y Scalers que viven en la memoria. | Ninguno |
| `/api/ai/reset-models` | `DELETE`| ⚠️ Resetea (vacía de RAM) todo el modelo y datos cargados. | **Uso delicado**, útil en pruebas para limpiar el estado sin apagar el servidor. |
