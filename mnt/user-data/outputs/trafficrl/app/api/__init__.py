from app.api.training import router as training_router
from app.api.predict  import router as predict_router

__all__ = ["training_router", "predict_router"]
