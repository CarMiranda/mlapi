from fastapi import FastAPI
from starter.ml import model as model_utils
from starter.api.views import router


def create_app():
    app = FastAPI()

    app.model = model_utils.load("model/model.pkl")
    app.lb = model_utils.load("model/label_encoder.pkl")

    app.include_router(router)

    return app
