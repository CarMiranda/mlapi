import pandas as pd
from starter.api.models import InferenceSchema
from starter.ml import model as model_utils
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/")
def index():
    return "Welcome!"


@router.post("/")
def request_inference(request: Request, inference_details: InferenceSchema):
    df = pd.DataFrame(
        {
            name.replace("_", "-"): [value]
            for name, value in inference_details.dict().items()
        }
    )
    preds = model_utils.inference(request.app.model, df)
    outs = request.app.lb.inverse_transform(preds)
    return outs[0]
