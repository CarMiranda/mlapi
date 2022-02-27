import json
import pandas as pd
from starter.api.models import InferenceSchema, User, UserInDB
from starter.ml import model as model_utils
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from starter.api.auth import (
    FAKE_USERS_DB,
    fake_hash_password,
    get_current_active_user,
)

router = APIRouter()
templates = Jinja2Templates(directory="starter/api/templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "example": json.dumps(InferenceSchema.schema()["example"], indent=4),
        },
    )


@router.post("/")
def request_inference(
    request: Request,
    inference_details: InferenceSchema,
    current_user: User = Depends(get_current_active_user),
):
    df = pd.DataFrame(
        {
            name.replace("_", "-"): [value]
            for name, value in inference_details.dict().items()
        }
    )
    preds = model_utils.inference(request.app.model, df)
    outs = request.app.lb.inverse_transform(preds)
    return outs[0]


@router.post("/token")
async def get_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = FAKE_USERS_DB.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    user = UserInDB(**user_dict)
    hashed_password = fake_hash_password(form_data.password)
    if not hashed_password == user.hashed_password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"access_token": user.username, "token_type": "bearer"}
