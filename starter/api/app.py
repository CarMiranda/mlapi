from this import d
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd

from starter.ml import model as model_utils

model = model_utils.load("model/model.pkl")
lb = model_utils.load("model/label_encoder.pkl")


class InferenceSchema(BaseModel):

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


app = FastAPI()


@app.get("/")
def index():
    return "Welcome!"


@app.post("/")
def request_inference(inference_details: InferenceSchema):
    df = pd.DataFrame(
        {
            name.replace("_", "-"): [value]
            for name, value in inference_details.dict().items()
        }
    )
    preds = model_utils.inference(model, df)
    outs = lb.inverse_transform(preds)
    return outs[0]


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
