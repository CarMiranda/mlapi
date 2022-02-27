import pytest
from fastapi.testclient import TestClient
from starter.api.app import InferenceSchema, app


@pytest.fixture
def client():
    return TestClient(app)


def test__get(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == "Welcome!"


def test__less50(client: TestClient):
    response = client.post(
        "/",
        json=InferenceSchema(
            age=39,
            workclass="State-gov",
            fnlgt=77516,
            education="Bachelors",
            education_num=13,
            marital_status="Never-married",
            occupation="Adm-clerical",
            relationship="Not-in-family",
            race="White",
            sex="Male",
            capital_gain=2174,
            capital_loss=0,
            hours_per_week=40,
            native_country="United-States",
        ).dict(),
    )

    assert response.status_code == 200
    assert response.json() == "<=50K"


def test__more50(client: TestClient):
    response = client.post(
        "/",
        json={
            "age": 52,
            "workclass": "Self-emp-not-inc",
            "fnlgt": 209642,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 45,
            "native-country": "United-States",
        },
    )

    assert response.status_code == 200
    assert response.json() == ">50K"
