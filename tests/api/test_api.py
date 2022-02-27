import pytest

from fastapi.testclient import TestClient
from starter.api.models import InferenceSchema, UserInDB
from starter.api.app import create_app
from starter.api.auth import FAKE_USERS_DB, get_db_user


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def user():
    return get_db_user(FAKE_USERS_DB, list(FAKE_USERS_DB)[0])


def test__get(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200


def test__less50(client: TestClient, user: UserInDB):
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
        headers={"Authorization": f"Bearer {user.username}"},
    )

    assert response.status_code == 200
    assert response.json() == "<=50K"


def test__more50(client: TestClient, user: UserInDB):
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
        headers={"Authorization": f"Bearer {user.username}"},
    )

    assert response.status_code == 200
    assert response.json() == ">50K"


def test__notoken(client: TestClient):
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

    assert response.status_code == 401


def test__get_token(client: TestClient, user: UserInDB):
    response = client.post(
        "/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "username": user.username,
            "password": user.hashed_password[10:],
        },
    )

    assert response.status_code == 200
    assert "access_token" in response.json()


def test__unkown_user(client: TestClient, user: UserInDB):
    response = client.post(
        "/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "username": user.username + "a",
            "password": user.hashed_password[10:],
        },
    )

    assert get_db_user(FAKE_USERS_DB, user.username + "a") is None
    assert response.status_code == 400


def test__wrong_password(client: TestClient, user: UserInDB):
    response = client.post(
        "/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"username": user.username, "password": user.hashed_password},
    )

    assert response.status_code == 400


def test__disabled_user(client: TestClient, user: UserInDB):

    FAKE_USERS_DB[user.username]["disabled"] = True
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
        headers={"Authorization": f"Bearer {user.username}"},
    )
    FAKE_USERS_DB[user.username]["disabled"] = False

    assert response.status_code == 400
