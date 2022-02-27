import requests

response = requests.post(
    url="https://mlapi-cm.herokuapp.com/",
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
    headers={"Authorization": f"Bearer johndoe"},
)
print(f"Status code: {response.status_code}")
print("Response content:")
print(response.content.decode("utf8"))
