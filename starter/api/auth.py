from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from starter.api.models import User, UserInDB

FAKE_USERS_DB = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    },
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderson",
        "email": "alice@example.com",
        "hashed_password": "fakehashedsecret2",
        "disabled": True,
    },
}


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def fake_hash_password(password: str):
    return "fakehashed" + password


def fake_decode_token(token):
    user_db = FAKE_USERS_DB.get(token, None)
    if user_db is None:
        return None
    else:
        return User(**user_db)


class NotAuthenticated(Exception):
    pass


async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_decode_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
