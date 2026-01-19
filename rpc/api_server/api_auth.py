import logging
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi.security.http import HTTPBasic

from deepfake.persistence import User
from deepfake.rpc.api_server.api_schemas import Token, UserCreate, UserOut, UserLogin
from deepfake.rpc.api_server.deps import get_api_config

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
SECRET_KEY = get_api_config().get("jwt_secret_key")
ACCESS_TOKEN_EXPIRE_MINUTES = 60

router_login = APIRouter()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

httpbasic = HTTPBasic(auto_error=False)
security = HTTPBasic()

# OAuth2 token URL
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


# Utility functions
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(username: str):
    return User.session.query(User).filter(User.username == username).first()

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception  # If auto_error=False and no token provided

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user

# Routes
@router_login.post("/auth/signup", response_model=UserOut, status_code=201, tags=["user"])
def signup(user_in: UserCreate):
    # Check if username or email exists
    if User.session.query(User).filter(User.username == user_in.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    if User.session.query(User).filter(User.email == user_in.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password)
    )
    User.session.add(user)
    User.session.commit()
    User.session.refresh(user)
    return user

@router_login.post("/auth/login", response_model=Token, tags=["user"])
def login(response: Response,form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="Lax",  # or "None" if cross-site
        secure=True      # only over HTTPS
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router_login.get("/users/me", response_model=UserOut, tags=["user"])
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
