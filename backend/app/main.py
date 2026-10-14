from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import upload


app = FastAPI()


# --------------------------------------------------
# CORS
# --------------------------------------------------

FRONTEND_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://bookish-acorn-jj4jxqp7xjxc5xwx-5173.app.github.dev",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Routes
# --------------------------------------------------

app.include_router(upload.router)