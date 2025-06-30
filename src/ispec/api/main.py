# src/ispec/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ispec.api.routes.routes import router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to specific frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)
