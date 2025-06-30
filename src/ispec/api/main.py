# src/ispec/api/main.py
from fastapi import FastAPI
from ispec.api.routes.routes import router

app = FastAPI()
app.include_router(router)
