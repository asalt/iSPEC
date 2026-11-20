"""Isolated dev demo: ordinary login and instrument routes, no other workloads."""

from fastapi import FastAPI
from ispec.api.routes import auth, instruments

app = FastAPI()
app.include_router(auth.router, prefix="/api")
app.include_router(instruments.router, prefix="/api")
