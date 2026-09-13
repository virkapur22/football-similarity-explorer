"""Vercel entrypoint for the existing FastAPI application."""

from src.api.app import app

__all__ = ["app"]
