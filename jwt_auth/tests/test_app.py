import os
# Set test database URL before importing app/models
os.environ["DATABASE_URL"] = "sqlite:///./test_sql_app.db"

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.models import Base, engine

# Use a separate test database
TEST_DB = "test_sql_app.db"

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.mark.asyncio
async def test_register():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/register", json={"email": "test@example.com", "password": "password123"})
    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"

@pytest.mark.asyncio
async def test_duplicate_register():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post("/register", json={"email": "test@example.com", "password": "password123"})
        response = await ac.post("/register", json={"email": "test@example.com", "password": "password123"})
    assert response.status_code == 409

@pytest.mark.asyncio
async def test_login_and_me():
    email = "login@example.com"
    password = "password123"
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Register
        await ac.post("/register", json={"email": email, "password": password})
        
        # Login
        login_response = await ac.post("/login", data={"username": email, "password": password})
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        
        # Get me
        headers = {"Authorization": f"Bearer {token}"}
        me_response = await ac.get("/me", headers=headers)
        assert me_response.status_code == 200
        assert me_response.json()["email"] == email
