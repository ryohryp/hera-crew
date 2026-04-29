# FastAPI JWT Authentication System

A simple and secure FastAPI application with JWT authentication.

## Features
- **User Registration**: `POST /register`
- **Login & JWT Issuance**: `POST /login`
- **Protected Profile**: `GET /me`
- **Password Hashing**: bcrypt
- **Persistence**: SQLAlchemy + SQLite
- **Asynchronous Testing**: pytest + httpx

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --reload
```

## Testing

```bash
# Run tests
pytest tests/test_app.py
```

## API Documentation
Once running, visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.
