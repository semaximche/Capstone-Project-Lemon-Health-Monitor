# Capstone API - Lemon Tree Disease Identification System

A FastAPI-based backend for plant disease identification using computer vision and LLM-powered recommendations.

## Overview

This API provides endpoints for:

- **Authentication**: Google OAuth integration for user management
- **Plant Management**: CRUD operations for plants in user orchards
- **Image Analysis**: Upload images and run disease detection pipeline
- **Monitoring**: Automated interval-based monitoring via IoT cameras
- **Camera Integration**: Connect to Home Assistant cameras

## Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Authentication**: Google OAuth 2.0 with JWT tokens
- **ML Pipeline**:
  - YOLOv11 for leaf detection
  - EfficientNetV2 for disease classification
  - LLM (OpenAI/Ollama) for recommendations
- **Object Storage**: S3/Minio/Firebase for image storage
- **IoT Integration**: Home Assistant for camera connectivity

## Project Structure

```
capstone_api/
├── capstone_api/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── settings.py          # Configuration settings
│   ├── core/                # Core utilities
│   │   ├── dependencies.py  # FastAPI dependencies
│   │   ├── security.py      # OAuth/JWT helpers
│   │   ├── exceptions.py    # Custom exceptions
│   │   └── o11y/           # Observability (logging)
│   ├── models/              # Pydantic schemas
│   │   ├── auth.py
│   │   ├── user.py
│   │   ├── plant.py
│   │   ├── analysis.py
│   │   ├── monitoring.py
│   │   ├── camera.py
│   │   └── common.py
│   ├── db/                  # Database models
│   │   ├── base.py
│   │   └── models.py
│   ├── routes/              # API endpoints
│   │   ├── health_router.py
│   │   └── v1/
│   │       ├── auth.py
│   │       ├── users.py
│   │       ├── plants.py
│   │       ├── analysis.py
│   │       ├── monitoring.py
│   │       └── cameras.py
│   ├── services/            # Business logic
│   │   ├── auth_service.py
│   │   ├── user_service.py
│   │   ├── plant_service.py
│   │   ├── analysis_service.py
│   │   ├── monitoring_service.py
│   │   └── camera_service.py
│   └── middlewares/         # HTTP middlewares
├── pyproject.toml
└── README.md
```

## API Endpoints

### Authentication (`/v1/auth`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/login` | Initiate Google OAuth flow |
| GET | `/callback` | OAuth callback handler |
| POST | `/logout` | Logout current user |
| GET | `/me` | Get current user info |
| POST | `/refresh` | Refresh access token |

### Users (`/v1/users`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/settings` | Get user settings |
| PUT | `/settings` | Update user settings |

### Plants (`/v1/plants`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/` | Create new plant |
| GET | `/` | List all plants |
| GET | `/{plant_id}` | Get plant details |
| PUT | `/{plant_id}` | Update plant |
| DELETE | `/{plant_id}` | Delete plant |
| GET | `/{plant_id}/analysis` | List plant analyses |

### Analysis (`/v1/analysis`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/` | Upload image and run analysis |
| GET | `/history` | Get analysis history with trends |
| GET | `/{analysis_id}` | Get analysis result |
| DELETE | `/{analysis_id}` | Delete analysis |

### Monitoring (`/v1/monitoring`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/configs` | List monitoring configs |
| POST | `/configs` | Create monitoring config |
| GET | `/configs/{config_id}` | Get config details |
| PUT | `/configs/{config_id}` | Update config |
| DELETE | `/configs/{config_id}` | Delete config |
| POST | `/configs/{config_id}/trigger` | Manually trigger monitoring |

### Cameras (`/v1/cameras`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | List cameras |
| POST | `/` | Register new camera |
| GET | `/{camera_id}` | Get camera details |
| DELETE | `/{camera_id}` | Remove camera |
| POST | `/{camera_id}/capture` | Capture image |
| POST | `/{camera_id}/test` | Test connection |

## Getting Started

### Prerequisites

- Python 3.12+
- uv (Python package manager)
- PostgreSQL
- (Optional) Home Assistant instance for IoT camera integration

### Installation

```bash
# Clone the repository
cd app

# Install dependencies using uv
uv sync

# Copy environment file and configure
cp .env.example .env

# Run database migrations (when implemented)
# uv run alembic upgrade head

# Start the development server
uv run uvicorn app.main:app --reload
```

### Environment Variables

Key environment variables (see `settings.py` for full list):

```env
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/capstone

# Google OAuth
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/v1/auth/callback

# JWT
JWT_SECRET_KEY=your-secret-key

# Home Assistant (optional)
HOME_ASSISTANT_URL=http://homeassistant.local:8123
HOME_ASSISTANT_TOKEN=your-long-lived-token

# LLM
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key
```

## Development Status

This is a skeleton implementation. All service methods raise `NotImplementedError` and need to be implemented:

- [ ] Database session management
- [ ] User authentication flow
- [ ] Plant CRUD operations
- [ ] Image upload and storage
- [ ] ML pipeline integration (YOLOv11, EfficientNetV2)
- [ ] LLM recommendation generation
- [ ] Home Assistant camera integration
- [ ] Background task scheduling for monitoring

## API Documentation

When the server is running, interactive API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT

