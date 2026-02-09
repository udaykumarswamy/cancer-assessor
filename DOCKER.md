# Docker Setup for Cancer Risk Assessment System

This guide explains how to build and run the cancer-assessor project using Docker.

## What's Included

The Docker image includes all trained models and pre-processed data:

- **ChromaDB Vector Store** (`vectorstore/`)
  - Pre-trained embeddings for NG12 guidelines
  - Clinical metadata indexes
  - Persistent storage ready to use
  
- **Cached Parsed PDF** (`data/cache/ng12_parsed.pkl`)
  - Pre-processed NG12 guideline document
  - Saves time on first startup

- **NG12 PDF Guide** (`data/ng12/ng12_suspected_cancer.pdf`)
  - Original NICE NG12 guidelines document

- **Patient Data** (`data/patients.json`)
  - Sample clinical data for testing

- **Frontend Build** 
  - Pre-built React/Vite frontend in `frontend/dist/`

## Prerequisites

- Docker & Docker Compose installed
- 4GB+ RAM available
- Optional: Google Cloud credentials (for full Vertex AI features)

## Quick Start

### 1. Build the Docker image

```bash
docker-compose build
```

### 2. Run with mock services (no GCP credentials needed)

```bash
USE_MOCK=true docker-compose up
```

The API will be available at: `http://localhost:8000`

API Documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 3. Run with full Vertex AI services

```bash
export GCP_PROJECT_ID=your-gcp-project-id
docker-compose up
```

## Environment Variables

Configure these in `docker-compose.yml` or with `-e` flag:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK` | `false` | Use mock services (no GCP needed) |
| `GCP_PROJECT_ID` | `cancer-assessor-poc` | Google Cloud project ID |
| `GCP_LOCATION` | `us-central1` | Vertex AI region |
| `WORKERS` | `4` | Number of FastAPI workers |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Common Commands

### Start in development mode
```bash
USE_MOCK=true docker-compose up --build
```

### View logs
```bash
docker-compose logs -f cancer-assessor
```

### Run specific command in container
```bash
docker-compose exec cancer-assessor python scripts/test_e2e_pipeline.py --mock
```

### Stop containers
```bash
docker-compose down
```

### Clean rebuild (remove volumes)
```bash
docker-compose down --volumes
docker-compose up --build
```

### Run in detached mode
```bash
USE_MOCK=true docker-compose up -d
```

## File Structure in Container

```
/app/
├── src/                          # Python source code
│   ├── api/                      # FastAPI routes
│   ├── services/                 # Business logic
│   ├── ingestion/                # Data ingestion (embeddings, chunking)
│   ├── llm/                      # LLM integration
│   └── config/                   # Configuration
├── vectorstore/                  # ChromaDB with trained embeddings ✓
├── data/
│   ├── cache/                    # Cached parsed PDF ✓
│   ├── ng12/                     # NG12 PDF guide ✓
│   └── patients.json             # Sample patient data ✓
├── frontend/
│   └── dist/                     # Pre-built React frontend ✓
├── scripts/                      # Utility scripts
├── tests/                        # Test suite
└── logs/                         # Application logs
```

## Testing

### Run pipeline test
```bash
docker-compose exec cancer-assessor \
  python scripts/test_e2e_pipeline.py --mock
```

### Run retrieval tests
```bash
docker-compose exec cancer-assessor \
  python -m pytest tests/ -v
```

## Performance Tuning

### Adjust worker count
```bash
WORKERS=8 docker-compose up
```

### Adjust resource limits
Edit `docker-compose.yml` and modify:
```yaml
deploy:
  resources:
    limits:
      cpus: '4'        # Change to desired number of CPUs
      memory: 8G       # Change to desired memory
```

## Troubleshooting

### Container won't start
```bash
docker-compose logs cancer-assessor
```

### Vector store not loading
The vectorstore is pre-loaded from `COPY` commands in the Dockerfile. If you need to update it:
```bash
docker-compose down --volumes
docker-compose up --build
```

### Out of memory
Increase available RAM or reduce `WORKERS`:
```bash
WORKERS=2 docker-compose up
```

### GCP authentication error
Use mock mode:
```bash
USE_MOCK=true docker-compose up
```

Or provide credentials via:
```bash
docker-compose run --rm -e GOOGLE_APPLICATION_CREDENTIALS=/etc/gcp/key.json \
  -v /path/to/key.json:/etc/gcp/key.json \
  cancer-assessor
```

## API Endpoints

Once running, test the API:

```bash
# Health check
curl http://localhost:8000/api/health

# Create assessment
curl -X POST http://localhost:8000/api/assess \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["persistent cough"],
    "age": 55,
    "duration_weeks": 3
  }'

# Retrieve guidelines
curl http://localhost:8000/api/search \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "lung cancer symptoms"
  }'
```

## Production Deployment

### Using Docker alone (single container)
```bash
docker build -t cancer-assessor:latest .
docker run -p 8000:8000 \
  -e GCP_PROJECT_ID=your-project \
  -e WORKERS=4 \
  cancer-assessor:latest
```

### Using Docker Compose
```bash
export GCP_PROJECT_ID=your-project
docker-compose -f docker-compose.yml up -d
```

### Push to registry
```bash
docker tag cancer-assessor:latest your-registry/cancer-assessor:latest
docker push your-registry/cancer-assessor:latest
```

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Test with mock mode: `USE_MOCK=true docker-compose up`
3. Review [README.md](README.md) for project details
4. Check [Dockerfile](Dockerfile) for build configuration
