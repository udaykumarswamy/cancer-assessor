# Cancer Risk Assessment System - Setup & Usage Guide

This guide explains how to set up and use the cancer-assessor system with or without Google Cloud Platform (GCP) credentials.

## Table of Contents

1. [Quick Start (No GCP Needed)](#quick-start-no-gcp-needed)
2. [Using Your Own GCP Credentials](#using-your-own-gcp-credentials)
3. [Running Tests](#running-tests)
4. [Troubleshooting](#troubleshooting)

---

## Quick Start (No GCP Needed)

The easiest way to get started - no credentials, no setup!

### Option A: Using Docker (Recommended)

```bash
# 1. Clone the repository
git clone <your-repo>
cd cancer-assessor

# 2. Build the Docker image
docker-compose build

# 3. Start with mock services (no GCP needed)
USE_MOCK=true docker-compose up

# 4. API is ready!
# - API: http://localhost:8000
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc

# 7. change dierctory to frontend
cd frontend

# 8. Install frontend dependencies
npm install

# 9. Run frontend
npm run dev

# - Frontend: http://localhost:3000
```

### Option B: Using Local Python (recommended)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start API with mock mode
USE_MOCK=true uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# save appropiate .env file

# 4. Vertex AI mode
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000


# 5. Access at http://localhost:8000/docs

# 6. Open New terminal

# 7. change dierctory to frontend
cd frontend

# 8. Install frontend dependencies
npm install

# 9. Run frontend
npm run dev

# 10. access UI at http://localhost:3000/ 


```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Search guidelines
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "lung cancer symptoms"}'

# Create assessment
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["persistent cough", "weight loss"],
    "age": 65,
    "duration_weeks": 4
  }'

# View API stats
curl http://localhost:8000/stats
```

---

## Using Your Own GCP Credentials

To use real Vertex AI embeddings instead of mock data, you'll need Google Cloud Platform credentials.

### Step 1: Get GCP Credentials

#### Option A: Create a New Service Account (Recommended)

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Select or create a GCP project
3. Navigate to **Service Accounts** → **Create Service Account**
4. Fill in the details and click **Create**
5. Grant these roles:
   - `Vertex AI Service Agent`
   - `Vertex AI User`
6. Click **Create Key** → **JSON** → **Create**
7. A `[project-name]-key.json` file will download

#### Option B: Use Existing Credentials

If you already have a GCP service account key, skip to Step 2.

### Step 2: Start with Your Credentials

#### Method 1: Environment Variable (Recommended)

```bash
# Set the credentials path
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/gcp-key.json

# Start the system (without USE_MOCK)
docker-compose up

# Or with local Python
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

#### Method 2: Docker Volume Mount

1. **Edit `docker-compose.yml`** and uncomment the GCP volume mount:

```yaml
volumes:
  # Mount for logs
  - ./logs:/app/logs
  
  # Mount for vector store updates
  - ./vectorstore:/app/vectorstore
  
  # Mount for data updates
  - ./data:/app/data
  
  # UNCOMMENT THIS LINE - replace /path/to/your/gcp-key.json
  - /path/to/your/gcp-key.json:/app/secrets/gcp-key.json:ro
```

2. **Add environment variable** to docker-compose.yml:

```yaml
environment:
  - USE_MOCK=false  # Use real Vertex AI
  - GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gcp-key.json
  - GCP_PROJECT_ID=your-gcp-project-id  # Optional
```

3. **Start the system:**

```bash
docker-compose up
```

#### Method 3: Direct Path Mount

```bash
docker-compose run --rm \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gcp-key.json \
  -v /path/to/your/gcp-key.json:/app/secrets/gcp-key.json:ro \
  cancer-assessor
```

### Step 3: Verify Real Credentials Are Working

```bash
# Check if using real Vertex AI (not mock)
curl http://localhost:8000/stats

# Look for:
# "mode": "real" (instead of "mock")
# Real Vertex AI embeddings (not fake)
```

## What's Pre-Included

You don't need to do any of this - it's already in the Docker image!

✅ **Vector Store** (161 chunks)
- Pre-trained ChromaDB embeddings
- NG12 guideline chunks
- Clinical metadata indexed

✅ **Cached Data**
- Parsed NG12 PDF (no parsing needed)
- Patient data samples
- Configuration files

✅ **Frontend**
- Pre-built React/Vite application
- Ready to serve at http://localhost:8000

✅ **Backend**
- FastAPI with all routes
- Mock embeddings (built-in)
- Real Vertex AI support (with credentials)

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `USE_MOCK` | `false` | Use mock embeddings (true) or real Vertex AI (false) |
| `GCP_PROJECT_ID` | `cancer-assessor-poc` | Your GCP project ID |
| `GCP_LOCATION` | `us-central1` | Vertex AI region |
| `API_HOST` | `0.0.0.0` | API host binding |
| `API_PORT` | `8000` | API port |
| `WORKERS` | `4` | FastAPI workers |
| `LOG_LEVEL` | `INFO` | Logging level |
| `GOOGLE_APPLICATION_CREDENTIALS` | _(none)_ | Path to GCP credentials JSON |

---

## Troubleshooting

### "GCP credentials not found" Error

**Solution:** Use mock mode:
```bash
USE_MOCK=true docker-compose up
```

### Port 8000 Already in Use

**Solution:** Kill the process or use a different port:
```bash
# Option 1: Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Option 2: Use different port
docker-compose run -p 8001:8000 cancer-assessor
```

### Vector Store Not Loading

**Solution:** It's pre-built in the Docker image. If you need to reinitialize:
```bash
docker-compose down --volumes  # Remove old data
docker-compose up --build       # Rebuild with new data
```

### OutOfMemory Error

**Solution:** Reduce workers:
```bash
WORKERS=2 docker-compose up
```

Or increase Docker memory in Desktop → Preferences → Resources.

### "Module not found" Error

**Solution:** Ensure all dependencies are installed:
```bash
# Using Docker
docker-compose build --no-cache

# Using local Python
pip install -r requirements.txt
```

---

## Directory Structure

```
cancer-assessor/
├── docker-compose.yml          # Docker compose config (edit for credentials)
├── Dockerfile                  # Docker build config
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
├── SETUP_GUIDE.md             # This file
├── SECURITY.md                # Security guidelines(excluded in build)
├── DOCKER.md                  # Docker documentation
│
├── src/                        # Python source code
│   ├── main.py               # FastAPI app
│   ├── api/                  # API routes
│   ├── services/             # Business logic
│   ├── ingestion/            # Data ingestion
│   └── config/               # Configuration
│
├── frontend/                  # React/Vite frontend source
│   ├── src/                  # Frontend source code
│   ├── package.json          # Frontend dependencies
│   └── dist/                 # Pre-built frontend (in Docker)
│
├── data/                      # Pre-included data
│   ├── ng12/                 # NG12 guideline PDF
│   ├── cache/                # Cached parsed data
│   └── patients.json         # Sample patient data
│
├── vectorstore/              # Pre-trained vector store
│   ├── chroma.sqlite3        # ChromaDB database
│   └── [UUID]/               # Embeddings data
│
└── tests/                     # Test suite
```

---

## Next Steps

1. **Explore the API** → Visit http://localhost:8000/docs for interactive documentation
2. **Run Tests** → `USE_MOCK=true docker-compose exec cancer-assessor pytest tests/ -v`
3. **Try the Frontend** → http://localhost:8000
4. **Check Out Examples** → See `scripts/` folder for example usage
5. **Read Documentation** → See README.md for architectural details

---

## Support

For questions or issues:
1. Check [TROUBLESHOOTING](#troubleshooting) section above
2. Review [DOCKER.md](DOCKER.md) for Docker-specific help
3. Check test files in `tests/` for usage examples


