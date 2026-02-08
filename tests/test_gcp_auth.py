#!/usr/bin/env python3
"""
GCP Authentication Test Script

Run this to verify your Google Cloud authentication is working
before using Vertex AI embeddings.

Usage:
    python test_gcp_auth.py

Prerequisites:
    1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
    2. Run: gcloud auth application-default login
    3. Set your project: gcloud config set project YOUR_PROJECT_ID
"""

import os
import sys

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)

def print_status(label, status, details=""):
    icon = "✅" if status else "❌"
    print(f"{icon} {label}")
    if details:
        print(f"   {details}")

def test_gcloud_cli():
    """Test if gcloud CLI is installed."""
    import shutil
    gcloud_path = shutil.which("gcloud")
    return gcloud_path is not None, gcloud_path or "Not found"

def test_application_default_credentials():
    """Test if application default credentials are set."""
    # Check standard locations
    home = os.path.expanduser("~")
    adc_paths = [
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
        os.path.join(home, ".config", "gcloud", "application_default_credentials.json"),
    ]
    
    for path in adc_paths:
        if path and os.path.exists(path):
            return True, path
    
    return False, "No credentials file found"

def test_google_auth_library():
    """Test if google-auth library can get credentials."""
    try:
        import google.auth
        credentials, project = google.auth.default()
        return True, f"Project: {project}"
    except Exception as e:
        return False, str(e)

def test_vertex_ai_connection():
    """Test actual connection to Vertex AI."""
    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel
        
        # Get project from environment or gcloud config
        project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        if not project_id:
            # Try to get from gcloud config
            import subprocess
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True, text=True
            )
            project_id = result.stdout.strip()
        
        if not project_id:
            return False, "No project ID found. Set GCP_PROJECT_ID env var or run 'gcloud config set project YOUR_PROJECT'"
        
        location = os.environ.get("GCP_LOCATION", "us-central1")
        
        print(f"   Connecting to project: {project_id}, location: {location}")
        
        vertexai.init(project=project_id, location=location)
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        
        # Try a simple embedding
        result = model.get_embeddings(["test"])
        embedding_dim = len(result[0].values)
        
        return True, f"Got embedding with {embedding_dim} dimensions"
        
    except ImportError as e:
        return False, f"Missing library: {e}. Run: pip install google-cloud-aiplatform vertexai"
    except Exception as e:
        return False, str(e)

def get_setup_instructions():
    """Return setup instructions."""
    return """
┌────────────────────────────────────────────────────────────┐
│  GCP SETUP INSTRUCTIONS                                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Install Google Cloud SDK:                               │
│     https://cloud.google.com/sdk/docs/install               │
│                                                             │
│  2. Authenticate with your Google account:                  │
│     $ gcloud auth login                                     │
│     $ gcloud auth application-default login                 │
│                                                             │
│  3. Set your project:                                       │
│     $ gcloud config set project YOUR_PROJECT_ID             │
│                                                             │
│  4. Enable Vertex AI API:                                   │
│     $ gcloud services enable aiplatform.googleapis.com      │
│                                                             │
│  5. (Optional) Set environment variables:                   │
│     $ export GCP_PROJECT_ID=your-project-id                 │
│     $ export GCP_LOCATION=us-central1                       │
│                                                             │
│  6. Install Python libraries:                               │
│     $ pip install google-cloud-aiplatform vertexai          │
│                                                             │
└────────────────────────────────────────────────────────────┘
"""

def main():
    print_header("GCP Authentication Test")
    
    all_passed = True
    
    # Test 1: gcloud CLI
    print("\n📋 Test 1: gcloud CLI")
    passed, details = test_gcloud_cli()
    print_status("gcloud CLI installed", passed, details)
    if not passed:
        all_passed = False
    
    # Test 2: Application Default Credentials
    print("\n📋 Test 2: Application Default Credentials")
    passed, details = test_application_default_credentials()
    print_status("Credentials file exists", passed, details)
    if not passed:
        all_passed = False
    
    # Test 3: google-auth library
    print("\n📋 Test 3: Google Auth Library")
    passed, details = test_google_auth_library()
    print_status("Can load credentials", passed, details)
    if not passed:
        all_passed = False
    
    # Test 4: Vertex AI connection
    print("\n📋 Test 4: Vertex AI Connection")
    passed, details = test_vertex_ai_connection()
    print_status("Can connect to Vertex AI", passed, details)
    if not passed:
        all_passed = False
    
    # Summary
    print_header("Summary")
    
    if all_passed:
        print("🎉 All tests passed! You're ready to use Vertex AI embeddings.")
        print("\nTo use real embeddings in your code:")
        print("  embedder = get_embedder(mock=False)")
    else:
        print("❌ Some tests failed. Follow the setup instructions below.")
        print(get_setup_instructions())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())