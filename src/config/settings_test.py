# test_settings.py
from settings import settings

# Print all settings
print(f"Project root: {settings.BASE_DIR}")
print(f"PDF path: {settings.pdf_path}")
print(f"Chunk size: {settings.CHUNK_SIZE_TOKENS} tokens")
print(f"LLM temp: {settings.LLM_TEMPERATURE}")

# Show how override works
import os
os.environ["CHUNK_SIZE_TOKENS"] = "256"

# Need to reload to pick up env var
new_settings = settings.__class__()
print(f"New chunk size: {new_settings.CHUNK_SIZE_TOKENS}")  # 256!
print("Settings test module executed successfully.")