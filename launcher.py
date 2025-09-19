import subprocess
import sys
import time

# Paths to your scripts
API_SCRIPT = "./lib/api.py"
STREAMLIT_APP = "./app/app.py"

# Command to run FastAPI (Uvicorn)
api_cmd = [sys.executable, "-m", "uvicorn", "lib.api:app", "--reload", "--host", "127.0.0.1", "--port", "8000"]

# Command to run Streamlit
streamlit_cmd = ["streamlit", "run", STREAMLIT_APP]

# Start the API server
print("Starting API server...")
api_process = subprocess.Popen(api_cmd)

# Small delay to let API start before launching Streamlit
time.sleep(2)

# Start the Streamlit app
print("Starting Streamlit app...")
streamlit_process = subprocess.Popen(streamlit_cmd)

# Wait for both processes to finish (optional)
try:
    api_process.wait()
    streamlit_process.wait()
except KeyboardInterrupt:
    print("Stopping both servers...")
    api_process.terminate()
    streamlit_process.terminate()
