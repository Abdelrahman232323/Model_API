# Use official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# ✅ Copy only requirements first (for caching)
COPY requirements.txt .

# ✅ Install dependencies first
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Then copy the rest of your project
COPY ./app ./app


ENV PYTHONPATH=/app

# Set the working directory for FastAPI
WORKDIR /app

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the API with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
