FROM python:3.10-slim

WORKDIR /app

# Add build tools for compiling dependencies
RUN apt-get update && apt-get install -y build-essential gfortran python3-dev

# Upgrade pip and install Cython early
RUN pip install --upgrade pip && pip install cython

# Copy requirements and install them
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

ENTRYPOINT ["python", "main.py"]
    