# # Use slim version of Python 3.10 for a smaller image size
# FROM python:3.10


# # Set the working directory inside the container
# WORKDIR /app



# # Install system dependencies for building packages
# RUN apt-get update && apt-get install -y build-essential gfortran

# # Copy requirements first (for better caching) and install dependencies
# COPY requirements.txt .

# RUN pip install -r requirements.txt


# # RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application files
# COPY . .

# # Ensure deploy.py is the correct entry point
# ENTRYPOINT ["python", "main.py"]


# ---------------------------------------------------------
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
    