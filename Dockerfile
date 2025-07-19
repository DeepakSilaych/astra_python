FROM python:3.11-slim

# Ensure stdout/stderr are unbuffered
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  libgomp1 \
  libgtk-3-0 \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libv4l-dev \
  libxvidcore-dev \
  libx264-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libatlas-base-dev \
  gfortran \
  wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create img directory
RUN mkdir -p img

# Expose the port that Cloud Run uses for health checks
EXPOSE 8080

# Default command (override in compose for web vs worker)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]