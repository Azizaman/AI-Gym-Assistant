# Use a base image with Python and Debian (for apt)
FROM python:3.11-slim

# Install FFmpeg and clean up to reduce image size
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set environment variable for cleaner logs
ENV PYTHONUNBUFFERED=1

# Expose the port Flask uses (Render autodetects, but this helps debugging)
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
