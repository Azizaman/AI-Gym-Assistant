FROM jrottenberg/ffmpeg:5.1-ubuntu2204 as ffmpeg

FROM python:3.11-slim

# Copy ffmpeg binaries from full image
COPY --from=ffmpeg /usr/bin/ffmpeg /usr/bin/ffmpeg
COPY --from=ffmpeg /usr/bin/ffprobe /usr/bin/ffprobe

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

CMD ["python", "app.py"]
