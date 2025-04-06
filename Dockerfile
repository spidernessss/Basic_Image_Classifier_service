FROM nvidia/cuda:11.8.0-base-ubuntu20.04

WORKDIR /app

# Install OpenGL libraries
RUN apt-get update && apt-get install -y libgl1-mesa-dev libglu1-mesa-dev libx11-dev libxext-dev libxrender-dev

# COPY main.py .
COPY requirements.txt .

# Install Python dependencies
RUN apt-get update && apt-get install -y python3-pip
# RUN apt-get update && apt-get install -y libglib2.0-0
RUN pip install "fastapi[standard]"
RUN pip install -r requirements.txt
# Without this I have errors with libgthread library
RUN pip install opencv-python-headless

EXPOSE 8010
# CMD ["uvicorn", "main:app", "--reload", "--host", "localhost", "--port", "8010"]
# CMD ["uvicorn", "main:app", "--reload", "--port", "8010"]
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8010"]

# To RUN Write this:
# docker run -it -p 8010:8010
# -v C:/Users/saido/Desktop/Service_for_classification/static:/app/static
# -v C:/Users/saido/Desktop/Service_for_classification/templates:/app/templates
# -v C:/Users/saido/Desktop/Service_for_classification/model:/app/model
# -v C:/Users/saido/Desktop/Service_for_classification/main.py:/app/main.py
# service3:latest
