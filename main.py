import zipfile
from urllib import request
from fastapi import FastAPI, Request, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.responses import HTMLResponse
from PIL import Image
from model import model_util
from model import image_classification
from model.image_classification import ImageClassifier


import tensorflow
import keras
import subprocess
import os
import shutil
import time
import cv2

# --- TO RUN DO THIS ---
# uvicorn main:app --reload --port 8010
# STANDART PORT AND THE LINE BELOW IS NOT WORKING!
# INTERNAL SERVER ERROR!
# fastapi dev --app app

app = FastAPI(title="Matrix Image Classification")

# Loading model
model = ImageClassifier()

# --- Configure Templates and Static Files ---
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Define Directory Paths ---
#TEMP_DIR = 'C:/Users/saido/Desktop/Service_for_classification/tmp'
#DATASET_DIR = 'C:/Users/saido/Desktop/Service_for_classification/tmp/dataset'
#INPUT_IMAGE_DIR = 'C:/Users/saido/Desktop/Service_for_classification/tmp/input_image'
#OUTPUT_IMAGE_DIR = 'C:/Users/saido/Desktop/Service_for_classification/tmp/output_image'

# --- Create temporary directories ---

#CURR_DIR = os.getcwd()
CURR_DIR = "C:/Users/saido/Desktop/Service_for_classification"
print(CURR_DIR)
TEMP_DIR = os.path.join(CURR_DIR, "tmp").replace('\\', '/')
print(TEMP_DIR)
DATASET_DIR = os.path.join(TEMP_DIR, "dataset").replace('\\', '/')
print(DATASET_DIR)
INPUT_IMAGE_DIR = os.path.join(TEMP_DIR, "input_image").replace('\\', '/')
print(INPUT_IMAGE_DIR)
OUTPUT_IMAGE_DIR = os.path.join(TEMP_DIR, "output_image").replace('\\', '/')
print(OUTPUT_IMAGE_DIR)
#TEMP_DIR = 'C:/Users/saido/Desktop/Service_for_classification/tmp'
#DATASET_DIR = 'C:/Users/saido/Desktop/Service_for_classification/tmp/dataset'
#INPUT_IMAGE_DIR = 'C:/Users/saido/Desktop/Service_for_classification/tmp/input_image'
#OUTPUT_IMAGE_DIR = 'C:/Users/saido/Desktop/Service_for_classification/tmp/output_image'
os.makedirs(DATASET_DIR)
os.makedirs(INPUT_IMAGE_DIR)
os.makedirs(OUTPUT_IMAGE_DIR)


DATABASE_SERVICE_URL = "http://127.0.0.1:8013"


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def post(request: Request):
    text = request.get("sent")


@app.post("/process")
async def process_image(dataset: UploadFile = File(...), input_image: UploadFile = File(...), requests=None):
    try:
        # --- Save dataset and image ---
        with open(os.path.join(INPUT_IMAGE_DIR,'input_image.jpg').replace('\\', '/'), "wb") as f:
            f.write(await input_image.read())
        with open(os.path.join(DATASET_DIR, "dataset.zip").replace('\\', '/'), "wb") as f:
            f.write(await dataset.read())
        # Unzip the dataset
        try:
            with zipfile.ZipFile(os.path.join(DATASET_DIR, "dataset.zip").replace('\\', '/'), 'r') as zip_ref:
                zip_ref.extractall(DATASET_DIR)
        except zipfile.BadZipFile:
            raise ValueError(f"The file dataset.zip is not a zip file or it is corrupted.")

        # Create Milvus database and collection
        url = f"{DATABASE_SERVICE_URL}//create_database_and_client"
        try:
            response = requests.post(url)  # Send a POST request with the vector data as JSON
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        except requests.exceptions.RequestException as e:
            print(f"Error create_database_and_client: {e}")

        # Add classes to the model
        for class_name in os.listdir(os.path.join(DATASET_DIR, "dataset").replace('\\', '/')):
            class_path = "dataset/" + class_name
            class_folder = os.path.join(DATASET_DIR, class_path).replace('\\', '/')
            print("class folder: ", class_folder)
            if os.path.isdir(class_folder):
                # MILVUS BEGIN
                url = f"{DATABASE_SERVICE_URL}/insert_vector"
                try:
                    vectors = model.get_class_vectors(class_name, class_folder)
                    for vector in vectors:
                        response = requests.post(url, json=vector)
                        response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"Error inserting vector: {e}")
                # MILVUS END
                # model.add_class(class_name, class_folder)
                print("Added class ", class_name, " to ", "class_folder\n")
        # Read the saved image using OpenCV
        image_path = os.path.join(INPUT_IMAGE_DIR, "input_image.jpg").replace('\\', '/')

        print("Image path: ", image_path)
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            print("File exists and is non-empty.")
        else:
            print("File does not exist or is empty.")

        image = cv2.imread(image_path)

        #img = Image.open(image_path)
        #img.show()  # This will open the image using the default image viewer
        print("Ok")
        print("Image is ", image)

        COLOR_GREEN = (100, 255, 100)

        # Get window sizes directly from the loaded image
        height, width, _ = image.shape
        window_size = (width, height)
        #name, dist = model.predict(image)

        image_features = model.extract_features(image)
        # MILVUS BEGIN
        try:
            response = requests.post(url,
                                     json={"query_vector": image_features.tolist(), "collection_name": "vectors_of_images"})
            response.raise_for_status()
            results = response.json()["results"]
            #find the best match
            best_match = max(results, key=lambda x: x["similarity"])
            if best_match["similarity"] > 0.8:
                name = best_match["class_name"]
                dist = best_match["distance"]
                print("The dist > 0.8")
                colour = COLOR_GREEN
                x = 45
                y = 45
                image = model.draw_label(image, x, y, window_size[0], window_size[1], name, colour)
                print(image)
            else:
                name = "Unknown"
                dist = 0
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with FastAPI endpoint: {e}")
            return "Error", 0
        except Exception as e:
            print(f"Error in image classification: {e}")
            return "Error", 0

        # Save the output image
        output_file_path = os.path.join(OUTPUT_IMAGE_DIR, "output_image.jpg").replace('\\', '/')
        cv2.imwrite(output_file_path, image)

        # out_img = Image.open(output_file_path)
        # out_img.show()  # This will open the image using the default image viewer

        print("Ok5")
        return JSONResponse({"output_file_path": output_file_path})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_IMAGE_DIR, filename).replace('\\', '/')
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename, media_type="image/jpg")
    else:
        return JSONResponse({"error": "File not found"}, status_code=404)

@app.on_event("shutdown")
async def shutdown_event():
    """
    Delete specified directories when the FastAPI application shuts down.
    """
    dirs_to_delete = ['tmp']
    for directory in dirs_to_delete:
        try:
            shutil.rmtree(directory)
            print(f"Directory '{directory}' deleted successfully.")
        except FileNotFoundError:
            print(f"Directory '{directory}' not found. Skipping deletion.")
