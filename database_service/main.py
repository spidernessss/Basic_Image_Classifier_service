import json
import traceback
from http.client import HTTPException
from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import JSONResponse

from milvus_functions import (create_database_and_client,
                              insert_vector, creating_collection, list_databases,
                              view_collection, single_vector_search, drop_collection,
                              drop_database, list_db_collections)
from pymilvus import (connections, utility, MilvusException, db,
                      FieldSchema, CollectionSchema, Collection,
                      DataType, MilvusClient)
import uvicorn

# --- TO RUN DO THIS ---
# cd database_service
# uvicorn main:app --reload --port 8013

app = FastAPI(title="Milvus FastAPI Service", description="Service for interacting with Milvus database.")

id_number = 0
client = None # Initialize client globally

@app.post("/create_database_and_client")
async def create_database_and_client_endpoint():
    global client
    client = create_database_and_client("service") # Connect on startup
    creating_collection(client, "vectors_of_images") # create collection on startup
    if client is None:
        raise HTTPException(status_code=500, detail="Milvus client not initialized")

@app.post("/insert_vector", status_code=201)
async def insert_vector_endpoint(vector: list, class_name, collection_name):
    global client
    if client is None:
        raise HTTPException(status_code=500, detail="Milvus client not initialized")
    status = insert_vector(vector, client, class_name, collection_name)
    if status == -1:
        raise HTTPException(status_code=500, detail="Error inserting vector")
    return {"message": "Vector inserted successfully", "insert_status": status}

@app.post("/search_vector")
async def search_vector_endpoint(query_vector: list):
    global client
    if client is None:
        raise HTTPException(status_code=500, detail="Milvus client not initialized")
    results = single_vector_search(client, "vectors_of_images", query_vector)
    if results == -1: # Assuming single_vector_search returns -1 on error
        raise HTTPException(status_code=500, detail="Error searching vectors")
    return {"results": json.loads(results)} # Return as parsed json

@app.get("/view_collection") # Example GET endpoint
async def view_collection_endpoint():
    global client
    if client is None:
        raise HTTPException(status_code=500, detail="Milvus client not initialized")
    view_collection(client, "vectors_of_images")
    return {"message": "Collection details printed to server console"}

@app.get("/list_databases")
async def list_databases_endpoint():
    global client
    if client is None:
        raise HTTPException(status_code=500, detail="Milvus client not initialized")
    databases = list_databases()
    return {"databases": databases}

@app.get("/list_db_collections/{database_name}")
async def list_databases_endpoint(database_name:str):
    global client
    if client is None:
        raise HTTPException(status_code=500, detail="Milvus client not initialized")
    try:
        collections = list_db_collections(database_name)
        return {"collections": collections}
    except MilvusException as e:
        print(f"MilvusException caught: {e}")
        raise HTTPException(status_code=500, detail=f"Error viewing collection: {e}")
    except Exception as e:
        print(f"Unexpected error in view_collection_endpoint: {e}, traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error")

@app.get("/view_collection/{collection_name}")
async def view_collection_endpoint(collection_name: str):
    global client
    if client is None:
        raise HTTPException(status_code=500, detail="Milvus client not initialized")
    try:
        description = view_collection(client, collection_name)
        return {"collection_details": description}
    except MilvusException as e:
        print(f"MilvusException caught: {e}")
        raise HTTPException(status_code=500, detail=f"Error viewing collection: {e}")
    except Exception as e:
        print(f"Unexpected error in view_collection_endpoint: {e}, traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error")

@app.delete("/drop_collection/{collection_name}")
async def drop_collection_endpoint(collection_name: str):
    global client
    if client is None:
        raise HTTPException(status_code=500, detail="Milvus client not initialized")
    try:
        drop_collection(client, collection_name)
    except MilvusException as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {e}")

@app.delete("/drop_database/{db_name}")
async def drop_database_endpoint(db_name: str):
    global client
    if client is None:
        raise HTTPException(status_code=500, detail="Milvus client is not initialized")
    try:
        drop_database(db_name=db_name)

    except MilvusException as e:
        raise HTTPException(status_code=500, detail=f"Error dropping database collection: {e}")




