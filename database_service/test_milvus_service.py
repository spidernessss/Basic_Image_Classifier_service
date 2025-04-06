import requests
import json
DATABASE_SERVICE_URL = "http://127.0.0.1:8013"

url1 = f"{DATABASE_SERVICE_URL}//create_database_and_client"
try:
    response = requests.post(url1)
    response.raise_for_status()
    print("The database was created")
except requests.exceptions.RequestException as e:
    print(f"Error create_database_and_client: {e}")

url2 = f"{DATABASE_SERVICE_URL}//list_databases"
try:
    response = requests.get(url2)
    response.raise_for_status()
    data = response.json()
    databases = data["databases"]
    print(databases)
except requests.exceptions.RequestException as e:
    print(f"Error listing databases: {e}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON response: {e}")
except KeyError as e:
    print(f"KeyError: 'databases' not found in JSON response: {e}")

url31 = f"{DATABASE_SERVICE_URL}//list_db_collections/service"
try:
    response = requests.get(url31)
    response.raise_for_status()
    data = response.json()
    collections = data["collections"]
    print(collections)
except requests.exceptions.RequestException as e:
    print(f"Error list_collections/service: {e}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON response: {e}")
except KeyError as e:
    print(f"KeyError: 'collections' not found in JSON response: {e}")

url3 = f"{DATABASE_SERVICE_URL}//view_collection/vectors_of_images"
try:
    response = requests.get(url3)
    response.raise_for_status()
    data = response.json()
    details = data["collection_details"]
    print(details)
except requests.exceptions.RequestException as e:
    print(f"Error view_collection/vectors_of_images: {e}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON response: {e}")
except KeyError as e:
    print(f"KeyError: 'collection_details' not found in JSON response: {e}")

