import json
import traceback

from pymilvus import (connections, utility, MilvusException, db,
                      FieldSchema, CollectionSchema, Collection,
                      DataType, MilvusClient)
id_number = 0

def create_database_and_client(db_name):
    conn = connections.connect(host="localhost", port="19530")
    if db_name not in db.list_database():
        database = db.create_database(db_name)
    client = MilvusClient(
        uri="http://localhost:19530",
        db_name="service"
    )
    return client

def insert_vector(vector, client: MilvusClient, class_name, collection_name):
    try:
        global id_number
        # Insert data
        data = [
            {"id": id_number, "class_name": f"{class_name}",
             "embedding": vector},
        ]
        id_number+=1
        insert_status = client.insert(
            collection_name=f"{collection_name}",
            data=data
        )
        return insert_status
    except MilvusException as e:
        print("Got Milvus exception: ", e)
        return -1

def creating_collection(client, collection_name):
    try:
        # Defining field schema
        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="primary id")
        class_name_field = FieldSchema(name="class_name", dtype=DataType.VARCHAR, max_length=256,
                                       description="name of the vector's class")
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128, description="vector")
        # Defining collection schema
        schema = CollectionSchema(fields=[id_field, class_name_field, embedding_field], auto_id=False,
                                  enable_dynamic_field=True,
                                  description="desc of a collection")
        # Creating a collection with the specified schema
        collection_name1 = collection_name
        collection1 = Collection(name=collection_name1, schema=schema, using='default', shards_num=2)

        # Set up the index parameters
        index_params = MilvusClient.prepare_index_params()
        # Add an index on the vector field.
        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name="vector_index",
            params={"nlist": 128}
        )
        # Create an index file
        client.create_index(
            collection_name=collection_name,
            index_params=index_params,
            sync=False  # Whether to wait for index creation to complete before returning. Defaults to True.
        )
    except MilvusException as e:
        print("Got Milvus exception: ", e)
        return -1

def list_databases():
    return db.list_database()

def list_db_collections(client, database_name: str):
    try:
        collections = client.list_collections(database_name=database_name)
        if collections:
            json_output = json.dumps({"collections": collections})
            print(json_output)
        else:
            print(json.dumps({"collections": []}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
    finally:
        connections.disconnect()

def view_collection(client, collection_name):
    res = client.describe_collection(
        collection_name=collection_name
    )
    return res

def single_vector_search(client, collection_name, query_vector):
    res = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=5,  # Max. number of search results to return
        search_params={"metric_type": "COSINE", "params": {}}
    )
    # Convert the output to a formatted JSON string

    entities = client.get_entities(collection_name=collection_name,
                                   expr=f"id in {tuple(entity.id for entity in res[0])}",
                                   output_fields=["id", "class_name", "embedding"])
    # Process the results
    processed_results = []
    for i in range(len(res[0])):
        entity = entities[i]
        processed_results.append({
            "id": entity.id,
            "class_name": entity.class_name,
            "embedding": entity.embedding,
            "distance": res[0][i].distance,
            "similarity": 1 - res[0][i].distance  # Cosine similarity is 1 - distance
        })
    #result = json.dumps(res, indent=4)
    #print(result)
    #return result
    print(processed_results)
    return processed_results

def drop_collection(client, collection_name):
    client.drop_collection(
        collection_name=f"{collection_name}"
    )

def drop_database(db_name):
    db.drop_database(f"{db_name}")