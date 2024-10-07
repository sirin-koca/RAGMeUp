from pymilvus import connections, Collection


# Connect to Milvus server
def connect_milvus():
    connections.connect(alias="default", host="localhost", port="19530")


# Search in Milvus collection
def search_milvus(vector, top_k=5):
    collection = Collection("neet_collection")
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search([vector], "embedding", search_params, limit=top_k)
    return results


# Drop Milvus collection (for resetting)
def drop_milvus_collection():
    collection = Collection("neet_collection")
    collection.drop()


# Example usage
if __name__ == "__main__":
    connect_milvus()
    print("Connected to Milvus")
