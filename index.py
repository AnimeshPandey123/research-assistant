from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
import pandas as pd
import time

# Connect to Elasticsearch
try:
    print("Connecting to Elasticsearch...")
    es = Elasticsearch("http://localhost:9200", 
        request_timeout=600,  # Increase from default 10 seconds
        max_retries=10,      # Allow more retries
        retry_on_timeout=True)
    if not es.ping():
        raise ValueError("Connection to Elasticsearch failed.")
    print("Connected to Elasticsearch")
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")
    exit(1)

# Generator function to stream documents from the dataset
def generate_documents(file_path, batch_size=10000):
    try:
        for chunk in pd.read_json(file_path, lines=True, chunksize=batch_size):
            for _, row in chunk.iterrows():
                yield {
                    "_index": "arxiv_v1",
                    "_id": row["id"],
                    "_source": {
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "categories": row["categories"],
                        "update_date": row["update_date"],
                        "submitter": row.get("submitter"),
                        "authors": row.get("authors"),
                        "comments": row.get("comments"),
                        "journal-ref": row.get("journal-ref"),
                        "doi": row.get("doi"),
                        "report-no": row.get("report-no"),
                        "license": row.get("license"),
                        "versions": row.get("versions"),
                        "authors_parsed": row.get("authors_parsed")
                    },
                }
    except Exception as e:
        print(f"Error reading the dataset or processing the documents: {e}")
        exit(1)

# Function to index documents using streaming bulk
def index_documents(file_path):
    start_time = time.time()
    total_docs = 0

    try:
        for success, info in streaming_bulk(es, generate_documents(file_path), chunk_size=1000):
            if not success:
                print(f"Error indexing a document: {info}")  # More verbose error info
            total_docs += 1
            if total_docs % 1000 == 0:
                elapsed = time.time() - start_time
                docs_per_second = total_docs / elapsed if elapsed > 0 else 0
                print(f"Indexed {total_docs} documents ({docs_per_second:.1f} docs/sec)")
    except Exception as e:
        print(f"Error during the bulk indexing process: {e}")
        exit(1)

    total_time = time.time() - start_time
    print(f"\nIndexing complete!")
    print(f"Total documents indexed: {total_docs}")
    print(f"Time taken: {total_time:.2f} seconds")
    print(f"Average speed: {total_docs / total_time:.2f} documents/second")

# Run the indexing process
index_documents("dataset/arxiv-metadata-oai-snapshot.json")
# Indexing complete!
# Total documents indexed: 2689088
# Time taken: 675.73 seconds
# Average speed: 3979.54 documents/second