from elasticsearch import Elasticsearch

# Initialize Elasticsearch client
es = Elasticsearch("http://localhost:9200")

def search_arxiv_papers(query, index_name="arxiv", size=10):
    """
    Search for arxiv papers using query expansion techniques in Elasticsearch.
    
    Parameters:
    - query: The user's search query
    - index_name: Name of your Elasticsearch index containing arxiv papers
    - size: Number of results to return
    
    Returns:
    - List of relevant papers
    """
    # Build a query that uses multiple fields with different weights
    search_body = {
        "size": size,
        "query": {
            "bool": {
                "should": [
                    # Exact match on title carries highest weight
                    {"match_phrase": {
                        "title": {
                            "query": query,
                            "boost": 3.0
                        }
                    }},
                    # Terms in title
                    {"match": {
                        "title": {
                            "query": query,
                            "boost": 2.0,
                            "fuzziness": "AUTO"
                        }
                    }},
                    # Terms in abstract
                    {"match": {
                        "abstract": {
                            "query": query,
                            "boost": 1.0
                        }
                    }},
                    # Match on author names
                    {"match": {
                        "authors": {
                            "query": query,
                            "boost": 1.5
                        }
                    }},
                    # Match on categories/topics
                    {"match": {
                        "categories": {
                            "query": query,
                            "boost": 1.8
                        }
                    }}
                ],
                # Optional: Add filters based on date, category, etc.
                "filter": []
            }
        },
        # Use highlighting to show where matches occurred
        "highlight": {
            "fields": {
                "title": {},
                "abstract": {"fragment_size": 150, "number_of_fragments": 3}
            }
        },
        # Use Elasticsearch's "more like this" feature for query expansion
        "suggest": {
            "term-suggestion": {
                "text": query,
                "term": {
                    "field": "title",
                    "suggest_mode": "always"
                }
            }
        }
    }
    
    # Execute search
    response = es.search(index=index_name, body=search_body)
    
    # Process results
    results = []
    for hit in response["hits"]["hits"]:

        paper = {
            "id": hit["_id"],
            "score": hit["_score"],
            "title": hit["_source"].get("title", ""),
            "abstract": hit["_source"].get("abstract", ""),
            "authors": hit["_source"].get("authors", []),
            "categories": hit["_source"].get("categories", []),
            "url": f"https://arxiv.org/pdf/{hit['_id']}",
            "year": hit["_source"].get("update_date", {})
        }
        
        # Add highlighted sections if available
        if "highlight" in hit:
            paper["highlights"] = hit["highlight"]
            
        results.append(paper)
    
    # Include query expansion suggestions
    suggestions = []
    if "suggest" in response:
        for suggestion in response["suggest"]["term-suggestion"]:
            for option in suggestion["options"]:
                suggestions.append(option["text"])
    
    return {
        "results": results,
        "suggestions": suggestions,
        "total": response["hits"]["total"]["value"] if isinstance(response["hits"]["total"], dict) else response["hits"]["total"]
    }


# print(search_arxiv_papers("machine learning algorithms"))