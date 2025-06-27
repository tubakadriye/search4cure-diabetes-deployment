
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
from db.mongodb_client import MONGODB_URI, MONGO_DB
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents import tool
from langchain_mongodb import MongoDBAtlasVectorSearch

DB_NAME = MONGO_DB
ATLAS_VECTOR_SEARCH_INDEX = "csv_vector_index"
CSV_COLLECTION= "records_embeddings"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"

embedding_model = GoogleGenerativeAIEmbeddings(
    model=GEMINI_EMBEDDING_MODEL,
    task_type="RETRIEVAL_DOCUMENT"
)

# Vector Store Intialisation
vector_store_csv_files = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_URI,
    namespace=DB_NAME + "." + CSV_COLLECTION,
    embedding=embedding_model,
    index_name=ATLAS_VECTOR_SEARCH_INDEX,
    text_key="combined_info",
)




@tool
def csv_files_vector_search_tool(query: str, k: int = 5):
    """
    Perform a vector similarity search on safety procedures.

    Args:
        query (str): The search query string.
        k (int, optional): Number of top results to return. Defaults to 5.

    Returns:
        list: List of tuples (Document, score), where Document is a record
              and score is the similarity score (lower is more similar).

    Note:
        Uses the global vector_store_csv_files for the search.
    """

    vector_search_results = vector_store_csv_files.similarity_search_with_score(
        query=query, k=k
    )
    return vector_search_results

## Generalized Record Document Creator
class GenericRecord(BaseModel):
    # Flexible for any CSV columns
    data: Dict[str, Any]
    combined_info: Optional[str] = None
    embedding: Optional[List[float]] = None
    dataset_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True



## Function to Create the Document
def create_generic_record_document(row: Dict[str, Any], dataset_id: str) -> dict:
    """
    Create a standardized document for any CSV record with flexible columns.

    Args:
        row (Dict[str, Any]): The actual CSV row as a dictionary.
        dataset_id (str): Reference to the parent dataset document.

    Returns:
        dict: Cleaned and standardized MongoDB document.
    """
    try:
        # Clean keys: remove leading/trailing whitespace, lowercase, etc. (optional)
        cleaned_row = {k.strip(): v for k, v in row.items()}

        # Create combined_info string for text search
        combined_info = " ".join(f"{k}: {v}" for k, v in cleaned_row.items())

        # Package into a generic Pydantic record
        record = GenericRecord(
            data=cleaned_row,
            combined_info=combined_info,
            dataset_id=dataset_id
        )

        return record.dict()
    except Exception as e:
        raise ValueError(f"Invalid record data: {e!s}")


# Tool to add new record
@tool
def create_new_record(new_record: Dict[str, any]) -> dict:
    """
    Create and validate a new generic CSV record.

    Args:
        new_record (dict): Dictionary containing a row from a CSV file. 
                           Must include 'dataset_id' as a key.

    Returns:
        dict: Validated and formatted record document.

    Raises:
        ValueError: If the input data is invalid or incomplete.

    Note:
        Uses Pydantic for data validation via create_generic_record_document function.
    """
    try:
        dataset_id = new_record.pop("dataset_id", None)
        if not dataset_id:
            raise ValueError("Missing required field 'dataset_id' in the new record.")

        document = create_generic_record_document(new_record, dataset_id)
        return document

    except Exception as e:
        raise ValueError(f"Error creating new record: {e}")
    

@tool
def hybrid_search_tool(query: str):
    """
    Perform a hybrid (vector + full-text) search on safety procedures.

    Args:
        query (str): The search query string.

    Returns:
        list: Relevant safety procedure documents from hybrid search.

    Note:
        Uses both vector_store_csv_files and record_text_search_index.
    """

    hybrid_search = MongoDBAtlasHybridSearchRetriever(
        vectorstore=vector_store_csv_files,
        search_index_name="record_text_search_index",
        top_k=5,
    )

    hybrid_search_result = hybrid_search.get_relevant_documents(query)

    return hybrid_search_result



