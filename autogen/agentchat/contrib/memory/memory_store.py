# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from graphrag_sdk import Attribute, Entity, Ontology, Relation

from ..graph_rag.document import Document
from ..graph_rag.falkor_graph_query_engine import FalkorGraphQueryEngine
from ..vectordb.base import Document as VectorDocument
from ..vectordb.chromadb import ChromaVectorDB
from ..vectordb.utils import get_logger

logger = get_logger(__name__)


class MemoryOperation(Enum):
    """Available memory operations"""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"


class MemoryType(Enum):
    """Memory storage types"""

    COMMON = "common"  # File-based document storage
    DOMAIN = "domain"  # Schema-based graph storage
    GENERAL = "general"  # Vector-based semantic storage


@dataclass
class SchemaConfig:
    """Domain knowledge schema configuration"""

    entities: List[Entity]
    relations: List[Relation]
    attributes: Optional[List[Attribute]] = None


@dataclass
class Memory:
    """Basic memory unit"""

    content: str
    metadata: Optional[Dict] = None
    timestamp: datetime = None
    id: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()


@dataclass
class MemoryRequest:
    """
    Unified memory operation request

    Args:
        operation: Type of operation to perform
        documents: Raw documents for common/domain storage
        vector_docs: Vector documents for semantic storage
        schema_config: Schema configuration for domain storage
        query: Query string for retrieval operations
        metadata: Additional metadata for the operation
        mode: Operation mode for updates ("replace" or "extend")
        collection_name: Target collection for vector operations
        n_results: Number of results to return for queries
        distance_threshold: Similarity threshold for vector queries
    """

    operation: MemoryOperation
    documents: Optional[List[Document]] = None
    vector_docs: Optional[List[VectorDocument]] = None
    schema_config: Optional[SchemaConfig] = None
    query: Optional[str] = None
    metadata: Optional[Dict] = None
    mode: Literal["replace", "extend"] = "extend"
    collection_name: Optional[str] = None
    n_results: int = 5
    distance_threshold: float = 0.7


class MemoryStore:
    """
    Unified storage system managing different types of memory.

    Supports:
    - Common Knowledge: File-based storage for raw documents
    - Domain Knowledge: Graph-based storage with schema/ontology
    - General Knowledge: Vector-based storage for semantic search

    Args:
        base_path: Base directory for storage
        graph_name: Name for the graph database
        graph_host: Host for the graph database
        graph_port: Port for the graph database
    """

    def __init__(
        self,
        base_path: str = "./memory_store",
        graph_name: str = "knowledge_graph",
        graph_host: str = "localhost",
        graph_port: int = 6379,
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        # Initialize storages
        self._init_common_storage()
        self._init_vector_storage(base_path)
        self._init_graph_storage(graph_name, graph_host, graph_port)

        logger.info(f"Initialized UnifiedMemoryStore at {base_path}")

    def _init_common_storage(self):
        """Initialize file-based document storage"""
        self.common_path = self.base_path / "common"
        self.common_path.mkdir(exist_ok=True)
        self.common_index = self._load_common_index()

    def _init_vector_storage(self, base_path: str):
        """Initialize vector storage with ChromaDB backend"""
        vector_path = str(Path(base_path) / "vector")

        self.vector_db = ChromaVectorDB(
            path=vector_path, metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 100, "hnsw:M": 64}
        )

        # Initialize default collection
        self.vector_db.create_collection("general_memories", get_or_create=True)

    def _init_graph_storage(self, graph_name: str, graph_host: str, graph_port: int):
        """Initialize graph storage with FalkorDB backend"""
        self.graph_name = graph_name
        self.graph_host = graph_host
        self.graph_port = graph_port
        self.graph_db = None  # Initialized with schema

    def _load_common_index(self) -> Dict:
        """Load common knowledge index from disk"""
        index_file = self.common_path / "index.json"
        if index_file.exists():
            return json.loads(index_file.read_text())
        return {}

    def _save_common_index(self):
        """Save common knowledge index to disk"""
        index_file = self.common_path / "index.json"
        index_file.write_text(json.dumps(self.common_index, indent=2))

    def operate_common_knowledge(self, request: MemoryRequest) -> Dict[str, Any]:
        """
        Handle operations for common knowledge storage.

        Manages file-based storage of raw documents with metadata.

        Args:
            request: Operation request containing documents and metadata

        Returns:
            Dict containing operation results and status
        """
        results = {}

        try:
            if request.operation == MemoryOperation.ADD and request.documents:
                for doc in request.documents:
                    doc_id = doc.id or f"doc_{datetime.now():%Y%m%d_%H%M%S}"
                    filepath = self.common_path / f"{doc_id}.txt"

                    # Store document
                    filepath.write_text(doc.content)
                    self.common_index[doc_id] = {
                        "type": str(doc.doctype),
                        "metadata": request.metadata or {},
                        "timestamp": datetime.now().isoformat(),
                    }
                    results[doc_id] = "added"
                self._save_common_index()

            elif request.operation == MemoryOperation.UPDATE and request.documents:
                for doc in request.documents:
                    if not doc.id or doc.id not in self.common_index:
                        continue

                    filepath = self.common_path / f"{doc.id}.txt"
                    filepath.write_text(doc.content)

                    if request.metadata:
                        self.common_index[doc.id]["metadata"].update(request.metadata)
                    results[doc.id] = "updated"
                self._save_common_index()

            elif request.operation == MemoryOperation.DELETE and request.documents:
                for doc in request.documents:
                    if not doc.id or doc.id not in self.common_index:
                        continue

                    filepath = self.common_path / f"{doc.id}.txt"
                    if filepath.exists():
                        filepath.unlink()
                    del self.common_index[doc.id]
                    results[doc.id] = "deleted"
                self._save_common_index()

            elif request.operation == MemoryOperation.QUERY:
                metadata_filter = request.metadata or {}
                for doc_id, info in self.common_index.items():
                    if all(info["metadata"].get(k) == v for k, v in metadata_filter.items()):
                        filepath = self.common_path / f"{doc_id}.txt"
                        if filepath.exists():
                            results[doc_id] = {"content": filepath.read_text(), **info}

        except Exception as e:
            logger.error(f"Error in common knowledge operation: {e}")
            results["error"] = str(e)

        return results

    def operate_domain_knowledge(self, request: MemoryRequest) -> Dict[str, Any]:
        """
        Handle operations for domain knowledge storage.

        Manages graph-based storage with schema/ontology support.

        Args:
            request: Operation request containing schema and documents

        Returns:
            Dict containing operation results and status
        """
        results = {}

        try:
            # Initialize/update schema
            if request.schema_config:
                ontology = Ontology()

                for entity in request.schema_config.entities:
                    ontology.add_entity(entity)
                for relation in request.schema_config.relations:
                    ontology.add_relation(relation)

                self.graph_db = FalkorGraphQueryEngine(
                    name=self.graph_name, host=self.graph_host, port=self.graph_port, ontology=ontology
                )
                results["schema_status"] = "initialized"

            if not self.graph_db:
                raise ValueError("Graph database not initialized. Schema configuration required.")

            if request.operation in [MemoryOperation.ADD, MemoryOperation.UPDATE]:
                if request.documents:
                    if request.mode == "replace":
                        self.graph_db.init_db(request.documents)
                    else:
                        existing_docs = self.graph_db.get_documents()
                        self.graph_db.init_db(existing_docs + request.documents)
                    results["status"] = "graph_updated"

            elif request.operation == MemoryOperation.QUERY and request.query:
                query_result = self.graph_db.query(request.query)
                if query_result:
                    results.update({"answer": query_result.answer, "source_documents": query_result.source_documents})

        except Exception as e:
            logger.error(f"Error in domain knowledge operation: {e}")
            results["error"] = str(e)

        return results

    def operate_general_knowledge(self, request: MemoryRequest) -> Dict[str, Any]:
        """
        Handle operations for general knowledge storage.

        Manages vector-based storage for semantic search using ChromaDB.

        Args:
            request: Operation request containing vector documents

        Returns:
            Dict containing operation results and status
        """
        collection_name = request.collection_name or "general_memories"
        results = {}

        try:
            if request.operation == MemoryOperation.ADD and request.vector_docs:
                self.vector_db.insert_docs(docs=request.vector_docs, collection_name=collection_name)
                results.update({"status": "documents_added", "count": len(request.vector_docs)})

            elif request.operation == MemoryOperation.UPDATE and request.vector_docs:
                self.vector_db.update_docs(docs=request.vector_docs, collection_name=collection_name)
                results.update({"status": "documents_updated", "count": len(request.vector_docs)})

            elif request.operation == MemoryOperation.DELETE and request.vector_docs:
                ids = [doc.id for doc in request.vector_docs if doc.id]
                self.vector_db.delete_docs(ids=ids, collection_name=collection_name)
                results.update({"status": "documents_deleted", "count": len(ids)})

            elif request.operation == MemoryOperation.QUERY and request.query:
                query_results = self.vector_db.retrieve_docs(
                    queries=[request.query],
                    collection_name=collection_name,
                    n_results=request.n_results,
                    distance_threshold=request.distance_threshold,
                    where=request.metadata,
                )
                results.update({"matches": query_results[0], "count": len(query_results[0])})  # First query's results

        except Exception as e:
            logger.error(f"Error in general knowledge operation: {e}")
            results["error"] = str(e)

        return results
