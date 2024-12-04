# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto
from typing import Any, List, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable

from pydantic import AnyUrl, FilePath

Metadata = Union[Mapping[str, Any], None]
Vector = Union[Sequence[float], Sequence[int]]
ItemID = Union[str, int]  # chromadb doesn't support int ids, VikingDB does
SourceLocation = Union[FilePath, AnyUrl, str]
Distance = float


class NodeType(str, Enum):

    DOCUMENT = "document"
    ENTITY = "entity"
    RELATION = "relation"
    MEMORY = "memory"
    OTHERS = "others"


class DocumentType(Enum):
    """
    Enum for supporting document type.
    """

    TEXT = auto()
    HTML = auto()
    PDF = auto()
    IMAGE = auto()
    AUDIO = auto()


class DatastoreType(str, Enum):

    VECTOR = "vector"
    GRAPH = "graph"
    SQL = "sql"


@runtime_checkable
class Node(Protocol):

    id: Optional[ItemID] = None
    metadata: Metadata
    content: Any = None

    @property
    def nodetype(self) -> NodeType:
        return self.metadata.get("nodetype", NodeType.OTHERS)


@runtime_checkable
class DB(Protocol):

    metadata: Metadata

    @property
    def db_type(self) -> str:
        return self.metadata.get("db_type", "unknown")

    def init(self, *args, **kwargs):
        pass


@runtime_checkable
class QueryEngine(Protocol):

    db: DB

    def init_db(self, *args, **kwargs):
        pass

    def add_records(self, new_records: List):
        pass

    def query(self, query: str, **kwargs):
        pass


@runtime_checkable
class Document(Protocol):
    """A Document is a record in the vector database.

    id: ItemID | the unique identifier of the document.
    content: str | the text content of the chunk.
    metadata: Metadata, Optional | contains additional information about the document such as source, date, etc.
    embedding: Vector, Optional | the vector representation of the content.
    """

    metadata: Metadata = {}
    content: Optional[object]

    @property
    def doctype(self) -> str:
        return self.metadata.get("doctype", DocumentType.TEXT).value


__all__ = [
    "Metadata",
    "Vector",
    "SourceLocation",
    "NodeType",
    "DocumentType",
    "DatastoreType",
    "Node",
    "DB",
    "QueryEngine",
    "Document",
]
