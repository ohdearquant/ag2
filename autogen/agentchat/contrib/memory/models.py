# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class CommonModel(BaseModel):
    """
    Model for common knowledge operations (file-based storage).
    Used for managing documentation, procedures, and reference materials.
    """

    operation: Literal["add", "update", "delete", "query"] = Field(
        description="Operation to perform on common knowledge"
    )
    category: str = Field(description="Category or type of the knowledge")
    content: Optional[str] = Field(default=None, description="Content to store or update")
    tags: List[str] = Field(default_factory=list, description="Searchable tags for categorization")
    metadata: Dict = Field(default_factory=dict, description="Additional structured information")
    filename: Optional[str] = Field(default=None, description="Document identifier for updates/deletes")

    model_config = ConfigDict(
        title="Common Knowledge Operations",
        description="Operations for file-based knowledge storage",
        json_schema_extra={
            "examples": [
                {
                    "operation": "add",
                    "category": "documentation",
                    "content": "API usage guide...",
                    "tags": ["api", "guide"],
                    "metadata": {"version": "1.0"},
                }
            ]
        },
        validate_default=True,
        extra="forbid",
    )


class DomainModel(BaseModel):
    """
    Model for domain knowledge operations (graph-based storage).
    Used for managing structured knowledge with relationships and schema.
    """

    operation: Literal["add", "update", "query"] = Field(description="Operation to perform on domain knowledge")
    content: List[str] = Field(default_factory=list, description="List of content to process and store")
    mode: Literal["replace", "extend"] = Field(
        default="extend", description="How to handle updates: replace all or extend existing"
    )
    query: Optional[str] = Field(default=None, description="Query string for information retrieval")

    model_config = ConfigDict(
        title="Domain Knowledge Operations",
        description="Operations for graph-based knowledge storage",
        json_schema_extra={
            "examples": [{"operation": "query", "query": "What is the relationship between X and Y?", "mode": "extend"}]
        },
        validate_default=True,
        extra="forbid",
    )


class GeneralModel(BaseModel):
    """
    Model for general memory operations (vector-based storage).
    Used for semantic storage and retrieval of varied information.
    """

    operation: Literal["add", "update", "delete", "query"] = Field(description="Operation to perform on general memory")
    content: Optional[str] = Field(default=None, description="Content to store or query against")
    metadata: Dict = Field(default_factory=dict, description="Additional memory attributes")
    memory_id: Optional[str] = Field(default=None, description="Memory identifier for updates/deletes")
    query_filter: Optional[Dict] = Field(default=None, description="Metadata filters for querying")

    model_config = ConfigDict(
        title="General Memory Operations",
        description="Operations for vector-based memory storage",
        json_schema_extra={
            "examples": [
                {
                    "operation": "add",
                    "content": "Meeting discussion about project timeline",
                    "metadata": {"type": "meeting_notes", "date": "2024-01-01"},
                }
            ]
        },
        validate_default=True,
        extra="forbid",
    )


class MemoryAnalysisModel(BaseModel):
    """
    Unified model for memory operations across all storage types.

    Coordinates operations across:
    - Common Knowledge: File-based storage for reference materials
    - Domain Knowledge: Graph-based storage for structured information
    - General Memory: Vector-based storage for semantic search
    """

    common: Optional[CommonModel] = Field(default=None, description="Operations for common knowledge storage")
    domain: Optional[DomainModel] = Field(default=None, description="Operations for domain knowledge storage")
    general: Optional[GeneralModel] = Field(default=None, description="Operations for general memory storage")

    model_config = ConfigDict(
        title="Memory Analysis Operations",
        description="Unified model for coordinating memory operations",
        json_schema_extra={
            "examples": [
                {
                    "common": {
                        "operation": "add",
                        "category": "documentation",
                        "content": "API guide...",
                        "tags": ["api"],
                    }
                },
                {"domain": {"operation": "query", "query": "How are components X and Y related?"}},
            ]
        },
        validate_default=True,
        extra="forbid",
        validate_assignment=True,
        frozen=False,
    )
