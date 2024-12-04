# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Optional

from autogen.extensions import RAG


@dataclass
class Document:
    """
    A wrapper of graph store query results.
    """

    doctype: RAG.DocumentType
    metadata: RAG.Metadata
    path_or_url: Optional[str] = ""

    @property
    def content(self):
        return self.content

    @content.setter
    def _(self, value):
        self.content = value
