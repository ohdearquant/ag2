# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import sys
from abc import ABC
from collections.abc import Iterable
from functools import wraps
from typing import Any, Callable, get_type_hints

from fast_depends import Depends as FastDepends
from fast_depends import inject
from fast_depends.dependencies import model

__all__ = [
    "BaseContext",
    "ChatContext",
    "Depends",
    "Field",
    "inject_params",
]


class BaseContext(ABC):
    """Base class for context classes.

    This is the base class for defining various context types that may be used
    throughout the application. It serves as a parent for specific context classes.
    """

    pass


class ChatContext(BaseContext):
    """ChatContext class that extends BaseContext.

    This class is used to represent a chat context that holds a list of messages.
    It inherits from `BaseContext` and adds the `messages` attribute.
    """

    messages: list[str] = []


def Depends(x: Any) -> Any:  # noqa: N802
    """Creates a dependency for injection based on the provided context or type.

    Args:
        x: The context or dependency to be injected.

    Returns:
        A FastDepends object that will resolve the dependency for injection.
    """
    if isinstance(x, BaseContext):
        return FastDepends(lambda: x)

    return FastDepends(x)


def _is_base_context_param(param: inspect.Parameter) -> bool:
    # param.annotation.__args__[0] is used to handle Annotated[MyContext, Depends(MyContext(b=2))]
    param_annotation = param.annotation.__args__[0] if hasattr(param.annotation, "__args__") else param.annotation
    return isinstance(param_annotation, type) and issubclass(param_annotation, BaseContext)


def _is_depends_param(param: inspect.Parameter) -> bool:
    return isinstance(param.default, model.Depends) or (
        hasattr(param.annotation, "__metadata__")
        and type(param.annotation.__metadata__) == tuple
        and isinstance(param.annotation.__metadata__[0], model.Depends)
    )


def _remove_params(func: Callable[..., Any], sig: inspect.Signature, params: Iterable[str]) -> None:
    new_signature = sig.replace(parameters=[p for p in sig.parameters.values() if p.name not in params])
    func.__signature__ = new_signature  # type: ignore[attr-defined]


def _remove_injected_params_from_signature(func: Callable[..., Any]) -> Callable[..., Any]:
    # This is a workaround for Python 3.9+ where staticmethod.__func__ is accessible
    if sys.version_info >= (3, 9) and isinstance(func, staticmethod) and hasattr(func, "__func__"):
        func = _fix_staticmethod(func)

    sig = inspect.signature(func)
    params_to_remove = [p.name for p in sig.parameters.values() if _is_base_context_param(p) or _is_depends_param(p)]
    _remove_params(func, sig, params_to_remove)
    return func


class Field:
    """Represents a description field for use in type annotations.

    This class is used to store a description for an annotated field, often used for
    documenting or validating fields in a context or data model.
    """

    def __init__(self, description: str) -> None:
        """Initializes the Field with a description.

        Args:
            description: The description text for the field.
        """
        self._description = description

    @property
    def description(self) -> str:
        return self._description


def _string_metadata_to_description_field(func: Callable[..., Any]) -> Callable[..., Any]:
    type_hints = get_type_hints(func, include_extras=True)

    for _, annotation in type_hints.items():
        if hasattr(annotation, "__metadata__"):
            metadata = annotation.__metadata__
            if metadata and isinstance(metadata[0], str):
                # Replace string metadata with DescriptionField
                annotation.__metadata__ = (Field(description=metadata[0]),)
    return func


def _fix_staticmethod(f: Callable[..., Any]) -> Callable[..., Any]:
    # This is a workaround for Python 3.9+ where staticmethod.__func__ is accessible
    if sys.version_info >= (3, 9) and isinstance(f, staticmethod) and hasattr(f, "__func__"):

        @wraps(f.__func__)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return f.__func__(*args, **kwargs)  # type: ignore[attr-defined]

        wrapper.__name__ = f.__func__.__name__

        f = wrapper
    return f


def inject_params(f: Callable[..., Any]) -> Callable[..., Any]:
    """Injects parameters into a function, removing injected dependencies from its signature.

    This function is used to modify a function by injecting dependencies and removing
    injected parameters from the function's signature.

    Args:
        f: The function to modify with dependency injection.

    Returns:
        The modified function with injected dependencies and updated signature.
    """
    # This is a workaround for Python 3.9+ where staticmethod.__func__ is accessible
    if sys.version_info >= (3, 9) and isinstance(f, staticmethod) and hasattr(f, "__func__"):
        f = _fix_staticmethod(f)

    f = _string_metadata_to_description_field(f)
    f = inject(f)
    f = _remove_injected_params_from_signature(f)

    return f
