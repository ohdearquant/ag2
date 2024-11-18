# by HaiyangLi,
# most of the codes are from https://github.com/lion-agi/lion-os
# APACHE LICENSE 2.0, copyright 2024, HaiyangLi

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from inspect import isclass
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel
from typing_extensions import Literal

T = TypeVar("T", bound=BaseModel)


# string_similarity
# copied from https://github.com/lion-agi/lion-os/blob/main/lion/libs/string_similarity.py
# copyright by HaiyangLi, APACHE LICENSE 2.0
def cosine_similarity(s1: str, s2: str) -> float:
    """Calculate the cosine similarity between two strings.

    Args:
        s1: First input string
        s2: Second input string

    Returns:
        float: Cosine similarity score between 0 and 1
    """
    if not s1 or not s2:
        return 0.0

    set1, set2 = set(s1), set(s2)
    intersection = set1.intersection(set2)

    if not set1 or not set2:
        return 0.0

    return len(intersection) / ((len(set1) * len(set2)) ** 0.5)


def hamming_similarity(s1: str, s2: str) -> float:
    """Calculate the Hamming similarity between two strings.

    The strings must be of equal length. Returns the proportion of positions
    at which corresponding symbols are the same.

    Args:
        s1: First input string
        s2: Second input string

    Returns:
        float: Hamming similarity score between 0 and 1
    """
    if not s1 or not s2 or len(s1) != len(s2):
        return 0.0

    matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
    return matches / len(s1)


def jaro_distance(s: str, t: str) -> float:
    """Calculate the Jaro distance between two strings.

    Args:
        s: First input string
        t: Second input string

    Returns:
        float: Jaro distance score between 0 and 1
    """
    s_len = len(s)
    t_len = len(t)

    if s_len == 0 and t_len == 0:
        return 1.0
    elif s_len == 0 or t_len == 0:
        return 0.0

    match_distance = (max(s_len, t_len) // 2) - 1
    match_distance = max(0, match_distance)  # Ensure non-negative

    s_matches = [False] * s_len
    t_matches = [False] * t_len

    matches = 0
    transpositions = 0

    # Identify matches
    for i in range(s_len):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, t_len)

        for j in range(start, end):
            if t_matches[j] or s[i] != t[j]:
                continue
            s_matches[i] = t_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(s_len):
        if not s_matches[i]:
            continue
        while not t_matches[k]:
            k += 1
        if s[i] != t[k]:
            transpositions += 1
        k += 1

    transpositions //= 2

    return (matches / s_len + matches / t_len + (matches - transpositions) / matches) / 3.0


def jaro_winkler_similarity(s: str, t: str, scaling: float = 0.1) -> float:
    """Calculate the Jaro-Winkler similarity between two strings.

    Args:
        s: First input string
        t: Second input string
        scaling: Scaling factor for common prefix adjustment

    Returns:
        float: Jaro-Winkler similarity score between 0 and 1

    Raises:
        ValueError: If scaling factor is not between 0 and 0.25
    """
    if not 0 <= scaling <= 0.25:
        raise ValueError("Scaling factor must be between 0 and 0.25")

    jaro_sim = jaro_distance(s, t)

    # Find length of common prefix (up to 4 chars)
    prefix_len = 0
    for s_char, t_char in zip(s, t):
        if s_char != t_char:
            break
        prefix_len += 1
        if prefix_len == 4:
            break

    return jaro_sim + (prefix_len * scaling * (1 - jaro_sim))


def levenshtein_distance(a: str, b: str) -> int:
    """Calculate the Levenshtein (edit) distance between two strings.

    Args:
        a: First input string
        b: Second input string

    Returns:
        int: Minimum number of single-character edits needed to change one
             string into the other
    """
    if not a:
        return len(b)
    if not b:
        return len(a)

    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    for i, j in product(range(1, m + 1), range(1, n + 1)):
        cost = 0 if a[i - 1] == b[j - 1] else 1
        d[i][j] = min(
            d[i - 1][j] + 1,  # deletion
            d[i][j - 1] + 1,  # insertion
            d[i - 1][j - 1] + cost,  # substitution
        )

    return d[m][n]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate the Levenshtein similarity between two strings.

    Converts Levenshtein distance to a similarity score between 0 and 1.

    Args:
        s1: First input string
        s2: Second input string

    Returns:
        float: Levenshtein similarity score between 0 and 1
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1 - (distance / max_len)


# Type definitions
SIMILARITY_ALGO_MAP: Dict[str, Callable[[str, str], float]] = {
    "jaro_winkler": jaro_winkler_similarity,
    "levenshtein": levenshtein_similarity,
    "sequence_matcher": lambda s1, s2: SequenceMatcher(None, s1, s2).ratio(),
    "hamming": hamming_similarity,
    "cosine": cosine_similarity,
}


SIMILARITY_TYPE = Literal[
    "jaro_winkler",
    "levenshtein",
    "sequence_matcher",
    "hamming",
    "cosine",
]


@dataclass(frozen=True)
class MatchResult:
    """Represents a string matching result."""

    word: str
    score: float
    index: int


def string_similarity(
    word: str,
    correct_words: Sequence[str],
    algorithm: SIMILARITY_TYPE | Callable[[str, str], float] = "jaro_winkler",
    threshold: float = 0.0,
    case_sensitive: bool = False,
    return_most_similar: bool = False,
) -> Optional[Union[str, List[str]]]:
    """Find similar strings using specified similarity algorithm."""
    if not correct_words:
        raise ValueError("correct_words must not be empty")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    # Convert inputs to strings
    compare_word = str(word)
    original_words = [str(w) for w in correct_words]

    # Handle case sensitivity
    if not case_sensitive:
        compare_word = compare_word.lower()
        compare_words = [w.lower() for w in original_words]
    else:
        compare_words = original_words.copy()

    # Get scoring function
    if isinstance(algorithm, str):
        score_func = SIMILARITY_ALGO_MAP.get(algorithm)
        if score_func is None:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    elif callable(algorithm):
        score_func = algorithm
    else:
        raise ValueError("algorithm must be a string specifying a built-in algorithm or " "a callable")

    # Calculate similarities
    results = []
    for idx, (orig_word, comp_word) in enumerate(zip(original_words, compare_words)):
        # Skip different length strings for hamming similarity
        if algorithm == "hamming" and len(comp_word) != len(compare_word):
            continue

        score = score_func(compare_word, comp_word)
        if score >= threshold:
            results.append(MatchResult(orig_word, score, idx))

    # Return None if no matches
    if not results:
        return None

    # Sort by score (descending) and index (ascending) for stable ordering
    results.sort(key=lambda x: (-x.score, x.index))

    # Return results
    if return_most_similar:
        return results[0].word

    # Filter exact matches for case sensitive comparisons
    if case_sensitive:
        max_score = results[0].score
        results = [r for r in results if r.score == max_score]

    return [r.word for r in results]


# copied from https://github.com/lion-agi/lion-os/blob/main/lion/integrations/pydantic_/break_down_annotation.py
# copyright by HaiyangLi, APACHE LICENSE 2.0
def break_down_pydantic_annotation(
    model: type[T], max_depth: int | None = None, current_depth: int = 0
) -> Dict[str, Any]:
    """
    Break down the type annotations of a Pydantic model into a dictionary.

    This function recursively processes Pydantic models, converting their
    field annotations into a dictionary structure. It handles nested models
    and lists of models.

    Args:
        model: The Pydantic model class to break down.
        max_depth: Maximum depth for recursion. None for no limit.
        current_depth: Current recursion depth (used internally).

    Returns:
        A dictionary representing the structure of the model's annotations.

    Raises:
        TypeError: If the input is not a Pydantic model.
        RecursionError: If max recursion depth is reached.

    Example:
        >>> from pydantic import BaseModel
        >>> class SubModel(BaseModel):
        ...     field1: int
        ...     field2: str
        >>> class MainModel(BaseModel):
        ...     sub: SubModel
        ...     items: list[SubModel]
        >>> result = break_down_annotation(MainModel)
        >>> print(result)
        {
            'sub': {'field1': <class 'int'>, 'field2': <class 'str'>},
            'items': [{'field1': <class 'int'>, 'field2': <class 'str'>}]
        }
    """

    if not _is_pydantic_model(model):
        raise TypeError("Input must be a Pydantic model")

    if max_depth is not None and current_depth >= max_depth:
        raise RecursionError("Maximum recursion depth reached")

    out: Dict[str, Any] = {}
    for k, v in model.__annotations__.items():
        origin = get_origin(v)
        if _is_pydantic_model(v):
            out[k] = break_down_pydantic_annotation(v, max_depth, current_depth + 1)
        elif origin is list:
            args = get_args(v)
            if args and _is_pydantic_model(args[0]):
                out[k] = [break_down_pydantic_annotation(args[0], max_depth, current_depth + 1)]
            else:
                out[k] = [args[0] if args else Any]
        else:
            out[k] = v

    return out


def _is_pydantic_model(x: Any) -> bool:
    return isclass(x) and issubclass(x, BaseModel)


# copied from https://github.com/lion-agi/lion-os/blob/main/lion/libs/parse.py
# copyright by HaiyangLi, APACHE LICENSE 2.0
def to_json(string: str | List[str], /, fuzzy_parse: bool = False) -> Union[List[Dict[str, Any]], Dict]:
    """Extract and parse JSON content from a string or markdown code blocks.

    This function attempts to parse JSON directly from the input string first.
    If that fails, it looks for JSON content within markdown code blocks
    (denoted by ```json).

    Args:
        string: Input string or list of strings to parse. If a list is provided,
               it will be joined with newlines.

    Returns:
        - A dictionary if a single JSON object is found
        - A list of dictionaries if multiple JSON objects are found
        - An empty list if no valid JSON is found

    Examples:
        >>> to_json('{"key": "value"}')
        {'key': 'value'}

        >>> to_json('''
        ... ```json
        ... {"key": "value"}
        ... ```
        ... ''')
        {'key': 'value'}

        >>> to_json('''
        ... ```json
        ... {"key1": "value1"}
        ... ```
        ... ```json
        ... {"key2": "value2"}
        ... ```
        ... ''')
        [{'key1': 'value1'}, {'key2': 'value2'}]
    """

    if isinstance(string, list):
        string = "\n".join(string)

    # Try direct JSON parsing first
    try:
        if fuzzy_parse:
            return fuzzy_parse_json(string)
        return json.loads(string)
    except Exception:
        pass

    # Look for JSON in markdown code blocks
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, string, re.DOTALL)

    if not matches:
        return []

    if len(matches) == 1:
        return json.loads(matches[0])

    if fuzzy_parse:
        return [fuzzy_parse_json(match) for match in matches]
    return [json.loads(match) for match in matches]


def fuzzy_parse_json(str_to_parse: str, /) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse a JSON string with automatic fixing of common formatting issues.

    Args:
        str_to_parse: The JSON string to parse

    Returns:
        The parsed JSON object as a dictionary

    Raises:
        ValueError: If the string cannot be parsed as valid JSON
        TypeError: If the input is not a string or the result is not a dict
    """
    if not isinstance(str_to_parse, str):
        raise TypeError("Input must be a string")

    if not str_to_parse.strip():
        raise ValueError("Input string is empty")

    try:
        return json.loads(str_to_parse)
    except Exception:
        pass

    cleaned = _clean_json_string(str_to_parse)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    try:
        fixed = fix_json_string(cleaned)
        return json.loads(fixed)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON string after all fixing attempts: {e}") from e


def _clean_json_string(s: str) -> str:
    """Clean and standardize a JSON string."""
    s = re.sub(r"(?<!\\)'", '"', s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r'([{,])\s*([^"\s]+):', r'\1"\2":', s)
    return s.strip()


def fix_json_string(str_to_parse: str, /) -> str:
    """Fix a JSON string by ensuring all brackets are properly closed.

    Args:
        str_to_parse: JSON string to fix

    Returns:
        Fixed JSON string with proper bracket closure

    Raises:
        ValueError: If mismatched or extra closing brackets are found
    """
    if not str_to_parse:
        raise ValueError("Input string is empty")

    brackets = {"{": "}", "[": "]"}
    open_brackets = []
    pos = 0
    length = len(str_to_parse)

    while pos < length:
        char = str_to_parse[pos]

        # Handle escape sequences
        if char == "\\":
            pos += 2  # Skip escape sequence
            continue

        # Handle string content
        if char == '"':
            pos += 1
            # Skip until closing quote, accounting for escapes
            while pos < length:
                if str_to_parse[pos] == "\\":
                    pos += 2  # Skip escape sequence
                    continue
                if str_to_parse[pos] == '"':
                    break
                pos += 1
            pos += 1
            continue

        # Handle brackets
        if char in brackets:
            open_brackets.append(brackets[char])
        elif char in brackets.values():
            if not open_brackets:
                raise ValueError(f"Extra closing bracket '{char}' at position {pos}")
            if open_brackets[-1] != char:
                raise ValueError(f"Mismatched bracket '{char}' at position {pos}")
            open_brackets.pop()

        pos += 1

    # Add missing closing brackets
    closing_brackets = "".join(reversed(open_brackets))
    return str_to_parse + closing_brackets


# TODO: add a recursive to dict into AG2
# check `recursive_to_dict` under https://github.com/lion-agi/lion-os/blob/main/lion/libs/parse.py

# TODO: add fuzzy matching key, fuzzy matching mapping, need to modify the implementation of to_dict from LION-OS
