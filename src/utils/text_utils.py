# src/utils/text_utils.py
import re
import string
import unicodedata
from typing import List, Optional


def preprocess_text(text: str) -> str:
    """
    Preprocess text for inference.

    Args:
        text: Input text

    Returns:
        Preprocessed text
    """
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Input text
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length] + '...'


def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenization function.

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Split on whitespace
    tokens = text.split()

    return tokens


def extract_features(text: str, brand: Optional[str] = None) -> str:
    """
    Extract features from product name and brand.

    Args:
        text: Product name
        brand: Product brand (optional)

    Returns:
        Combined features
    """
    # Preprocess name
    processed_text = preprocess_text(text)

    # Process brand if provided
    if brand:
        processed_brand = preprocess_text(brand)
        # Combine name and brand
        return f"{processed_text} {processed_brand}"

    return processed_text


def extract_categories(category_hierarchy: str) -> List[str]:
    """
    Extract categories from a hierarchy.

    Args:
        category_hierarchy: Category hierarchy (e.g., "Electronics > Smartphones > Accessories")

    Returns:
        List of categories
    """
    # Split by separator
    categories = category_hierarchy.split(' > ')

    return categories


def get_main_category(category_hierarchy: str) -> str:
    """
    Get the main (top-level) category from a hierarchy.

    Args:
        category_hierarchy: Category hierarchy (e.g., "Electronics > Smartphones > Accessories")

    Returns:
        Main category
    """
    categories = extract_categories(category_hierarchy)
    return categories[0] if categories else ""


def get_subcategory(category_hierarchy: str, level: int = 1) -> Optional[str]:
    """
    Get a subcategory at a specific level from a hierarchy.

    Args:
        category_hierarchy: Category hierarchy (e.g., "Electronics > Smartphones > Accessories")
        level: Level of the subcategory (0-based index)

    Returns:
        Subcategory or None if not available
    """
    categories = extract_categories(category_hierarchy)
    if level < len(categories):
        return categories[level]
    return None
