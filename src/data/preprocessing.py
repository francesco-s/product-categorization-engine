# src/data/preprocessing.py
import re
import string
import unicodedata
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk

# Initialize logger
logger = logging.getLogger(__name__)


# Download NLTK resources function
def download_nltk_resources():
    """Download necessary NLTK resources."""
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            logger.warning(f"Could not download NLTK resource {resource}: {e}")


# Call download function
download_nltk_resources()

# Import NLTK modules after downloading resources
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError as e:
    logger.error(f"Error importing NLTK modules: {e}")
    # Fall back to basic tokenization if NLTK import fails
    stopwords = None
    WordNetLemmatizer = None


    def word_tokenize(text):
        return text.split()


class TextPreprocessor:
    """
    Text preprocessing class for product categorization.

    Implements the Strategy Pattern, allowing different preprocessing strategies.
    """

    def __init__(
            self,
            lowercase: bool = True,
            remove_punctuation: bool = True,
            remove_special_chars: bool = True,
            remove_stopwords: bool = True,
            lemmatize: bool = True,
            language: str = "english"
    ):
        """
        Initialize text preprocessor.

        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_special_chars: Whether to remove special characters
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            language: Language for stopwords
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_special_chars = remove_special_chars
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.language = language

        # Initialize components if needed
        if remove_stopwords and stopwords is not None:
            try:
                self.stop_words = set(stopwords.words(language))
            except Exception as e:
                logger.warning(f"Could not load stopwords for {language}: {e}")
                self.stop_words = set()
        else:
            self.stop_words = set()

        if lemmatize and WordNetLemmatizer is not None:
            try:
                self.lemmatizer = WordNetLemmatizer()
            except Exception as e:
                logger.warning(f"Could not initialize lemmatizer: {e}")
                self.lemmatize = False

    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text string.

        Args:
            text: Text to preprocess

        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove special characters
        if self.remove_special_chars:
            # Normalize unicode characters
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            # Remove non-alphanumeric characters (except spaces)
            text = re.sub(r'[^\w\s]', '', text)

        # Tokenize with fallback if NLTK fails
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            logger.warning(f"Error in word_tokenize: {e}. Using fallback tokenization.")
            tokens = text.split()

        # Remove stopwords
        if self.remove_stopwords and self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Lemmatize
        if self.lemmatize and hasattr(self, 'lemmatizer'):
            try:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            except Exception as e:
                logger.warning(f"Error in lemmatization: {e}. Skipping.")

        # Join tokens back into a string
        text = ' '.join(tokens)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def preprocess_df(
            self,
            df: pd.DataFrame,
            text_column: str,
            brand_column: Optional[str] = None,
            combine_columns: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess text column(s) in a dataframe.

        Args:
            df: Input dataframe
            text_column: Name of the text column
            brand_column: Name of the brand column (optional)
            combine_columns: Whether to combine text and brand columns

        Returns:
            Dataframe with preprocessed text
        """
        # Create a copy to avoid modifying the original
        df_processed = df.copy()

        # Preprocess the text column
        logger.info(f"Preprocessing {text_column} column")
        df_processed[f'processed_{text_column}'] = df_processed[text_column].apply(self.preprocess)

        # Preprocess the brand column if provided
        if brand_column and brand_column in df.columns:
            logger.info(f"Preprocessing {brand_column} column")
            df_processed[f'processed_{brand_column}'] = df_processed[brand_column].apply(self.preprocess)

            # Combine columns if requested
            if combine_columns:
                logger.info(f"Combining {text_column} and {brand_column} columns")
                df_processed['combined_text'] = (
                        df_processed[f'processed_{text_column}'] + ' ' +
                        df_processed[f'processed_{brand_column}']
                )
        else:
            if brand_column:
                logger.warning(f"Brand column '{brand_column}' not found in dataframe")

            # If no brand column or combining not requested, just use processed text
            if not combine_columns:
                logger.info(f"Using only {text_column} column")
                df_processed['combined_text'] = df_processed[f'processed_{text_column}']

        return df_processed


class ProductDataProcessor:
    """
    Data processor for the product categorization task.
    """

    def __init__(
            self,
            text_preprocessor: Optional[TextPreprocessor] = None,
            test_size: float = 0.15,
            val_size: float = 0.15,
            random_state: int = 42
    ):
        """
        Initialize data processor.

        Args:
            text_preprocessor: Text preprocessor instance
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            random_state: Random seed for reproducibility
        """
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.category_to_id = {}
        self.id_to_category = {}
        self.num_classes = 0

    def process(
            self,
            df: pd.DataFrame,
            name_column: str = 'name',
            brand_column: str = 'brand',
            category_column: str = 'categories',
            save_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process the product dataframe.

        Args:
            df: Input dataframe
            name_column: Name of the product name column
            brand_column: Name of the brand column
            category_column: Name of the category column
            save_path: Path to save processed data (optional)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Processing dataset with {len(df)} rows")

        # Validate required columns
        required_columns = [name_column]
        if brand_column:
            required_columns.append(brand_column)
        if category_column:
            required_columns.append(category_column)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for missing values in required columns
        missing_values = df[required_columns].isnull().sum()
        logger.info(f"Missing values:\n{missing_values}")

        # Drop rows with missing values in required columns
        df_clean = df.dropna(subset=required_columns)
        logger.info(f"Dataset after dropping missing values: {len(df_clean)} rows")

        # Preprocess text columns
        if brand_column:
            df_processed = self.text_preprocessor.preprocess_df(
                df_clean,
                text_column=name_column,
                brand_column=brand_column,
                combine_columns=True
            )
        else:
            # Only process name column if brand is not available
            df_processed = self.text_preprocessor.preprocess_df(
                df_clean,
                text_column=name_column,
                brand_column=None,
                combine_columns=False
            )
            # Set combined text to processed name
            df_processed['combined_text'] = df_processed[f'processed_{name_column}']

        # Encode category labels
        self._encode_categories(df_processed[category_column])

        # Add encoded categories to dataframe
        df_processed['category_id'] = df_processed[category_column].map(self.category_to_id)

        # Split data into train, validation, and test sets
        train_df, test_df = train_test_split(
            df_processed,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df_processed['category_id']
        )

        train_df, val_df = train_test_split(
            train_df,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state,
            stratify=train_df['category_id']
        )

        logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # Save processed data if path is provided
        if save_path:
            logger.info(f"Saving processed data to {save_path}")
            train_df.to_csv(f"{save_path}/train.csv", index=False)
            val_df.to_csv(f"{save_path}/val.csv", index=False)
            test_df.to_csv(f"{save_path}/test.csv", index=False)

            # Save category mappings
            category_mappings = pd.DataFrame({
                'category': list(self.category_to_id.keys()),
                'id': list(self.category_to_id.values())
            })
            category_mappings.to_csv(f"{save_path}/category_mappings.csv", index=False)

        return train_df, val_df, test_df

    def _encode_categories(self, categories: pd.Series) -> None:
        """
        Encode category labels.

        Args:
            categories: Series of category labels
        """
        unique_categories = categories.unique()
        logger.info(f"Found {len(unique_categories)} unique categories")

        # Create category mappings
        self.category_to_id = {
            category: idx for idx, category in enumerate(unique_categories)
        }
        self.id_to_category = {
            idx: category for category, idx in self.category_to_id.items()
        }
        self.num_classes = len(self.category_to_id)

    def get_category_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Get category mappings.

        Returns:
            Tuple of (category_to_id, id_to_category)
        """
        return self.category_to_id, self.id_to_category

    def get_num_classes(self) -> int:
        """
        Get number of classes.

        Returns:
            Number of unique categories
        """
        return self.num_classes


def preprocess_text(text: str) -> str:
    """
    Simple text preprocessing function for inference.

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
