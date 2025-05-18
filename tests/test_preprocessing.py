# tests/test_preprocessing.py

import sys
import os
import unittest
import pandas as pd
from unittest.mock import patch, Mock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import TextPreprocessor, ProductDataProcessor, preprocess_text


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for the TextPreprocessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = TextPreprocessor(
            lowercase=True,
            remove_punctuation=True,
            remove_special_chars=True,
            remove_stopwords=True,
            lemmatize=True
        )

        # Sample texts
        self.sample_texts = [
            "Apple iPhone 14 Pro Max, 256GB, Deep Purple",
            "Samsung Galaxy S23 Ultra, 512GB, Phantom Black",
            "Logitech MX Master 3 Wireless Mouse",
            "Nike Men's Air Force 1 '07 Sneaker"
        ]

    def test_lowercase_conversion(self):
        """Test lowercase conversion."""
        text = "Apple iPhone 14 PRO Max"
        processed = self.processor.preprocess(text)

        # Check if all characters are lowercase
        self.assertEqual(processed, processed.lower())

    def test_punctuation_removal(self):
        """Test punctuation removal."""
        text = "Apple iPhone, 14 Pro Max (256GB) - Deep Purple!"
        processed = self.processor.preprocess(text)

        # Check if punctuation is removed
        import string
        for char in string.punctuation:
            self.assertNotIn(char, processed)

    def test_special_chars_removal(self):
        """Test special characters removal."""
        text = "Samsung™ Galaxy® S23 Ultra 😊"
        processed = self.processor.preprocess(text)

        # Check if special characters are removed
        self.assertNotIn("™", processed)
        self.assertNotIn("®", processed)
        self.assertNotIn("😊", processed)

    def test_stopwords_removal(self):
        """Test stopwords removal."""
        text = "This is the new iPhone from Apple and it has a great camera"
        processed = self.processor.preprocess(text)

        # Check if common stopwords are removed
        self.assertNotIn(" is ", processed)
        self.assertNotIn(" the ", processed)
        self.assertNotIn(" and ", processed)
        self.assertNotIn(" it ", processed)
        self.assertNotIn(" a ", processed)

    def test_lemmatization(self):
        """Test lemmatization."""
        # This is a basic test, since we're mocking NLTK
        text = "running shoes for women"
        with patch('src.data.preprocessing.WordNetLemmatizer') as mock_lemmatizer:
            mock_instance = Mock()
            mock_instance.lemmatize.side_effect = lambda x: x.replace("running", "run")
            mock_lemmatizer.return_value = mock_instance

            processor = TextPreprocessor(lemmatize=True)
            processed = processor.preprocess(text)

            # Check if lemmatization was applied
            mock_instance.lemmatize.assert_called()

    def test_preprocess_df(self):
        """Test DataFrame preprocessing."""
        # Create sample DataFrame
        df = pd.DataFrame({
            'name': self.sample_texts,
            'brand': ['Apple', 'Samsung', 'Logitech', 'Nike']
        })

        # Process DataFrame
        df_processed = self.processor.preprocess_df(
            df,
            text_column='name',
            brand_column='brand',
            combine_columns=True
        )

        # Check if new columns were created
        self.assertIn('processed_name', df_processed.columns)
        self.assertIn('processed_brand', df_processed.columns)
        self.assertIn('combined_text', df_processed.columns)

        # Check if combined text contains both name and brand
        for idx, row in df_processed.iterrows():
            self.assertIn(row['processed_brand'], row['combined_text'])

    def test_empty_input(self):
        """Test handling of empty input."""
        text = ""
        processed = self.processor.preprocess(text)

        # Check if empty input is handled gracefully
        self.assertEqual(processed, "")

    def test_non_string_input(self):
        """Test handling of non-string input."""
        text = 12345
        processed = self.processor.preprocess(text)

        # Check if non-string input is handled gracefully
        self.assertIsInstance(processed, str)


class TestProductDataProcessor(unittest.TestCase):
    """Test cases for the ProductDataProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.text_preprocessor = TextPreprocessor()
        self.processor = ProductDataProcessor(
            text_preprocessor=self.text_preprocessor,
            test_size=0.2,
            val_size=0.2,
            random_state=42
        )

        # Sample data
        self.sample_df = pd.DataFrame({
            'name': [
                "Apple iPhone 14 Pro Max, 256GB",
                "Samsung Galaxy S23 Ultra, 512GB",
                "Logitech MX Master 3 Wireless Mouse",
                "Nike Men's Air Force 1 '07 Sneaker",
                "ASUS ROG Strix G15 Gaming Laptop"
            ],
            'brand': [
                "Apple", "Samsung", "Logitech", "Nike", "ASUS"
            ],
            'categories': [
                "Electronics > Smartphones",
                "Electronics > Smartphones",
                "Electronics > Computer Accessories",
                "Fashion > Men > Shoes",
                "Electronics > Laptops"
            ]
        })

    def test_category_encoding(self):
        """Test category encoding."""
        # Process data
        self.processor._encode_categories(self.sample_df['categories'])

        # Check if category mappings were created
        self.assertGreater(len(self.processor.category_to_id), 0)
        self.assertGreater(len(self.processor.id_to_category), 0)
        self.assertEqual(self.processor.num_classes, len(self.sample_df['categories'].unique()))

        # Check if mappings are consistent
        for category, idx in self.processor.category_to_id.items():
            self.assertEqual(self.processor.id_to_category[idx], category)

    def test_process_data_splitting(self):
        """Test data processing and splitting."""
        # Process data
        train_df, val_df, test_df = self.processor.process(
            self.sample_df,
            name_column='name',
            brand_column='brand',
            category_column='categories'
        )

        # Check if data was split correctly
        total_rows = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_rows, len(self.sample_df))

        # Check for category consistency
        self.assertIn('category_id', train_df.columns)
        self.assertIn('category_id', val_df.columns)
        self.assertIn('category_id', test_df.columns)

    def test_get_category_mappings(self):
        """Test retrieval of category mappings."""
        # Process data
        self.processor._encode_categories(self.sample_df['categories'])

        # Get mappings
        category_to_id, id_to_category = self.processor.get_category_mappings()

        # Check if mappings match internal state
        self.assertEqual(category_to_id, self.processor.category_to_id)
        self.assertEqual(id_to_category, self.processor.id_to_category)


class TestPreprocessText(unittest.TestCase):
    """Test cases for the preprocess_text function."""

    def test_basic_preprocessing(self):
        """Test basic preprocessing functionality."""
        text = "Apple iPhone 14 Pro Max, 256GB, Deep Purple!"
        processed = preprocess_text(text)

        # Check if basic preprocessing was applied
        self.assertEqual(processed, processed.lower())
        self.assertNotIn(",", processed)
        self.assertNotIn("!", processed)

    def test_non_string_input(self):
        """Test handling of non-string input."""
        text = 12345
        processed = preprocess_text(text)

        # Check if non-string input is handled gracefully
        self.assertIsInstance(processed, str)

    def test_empty_input(self):
        """Test handling of empty input."""
        text = ""
        processed = preprocess_text(text)

        # Check if empty input is handled gracefully
        self.assertEqual(processed, "")


if __name__ == '__main__':
    unittest.main()
