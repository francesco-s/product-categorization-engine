import nltk
import ssl
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_nltk_data():
    """Download NLTK data packages required for text processing."""

    # Required NLTK data packages
    packages = [
        'punkt',  # For tokenization
        'punkt_tab',
        'stopwords',  # For stopword removal
        'wordnet',  # For lemmatization
        'averaged_perceptron_tagger'  # For POS tagging
    ]

    # Handle SSL certificate issues that sometimes occur with NLTK downloads
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Download each package
    for package in packages:
        try:
            logger.info(f"Downloading NLTK package: {package}")
            nltk.download(package)
            logger.info(f"Successfully downloaded {package}")
        except Exception as e:
            logger.error(f"Error downloading {package}: {e}")


if __name__ == "__main__":
    logger.info("Starting NLTK data download")
    download_nltk_data()
    logger.info("NLTK data download completed")