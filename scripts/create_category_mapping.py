import os
import sys
import pandas as pd
import json
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create a category mapping file from processed data')

    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--output_file', type=str, default='data/processed/category_mapping.json',
                        help='Output file for category mapping')

    return parser.parse_args()


def main():
    """Main function to create category mapping."""
    args = parse_arguments()

    # Check for mapping file first
    if os.path.exists(os.path.join(args.data_dir, 'category_mappings.csv')):
        # Load from category_mappings.csv
        print(f"Loading category mappings from {os.path.join(args.data_dir, 'category_mappings.csv')}")
        mappings_df = pd.read_csv(os.path.join(args.data_dir, 'category_mappings.csv'))

        # Create mapping dictionary
        mapping = {}
        for _, row in mappings_df.iterrows():
            mapping[str(row['id'])] = row['category']

        print(f"Found {len(mapping)} category mappings")
    else:
        # Try to load from training data
        train_file = os.path.join(args.data_dir, 'train.csv')
        if not os.path.exists(train_file):
            print(f"Error: Training file not found at {train_file}")
            sys.exit(1)

        print(f"Loading category data from {train_file}")
        train_df = pd.read_csv(train_file)

        # Check if we have the right columns
        if 'category_id' not in train_df.columns or 'categories' not in train_df.columns:
            print("Error: Required columns not found in training data (need 'category_id' and 'categories')")
            print(f"Available columns: {train_df.columns.tolist()}")
            sys.exit(1)

        # Create mapping
        mapping = {}
        categories = train_df[['category_id', 'categories']].drop_duplicates()
        for _, row in categories.iterrows():
            mapping[str(int(row['category_id']))] = row['categories']

        print(f"Created {len(mapping)} category mappings from training data")

    # Save mapping to file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"Category mapping saved to {args.output_file}")
    print("Sample mappings:")
    for i, (key, value) in enumerate(list(mapping.items())[:5]):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()