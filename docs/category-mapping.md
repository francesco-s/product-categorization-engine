# Category Mapping in the Product Categorization API

This document explains how category mapping works in the product categorization system and provides guidance on creating and managing category mappings.

## Understanding Category Mapping

### What is Category Mapping?

Category mapping is the translation between:
- **Numeric category IDs** (used internally by the model)
- **Human-readable category names** (displayed to users)

Example mapping:
```json
{
  "0": "Electronics > Smartphones & Accessories > Smartphones",
  "1": "Electronics > Computers & Accessories > Laptops",
  "2": "Fashion > Men > Shoes > Athletic",
  "3": "Home & Kitchen > Kitchen Appliances"
}
```

### Why Is Category Mapping Important?

Without proper category mapping, the API will return generic labels like "Category 0" instead of meaningful names like "Electronics > Smartphones".

## How Category Mapping Works

The system uses a three-tiered approach to category mapping:

### 1. Primary Method: Model Checkpoint

During training, the category mapping is saved in the model checkpoint:

```python
# In training script
checkpoint['id_to_category'] = id_to_category
torch.save(checkpoint, model_path)
```

When the API loads the model, it retrieves this mapping:

```python
# In API
id_to_category = checkpoint.get('id_to_category', {})
```

### 2. Fallback Method: External Mapping File

If the model checkpoint doesn't contain mappings, the API looks for an external JSON file:

```python
# Location: data/processed/category_mapping.json
external_mapping_path = "data/processed/category_mapping.json"
if not id_to_category and os.path.exists(external_mapping_path):
    with open(external_mapping_path, 'r') as f:
        id_to_category = json.load(f)
```

### 3. Last Resort: Default Names

If no mapping is available, the API generates generic category names:

```python
if not id_to_category and num_classes > 0:
    id_to_category = {str(i): f"Category {i}" for i in range(num_classes)}
```

## Creating an External Category Mapping

If your model doesn't have proper category names, you can create an external mapping file:

### Method 1: Using the Provided Script

1. Run the mapping creation script:
   ```bash
   python scripts/create_category_mapping.py
   ```

2. This script will:
   - First look for `category_mappings.csv` in the processed data directory
   - If not found, extract mappings from the training data in `train.csv`
   - Save the mapping to `data/processed/category_mapping.json`

### Method 2: Manual Creation

You can manually create the mapping file:

1. Create a JSON file at `data/processed/category_mapping.json`
2. Define the mappings in the following format:
   ```json
   {
     "0": "Electronics > Smartphones",
     "1": "Electronics > Laptops",
     "2": "Fashion > Shoes",
     "3": "Home & Kitchen"
   }
   ```
3. Save the file

## Troubleshooting Category Mappings

If you're still seeing generic labels like "Category 0" in API responses:

### Check the API Logs

The API logs detailed information about category mappings:

```
INFO:__main__:Loaded category mapping from checkpoint: {'0': 'Electronics', '1': 'Fashion'}
INFO:__main__:Sample category mappings:
INFO:__main__:  0: Electronics
INFO:__main__:  1: Fashion
```

### Verify External Mapping File

Ensure your mapping file exists and is properly formatted:

```bash
# Check if file exists
ls -la data/processed/category_mapping.json

# Examine the file contents
cat data/processed/category_mapping.json
```

### Regenerate the Mapping

If the mapping file exists but contains errors, regenerate it:

```bash
# Force regeneration from training data
python scripts/create_category_mapping.py --data_dir data/processed --output_file data/processed/category_mapping.json
```

## Advanced: Customizing Category Names

You can modify the category names in the mapping file to make them more user-friendly:

1. Edit the `category_mapping.json` file
2. Change the category names as needed
3. Save the file
4. Restart the API

Example customization:
```json
# Original
{
  "0": "Electronics > Smartphones & Accessories > Smartphones",
  "1": "Electronics > Computers & Accessories > Laptops"
}

# Customized (simpler names)
{
  "0": "Smartphones",
  "1": "Laptops"
}
```
