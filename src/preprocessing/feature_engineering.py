import pandas as pd
import numpy as np
import re
from io import StringIO
import os

# --- Definitive Unit Consolidation Map ---
# This map aggressively groups messy unit strings into clean, standard categories.
UNIT_CONSOLIDATION_MAP = {
    # Weight (Ounces)
    'ounce': 'unit_weight_oz', 'ounces': 'unit_weight_oz', 'oz': 'unit_weight_oz',
    # Weight (Pounds)
    'pound': 'unit_weight_lb', 'pounds': 'unit_weight_lb', 'lb': 'unit_weight_lb',
    # Weight (Grams)
    'gram': 'unit_weight_g', 'grams': 'unit_weight_g', 'gr': 'unit_weight_g', 'g': 'unit_weight_g',
    # Weight (Kilograms)
    'kg': 'unit_weight_kg',
    # Volume (Fluid Ounces)
    'fl oz': 'unit_volume_oz', 'fz': 'unit_volume_oz', 'fl. oz.': 'unit_volume_oz', 'fluid': 'unit_volume_oz',
    # Volume (Milliliters)
    'ml': 'unit_volume_ml', 'milliliter': 'unit_volume_ml', 'millilitre': 'unit_volume_ml',
    # Volume (Liters)
    'liter': 'unit_volume_l', 'liters': 'unit_volume_l', 'ltr': 'unit_volume_l',
    # Count / Each (the most common category for items sold individually or in packs)
    'count': 'unit_count', 'ct': 'unit_count', 'ea': 'unit_count', 'each': 'unit_count',
    'pack': 'unit_count', 'packs': 'unit_count', 'pk': 'unit_count', 'bottle': 'unit_count',
    'can': 'unit_count', 'bag': 'unit_count', 'box': 'unit_count', 'pouch': 'unit_count',
    'jar': 'unit_count', 'piece': 'unit_count', 'units': 'unit_count'
}


def clean_and_consolidate_unit(unit_str: str) -> str:
    """Uses the consolidation map to clean the unit string into a standard category."""
    if not isinstance(unit_str, str):
        return 'unit_unknown'
    # A simple loop to find a match in the keys of the map
    unit_str = unit_str.lower().strip()
    for key, value in UNIT_CONSOLIDATION_MAP.items():
        if key == unit_str:
            return value
    return 'unit_unknown'


def parse_text_content(catalog_content: str) -> tuple:
    """Parses the main text block into its constituent parts."""
    item_name, bullet_points, description, value, unit = "", "", "", np.nan, "unknown"
    if not isinstance(catalog_content, str):
        return item_name, bullet_points, description, value, unit

    name_match = re.search(r"Item Name:\s*(.*?)\n", catalog_content, re.IGNORECASE)
    if name_match: item_name = name_match.group(1).strip()
    bullets = re.findall(r"Bullet Point \d+:\s*(.*?)\n", catalog_content, re.IGNORECASE)
    bullet_points = " ".join([b.strip() for b in bullets])
    desc_match = re.search(r"Product Description:\s*(.*)", catalog_content, re.DOTALL | re.IGNORECASE)
    if desc_match: description = re.sub(r'<br\s*/?>', ' ', desc_match.group(1).strip())
    value_match = re.search(r"Value:\s*([\d\.]+)", catalog_content, re.IGNORECASE)
    if value_match:
        try: value = float(value_match.group(1))
        except ValueError: value = np.nan
    # Use a more general regex for unit to catch more variations initially
    unit_match = re.search(r"Unit:\s*([\w\s\.]+)", catalog_content, re.IGNORECASE)
    if unit_match: unit = unit_match.group(1).strip()
    return item_name, bullet_points, description, value, unit


def extract_pack_multiplier(text: str) -> float:
    """Extracts pack size (e.g., 'Pack of 12' -> 12.0) from the item name."""
    if not isinstance(text, str): return 1.0
    text = text.lower()
    patterns = [r'\(pack of (\d+)\)', r'pack of (\d+)', r'(\d+)\s*count', r'(\d+)\s*ct', r'pk-\s*(\d+)', r'(\d+)\s*pk', r'case of (\d+)', r'(\d+)\s*pack']
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            num = float(match.group(1))
            if num > 0: return num
    return 1.0


def extract_brand(item_name: str) -> str:
    if not isinstance(item_name, str) or not item_name:
        return "unknown"

    KNOWN_MULTI_WORD_BRANDS = [
        "Gift Basket Village", "Bear Creek Country Kitchens", "Harry & David",
        "Joe T Garcia's", "L'Oréal Paris", "Old El Paso", "Seventh Generation",
        "Stretch Island", "Simply Asia", "Crystal Light", "Coca-Cola", "YumEarth"
    ]
    for brand in KNOWN_MULTI_WORD_BRANDS:
        if item_name.startswith(brand):
            return brand

    words = item_name.split()
    if not words:
        return "unknown"

    first_word = words[0]
    
    if len(words) == 1:
        return first_word

    second_word = words[1]
    
    # List of common descriptive words that are unlikely to be part of a brand name.
    COMMON_DESCRIPTIVE_WORDS = [
        "organic", "gourmet", "spiced", "classic", "gluten", "vegan", "sugar-free",
        "hot", "mini", "snacks", "premium", "natural", "light"
    ]
    
    # If the second word is a common descriptor, the brand is likely just the first word.
    if second_word.lower().strip(',') in COMMON_DESCRIPTIVE_WORDS:
        return first_word
    return f"{first_word} {second_word}"


def create_keyword_flags(text: str) -> pd.Series:
    """Creates boolean (0/1) flags for valuable marketing keywords."""
    if not isinstance(text, str): text = ""
    text = text.lower()
    flags = {
        'is_organic': 'organic' in text, 'is_gluten_free': 'gluten free' in text or 'gluten-free' in text,
        'is_vegan': 'vegan' in text, 'is_kosher': 'kosher' in text,
        'is_non_gmo': 'non-gmo' in text or 'no gmo' in text,
        'is_sugar_free': 'sugar-free' in text or 'sugar free' in text or 'zero sugar' in text
    }
    return pd.Series(flags, dtype=int)


def run_feature_engineering(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    This is the definitive feature engineering pipeline.
    It generates all tabular features and preserves columns needed for other models.
    """
    print("--- Starting Definitive Feature Engineering Pipeline ---")

    # === Stage 1: Parsing and Cleaning ===
    print("1. Parsing and cleaning raw text...")
    parsed_cols = df['catalog_content'].apply(lambda x: pd.Series(parse_text_content(x)))
    parsed_cols.columns = ['item_name', 'bullet_points', 'description', 'value', 'raw_unit']
    df = pd.concat([df.drop('catalog_content', axis=1), parsed_cols], axis=1)
    
    # Clean base columns before feature creation
    df['item_name'] = df['item_name'].fillna('missing')
    df['value'] = df['value'].fillna(df['value'].median())
    df['unit'] = df['raw_unit'].apply(clean_and_consolidate_unit)

    # === Stage 2: Feature Creation ===
    print("2. Creating core numerical and text features...")
    # The 'brand' column is kept as-is, ready for target encoding in the modeling phase
    df['brand'] = df['item_name'].apply(extract_brand).fillna('unknown')
    
    df['pack_multiplier'] = df['item_name'].apply(extract_pack_multiplier)
    df['normalized_quantity'] = df['value'] * df['pack_multiplier']
    
    # This text column is for the NLP model (Teammate B)
    df['text_for_embedding'] = df['item_name'] + ' [SEP] ' + df['bullet_points'].fillna('')
    
    # These flags are for the tabular model
    keyword_flags = df['text_for_embedding'].apply(create_keyword_flags)
    df = pd.concat([df, keyword_flags], axis=1)

    # === Stage 3: One-Hot Encoding for CONSOLIDATED Units ===
    print("3. One-hot encoding for consolidated 'unit' feature...")
    df = pd.concat([df, pd.get_dummies(df['unit'])], axis=1)

    # === Stage 4: Finalizing the DataFrame ===
    print("4. Finalizing feature set for all models...")
    if is_train and 'price' in df.columns:
        # This is the target variable for all models
        df['price_log1p'] = np.log1p(df['price'])
    
    # Drop intermediate columns that are no longer needed
    columns_to_drop = ['item_name', 'bullet_points', 'description', 'unit', 'raw_unit']
    df_processed = df.drop(columns=columns_to_drop)
    df_processed.columns = [str(col) for col in df_processed.columns]

    print("\n--- Feature Engineering Complete. Data is ready for all team members. ---")
    return df_processed

# --- Example Usage ---
if __name__ == '__main__':
    # Define paths according to your project structure
    output_path = 'processed_data'
    File_Path = 'dataset/test.csv'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_filename = os.path.join(output_path, 'processed_features_test.csv')

    

    df_raw = pd.read_csv(File_Path)
    df_raw.loc[0, 'catalog_content'] = "Item Name: A\n Unit: Count"
    df_raw.loc[1, 'catalog_content'] = "Item Name: B\n Unit: ounce"
    df_raw.loc[2, 'catalog_content'] = "Item Name: C\n Unit: lb"
    df_raw.loc[3, 'catalog_content'] = "Item Name: D\n Unit: fl oz"
    df_raw.loc[4, 'catalog_content'] = "Item Name: E\n Unit: grams"
    
    # Run the full pipeline
    df_final = run_feature_engineering(df_raw, is_train=False)
    
    # Save the output
    df_final.to_csv(output_filename, index=False)
    print(f"\nSuccessfully saved final features to: {output_filename}")
    
    # Display the final structure
    print("\n--- Final, Clean DataFrame Head ---")
    print(df_final.head())
    print("\n--- Final, Clean DataFrame Info ---")
    df_final.info()