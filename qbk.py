import pandas as pd
import re
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "INSERT_API_KEY"))

def normalize_description(desc):
    desc = re.sub(r'\*[A-Z0-9]{8,10}', '', desc, flags=re.IGNORECASE)
    desc = re.sub(r'[A-Z0-9]{7,10}', '', desc, flags=re.IGNORECASE)
    desc = re.sub(r'\d+', '', desc)
    desc = re.sub(r'[^\w\s]', '', desc)
    desc = re.sub(r'\bAM\b|\bPM\b', '', desc, flags=re.IGNORECASE)
    return desc.strip()

def load_categories(file_path):
    categories = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 2:  # Ensure there are exactly two parts
                category, nature = parts[0].strip(), parts[1].strip()
                categories[category] = nature
    return categories

def categorize_transaction(original_desc, normalized_desc, categories, business_type="Convenience Store"):
    try:
        category_keys = list(categories.keys())
        prompt = f"""
        Business Type: {business_type}. Based on the business type and the transaction descriptions below, categorize the transaction into one of the following categories: {', '.join(category_keys)}. Consider the context of the transaction.

        Original Description: '{original_desc}'
        Normalized Description: '{normalized_desc}'

        What is the most appropriate category for this transaction?
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Thinking like a human accountant, given that the business type is {business_type}, categorize the following transaction: {prompt}"},
                {"role": "user", "content": f"Considering the business is categorized under NAICS as {business_type}, which category does this transaction belong to?"}
            ],
            temperature=0
        )
        text = "\u200B\n\n" + response.choices[0].message.content.strip()
        matched_category = next((cat for cat in category_keys if cat.lower() in text.lower()), "Miscellaneous")
        return matched_category
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Miscellaneous"

categories = load_categories('categories.txt')

df = pd.read_csv('input.csv')

df['Normalized Description'] = df['Description'].apply(normalize_description)

unique_normalized = df.drop_duplicates(subset=['Normalized Description'])
unique_normalized = unique_normalized.head(200)
unique_normalized['Category'] = unique_normalized.apply(lambda row: categorize_transaction(row['Description'], row['Normalized Description'], categories), axis=1)

df = df.merge(unique_normalized[['Normalized Description', 'Category']], on='Normalized Description', how='left')

df['Debit'] = df['Amount'].apply(lambda x: x if x < 0 else 0)
df['Credit'] = df['Amount'].apply(lambda x: x if x > 0 else 0)

grouped = df.groupby(['Normalized Description', 'Category']).agg({
    'Debit': 'sum',
    'Credit': 'sum',
    'Description': 'first'
}).reset_index()

grouped.rename(columns={'Description': 'Sample Original Description'}, inplace=True)

grouped.to_csv('output_summary.csv', index=False)

print("Processing complete. The summarized transactions with sample original descriptions are saved in 'output_summary.csv'.")
