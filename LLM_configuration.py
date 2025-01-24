
import pandas as pd
import requests
import json
import os
import time

# API details
API_KEY ='API_KEY'  # Make sure to replace with your actual API key

API_URL = "https://api.openai.com/v1/chat/completions"

# Input and output directories
input_dir = 'batches'
output_dir = 'batches_results'
os.makedirs(output_dir, exist_ok=True)

# Function to analyze sentiment for a batch
def analyze_sentiment_batch(batch):
    texts = batch['text'].tolist()
    prompt = "Analyze the sentiment of the following texts as Positive, Negative, or Neutral:\n" + "\n".join(
        [f"{i+1}. {text}" for i, text in enumerate(texts)]
    )
    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500
            },
        )
        response.raise_for_status()
        sentiments = response.json()['choices'][0]['message']['content'].strip().split("\n")
        return sentiments
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:  # Rate limit
            print("Rate limit hit. Retrying in 5 seconds...")
            time.sleep(5)
            return analyze_sentiment_batch(batch)  # Retry
        else:
            print(f"HTTP error: {e}")
            return ["Error"] * len(batch)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ["Error"] * len(batch)

# Process each batch in the input directory
batch_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
for batch_file in batch_files:
    print(f"Processing {batch_file}...")
    batch_path = os.path.join(input_dir, batch_file)

    # Load batch data
    batch_data = pd.read_json(batch_path)

    # Analyze sentiment for the batch
    sentiments = analyze_sentiment_batch(batch_data)

    # Add sentiments to the batch
    batch_data['Sentiment'] = sentiments
# Remove numeric prefixes from Sentiment
    batch_data['Sentiment'] = batch_data['Sentiment'].str.replace(r'^\d+\.\s*', '', regex=True)

    # Save the processed batch results
    result_path = os.path.join(output_dir, batch_file)
    batch_data.to_json(result_path, orient='records', indent=4)
    print(f"Results saved to {result_path}")

print("All batches processed successfully.")
