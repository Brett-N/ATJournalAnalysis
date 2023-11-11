import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Constants
PLACEHOLDER = "PLACEHOLDER_FOR_MISSING_VALUE"

def process_excel_with_sentiment_analysis(file_path: str):
    df = pd.read_excel(file_path)

    # Convert lines that contain only spaces to NaN
    df['Journal Story'] = df['Journal Story'].apply(lambda x: x.strip() if isinstance(x, str) else x).replace("", None)

    # Replace missing values with a unique placeholder
    df['Journal Story'].fillna(PLACEHOLDER, inplace=True)

    # Initialize the BERT tokenizer and model for sentiment analysis
    # Note: 'bert-base-uncased' and 'nlptown/bert-base-multilingual-uncased-sentiment' are common models for sentiment analysis, but you might want to choose one that fits your data better.
    model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Use the model for sentiment prediction
    sentiments = []
    for entry in df['Journal Story']:
        if entry == PLACEHOLDER:
            sentiments.append(None)  # Set the sentiment to None (or "" if you prefer) for placeholder values
            continue

        inputs = tokenizer.encode_plus(entry, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        sentiments.append(predicted_class)  # This gives a sentiment score from 0 to 4 (for the 'nlptown' model)
    
    df['Sentiment'] = sentiments
    
    # Replace the placeholder value with an empty string
    df['Journal Story'] = df['Journal Story'].replace(PLACEHOLDER, "")

    # Save the updated dataframe
    output_filename = file_path.replace(".xlsx", "_SentimentAnalysis.xlsx")
    df.to_excel(output_filename, index=False)
    print(f"Analysis completed. Output saved to: {output_filename}")

if __name__ == "__main__":
    input_file = input("Enter the path to your Excel file: ")
    process_excel_with_sentiment_analysis(input_file)