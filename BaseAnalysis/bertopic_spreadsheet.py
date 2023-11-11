from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def process_excel_with_bertopic(file_path: str):
    df = pd.read_excel(file_path)

    # Ensure there are no missing values
    if df['Journal Story'].isnull().any():
        print("Warning: Found missing values in 'Journal Story' column. Dropping them.")
        df.dropna(subset=['Journal Story'], inplace=True)

    # Convert the journal stories into embeddings
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(df['Journal Story'].tolist(), show_progress_bar=True)

    # Create topics from the embeddings using CountVectorizer to remove stop words
    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics="auto", low_memory=True, calculate_probabilities=False, n_gram_range=(1, 2), min_topic_size=5)
    topics, _ = topic_model.fit_transform(df['Journal Story'].tolist(), embeddings=embeddings)

    # Get topics and their associated words
    topics_overview = topic_model.get_topics()
    topic_words = {topic: [word[0] for word in words] for topic, words in topics_overview.items()}

    # Map topic numbers to topic words
    df['Topic'] = topics
    df['Topic Words'] = df['Topic'].apply(lambda x: ', '.join(topic_words.get(x, [])))

    # Save the updated dataframe
    output_filename = file_path.replace(".xlsx", "_TopicAnalysis.xlsx")
    df.to_excel(output_filename, index=False)
    print(f"Analysis completed. Output saved to: {output_filename}")

if __name__ == "__main__":
    input_file = input("Enter the path to your Excel file: ")
    process_excel_with_bertopic(input_file)