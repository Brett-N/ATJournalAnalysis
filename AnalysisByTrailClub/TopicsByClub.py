import os
import plotly.io as pio
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException
from collections import defaultdict

# Setting a seed to make language detection deterministic
DetectorFactory.seed = 0

PLACEHOLDER = "PLACEHOLDER_FOR_MISSING_VALUE"
DESIRED_TOPICS = 800

def is_english(text):
    """Detect if a given text is in English"""
    try:
        return detect(text) == 'en'
    except LangDetectException:  # Handle cases where detection fails
        return False

def process_excel_with_bertopic(file_path: str):
    df = pd.read_excel(file_path)
    
    # Extract unique trail clubs and names
    unique_trail_clubs = df['Trail Club'].unique().tolist()
    unique_names = df['Hiker trail name'].unique().tolist()

    # List of additional words to be removed
    additional_stop_words = ["umbel", "rich", "wandershue.wordpress.", "com"]
    all_stop_words = unique_names + additional_stop_words

    # Remove names and additional stop words from Journal Story column
    def remove_words_from_text(text, words_list):
        for word in words_list:
            text = text.replace(word, "")
        return text

    df['Journal Story'] = df['Journal Story'].apply(lambda x: remove_words_from_text(x, all_stop_words) if isinstance(x, str) else x)

    # Convert lines that contain only spaces to NaN
    df['Journal Story'] = df['Journal Story'].apply(lambda x: x.strip() if isinstance(x, str) else x).replace("", None)

    # Filter out rows with Journal Story that are either blank, contain only stop words, or not in English
    mask_valid_stories = df['Journal Story'].apply(
        lambda x: x and any(word not in all_stop_words for word in x.split()) and is_english(x) if isinstance(x, str) else False
    )
    valid_stories_df = df[mask_valid_stories].copy()

    # Initialize topic_frequencies dictionary
    topic_frequencies = {trail_club: {} for trail_club in unique_trail_clubs}

    for trail_club in unique_trail_clubs:
        club_df = valid_stories_df[valid_stories_df['Trail Club'] == trail_club]
        stories_for_embedding = club_df['Journal Story'].tolist()

        # Convert the journal stories into embeddings
        model = SentenceTransformer("all-mpnet-base-v2")
        embeddings = model.encode(stories_for_embedding, show_progress_bar=True)

        # Modify vectorizer to include names and additional stop words in the stop words
        stop_words = list(ENGLISH_STOP_WORDS) + all_stop_words
        vectorizer_model = CountVectorizer(stop_words=stop_words)

        # Create the BERTopic model
        topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics="auto", low_memory=True, calculate_probabilities=False, n_gram_range=(1, 2))
        topics, _ = topic_model.fit_transform(stories_for_embedding, embeddings=embeddings)

        # Populate topic_frequencies dictionary
        for topic in topics:
            topic_frequencies[trail_club][topic] = topic_frequencies[trail_club].get(topic, 0) + 1

        # Associate topics to stories in the dataframe
        valid_stories_df.loc[club_df.index, 'Topic'] = topics

    # Visualization
    fig_barchart = topic_model.visualize_barchart(top_n_topics=10)  # Visualizing top 10 topics for simplicity, you can adjust
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_heatmap = topic_model.visualize_heatmap()

    # Save visualizations
    fig_barchart.write_html("barchart.html")
    fig_hierarchy.write_html("hierarchy.html")
    fig_heatmap.write_html("heatmap.html")

    # Get topics and their associated words
    topics_overview = topic_model.get_topics()
    topic_words = {topic: [word[0] for word in words] for topic, words in topics_overview.items()}
    valid_stories_df['Topic Words'] = valid_stories_df['Topic'].apply(lambda x: ', '.join(topic_words.get(x, [])))

    # After generating topics, merge the results back into the original dataframe
    df.loc[mask_valid_stories, 'Topic'] = valid_stories_df['Topic']
    df.loc[mask_valid_stories, 'Topic Words'] = valid_stories_df['Topic Words']

    # Save the updated dataframe
    output_filename = file_path.replace(".xlsx", "_TopicAnalysis.xlsx")
    df.to_excel(output_filename, index=False)
    print(f"Analysis completed. Output saved to: {output_filename}")

    # Print trail club topic differences
    sorted_topic_freqs = defaultdict(dict)
    for club, freqs in topic_frequencies.items():
        sorted_topic_freqs[club] = dict(sorted(freqs.items(), key=lambda item: item[1], reverse=True))
    
    print("Topic differences among trail clubs:")
    for club, freqs in sorted_topic_freqs.items():
        print(f"\nTrail Club: {club}")
        for topic, count in freqs.items():
            print(f"Topic {topic} ({', '.join(topic_words.get(topic, []))}): {count} occurrences")

if __name__ == "__main__":
    input_file = input("Enter the path to your Excel file: ")
    process_excel_with_bertopic(input_file)