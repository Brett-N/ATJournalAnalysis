from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Function to extract topics and their frequencies
def extract_topics(topic_model, docs):
    topics, _ = topic_model.fit_transform(docs)
    topic_freq = topic_model.get_topic_freq().head(4)
    return topic_model.get_topics(), topic_freq

# Function to save topics to CSV
def save_to_csv(location, emotion, entries, topics, topic_freq, filename):
    data = {
        'Location': location,
        'Number of Journals': len(entries),
    }

    # Add topic words for the top three topics
    for i, topic in enumerate(topic_freq['Topic'][1:4]):  # skipping -1
        words = " ".join([word for word, _ in topics[topic]])
        data[f'Topic {i+1} Words'] = words

    # Add journal entries
    for i, entry in enumerate(entries.head(5)):
        data[f'Journal Entry {i+1}'] = entry

    # Append to the corresponding emotion CSV
    pd.DataFrame([data]).to_csv(filename, mode='a', index=False, header=not pd.io.common.file_exists(filename))

# Get the input file path from the user
input_file_path = input("Enter the path to the .xlsx file: ")

# Read the data
df = pd.read_excel(input_file_path)

# Initialize CountVectorizer
count_vectorizer = CountVectorizer()

# Get the unique emotions
emotions = df['label'].unique()

# Process each emotion
for emotion in emotions:
    emotion_df = df[df['label'] == emotion]
    locations = emotion_df['Location Name'].unique()

    # CSV filename for the emotion
    csv_filename = f"{emotion}_analysis.csv"

    # Process each location for the current emotion
    for location in locations:
        location_entries = emotion_df[emotion_df['Location Name'] == location]['Journal Story']

        if len(location_entries) < 2:
            print(f"Not enough journal entries for location: {location} with emotion: {emotion}")
            continue

        # Initialize BERTopic with CountVectorizer
        topic_model = BERTopic(vectorizer_model=count_vectorizer)

        # Perform BERTopic modeling
        try:
            topics, topic_freq = extract_topics(topic_model, location_entries)
            save_to_csv(location, emotion, location_entries, topics, topic_freq, csv_filename)
        except ValueError as e:
            print(f"An error occurred with location: {location} for emotion: {emotion}")
            print(str(e))
            continue

print("Topic modeling completed.")