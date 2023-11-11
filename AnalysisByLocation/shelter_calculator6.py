import os
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import plotly.express as px
nltk.download('words')
from nltk.corpus import words
import plotly.graph_objects as go
from fuzzywuzzy import process
from tqdm import tqdm

english_words_set = set(words.words())

MIN_TOPICS = 500

def plot_top_topics_by_location_table(df, emotional_label, df_original, topic_model):
    # Filter based on emotional label
    df_filtered = df[df['emotional_label'] == emotional_label]

    # Group by location
    grouped = df_filtered.groupby(['Latitude', 'Longitude'])

    # Prepare the CSV data
    csv_data = []

    for (lat, lon), group in grouped:
        # Find the top 3 topics in this group
        top_topics = group['Topic'].value_counts().head(3)

        row = {'Latitude': lat, 'Longitude': lon}
        for i, (topic, count) in enumerate(top_topics.items(), start=1):
            if i > 3:
                break

            # Topic words
            topic_words = ', '.join([word[0] for word in topic_model.get_topic(topic)])
            row[f'Topic {i} Words'] = topic_words

            # Journal entries for the topic
            topic_entries = group[group['Topic'] == topic].head(5)
            for j, entry in enumerate(topic_entries['Journal Index'], start=1):
                original_entry = df_original.loc[entry, 'Journal Story']
                row[f'Topic {i} Entry {j}'] = original_entry

        csv_data.append(row)

    # Convert to DataFrame and save to CSV
    csv_df = pd.DataFrame(csv_data)
    csv_filename = f"TopTopics_{emotional_label}.csv"
    csv_df.to_csv(csv_filename, index=False)
    print(f"CSV file saved: {csv_filename}")

def process_excel_with_bertopic(file_path: str):
    df = pd.read_csv(file_path)

    # Remove rows where 'Latitude' is -1
    df = df[df['Latitude'] != -1]

    df['Journal Index'] = df.index

    df_original = df.copy()

    # Extract unique names
    unique_names = df['Hiker trail name'].unique().tolist()

    additional_stop_words = ["umbel", "wandershue.wordpress.", 
                            "com", "2016", "okay", "webcatcher", "walsch", "neale", "donald", 
                            "trimpi", "ood", "timejust", "2189", "jeff", "walker", "day174",
                            "recons", "tonto", "samsung", "s6", "smartphone", "homefront", 
                            "entries", "iphone", "sent", "httpwwwtrailjournalsj", "ive", "jerard",
                            "towlie", "ot", "norsemen", "1am", "david", "zeros", "today", "frodo", 
                            "aloha", "dot", "franky", "pamela", "inyo", "scotsman" "refers", "unaka", 
                            "westy", "id", "hobbit", "way", "feet", "lt", "vandeventer", "macgyver", 
                            "macyver", "sean", "hops", "fi", "rick", "clingmans", "thor", "naomi",
                              "floaty", "rusty", "730pm", "went", "got", "clicks", "selah", 
                            "hang", "going", "nom", "tellico", "noc", "maverick", "wesser", "mary",
                              "mongo", "duane", "otis", "rouse", "colonel", "jimmy", 
                            "franconia", "quinn", "charlie", "topsy", "stratton", "killington", "robby",
                              "antonio", "siblings", "mahoosuc", 
                            "0s", "raveyard", "ha", "bitmaybe", "newfound", "brewha", "menassas", "choking", 
                            "consist", "krispy", "orp", "cologne", "engaged", "carl", "outfittter", "pacmans",
                              "stuffwent", "sassafras", "ct", "riga", 
                             "everett", "cts", "jeep", "roller", "90s", "polk", "moderating", "dunking", 
                             "snickers", "im", "connor", "waynesboro", 
                             "od", "ods", "bert", "becky", "bag", "jrk", "cmc", "david", "exists", "dam", "lonnie", 
                             "utah", "ria", "bc", "noah", "qb", "norbert", "boogied", "brian", "va", "david", "ap",
                               "thomas", "rayson", "rogers", "rover", 
                             "dans","windsor", "mi" "anker", "lack", "spaceholder", "gaurantees", "e1", "thor", 
                             "briface", "nat", "musician", "answer", "barrel",  "reylock", "707", "limsik",
                               "ricktic", "cokebig", "fd", "ricola", 
                             "antibiic", "poke", "tricorner", "pan", "travr", "monument", "tt", "0", "guy",
                               "sa", "facebookatthuraven", "hanover", 
                             "vt", "maine", "nch", "mi", "liter", "poke", "travr", "mich", "eezer", "cosby",
                               "77", "inches", "pad", "met", "silers", "19", "eve", "sme", 
                             "watauga", "iris", "tejust", "begins", "proje", "alastair", "rearviews", "couts",
                               "quill", "cartrge", "amtrak",  
                             "phil", "asked", "told", "als", "pa", "fol", "mike", "cong", "smd", "fort", "add",
                             "te", "ve", "twain", "mark", "tn", "just", "niceably", "alaxy", "shot", "video", 
                             "morning",
                             "fostep", "jerry", "al", "porky", "transcription", "bobby", "fol","coelho", "wake",
                               "tigger",
                              "cog", "york", "jenny", "olivia", "ak", "franklin", "frank", "ralph", "billion",
                             "joe", "sago", "eric", "uva", "sub", "suds", "po", "noel", "wir",
                             "ich", "es", "ist", "den", "das", "sind", "nach", "sie", "tag", "wo",
                              "whistler", "earl", "la", "vie", "en", "belle", "mon", "se", "sur", "soleil",
                                "tu", "dan", "skippy",
                              "bob", "huck", "ginny", "er", "morgan", "ra", "bis", "ra", "paar", "wird",
                                "mir", "um", "ginny",
                               "holly", "harry", "allison", "morgen", "gut", "mein", "bin", "muss", "da" ]

    all_stop_words = unique_names + additional_stop_words

    def remove_words_from_text(text, words_list):
        text_split = text.split()
        cleaned_text = [word for word in text_split if word.lower() not in words_list and word.lower() in english_words_set]
        return ' '.join(cleaned_text)

    df['Journal Story'] = df['Journal Story'].apply(lambda x: remove_words_from_text(x, all_stop_words) if isinstance(x, str) else x)
    df['Journal Story'] = df['Journal Story'].apply(lambda x: x.strip() if isinstance(x, str) else x).replace("", None)

    stories_for_embedding = df['Journal Story'].dropna().tolist()

    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(stories_for_embedding, show_progress_bar=True)

    stop_words = list(ENGLISH_STOP_WORDS) + all_stop_words
    vectorizer_model = CountVectorizer(stop_words=stop_words)

    topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics=MIN_TOPICS, calculate_probabilities=False, n_gram_range=(1, 2), min_topic_size=5)
    topics, _ = topic_model.fit_transform(stories_for_embedding, embeddings=embeddings)

    df['Topic'] = -1
    df.loc[df['Journal Story'].notna(), 'Topic'] = topics

    topic_words_dict = topic_model.get_topics()
    df['Topic Words'] = df['Topic'].apply(lambda x: ', '.join([word[0] for word in topic_words_dict[x]]) if x in topic_words_dict else "")

    # Loop over unique emotional_labels and generate a table for each one.
    unique_labels = df['emotional_label'].unique().tolist()
    for emotional_label in unique_labels:
        plot_top_topics_by_location_table(df, emotional_label, df_original, topic_model)

    # Saving the topic modeled data
    output_filename = file_path.replace(".csv", "_TopicAnalysis.csv")
    df.to_csv(output_filename, index=False)
    print(f"Analysis completed. Output saved to: {output_filename}")


if __name__ == "__main__":
    input_file = input("Enter the path to your CSV file: ")
    process_excel_with_bertopic(input_file)