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

def plot_top_topics_by_location_table(df, label, df_original):
    # Filter by label first
    df_label = df[df['label'] == label]

    # Modify the aggregation functions to collect the journal stories and their indices
    aggregation_functions = {
      'Topic': lambda x: list(x)[:3],  # Take up to the first 3 topics
      'Topic Words': lambda x: list(x)[:3],  # Take up to the first 3 topic words
      'Journal Story': lambda x: list(x)[:5],  # Take up to the first 5 journal stories
      'Journal Index': lambda x: list(x.index)[:5]  # Take the indices of the up to first 5 journal stories
    }
    grouped = df_label.groupby(['Location Name', 'Latitude', 'Longitude']).agg(aggregation_functions).reset_index()


    # Create the CSV data dict
    csv_data = {
        "Location Name": grouped['Location Name'],
        "Latitude": grouped['Latitude'],
        "Longitude": grouped['Longitude']
    }

    # Dynamically create columns for topics and journal entries
    for i in range(1, 4):  # For each topic (up to 3 topics)
        topic_word_col = f"Topic {i} Words"
        csv_data[topic_word_col] = grouped['Topic Words'].apply(lambda x: x[i-1] if len(x) >= i else None)

        # Add columns for journal entries for each topic
        for j in range(1, 6):  # For each journal entry (up to 5 per topic)
          journal_col = f"Topic {i} Journal Entry {j}"
          # Now, use the indices to access the original journal stories from df_original
          csv_data[journal_col] = grouped['Journal Index'].apply(
              lambda indices: df_original.at[indices[j-1], 'Journal Story'] if j <= len(indices) else None
          )

    # Create DataFrame from the csv_data dictionary
    df_csv = pd.DataFrame(csv_data)

    # Save to CSV
    csv_filename = f"TopTopics_{label}.csv"
    df_csv.to_csv(csv_filename, index=False)
    print(f"Table for {label} saved to {csv_filename}")

def process_excel_with_bertopic(file_path: str):
    df = pd.read_excel(file_path)

    df['Journal Index'] = df.index

    # Filter rows with "No Club" in the "Trail club" column
    df = df[df['Trail club'] != 'No Club']

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

    output_filename = file_path.replace(".xlsx", "_TopicAnalysis.xlsx")
    df.to_excel(output_filename, index=False)
    print(f"Analysis completed. Output saved to: {output_filename}")

    # Loop over unique labels and generate a table for each one.
    unique_labels = df['label'].unique().tolist()
    for label in unique_labels:
      plot_top_topics_by_location_table(df, label, df_original)


if __name__ == "__main__":
    input_file = input("Enter the path to your Excel file: ")
    process_excel_with_bertopic(input_file)