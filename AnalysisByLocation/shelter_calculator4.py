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

english_words_set = set(words.words())

MIN_TOPICS = 500

def plot_top_topics_by_location_table(df, label, original_texts):
    # Filter by label first
    df_label = df[df['label'] == label]

    # Group by 'Location Name', 'Latitude', 'Longitude', 'Topic', 'Topic Words', 'Journal Story' to get count of each topic for each location
    grouped = df_label.groupby(['Location Name', 'Latitude', 'Longitude', 'Topic', 'Topic Words', 'Journal Story']).size().reset_index(name='Counts')
    
    # Calculate total number of journal entries for each location
    total_journals = grouped.groupby(['Location Name', 'Latitude', 'Longitude']).size().reset_index(name='Number of Journals')
    
    # Merge the total journals count back to the grouped dataframe
    grouped = pd.merge(grouped, total_journals, on=['Location Name', 'Latitude', 'Longitude'])
    
    # Sort by counts
    sorted_grouped = grouped.sort_values(['Location Name', 'Counts'], ascending=[True, False])
    
    # For each location, select the top 3 topics
    top_topics_per_location = sorted_grouped.groupby('Location Name').head(3)

    # Create lists to hold data for the table
    locations = []
    latitudes = []
    longitudes = []
    topics = []
    number_of_journals = []
    journal_entries = {i: [] for i in range(1, 6)}

    for (location, lat, lon), group in top_topics_per_location.groupby(['Location Name', 'Latitude', 'Longitude']):
        locations.append(location)
        latitudes.append(lat)
        longitudes.append(lon)
        topics.append(', '.join(group['Topic Words'].tolist()))
        number_of_journals.append(group['Number of Journals'].values[0])  # all rows in the group have the same 'Number of Journals'
        
        # Retrieve original journal entries
        for i, entry in enumerate(group['Journal Story'].tolist(), start=1):
            if i > 5:
                break
            original_entry = original_texts.get(entry, entry)  # Retrieve original text if available
            journal_entries[i].append(original_entry)

    # Add None for missing values
    max_length = max(len(locations), len(latitudes), len(longitudes), len(topics), len(number_of_journals), *map(len, journal_entries.values()))
    for key in journal_entries:
        while len(journal_entries[key]) < max_length:
            journal_entries[key].append(None)

    # Create Plotly table without journal entries (for visualization)
    fig = go.Figure(data=[go.Table(header=dict(values=['Location Name', 'Latitude', 'Longitude', 'Top Topics', 'Number of Journals']),
                                   cells=dict(values=[locations, latitudes, longitudes, topics, number_of_journals]))])

    fig.show()

    # Save to csv
    csv_data = {
        "Location Name": locations,
        "Latitude": latitudes,
        "Longitude": longitudes,
        "Top Topics": topics,
        "Number of Journals": number_of_journals
    }

    for key, value in journal_entries.items():
        csv_data[f"Journal Entry {key}"] = value

    df_csv = pd.DataFrame(csv_data)
    csv_filename = f"TopTopics_{label}.csv"
    df_csv.to_csv(csv_filename, index=False)
    print(f"Table for {label} saved to {csv_filename}")

def process_excel_with_bertopic(file_path: str):
    df = pd.read_excel(file_path)

    # Filter rows with "No Club" in the "Trail club" column
    df = df[df['Trail club'] != 'No Club']

    # Before calling the BERTopic model, create a dictionary to map processed texts back to original texts
    original_texts = {original_text: original_text for original_text in df['Journal Story']}

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
    # Example of how to call the plot_top_topics_by_location_table function with the original_texts dictionary
    for label in unique_labels:
        plot_top_topics_by_location_table(df, label, original_texts)


if __name__ == "__main__":
    input_file = input("Enter the path to your Excel file: ")
    process_excel_with_bertopic(input_file)