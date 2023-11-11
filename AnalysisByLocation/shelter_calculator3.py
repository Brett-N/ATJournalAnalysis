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

def plot_top_topics_by_location_table(df, label):
    # Filter by label first
    df_label = df[df['label'] == label]

    # Group by 'Location Name', 'Latitude', 'Longitude', 'Topic Words' to get count of each topic for each location
    grouped = df_label.groupby(['Location Name', 'Latitude', 'Longitude', 'Topic', 'Topic Words']).size().reset_index(name='Counts')
    
    # Sort by counts
    sorted_grouped = grouped.sort_values(['Location Name', 'Counts'], ascending=[True, False])
    
    # For each location, select the top 3 topics
    top_topics_per_location = sorted_grouped.groupby('Location Name').head(3)

    # Create lists to hold data for the table and CSV
    locations = []
    latitudes = []
    longitudes = []
    topics = []
    journal_entries_lists = []

    for (location, lat, lon), group in top_topics_per_location.groupby(['Location Name', 'Latitude', 'Longitude']):
        locations.append(location)
        latitudes.append(lat)
        longitudes.append(lon)
        topics.append(', '.join(group['Topic Words'].tolist()))

        # Fetch journal entries for the top topics
        top_topics = group['Topic'].tolist()
        journal_entries = df_label[(df_label['Location Name'] == location) & (df_label['Topic'].isin(top_topics))]['Journal Story'].tolist()
        journal_entries_lists.append(' | '.join(map(str, journal_entries)))  # Join entries and convert all to string

    # Create DataFrame for CSV
    data = {
        'Location Name': locations,
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Top Topics': topics,
        'Journal Entries': journal_entries_lists,
    }
    csv_df = pd.DataFrame(data)

    # Save to csv
    csv_filename = f"TopTopics_{label}.csv"
    csv_df.to_csv(csv_filename, index=False)
    print(f"Table for {label} saved to {csv_filename}")

    # Create Plotly table without journal entries (for visualization)
    fig = go.Figure(data=[go.Table(header=dict(values=['Location Name', 'Latitude', 'Longitude', 'Top Topics']),
                                   cells=dict(values=[locations, latitudes, longitudes, topics]))])

    fig.show()

def process_excel_with_bertopic(file_path: str):
    df = pd.read_excel(file_path)

    # Filter rows with "No Club" in the "Trail club" column
    df = df[df['Trail club'] != 'No Club']

    # Extract unique names
    unique_names = df['Hiker trail name'].unique().tolist()

    additional_stop_words = ["umbel", "rich", "wandershue.wordpress.", 
                             "com", "2016", "okay", "webcatcher", "walsch", 
                             "neale", "donald", "trimpi", "ood", "timejust", 
                             "2189", "jeff", "walker", "day174", "recons", 
                             "tonto", "samsung", "s6", "smartphone", "homefront", 
                             "entries", "iphone", "sent", "httpwwwtrailjournalsj", 
                             "ive", "jerard", "towlie", "ot", "norsemen", "1am", 
                             "david", "zeros", "today", "frodo", "aloha", "dot", 
                             "franky", "pamela", "inyo", "scotsman" "refers", "unaka", 
                             "westy", "id", "hobbit", "way", "feet", "lt", "vandeventer", 
                             "macgyver", "macyver", "sean","boys", "hops", "fi", "rick", 
                             "clingmans", "thor", "naomi", "floaty", "rusty", "730pm", "went", 
                             "got", "clicks", "documentary", "freak", "brother", "low", "selah", 
                             "hang", "going", "nom", "tellico", "noc", "maverick", "wesser", "mary", 
                             "mongo", "duane", "otis", "anniversary", "rouse", "colonel", "jimmy", 
                             "promise", "franconia", "quinn", "charlie", "topsy", "stratton", 
                             "killington", "robby","syrup", "antonio", "siblings", "mahoosuc", 
                             "0s", "raveyard", "ha", "pearisburg", "zero", "day", "bitmaybe", 
                             "atlinburg", "newfound", "brewha", "menassas", "choking", "liters", 
                             "consist", "krispy", "orp", "cologne", "engaged", "carl", "outfittter", 
                             "pacmans", "stuffwent", "suprisingly", "sassafras", "vermont", "ct", "riga", 
                             "everett", "cts", "rutland", "connecticut", "liters", "drank", "jeep", "roller", 
                             "90s", "polk", "moderating", "dunking", "snickers", "im", "connor", "waynesboro", 
                             "od", "ods", "bert", "becky", "bag", "jrk", "cmc", "david", "exists", "dam", "lonnie", 
                             "utah", "ria", "shrink", "bc", "noah", "qb", "norbert", "boogied", "pepsi", "digestive", 
                             "brian", "va", "david", "ap", "thomas", "rayson", "rogers", "toothbrush", "rover", "bake", 
                             "oven", "folks", "dans","windsor", "mi" "anker", "lack", "phone", "spaceholder", 
                             "gaurantees", "e1", "devote", "upgraded", "update", "sorry", "thor", "puppy", 
                             "briface", "nat", "musician", "answer", "barrel" "hanover", "williamstown", 
                             "reylock", "hampshire", "707", "limsik", "ricktic", "cokebig", "fd", "ricola", 
                             "antibiic", "poke", "tricorner", "inches", "pan", "travr", "washington", 
                             "monument", "tt", "0", "guy", "sa", "facebookatthuraven", "hanover", 
                             "vt", "maine", "nch", "mi", "liter", "poke", "travr", "mich", "damascus", 
                             "eezer", "cosby", "77", "inches", "pad", "met", "silers", "19", "eve", "sme", 
                             "watauga", "iris", "tejust", "begins", "proje", "alastair", "rearviews", "couts", 
                             "civil", "quill", "nobo", "intelligence", "cartrge", "sawyer", "amtrak", "roan", 
                             "phil", "asked", "told", "als", "pa", "fol", "mike", "cong", "smd", "fort", "add",
                             "te", "ve", "twain", "mark", "tn", "just", "niceably", "alaxy", "shot", "video", "morning",
                             "fostep", "jerry", "al", "porky", "transcription", "bobby", "fol","coelho", "wake", "tigger",
                              "cog", "york", "jenny", "olivia", "ak", "franklin", "frank", "ralph", "billion",
                              "airport", "joe", "sago", "eric", "uva", "sub", "suds", "po", "noel", "wir",
                              "die", "ich", "es", "ist", "den", "das", "sind", "nach", "sie", "tag", "wo",
                              "whistler", "earl", "la", "vie", "en", "belle", "mon", "se", "sur", "soleil", "tu", "dan", "skippy",
                              "bob", "huck", "ginny", "er", "morgan", "hat", "ra", "bis", "ra", "paar", "wird", "mir", "um", "ginny",
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
        plot_top_topics_by_location_table(df, label)


if __name__ == "__main__":
    input_file = input("Enter the path to your Excel file: ")
    process_excel_with_bertopic(input_file)