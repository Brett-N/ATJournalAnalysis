import os
import plotly.io as pio
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import pandas as pd
import nltk
nltk.download('words')
from nltk.corpus import words
import plotly.graph_objects as go

PLACEHOLDER = "PLACEHOLDER_FOR_MISSING_VALUE"
DESIRED_TOPICS = 30
english_words_set = set(words.words())

def process_excel_with_bertopic(file_path: str):
    df = pd.read_excel(file_path)

    # Filter rows with "No Club" in the "Trail club" column
    df = df[df['Trail club'] != 'No Club']

    # Extract unique names and location names
    unique_names = df['Hiker trail name'].unique().tolist()
    unique_locations = df['Location Name'].unique().tolist()  # Change 'Trail club' to 'Location Name'

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

    # Here, after processing the journal stories, we check if they are only made up of stop words and/or spaces
    mask_valid_stories = df['Journal Story'].apply(lambda x: x and any(word.lower() not in all_stop_words for word in x.split()) if isinstance(x, str) else False)
    valid_stories_df = df[mask_valid_stories].copy()

    stories_for_embedding = valid_stories_df['Journal Story'].tolist()

    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(stories_for_embedding, show_progress_bar=True)

    stop_words = list(ENGLISH_STOP_WORDS) + all_stop_words
    vectorizer_model = CountVectorizer(stop_words=stop_words)

    topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics=DESIRED_TOPICS, low_memory=True, calculate_probabilities=False, n_gram_range=(1, 2), min_topic_size=5)
    topics, _ = topic_model.fit_transform(stories_for_embedding, embeddings=embeddings)

    # Step 1: Initialize the column with placeholder values
    df['Topic'] = -1

    # Step 2: Assign topics only to the rows that have valid stories
    df.loc[mask_valid_stories, 'Topic'] = topics

    # Check if locations have 5 or more topics
    location_topic_counts = df[df['Topic'] != -1].groupby('Location Name')['Topic'].nunique()  # Change 'Trail club' to 'Location Name'
    for location, topic_count in location_topic_counts.items():  # Change 'Trail club' to 'Location Name'
        if topic_count < 5:
            print(f"Location '{location}' has fewer than 5 topics.")  # Change 'Trail club' to 'Location Name'

    topic_frequencies = df.groupby('Location Name')['Topic'].value_counts(normalize=True).unstack().fillna(0)
    average_frequencies = topic_frequencies.mean()

    overall_most_frequent_topics_per_location = {}

    for location in unique_locations:
        location_specific_frequencies = topic_frequencies.loc[location]
        most_frequent = location_specific_frequencies.nlargest(5).index.tolist()
        overall_most_frequent_topics_per_location[location] = most_frequent

    print(overall_most_frequent_topics_per_location)
    
    # Visualization (remains the same)
    fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_heatmap = topic_model.visualize_heatmap()

    fig_barchart.write_html("barchart.html")
    fig_hierarchy.write_html("hierarchy.html")
    fig_heatmap.write_html("heatmap.html")

    MIN_WORDS_IN_TOPIC = 3  # Set your threshold here

    # Create a dictionary of topics and their words, but filter out topics with fewer than MIN_WORDS_IN_TOPIC words
    topic_words = {}
    for location_topics in df['Topic']:
        if isinstance(location_topics, list):  # Check if it's a list of topics
            for t in location_topics:
                words_list = [word[0] for word in topic_model.get_topic(t)]  # Extract words from tuples
                if len(words_list) >= MIN_WORDS_IN_TOPIC:
                    topic_words[t] = ', '.join(words_list)  # Join the words into a single string

    # Apply the topic words to each row in the DataFrame
    df['Topic Words'] = df['Topic'].apply(lambda x: ', '.join([topic_words.get(t, "") for t in x]) if isinstance(x, list) else "")

    output_filename = file_path.replace(".xlsx", "_TopicAnalysis.xlsx")
    df.to_excel(output_filename, index=False)
    print(f"Analysis completed. Output saved to: {output_filename}")
    visualize_with_plotly(overall_most_frequent_topics_per_location, topic_words)
    
import plotly.graph_objects as go

def visualize_with_plotly(data, topic_words):
    # Filter out "No Club," topic -1, and topic 0 from the data
    filtered_data = {key: [val for val in value if val not in [-1, 0]] for key, value in data.items() if key != 'No Club'}

    # Sort topics for each location by frequency in descending order
    sorted_data = {key: sorted(value, key=lambda x: -len(topic_words.get(x, []))) for key, value in filtered_data.items()}

    # Convert the filtered dictionary data into two lists: clubs and topic counts
    locations = list(sorted_data.keys())
    topic_nums = [len(val) for val in sorted_data.values()]

    # Create the bar chart with a visually appealing color
    fig = go.Figure(data=[
        go.Bar(
            x=locations,
            y=topic_nums,
            marker_color='limegreen'  # Change the color to a visually appealing one
        )
    ])

    # Set the layout properties for the main chart
    fig.update_layout(
        title='Most Frequent Topics Per Club (Joy)',
        xaxis=dict(
            title='Club',
            tickfont_size=12,  # Increase font size for better readability
            tickangle=45
        ),
        yaxis=dict(
            title='Number of Most Frequent Topics',
            tickfont_size=14
        ),
        plot_bgcolor='white'  # Set background color to white for a clean look
    )

    # Create a table to display topics and their associated words with custom fonts and colors
    table_data = []
    for location in locations:
        topics_and_words = []
        for topic in sorted_data[location]:
            topics_and_words.append(f"Topic {topic}: {', '.join(topic_words.get(topic, []))}")
        table_data.append([location, "<br>".join(topics_and_words)])

    table = go.Figure(data=[go.Table(
        header=dict(
            values=["Location", "Topics and Words"],
            fill=dict(color='limegreen'),  # Header background color
            align=['center', 'center'],  # Header text alignment
            font=dict(size=14, color='black')  # Header font size and color
        ),
        cells=dict(
            values=list(zip(*table_data)),
            fill=dict(color='white'),  # Cell background color
            align=['center', 'left'],  # Cell text alignment
            font=dict(size=14, color='black'),  # Cell font size and color
            line=dict(color='black', width=1)  # Add black cell outlines
        ),
        columnwidth=[100, 500]
    )])

    # Set the layout properties for the table
    table.update_layout(
        title='Topics and Words Per Club',
        margin=dict(t=20)  # Adjust the top margin to separate from the main chart
    )

    # Show the main chart and the table
    fig.show()
    table.show()
    fig.write_html("most_frequent_topics_per_club.html")


if __name__ == "__main__":
    input_file = input("Enter the path to your Excel file: ")
    process_excel_with_bertopic(input_file)