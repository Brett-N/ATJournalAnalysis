import os
import plotly.io as pio
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException
import nltk
from nltk.corpus import words

# Setting a seed to make language detection deterministic
DetectorFactory.seed = 0

PLACEHOLDER = "PLACEHOLDER_FOR_MISSING_VALUE"
DESIRED_TOPICS = 100

# Download the words dataset
nltk.download('words')

# Create a set of English words for faster lookup
english_words = set(words.words())

def is_english(text):
    """Detect if a given text is in English"""
    try:
        return detect(text) == 'en'
    except LangDetectException:  # Handle cases where detection fails
        return False

def is_valid_word(word):
    """Check if a word is a valid English word"""
    return word.lower() in english_words

def process_excel_with_bertopic(file_path: str):
    df = pd.read_excel(file_path)

    # Extract unique names
    unique_names = df['Hiker trail name'].unique().tolist()
    
    # List of additional words to be removed
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
                             "te", "ve", "twain", "mark", "tn", "just", "niceably", "alaxy", "shot", "video", "morning"
                             ,"fostep", "jerry", "al", "porky", "transcription", "bobby", "fol","coelho", "wake", ]
    all_stop_words = unique_names + additional_stop_words

    # Remove names, additional stop words, and non-English words from Journal Story column
    def remove_words_from_text(text, words_list):
        text = ' '.join([word for word in text.split() if is_valid_word(word)])  # Keep only valid English words
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

    stories_for_embedding = valid_stories_df['Journal Story'].tolist()

    # Convert the journal stories into embeddings
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(stories_for_embedding, show_progress_bar=True)

    # Modify vectorizer to include names and additional stop words in the stop words
    stop_words = list(ENGLISH_STOP_WORDS) + all_stop_words
    vectorizer_model = CountVectorizer(stop_words=stop_words)

    min_topic_size = 5
    topics_overview = {}
    while len(topics_overview) < DESIRED_TOPICS and min_topic_size > 1:
        topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics=None, low_memory=True, calculate_probabilities=False, n_gram_range=(1, 2), min_topic_size=min_topic_size)
        topics, _ = topic_model.fit_transform(stories_for_embedding, embeddings=embeddings)
        topics_overview = topic_model.get_topics()
        min_topic_size -= 1

    # Visualization
    fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_heatmap = topic_model.visualize_heatmap()

    # Save visualizations
    fig_barchart.write_html("barchart.html")
    fig_hierarchy.write_html("hierarchy.html")
    fig_heatmap.write_html("heatmap.html")

    # Get topics and their associated words
    topic_words = {topic: [word[0] for word in words] for topic, words in topics_overview.items()}
    valid_stories_df['Topic'] = topics
    valid_stories_df['Topic Words'] = valid_stories_df['Topic'].apply(lambda x: ', '.join(topic_words.get(x, [])))

    # After generating topics, merge the results back into the original dataframe
    df.loc[mask_valid_stories, 'Topic'] = valid_stories_df['Topic']
    df.loc[mask_valid_stories, 'Topic Words'] = valid_stories_df['Topic Words']

    # Save the updated dataframe
    output_filename = file_path.replace(".xlsx", "_TopicAnalysis.xlsx")
    df.to_excel(output_filename, index=False)
    print(f"Analysis completed. Output saved to: {output_filename}")

if __name__ == "__main__":
    input_file = input("Enter the path to your Excel file: ")
    process_excel_with_bertopic(input_file)