from IPython.display import display, HTML
import spacy
from spacy.lang.en import English
from spacy import displacy
spacy_nlp = spacy.load('en_core_web_sm')
eng_tokenizer = English().tokenizer

def convert_to_spacy_format(text, tokens, labels, trailing_whitespace):
    """
    Converts tokenized input data into a format compatible with spaCy's displaCy visualizer.
    
    This function processes a piece of text along with corresponding tokens, entity labels,
    and trailing whitespace information, organizing them into a dictionary structure that
    spaCy can use to visualize named entities. It distinguishes between 'B-' (beginning of an entity)
    and 'I-' (inside an entity) labels to accurately represent entities in the text.
    
    Parameters:
    - text (str): The original text input.
    - tokens (list of str): Tokens extracted from the text.
    - labels (list of str): The BIO (Begin, Inside, Outside) labels for each token.
    - trailing_whitespace (list of str): A list indicating whether a token is followed by a whitespace.
      Each element is a string representation of a boolean ('True' or 'False').
    
    Returns:
    - list of dict: A list containing a single dictionary with keys 'text', 'ents', and 'title',
      where 'ents' is a list of entities represented as dictionaries with 'start', 'end', and 'label' keys.
    """
    ents = []  # To store entity dictionaries
    start = 0  # Position tracker for the start of each token in the text
    
    for i, (token, label, space) in enumerate(zip(tokens, labels, trailing_whitespace)):
        if label.startswith('B-') or label.startswith('I-'):
            #label_type = label[2:]  # Extract entity type from label
            label_type = label
            token_start = text.find(token, start)  # Find the start index of the token in text
            token_end = token_start + len(token)  # Calculate the end index of the token
            
            # If it's a 'B-' label or the first 'I-' label following non-matching or 'O' labels, start a new entity
            if label.startswith('B-') or (label.startswith('I-') and (i == 0 or not labels[i-1].endswith(label_type))):
                ents.append({"start": token_start, "end": token_end, "label": label_type})
            # If it's an 'I-' label continuing an entity, extend the last entity's end index
            elif label.startswith('I-') and ents and ents[-1]["label"] == label_type:
                ents[-1]["end"] = token_end
            
            start = token_end + (1 if space == 'True' else 0)  # Update start position for next token
    
    return [{"text": text, "ents": ents, "title": None}]

def visualize_ner(row):
    """
    Visualizes named entities in a piece of text using spaCy's displaCy visualizer with custom CSS styling.
    
    This function takes a row from a DataFrame containing the text, tokens, labels, and trailing whitespace,
    converts it into a format compatible with spaCy, and then visualizes the named entities in a Jupyter Notebook.
    Custom CSS is applied to modify the appearance of the entity annotations in the visualization.
    
    Parameters:
    - row (pd.Series): A Series object containing the columns 'text', 'tokens', 'labels', and 'trailing_whitespace'
      from a DataFrame. Each of these is expected to be a list except for 'text', which is a string.
    
    Returns:
    - None: This function does not return a value but instead displays the entity visualization in a Jupyter Notebook.
    """
    display_text = row['text'].values[0] 
    display_labels = row['labels'].values[0] 
    trailing_whitespace = row['trailing_whitespace'].values[0] 
    tokens = row['tokens'].values[0] 

    display_text = display_text.replace("\n\n", "\r\n")
    ex = convert_to_spacy_format(display_text, tokens, display_labels, trailing_whitespace)

    custom_css = """
                <style>    
                    /* Customizing entity appearance */
                    .entities {
                        font-size: 11px !important;
                        font-family: Verdana !important;
                        line-height: 1.25 !important;
                        border-radius: 10px !important; /* Rounded corners */
                        background-color: #f9f9f9 !important; /* Very light gray background */
                        padding: 20px 15px !important; /* Adjust padding */
                    }
                    /* Customizing entity appearance */
                    .entity {
                        font-size: 10px !important;
                        padding: 0.2em 0.4em !important;
                        font-family: Verdana !important;
                        font-weight: bold !important;
                        
                    }
                </style>
                """

    options = {"colors": {"B-NAME": "#748CAB", "I-NAME": "#527184", "URL_PERSONAL": "#FFFC31", "ID_NUM": "#E94F37", "B-EMAIL": "#F8B195", "I-EMAIL": "#F8B195", "B-STREET_ADDRESS": "#BDBF09", "I-STREET_ADDRESS": "#CDADE6", "B-PHONE_NUM": "#BD8A3D",
                           "I-PHONE_NUM": "#D96C06", "USERNAME": "#2292A4"}}

    # Inject custom CSS
    display(HTML(custom_css))

    spacy.displacy.render(ex, style="ent", manual=True, jupyter=True, options=options)

def visualize_ner_pred_text(text, text_labels):
    # CSS for visual enhancement in displaCy
    custom_css = """
    <style>    
        /* Customizing entity appearance */
        .entities {
            font-size: 11px !important;
            font-family: Verdana !important;
            line-height: 1.25 !important;
            border-radius: 10px !important; /* Rounded corners */
            background-color: #f9f9f9 !important; /* Very light gray background */
            padding: 20px 15px !important; /* Adjust padding */
        }
        /* Customizing entity appearance */
        .entity {
            font-size: 10px !important;
            padding: 0.2em 0.4em !important;
            font-family: Verdana !important;
            font-weight: bold !important;
            color: black !important; /* Ensuring text in entities is black */
        }
        /* Overriding default styling for non-entity text */
        .entities .entity:not([style]), .entities span:not(.entity) {
            color: black !important; /* Making non-entity text black */
        }
    </style>
    """

    # Entity colors
    options = {"colors": {"B-NAME": "#748CAB", "URL_PERSONAL": "#FFFC31", "ID_NUM": "#E94F37", "B-EMAIL": "#F8B195", "STREET_ADDRESS": "#BDBF09", "I-PHONE_NUM": "#D96C06", "USERNAME": "#2292A4"}}

    # '#57634B', '#D4793A', '#527184'],  
    # 'four_colors': ['#85977D', '#8498A5', '#587F86', '#BD8A3D'],  
    # 'five_colors': ['#57634B', '#D4793A', '#527184', '#CFA802', '#BBB599'], 
    # 'twelve_colors': ['#57634B', '#85977D', '#8498A5', '#527184', '#E9B649', '#BD8A3D', '#D4793A', 
    #                   '#7D1F1D', '#BB6D71', '#BBB599', '#BE477D', '#CDADE6'

    # Display the custom CSS
    display(HTML(custom_css))

    # Load a spaCy model
    nlp = spacy.load('en_core_web_sm')

    # Process the text using spaCy to get a doc
    doc = nlp(text)

    # Set the entities manually
    entities = []
    last_end = 0
    for text_label, entity_label in text_labels:
        if entity_label != "O":
            start = doc.text.find(text_label, last_end)
            if start == -1:  # If the label is not found beyond last end
                continue
            end = start + len(text_label)
            span = doc.char_span(start, end, label=entity_label)
            if span is not None:
                entities.append(span)
                last_end = end  # Update last_end to be the end of the current span

    # Setting the custom entities
    doc.ents = entities

    # Use displaCy to render the document with custom settings
    html = displacy.render(doc, style="ent", options=options)
    display(HTML(html))