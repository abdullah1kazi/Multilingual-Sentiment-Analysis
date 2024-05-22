import streamlit as st

def app():
    st.title("Language Capabilities")
    st.write("""
    ## Supported Languages
    Our Multilingual Sentiment Analysis model supports a wide range of languages, including but not limited to:
    
    - Arabic
    - Chinese
    - Dutch
    - English
    - French
    - German
    - Italian
    - Japanese
    - Korean
    - Portuguese
    - Russian
    - Spanish
    
    The model leverages the `bert-base-multilingual-uncased` pre-trained model from Hugging Face, which is capable of understanding and processing text in these languages effectively.
    
    ## How It Works
    The sentiment analysis model classifies input texts into one of three sentiment categories:
    
    - **Negative**
    - **Neutral**
    - **Positive**
    
    By analyzing the context and tone of the text, the model can provide valuable insights into the sentiment expressed in multiple languages, making it a powerful tool for global businesses.
    
    ## Example Usage
    Here is an example of how you can use the model to analyze sentiment in different languages:
    
    ```python
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    
    model_name = "Akazi/bert-base-multilingual-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    texts = ["This is a great product!", "C'est un excellent produit!", "Este es un gran producto!"]
    predictions = classifier(texts)
    print(predictions)
    ```
    
    The above code will output the sentiment predictions for the provided texts in English, French, and Spanish, respectively.
    """)

# Add CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)
