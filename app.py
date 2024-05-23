import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
import matplotlib.pyplot as plt
import base64

# Function to test the model
def test_model(source, identifier_or_directory, texts):
    try:
        if source == 'huggingface':
            tokenizer = AutoTokenizer.from_pretrained(identifier_or_directory)
            model = AutoModelForSequenceClassification.from_pretrained(identifier_or_directory)
        elif source == 'local':
            tokenizer = AutoTokenizer.from_pretrained(identifier_or_directory)
            model = AutoModelForSequenceClassification.from_pretrained(identifier_or_directory)
        else:
            raise ValueError("Source must be 'huggingface' or 'local'")
        
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        predictions = classifier(texts)
        return predictions
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

# Function to generate download link for the results
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_analysis_results.csv">Download CSV file</a>'
    return href

# Streamlit app
st.set_page_config(page_title="Multilingual Sentiment Analysis", layout="wide")
st.image("logo.png", width=150)
st.title("Multilingual Sentiment Analysis for Businesses")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Home", "Language Capabilities"])

if page == "Home":
    st.write("""
    Welcome to our Multilingual Sentiment Analysis tool. This application leverages state-of-the-art NLP models to analyze the sentiment of texts in various languages. Ideal for businesses looking to gain insights into customer feedback, social media comments, and more.
    """)

    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    source = st.sidebar.selectbox("Select the model source", ["huggingface", "local"])
    if source == 'huggingface':
        model_id = st.sidebar.text_input("Enter the Hugging Face model identifier", "Akazi/bert-base-multilingual-uncased")
    else:
        model_id = st.sidebar.text_input("Enter the path to the local model directory", "sentiment-model")

    # Instructions
    st.write("## Instructions")
    st.write("""
    1. Select the model source from the sidebar.
    2. If using a Hugging Face model, enter the model identifier. If using a local model, enter the directory path.
    3. Enter the texts you wish to analyze, with each text on a new line.
    4. Click 'Analyze Sentiment' to get the sentiment analysis results.
    5. Download the results as a CSV file if needed.
    6. Alternatively, upload your own CSV file for analysis.
    """)

    # File upload for user CSV
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV:")
        st.write(df)

        # Select column containing text
        text_column = st.selectbox("Select the column containing text data", df.columns)

        # Checkbox to analyze the uploaded data
        if st.checkbox("Analyze Uploaded CSV"):
            with st.spinner("Analyzing..."):
                texts = df[text_column].tolist()
                results = test_model(source, model_id, texts)
                if results:
                    sentiments = [result['label'] for result in results]
                    confidences = [result['score'] for result in results]
                    df['Sentiment'] = sentiments
                    df['Confidence'] = confidences

                    st.write("## Results")
                    st.write(df)

                    # Download link for the updated CSV
                    st.markdown(get_table_download_link(df), unsafe_allow_html=True)

                    # Sentiment distribution
                    st.write("### Sentiment Distribution")
                    sentiment_counts = df['Sentiment'].value_counts()
                    fig, ax = plt.subplots()
                    sentiment_counts.plot(kind='bar', ax=ax)
                    ax.set_title('Sentiment Distribution')
                    ax.set_xlabel('Sentiment')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)

    # Example business use cases
    st.write("## Example Business Use Cases")
    use_case = st.selectbox("Select a business use case", ["Customer Feedback Analysis", "Social Media Monitoring", "Market Research", "Employee Feedback"])

    example_texts = {
        "Customer Feedback Analysis": ["The product is fantastic, really love it!", "Not satisfied with the quality of the service."],
        "Social Media Monitoring": ["#BrandX is amazing! Best purchase ever!", "I'm very disappointed with #BrandX's customer service."],
        "Market Research": ["Competitor Y's new product seems to be getting positive reviews.", "Many customers are complaining about Competitor Z's latest update."],
        "Employee Feedback": ["I feel valued and appreciated at work.", "The workload is too high and stressful."]
    }

    st.write(f"### Example Texts for {use_case}")
    texts = st.text_area("Enter the texts to analyze (one per line)", "\n".join(example_texts[use_case]))
    text_list = texts.split('\n')

    # Analyze button
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing..."):
            results = test_model(source, model_id, text_list)
        
        st.write("## Results")
        if results:
            result_data = []
            for i, result in enumerate(results):
                result_data.append({
                    'Text': text_list[i],
                    'Sentiment': result['label'],
                    'Confidence': f"{result['score']:.2f}"
                })
            
            df_results = pd.DataFrame(result_data)
            st.table(df_results)
            
            # Download link
            st.markdown(get_table_download_link(df_results), unsafe_allow_html=True)
            
            # Sentiment distribution
            st.write("### Sentiment Distribution")
            sentiment_counts = df_results['Sentiment'].value_counts()
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', ax=ax)
            ax.set_title('Sentiment Distribution')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            st.pyplot(fig)

    # Business Applications
    st.write("## Business Applications")
    st.write("""
    Our Multilingual Sentiment Analysis tool can be leveraged in various business contexts, including:
    - **Customer Feedback Analysis:** Understand customer sentiment from reviews, feedback forms, and support tickets to improve products and services.
    - **Social Media Monitoring:** Track brand sentiment on social media platforms to gauge public perception and address issues proactively.
    - **Market Research:** Analyze consumer sentiment towards competitors' products and industry trends.
    - **Employee Feedback:** Assess sentiment in employee surveys and feedback to enhance workplace satisfaction and productivity.
    """)

    # Contact Information
    st.write("## Contact Us")
    st.write("""
    If you have any questions or need support, please contact us at one of the following emails: akazi@ucdavis.edu, mmaanvee@gmail.com, goyalavantika@gmail.com, chandra.shivank@gmail.com.
    """)

    # Footer
    st.write("""
    ---
    This app uses BERT-based models for multilingual sentiment analysis.
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

elif page == "Language Capabilities":
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


