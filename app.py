import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Define the test_model function
def test_model(source, identifier_or_directory, texts):
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

# Streamlit app
st.set_page_config(page_title="Multilingual Sentiment Analysis", layout="wide")
st.title("Multilingual Sentiment Analysis for Businesses")
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
""")

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
    result_data = []
    for i, result in enumerate(results):
        result_data.append({
            'Text': text_list[i],
            'Sentiment': result['label'],
            'Confidence': f"{result['score']:.2f}"
        })
    
    st.table(result_data)

# Business Applications
st.write("## Business Applications")
st.write("""
Our Multilingual Sentiment Analysis tool can be leveraged in various business contexts, including:
- **Customer Feedback Analysis:** Understand customer sentiment from reviews, feedback forms, and support tickets to improve products and services.
- **Social Media Monitoring:** Track brand sentiment on social media platforms to gauge public perception and address issues proactively.
- **Market Research:** Analyze consumer sentiment towards competitors' products and industry trends.
- **Employee Feedback:** Assess sentiment in employee surveys and feedback to enhance workplace satisfaction and productivity.
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
