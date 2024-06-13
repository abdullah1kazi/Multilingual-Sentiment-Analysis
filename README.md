# Multilingual Sentiment Analysis Model

This repository contains a fine-tuned sentiment analysis model based on the BERT base multilingual model and a Streamlit web application to demonstrate its capabilities. The model is trained on a dataset with texts labeled as 'negative', 'neutral', or 'positive'.

## Model Description

The model is built using the Hugging Face Transformers library and the `bert-base-multilingual-uncased` pre-trained model. It is fine-tuned on a sentiment analysis task, allowing it to classify input texts into one of three sentiment categories: negative, neutral, or positive.

## Usage

To use this model, you can load it from the Hugging Face Model Hub using the following code:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_name = "Akazi/bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
texts = ["This is a great product!", "I'm not sure about this."]
predictions = classifier(texts)
print(predictions)
```

This code will load the model and tokenizer from the Hugging Face Model Hub, create a text classification pipeline, and use it to classify the provided texts. The output will be a list of dictionaries, where each dictionary contains the predicted label and score for the corresponding input text.

## Model Performance

The model was evaluated on a held-out test dataset, and the following metrics were obtained:

- **Accuracy**: 0.7061
- **Precision**: 0.7038
- **Recall**: 0.7040
- **F1-score**: 0.7038

Please note that the performance may vary depending on the input data and domain.

## Training Data

The model was trained on a dataset consisting of text samples labeled as 'negative', 'neutral', or 'positive'. The dataset was preprocessed by mapping the textual labels to numeric values (0 for 'negative', 1 for 'neutral', and 2 for 'positive').

## Streamlit Web Application

A Streamlit web application is also included in this repository to provide an interactive interface for using the sentiment analysis model. The app allows users to input texts and get sentiment predictions in real-time.

### Running the Streamlit App Locally

To run the Streamlit app locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multilingual-sentiment-analysis.git
   ```

2. Navigate to the project directory:
   ```bash
   cd multilingual-sentiment-analysis
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

The app will be accessible at `http://localhost:8501`.

### Streamlit Web Application Link

You can also access the deployed Streamlit app here: [Streamlit App](https://bax453.streamlit.app/)

## License

This model is licensed under the [MIT License](LICENSE).

## Credits

This model and web application were developed by Abdullah Kazi, Maanvee Mehrotra, Avantika Goyal (goyalavantika@gmail.com), Shivank Chandra, and Anurag Vedagiri(anurag.vedagiri@gmail.com) using the Hugging Face Transformers library and the BERT base multilingual model.
