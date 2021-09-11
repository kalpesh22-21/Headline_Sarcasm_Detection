* The dataset is collected from two news websites, theonion.com and huffingtonpost.com and Labeled.
* Data Preparation - Pre-processed the text using NLTK,Spacy etc.
* Tokenized the the headline and used GLove embeddings.
* Trained a deep learning Bi-Directional LSTM model.
* Pickle the model
* Creating a Flask API Endpoint to classify desccriptions using the pretrained model. 

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:**  Pandas, numpy, sklearn,nltk, matplotlib, flask, flask_bootstrap, pickle, tensorflow. 

### Class Distribution
<img target="_blank" src="https://github.com/kalpesh22-21/NewsHeadline_Sarcasm_Detection/blob/main/Data%20Distribution.png" width=270>

### Sarcastic Word Cloud
<img target="_blank" src="https://github.com/kalpesh22-21/NewsHeadline_Sarcasm_Detection/blob/main/Sarcastic%20Word%20Cloud.png" width=400>


## Model performance
The Bi-directional Model trained on the corpus delivered an accuracy of 85% on the test data.

## Deployment
In this step, I built a flask API endpoint that was hosted on a local webserver.
The comments were then passed to complete pipeline where the text was cleaned, tokenized and parsed though the model to evaluate the result.

## Front End
<img target="_blank" src="https://github.com/kalpesh22-21/NewsHeadline_Sarcasm_Detection/blob/main/Front%20End.png" width=600>

<img target="_blank" src="https://github.com/kalpesh22-21/NewsHeadline_Sarcasm_Detection/blob/main/Prediction.png" width=600>
