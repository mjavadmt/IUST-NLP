# IUST NLP Course
This repository contains NLP course Workshops and Exam questions


## Baseline MiniVAQ
Visual Question Answering(VAQ) is implemented with extracting features of image with *ResNet* and extracting features of question with *sentence transformers(distilbert-base-nli-mean-tokens)* and then concatenating these and with fully connected layer at the end of network will lead to softmax classification between labels.

## MiniVQA Custom Transformer
Custom Transformer can help us to dig through the details of transformer model. Image features were extracted before and saved into to the pickle file and then pretrained embeddings of BERT were used to tokenize and initialize our question embedding. We implemented custom positional encoder, and we could use any number of hidden layers other than 12(default BERT 12) and hidden size other than 768(default BERT 768). supermacy of this implementation compared to previous VQA is that instead of predicting label for each question, it generates answer to each question which is fantastic.

## HF_models
Fine-tuning BERT on a sample dataset and then push the model and tokenizer to HuggingFace.

## HuggingFace
Learning how to work with BERT tokenizer, methods such batch_encode, decode and what are the *CLS and SEP* tokens are. Fine-Tunes imdb dataset with two approaches: 
-  implementing from scrach with for loop on epochs and batch size
-  Built in *trainer* object
then Implemented Sentence Generation on GPT2

## VWSD
This workshop aimed to solve ambiguity for a word that has more than one meaning. for each word that has two meaning we provided some collocation which used for that word and then show corresponding image to that expression which word was used. It was implemented based on AltClip processor. dataset has a expression and 10 images besides it our task is to predict which image belongs to that collocation.

## PyTorch
Learning PyTorch basics, fucntion and computational graph it generates for gradient calculating.

## Numpy
Learning Numpy basics, function and training logistic regression with numpy fully implmenetation for predicting if a word is noun or verb. 

## EXAM Q1_Embedding
Load BERT pretrained embeddings and then gets embedding average based on occurence of each token on our dataset. After that we find the most *k* similar words to a specific word based on Cosine Similary between BERT word embeddings.

## EXAM Q2_Crawling
Crawl StackOverFlow to extract just solution to the question and save them in a list.

## EXAM Q3_A1
Load GloVe pretrained embeddings and then run word analogy like King - Man + Woman = Queen. And also I checked the word vector biases for female and male context.

## EXAM Q6_Hugging_Face
Fine Tune distil-bert-cased on a sample dataset with *trainer* object

## EXAM Q7_Numpy_HuggingFace
Implement Softmax with different temperature. Also generate GPT-2 large sentence with different parameters setting.
