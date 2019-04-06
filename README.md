# Machine Comprehension
Machine(reading) comprehension using SQuAD dataset

I will be using BERT model for QuestionAnswering.

I have taken the code from this [repo](https://github.com/huggingface/pytorch-pretrained-BERT)

## Overivew

This repository comprises implementation of Question Answering using the pretrained BERT model.

- [What is BERT?](#what-is-bert)
- [BERT for Question Answering](#bert-for-question-answering)
- [What is SQuAD?](#what-is-squad)
- [Reading the SQuAD data](#reading-the-squad-data)
- [Converting into features](#converting-into-features)
- Training
- Evaluation
- Fine tuning the answers
- End Notes


### What is BERT?

BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.

You can read more from the following resources: 
- [paper](https://arxiv.org/abs/1810.04805)
- [blog](http://jalammar.github.io/illustrated-bert/)
- [official code repository](https://github.com/google-research/bert#what-is-bert)

Results were all obtained with almost no task-specific neural network architecture design.

### BERT for Question Answering

In the paper, they suggested the architecture for Question Answering Tasks:

![model](./images/base_model.png)

Key steps:

- Question is treated as sequence 1 and Paragraph is treated a sequence 2
- Predicition of Start and End positions of the answer are from the Paragraph tokens.
- To get that predicitions, we add a linear layer which takes each token output and output 2 values, indicating the logits for start and end positions.

The above mentioned linear layer on top BERT is already implemented as [`BertForQuestionAnswering`](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L1130)

### What is SQuAD?

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

The dataset can be dowloaded from [here](https://rajpurkar.github.io/SQuAD-explorer/)

See a sample squad data [here](./samples/squad_sample.md)

### Reading the SQuAD data

Each data point in Squad has

- A entry: A wikipedia article
- Each entry has multiple paragraphs
- Each paragraph multiple questions
- Each question has a single / no answer

Converting the squad data into `SquadExample` instances are done as the following.

![data reading](./images/read_data.png)

Main steps involved in data reading are :

- for each paragraph
    - creating the tokens for each paragraph
    - creating the character to token offset
    - for each question in the paragraph
        - creating the answer text from the given answer offset positions
        - get the start position of the answer
        - get the end position of the answer
        - create the `SquadExample` using the above data

### Converting into features

Once the data is read, the next step would be to convert the `SquadExample` into `InputFeatures` so that it will suitable for the model to process.

Convertion of examples into features are done as the following.

![features](./images/load_examples.png)

Main steps involved in convertion of examples into features are :
