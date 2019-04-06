# Machine Comprehension
Machine(reading) comprehension using SQuAD dataset

I will be using BERT model for QuestionAnswering.

I have taken the code from this [repo](https://github.com/huggingface/pytorch-pretrained-BERT)

## Overivew

This repository comprises implementation of Question Answering using the pretrained BERT model.

- [What is BERT?](# What is BERT?)
- BERT for Question Answering
- What is SQuAD?
- Reading the SQuAD data
- Converting into features
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

![img](./images/base_model.png)

Key steps:

- Question is treated as sequence 1 and Paragraph is treated a sequence 2
- Predicition of Start and End positions of the answer are from the Paragraph tokens.
- To get that predicitions, we add a linear layer which takes each token output and output 2 values, indicating the logits for start and end positions.

The above mentioned linear layer on top BERT is already implemented as [`BertForQuestionAnswering`](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L1130)

