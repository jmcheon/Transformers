# BERT: Pre-training for Deep Bidirectional Transformers for Language Understanding 

<div align="center">

  <img alt="BERT procedures" src="./assets/bert_procedures.png" width=800 height=400/>
  <br/>
  <figcaption>Figure 1: BERT procedures</figcaption>

</div>

</br>

## BERT Objective


1. Multi-Mask Language Model

	- Loss: Cross-entropy loss

2. Next Sentence Prediction

	- Loss: Binary loss

## BERT Input Representation

<div align="center">

  <img alt="BERT input representation" src="./assets/bert_input_representation.png" width=800 height=250/>
  <br/>
  <figcaption>Figure 2: BERT input representation</figcaption>

</div>

The input embeddings are the sum of three embeddings

- **token embeddings**: there are `CLS` token to indicate the beginning of the sentence and `SEP` token to indicate the end of the sentence
	- [CLS]: a special classification token added in front of every input
	- [SEP]: a sepcial separator token
- **segment embeddings**: it allows to indicate whether it sentence `a` or `b`
- **posision embeddings**: it allows to indicate the word's position in the sentence

</br>

# References

- https://arxiv.org/abs/1810.04805