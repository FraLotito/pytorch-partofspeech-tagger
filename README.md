# partofspeech-tagger
Part-of-speech tagger with character-level features using PyTorch.

I extended this [tutorial](http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging) adding character-level embeddings and a new LSTM layer.

#### Requirements
You need Python, PyTorch and Numpy.

#### Usage
```
python post.py
```

#### Example

##### Input
```
The dog ate the apple
```
##### Output
```
The: DET
dog: NN
ate: V
the: DET
apple: NN
```
