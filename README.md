###Language Translation with deep learning 

### Project purpose

For this project we build a RNN sequence-to-sequence learning in Keras to translate English language to chinese/japanese/korean language.

### Language and Dataset

Since I am in version of english, I choose to translate english to chinese/japanese/korean however our system is pretty general and accepts any other language pair (e.g. english/french). 

For such translation , data is downloaded from Kaggle and Data.world

### What is Sequence-to-sequence learning?

Sequence-to-sequence learning (Seq2Seq) is about training models to convert sequences from one domain to sequences in another domain. 

### Work flow of the model: 

1. We start with input sequences from a domain (e.g. English sentences) and correspding target sequences from another domain
    (e.g. chinese/japanese/korean language sentences).

2. An encoder LSTM turns input sequences to 2 state vectors (we keep the last LSTM state and discard the outputs).

3. A decoder LSTM is trained to turn the target sequences into the same sequence but offset by one timestep in the future.Is uses as initial state the state vectors from the encoder.     Effectively, the decoder learns to generate `targets[t+1...]` given `targets[...t]`, conditioned on the input sequence.
	
4. In inference mode, when we want to decode unknown input sequences, we:
    * Encode the input sequence into state vectors
    * Start with a target sequence of size 1 (just the start-of-sequence character)
    *	Feed the state vectors and 1-char target sequence to the decoder to produce predictions for the next character
    * Sample the next character using these predictions (we simply use argmax).
    * Append the sampled character to the target sequence
    * Repeat until we generate the end-of-sequence character or we hit the character limit.



