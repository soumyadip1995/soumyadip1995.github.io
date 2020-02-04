## **Machine Translation in Recurrent Neural Networks**
### **What is Sequence To Sequence Modeling**

> "Recurrent networks with recurrent connections between hidden units, 
that read an entire sequence and then produce a single output."— Page 372, Deep Learning, 2016.


By the help of an Recurrent Neural Network, a Seq2seq turns one sequence into another sequence.
Often times a sequence to sequence model can also use LSTMs or GRUs to avoid the problem of vanishing gradient. 

The context for each item (words, letters, time-series) is the output from the previous step. 
The primary components in a seq2seq model are one encoder and one decoder network. 
The encoder turns each item into a corresponding hidden vector containing the item and its context. The decoder reverses the process, turning the vector into an output item, using the previous output as the output context.

![alt text](https://3.bp.blogspot.com/-3Pbj_dvt0Vo/V-qe-Nl6P5I/AAAAAAAABQc/z0_6WtVWtvARtMk0i9_AtLeyyGyV6AI4wCLcB/s1600/nmt-model-fast.gif)

*Image from:- google.github.io*

The algorithm was developed by Google for use in machine translation.
In 2019, Facebook announced its use in solving differential equations.


### **Machine Translation- Definition**

Machine translation is the task of automatically converting source text in one language to text in another language.

> "In a machine translation task, the input already consists of a sequence of symbols in some language,
and the computer program must convert this into a sequence of symbols in another language." — Page 98, Deep Learning, 2016.

If we are given a sequence of text in a source language, there is no single best translation of that
text to another language. This is because of the natural flexibility of human language. 
This is one of the greater challenges in Automatic Machine Translation, perhaps the most difficult in Artificial Intelligence.

### **How the Sequence To Sequence Model Works**

The model consists of 3 parts: 
* Encoder.
* Intermediate (encoder) vector and 
* Decoder.

### **Encoder**

An Encoder is a stack of several recurrent units (often times LSTM or GRU cells are used for better performance) where each accepts a single element of the input sequence, collects information for that element and propagates it forward.

![alt text](https://pytorch.org/tutorials/_images/seq2seq_ts.png)
*Image from:- pytorch seqence to sequence*

In a question-answering problem, the input sequence is a collection of all words from the question. 
Each word is represented as $x_i$ where $i$ is the order of that word.
The hidden states $h_i$ are computed using the formula:


$h_t= f(W^{(hh)}h_{t-1} + W^{(hx)}X_{t})....(1)$

The above formula represents the result of a  recurrent neural network. 
As you can see, we are applying the appropriate weights to the previous hidden state $h_{t-1}$ and the input vector $x_t$.

We first compute the word embeddings for the input one hot word vectors and then 
send the embedded words to translate. The embedded word vector is multiplied by some weight matrix $W(hx)$. 
The previous calculated hidden state (which is the previous output of RNN node) is multiplied by a different
weight matrix W(hh). The results of these 2 multiplications are then added together and non linearity like Relu/tanh is applied.
This is now our hidden state $h$.

*Also Note:- The sentence could be different lengths so it must also have a stop token (e.g a full stop)
which indicates that the end of the sentence has been reached.*



### **Encoder-Vector**

This is the final hidden state produced from the encoder part of the model. It is calculated using the formula (1) above.
This vector aims to encapsulate the information for all input elements in order to help the decoder make accurate predictions.
It acts as the initial hidden state of the decoder part of the model.


### **Decoder**

Once the stop token is reached, the decoder RNN node will begin producing output vectors.
A decoder is a stack of several recurrent units where each predicts an output $y_t$ at a time step $t$.
Each recurrent unit accepts a hidden state from the previous unit and produces and output as well as its own hidden state.
In a question-answering problem, the output sequence is a collection of all words from the answer.
Each word is represented as $y_i$ where $i$ is the order of that word.

Any hidden state $h_i$ is computed using the formula:

$h_t= f(W^{(hh)}h_{t-1})...(2)$

As seen from the Equation (2), the previous hidden state is being used to compute the next one.

The output $y_t$ at time step $t$ is computed using the formula:

$y_{t} = softmax(W^Sh_{t})$

The outputs using the hidden state at the current time step together 
with the respective weight $W(S)$ is calculated. Softmax is used to create a probability vector which will help 
determine the final output.
