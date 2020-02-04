# **Machine Translation in Recurrent Neural Networks**
{: class="table-of-content"}

- TOC {:toc}


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

The output  $y_t$ at time step $t$ is computed using the formula:

$y_{t} = softmax(W^Sh_{t})$

The outputs using the hidden state at the current time step together 
with the respective weight $W(S)$ is calculated. Softmax is used to create a probability vector which will help 
determine the final output.


### **Improvements on the current model**

What are the changes that can be done if improving the current model is in question.

Train different RNN weights for encoding and decoding. In the model above, the same RNN node doing
both encoding and decoding which is clearly not optimal for learning.

At decoder stage rather than just have the previous hidden stage as input, like in the above, 
the last hidden stage of the encoder (call this C in the diagram below) can also be included. 
The last predicted output word vector is also included. This should help the model to know it 
just output a certain word and not to output that word again. This C vector is known as the context vector, 
which we will see when we talk about attention mechanisms later on.

![alt text](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/assets/rnn3.PNG)

*Img:- Encoder-Decoder Recurrent Neural Network Model.
Taken from “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation Img Credits:-leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/assets/rnn3.PNG”*

Now,  the decoder node will  have 3 weight matrices in it ( one of previous hidden state h,
one for last predicted word vector y and one for the last encoder hidden state c) which is multiplied by the corresponding input and then sum up to get the decoder output.

*Also Note:-Important to remember that the weight matrix that is used to multiply the inputs in each step of the encoder is the exact same, it is not different for different time steps.*

## **Statistical Machine Translation**

Statistical machine translation, or SMT for short, is the use of statistical models that
learn to translate text from an input or a source language to a target language by analyzing
existing human translations (known as bilingual text corpora). In contrast to the
Rules Based Machine Translation (RBMT) approach that is usually word based, most mondern SMT systems are phrased based

> "Statistical Machine Translation (SMT) has been the dominant translation paradigm for decades.
Practical implementations of SMT are generally phrase-based systems (PBMT) which translate sequences of words
or phrases where the lengths may differ." — Google’s Neural Machine Translation System: Bridging the Gap
between Human and Machine Translation, 2016.

The idea behind statistical machine translation comes from information theory.
A document is translated according to the probability distribution 

$${\p(e|f)}p(e|f)$$ that a string ${\e}$ in the target language
(for example, English) is the translation of a string ${\f}$ in the source language (for example, German).



![alt text](https://www.researchgate.net/profile/Karan_Singla/publication/279181014/figure/fig2/AS:294381027381252@1447197314391/Basic-Statistical-Machine-Translation-Pipeline.png)

***A Basic Statistical Machine Translation Pipeline** 

*Img Credits:- researchgate.net/profile/Karan_Singla/publication/279181014/figure/fig2/AS:294381027381252@1447197314391/Basic-Statistical-Machine-Translation-Pipeline.png*


Where such corpora are available, good results can be achieved translating similar texts, 
but such corpora are still rare for many language pairs. The first statistical machine translation software was 
CANDIDE from IBM. Google used SYSTRAN (Founded by Dr. Peter Toma in 1986, is one of the oldest Machine Translation Companies) for several years, but switched to a statistical translation method in 2007.

In 2005, Google improved its internal translation capabilities by using approximately 200 billion words
from United Nations materials to train their system.

SMT's biggest downfall includes it being dependent upon huge amounts of parallel texts.

## **BackGround on Neural Machine Translations**

Back in the day, traditional phrase-based translation systems performed their task by breaking up 
sentences into multiple chunks and then translating them phrase-by-phrase. This led to disfluency in the 
translation outputs and was not quite like how humans translate. Humans read the entire source sentence, 
understand its meaning, and then produce a translation. Neural Machine Translation (NMT) mimics this method.

![alt text](https://github.com/tensorflow/nmt/raw/master/nmt/g3doc/img/encdec.jpg)

*Figure:-Encoder-decoder architecture – example of a general approach for NMT. An encoder converts a source sentence into a "meaning" vector which is passed through a decoder to produce a translation.Img taken from :- github.com/tensorflow/nmt*

Specifically, an NMT system first reads the source sentence using an encoder to build a "thought" vector. (context vector)

> " Thought vectors, Geoffrey Hinton explained, work at a higher level by extracting something closer to
actual meaning. The technique works by ascribing each word a set of numbers (or vector) that define its position
in a theoretical “meaning space” or cloud. A sentence can be looked at as a path between these words, which can
in turn be distilled down to its own set of numbers, or thought vector." —theguardian.com/science/2015/may/21/google-a-step-closer-to-developing-machines-with-human-like-intelligence


- Usually an RNN is used for both the encoder and decoder.

- A sequence of numbers represents the sentence meaning; 
A decoder then, processes the sentence vector to emit a translation, as shown in the above Figure. 
This is often referred to as the encoder-decoder architecture. 

- In this manner, NMT addresses the translation problem in the traditional phrase-based approach: it can
capture long-range dependencies in languages, e.g., syntax structures; etc., and produce much more fluent
translations as demonstrated by Google Neural Machine Translation systems.

![alt text](https://1.bp.blogspot.com/-TAEq5oc14jQ/V-qWTeqaA7I/AAAAAAAABPo/IEmOBO6x7nIkzLqomgk_DwVtzvpEtJF1QCLcB/s640/img3.png)

*An example of a translation produced by our system for an input sentence sampled from a news site. Go here for more examples of translations for input sentences sampled randomly from news sites and books. Img Credit:- ai.googleblog.com/2016/09/a-neural-network-for-machine.html*

NMT models vary in terms of their exact architectures. A natural choice for sequential data is the recurrent neural 
network (RNN), used by most NMT models. 

*Also Note:- I am basing this NMT portion on the thesis of Thang Luong , 2016 on Neural Machine Translation and the Tensorflow Tutorial associated with it*

