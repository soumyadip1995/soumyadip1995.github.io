## **Machine Translation in Recurrent Neural Networks**

1. TOC
{:toc}


> - Updates:- Updated Explanation of Attention Mechanisms.
  - Updates:- Updated Self Attention

## **What is Sequence To Sequence Modeling**

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

## **How the Sequence To Sequence Model Works**

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
A document is translated according to the probability distribution:- 

$${p(e|f)}p(e|f)$$ that a string ${e}$ in the target language
(for example, English) is the translation of a string ${f}$ in the source language (for example, German).



![alt text](https://www.researchgate.net/profile/Karan_Singla/publication/279181014/figure/fig2/AS:294381027381252@1447197314391/Basic-Statistical-Machine-Translation-Pipeline.png)

***A Basic Statistical Machine Translation Pipeline** 

*Img Credits:- researchgate.net/profile/Karan_Singla
/publication/279181014/figure
/fig2/AS:294381027381252@1447197314391/Basic-Statistical-Machine-Translation-Pipeline.png*


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


## **The Attention Mechanism**

To build state-of-the-art neural machine translation systems, we will need more "secret ingredient": 
the attention mechanism, which was first introduced by Bahdanau et al., 2015, then later refined by Luong et al., 2015 and others. 
The key idea of the attention mechanism is to establish direct short-cut connections between the target and the source by paying "attention" to relevant source content as one translates. 

The Attention Mechanism in Deep Learning is based on directing your focus to certain factors when the data is being processed.
To get you up to speed you can check [this blog](https://towardsdatascience.com/word-level-english-to-marathi-neural-machine-translation-using-seq2seq-encoder-decoder-lstm-model-1a913f2dc4a7)


A nice by product of the attention mechanism is an **alignment score** between the source and target sentences
(as shown in the Figure below). This shows the correlation between source and target words.

The relevant equations will be explained later on.

![alt text](https://github.com/tensorflow/nmt/raw/master/nmt/g3doc/img/attention_vis.jpg)

*Figure:-Attention visualization – example of the alignments between source and target sentences.
Image is taken from (Bahdanau et al., 2015).*

In a vanilla seq2seq model, the last source state is passed from the encoder to the decoder when the decoding process starts.
This procedure works well for short and medium-length sentences; however, for long sentences, 
the single fixed-size hidden state becomes an information bottleneck.
So,  instead of discarding all of the hidden states computed in the source RNN, the attention mechanism provides an
approach that allows the decoder to "peek" at them (treating them as a dynamic memory of the source information).
By doing so, the attention mechanism improves the translation of longer sentences. 


**Disadvantage of fixed size context vector**

A critical and apparent disadvantage of fixed-length context vector design is the incapability of remembering long sentences.
Often times it is seen that it has forgotten the first part once it completes processing the whole input.
The attention mechanism was born (Bahdanau et al., 2015) to resolve this problem.


### **Definition of Attention Mechanisms**

Now lets try and understand what exactly does all of the above mean and try to define attention mechanisms. 
Lets say that we have an input sequence of a certain length and we are trying  to output a sequence of a certain length. 

- The encoder is a Bidirectional RNN with a forward hidden state and the backward hidden state.
-  A concatenation of these two states, represents the current encoder state. 
- The idea is to include the preceding word and the following word while annoting one word.(which you will see when we try and understand the equations of attention mechanisms). 

- The context vector is a summation of hidden states of the encoder or the input sequence and the alignment weight (weighted by alignment scores).
- The decoder has hidden states for the output at output positions, say $t$.

- The alignment model assigns a score to the pair based on encoder state and the hidden decoder states. 
(explained in more detail in the equations).

- In Bahdanau’s paper, the alignment score is parametrized by a feed-forward network with a single hidden layer.

- In the score function, tanh is used as the non-linear activation function:

Now, look at the above figure to see the correlation between the source and the target words.



### **Different types of Attention Mechanisms**

An instance of the attention mechanism proposed in (Luong et al., 2015), which has been used in several state-of-the-art systems including open-source toolkits such as OpenNMT.

![alt text](https://opennmt.net/simple-attn.png)

*Img taken from:-//opennmt.net/*

With the help of the attention, the dependencies between source and target sequences are not restricted by the in-between distance. Given the big improvement by attention in machine translation, it soon got extended into the computer vision field (Xu et al. 2015) and people started exploring various other forms of attention mechanisms (Luong, et al., 2015; Britz et al., 2017; Vaswani, et al., 2017).

### **The Encoder and Decoder model**

The current target hidden state is compared with all source states to derive attention weights can be visualized as in Figure:- Attention Visualization- Above).

![alt text](https://github.com/tensorflow/nmt/raw/master/nmt/g3doc/img/attention_mechanism.jpg)

*Figure 5. Attention mechanism – example of an attention-based NMT system as described in (As in Luong et al., 2015).*


As illustrated in Figure 5, the attention computation happens at every decoder time step. It consists of the following stages:

- The current target hidden state is compared with all source states to derive attention weights (can be visualized  in the Attention Visualization picture).
- Based on the attention weights we compute a context vector as the weighted average of the source states.
- Combine the context vector with the current target hidden state to yield the final attention vector.
The attention vector is fed as an input to the next time step (input feeding).

## **Explanation of Attention Mechanisms**


Now, allow me to try and explain what exactly is going on here. I am going to use lilianweng.github.io's [blog post](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#definition) about Attention Mechanism to try and explain the equations in detail.

Let's assume that we have a source sequence $\mathbf{x}$ of length $n$ and an output sequence $\mathbf{y}$ of length $m$.

$$\begin{aligned}
\mathbf{x} &= [x_1, x_2, \dots, x_n] \\
\mathbf{y} &= [y_1, y_2, \dots, y_m]
\end{aligned}$$

Here, $\mathbf{x}$ and $\mathbf{y}$ are vectors. The encoder is a RNN with a forward hidden state $\overrightarrow{\boldsymbol{h}}_s$ and a backward one $\overleftarrow{\boldsymbol{h}}_s$. 

Considering $s$ as the source position where, $s=1,2...,n$
. Concatenation of the two represents the encoder state. **The idea is to include the preceding word and the following word while annoting one word.** So, $\boldsymbol{h}_s$ will be.

 $$\boldsymbol{h}_s = [\overrightarrow{\boldsymbol{h}}_s^\top; \overleftarrow{\boldsymbol{h}}_s^\top]^\top, s=1,\dots,n$$

Now, the decoder has a hidden state $$\boldsymbol{h}_t=f(\boldsymbol{h}_{t-1}, y_{t-1}, \mathbf{c}_t)$$ 

for the output word at position  $t$, where $t=0,1,2...,m$. where the context vector $\mathbf{c}_t$

is a sum of hidden states of the input sequence, 
weighted by alignment scores:



$$\begin{aligned}
\mathbf{c}_t &= \sum_{s=1}^n \alpha_{t,s} \boldsymbol{h}_s & \small{\text{; Context vector for output }y_t}\end{aligned}$$



The alignment model assigns a score $\alpha_{t,s}$ in the above equation to the pair of source at position $s$ and output/target at position $t$, based on how well they match.

**The set of $$\{\alpha_{t, s}\}$$ are weights defining how much of each source hidden state should be considered for each output.**


$$\begin{aligned}\alpha_{t,s} &= \text{align}(y_t, x_s) & \small{\text{; How well two words }y_t\text{ and }x_s\text{ are aligned.}}\\
&= \frac{\exp(\text{score}(\boldsymbol{h}_{t-1}, \boldsymbol{h}_s))}{\sum_{s'=1}^n \exp(\text{score}(\boldsymbol{h}_{t-1}, \boldsymbol{h}_{s'}))} & \small{\text{; Softmax of some predefined alignment score.}}.
\end{aligned}$$



The score function is therefore in the following form, given that tanh is used as the non-linear activation function: (Bahdanau's additive style as seen in the equation below).


![alt text](https://github.com/tensorflow/nmt/raw/master/nmt/g3doc/img/attention_equation_1.jpg)

*Image taken from Github/Tensorflow/nmt*

*Credits go to:- Github/Tensorflow/nmt, lilianweng.github.io*

$$\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \mathbf{v}_a^\top \tanh(\mathbf{W}_a[\boldsymbol{s}_t; \boldsymbol{h}_i])$$


**[Additive/concat]**


where both $\mathbf{v}_a$  and $\mathbf{W}_a$ are weight matrices to be learned in the alignment model.

**So, the Three Equations are:-**

- **Context Vector for Output $y_{t}$**

$$\begin{aligned}
\mathbf{c}_t &= \sum_{s=1}^n \alpha_{t,s} \boldsymbol{h}_s\end{aligned}$$

- **Attention Weights:-**

$$\begin{aligned}\alpha_{t,s} &= \text{align}(y_t, x_s)\
&= \frac{\exp(\text{score}(\boldsymbol{h}_{t-1}, \boldsymbol{h}_s))}{\sum_{s'=1}^n \exp(\text{score}(\boldsymbol{h}_{t-1}, \boldsymbol{h}_{s'}))} .
\end{aligned}$$

- **Attention Vector:-**

$$\begin{aligned}a_{t} &= \text{f}(c_t, h_t)\end{aligned}$$



### **A summary of Popular Attention Mechanisms**


Below is a summary of popular attention mechanisms and corresponding alignment score functions:

- Name:- **Content-base attention**
  - Citation:- 	Graves et.al 2014
  - Alignment score function:-
$$\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \text{cosine}[\boldsymbol{s}_t, \boldsymbol{h}_i]$$

- Name:- **Additive/concat**
  - Citation:- 	Bahdanau2015
  - Alignment Score Function:-
$$\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \mathbf{v}_a^\top \tanh(\mathbf{W}_a[\boldsymbol{s}_t; \boldsymbol{h}_i])$$

- Name:- **Location**
  - Citation:- Luong2015
  - Alignment Score Function:-
$$\alpha_{t,i} = \text{softmax}(\mathbf{W}_a \boldsymbol{s}_t)$$

- Name:- **General**
  - Citation:- 	Luong2015
  - Alignment Score Function:-
$$\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \boldsymbol{s}_t^\top\mathbf{W}_a\boldsymbol{h}_i$$ 

where $$\boldsymbol{W}_a$$  is a trainable weight matrix in the attention layer.


## **PseudoCode for Luong StyleAttention Mechanism**

- Bahdanau attention mechanism proposed only the concat score function
- Luong-style attention uses the current decoder output to compute the alignment vector, whereas
Bahdanau’s uses the output of the previous time step.


In case of Luong , we need to take the dot product of the weight matrix  $\mathbf{W}$ and the hidden state of the Encoder. What layer can do a dot product? It’s the Dense layer:

Below is the code by Machinetalk.org, just the LuongAttention function. See the full code in details in the MachineTalk.org blog on [Neural Machine Translation with Attention Mechanism](https://machinetalk.org/2019/03/29/neural-machine-translation-with-attention-mechanism/)

For Bahdanau's Attention Mechanism visit Tensorflow's NMT [blog](https://www.tensorflow.org/tutorials/text/nmt_with_attention#top_of_page)

*credits=Machinetalk.org*


```
class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(rnn_size)
        ## We need to implement the forward pass.
        ## Note that we have to pass in the encoder’s output this time around. 
        ## The first thing to do is to compute the score.
        ## It’s the dot product of the current decoder’s output..
        ##.. and the output of the Dense layer. 


   def call(self, decoder_output, encoder_output):
        # Dot score: h_t (dot) Wa (dot) h_s
        # encoder_output shape: (batch_size, max_len, rnn_size)
        # decoder_output shape: (batch_size, 1, rnn_size)
        # score will have shape: (batch_size, 1, max_len)
        score = tf.matmul(decoder_output, self.wa(encoder_output),
                transpose_b=True)
  
        ## We then compute the Alignment Vector
        # alignment vector a_t
        alignment = tf.nn.softmax(score, axis=2)

        ## We compute the context vector
        # context vector c_t is the average sum of encoder output
        context = tf.matmul(alignment, encoder_output)

        return context, alignment
```

## **Self Attention**

Self-attention, also known as intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the same sequence. A self-attention module takes in $n$ inputs, and returns $n$ outputs. In layman’s terms, the self-attention mechanism allows the inputs to interact with each other (“self”) and find out who they should pay more attention to (“attention”). The outputs are aggregates of these interactions and attention scores. Self Attention has been shown to be very useful in machine reading or image description generation. 

In Self-Attention if the input is, for example, a sentence, then each word in the sentence needs to undergo Attention computation. The goal is to learn the dependencies between the words in the sentence and use that information to capture the internal structure of the sentence. Self attention has really improved the ability of context derivation. [B.Yang, et al., 2018:](https://www.aclweb.org/anthology/D18-1475.pdf)

![alt text](https://lilianweng.github.io/lil-log/assets/images/cheng2016-fig1.png)

*The current word is in red and the size of the blue shade indicates the activation level. (Image source: Cheng et al., 2016, lilianweng.github.io)*



We'll cover more ground on self attention when we are discussing Transformer networks.

## **Neural Turing Machine**


A Turing machine is a mathematical model of computation that defines an abstract machine.
The Turing Machine was proposed by Alan Turing in 1936.  The machine operates on an infinite memory tape . 
The tape has countless number of "discrete cells" on it, each filled with a symbol: 0, 1 or blank (“ “). The operation head can move left/right on the tape.

![alt text](https://i1.wp.com/makezine.com/wp-content/uploads/2010/03/turingfull560.jpg?resize=1200%2C670&strip=all&ssl=1)

*Img:- Turing machine: a tape + a head that handles the tape. Img source: i1.wp.com/makezine.com/*

 With this model, Turing was able to answer two questions in the negative: 
 (1) Does a machine exist that can determine whether any arbitrary machine on its tape is "circular" (e.g., freezes, or fails to continue its computational task); similarly,
 (2) does a machine exist that can determine whether any arbitrary machine on its tape ever prints a given symbol.

Theoretically a Turing machine can simulate any computer algorithm, irrespective of how complex or expensive the procedure might be.
The infinite memory gives a Turing machine an edge to be mathematically limitless. However, infinite memory is not feasible in real modern computers and then we only consider Turing machine as a mathematical model of computation.

### **NTM**

**Neural Turing Machine** (NTM, Graves, Wayne & Danihelka, 2014) is a model architecture for coupling a neural network with external memory storage. The memory mimics the Turing machine tape and the neural network controls the operation heads to read from or write to the tape. However, the memory in NTM is finite, and thus it probably looks more like a “Neural von Neumann Machine”.

NTM contains two major components, a controller neural network and a memory bank. 
- Controller: is in charge of executing operations on the memory. It can be any type of neural network, feed-forward or recurrent.
-  Memory: stores processed information. It is a matrix of size 
$N * M$
, containing $N$ vector rows and each has *M*
 dimensions.

In one update iteration, the controller processes the input and interacts with the memory bank accordingly to generate output. The interaction is handled by a set of parallel read and write heads.

![alt text](https://lilianweng.github.io/lil-log/assets/images/NTM.png)

*A neural Turing Machine:- Image used from lilianweng.github.io*

*Credits for NTM:- Wikipedia, lilianweng.github.io*


## **Credits/Citations**


- [Statistical Machine Translations- Wikipedia](https://en.wikipedia.org/wiki/Statistical_machine_translation#Language_models)
- [Lilianweng.github.io](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#definition)
- [Neural Machine Translation- Wikipedia](https://en.wikipedia.org/wiki/Neural_machine_translation)
- [NMT- Machine Learning Mastery](https://machinelearningmastery.com/introduction-neural-machine-translation/)
- [Machine Talk- NMT Code (Machinetalk.org)](https://machinetalk.org/2019/03/29/neural-machine-translation-with-attention-mechanism/)
- [Tensorflow NMT with Attention](https://www.tensorflow.org/tutorials/text/nmt_with_attention)
- [Leonardoraujosantos.github.io](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/recurrent_neural_networks/machine-translation-using-rnn.html)
- [Understanding Encoding to Decoding Sequence Modeling- Medium post](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)
- [lmthang thesis on Machine Translation- Github.](https://github.com/lmthang/thesis/blob/master/thesis.pdf)
- [NMT Tensorflow Github](https://github.com/tensorflow/nmt#inference--how-to-generate-translations)
- [NextJournal- Machine Translation](https://nextjournal.com/gkoehler/machine-translation-seq2seq-cpu)
- Neural Machine Translation by jointly learning to align and translate - [ Bahdanau et al., 2015.](https://arxiv.org/pdf/1409.0473.pdf)
- Effective Approaches to Attention-based Neural Machine Translation- [Luong, et al., 2015; ](https://arxiv.org/pdf/1508.04025.pdf)
- Massive Exploration of Neural Machine Translation Architectures- [Britz et al., 2017:](https://arxiv.org/abs/1703.03906)
- Attention Is All You Need- [Vaswani et al., 2017](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
- Long Short-Term Memory-Networks for Machine Reading- [Cheng et al., 2016](https://arxiv.org/pdf/1601.06733.pdf)
- [Neural Turing Machine- Wikipedia](https://en.wikipedia.org/wiki/Neural_Turing_machine)
