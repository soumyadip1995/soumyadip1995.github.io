## **Effective Accelerationism**

## **Introduction**

<p align="center">
  <img src="https://pbs.twimg.com/profile_images/1747680364531785728/Ph1LX3zl_400x400.jpg" />
</p>



The main idea for Accelerationism comes from Nick Land. One of my all time favourite quotes from him - "Nothing human makes it out of the near-future". For me, it is both interesting and terrifying. Interesting in the sense that there is a good possibility that evolution of AI will one day catch upto human evolution in terms of IQ, cognitive abilities and therefore, we no longer have to depend on our deteriorating limbic and cortical systems to make technological progress. Our consciousness can merge at will with AI, climb up the Kardashev scale and live till the end of entropy. It is just our minds will be transferred into a different evolving vessel. But then again, this is all wishful thinking. It is also terrifying thinking that we don't get out at all and become a mere control point in the feedback loop of capitalism that Nick land mentions. The human element disappears all together.

Since I am a hopeful optimist, I would like to believe in the former. As of now, I am leaning towards a post modernist point of view when I think about acceleration. An information theory/thermodynamic approach.

A lot of progress in our modern day engineering comes from 20th century physics. Think about how first principles works. As you traverse down the binary tree (i.e., pick something apart - bare bones), you gain more intuition and then you work your way up to the root node. Most of our modern day engineering accomplishments come from thinking from first principles theory. As we gain more intuition, we apply engineering skills to make something more efficient. This is a modernist approach.

Even though I am trained as an engineer, my thought process has evolved in the last few years. I got into AI and information theory right out of college and it has definitely had a profound impact on how I approach solving problems.
Information theory/ thermodynamic approach suggests that you look at a problem as an isolated system. You observe for gain or loss in entropy by depending on the probability of "states". The entropy of a system can be defined as the lack of information or uncertainty in the system. This is a post-modernist approach, where uncertainty of a system can be used to gain more information about the internal configuration of a system, rather than breaking it down by first principles.
This might just be a point of contention for many and you are free to debate.
Very rarely in my 29 years on planet earth, there have been a few days which has been significant in changing the way I think. I would like to list a few.

## **A few significant days**


### Day 0:-

Back  when I was just a lowly undergrad , logic gates were incredibly revolutionary to me. Given two inputs you could perform an operation like AND, OR, NOR and you could get a single output "state".  Also, you could try so many different combinations and get a result.  The applications knew no bounds. If only I had a clue what I would stumble upon..!

### Day 1:-

Compression = Intelligence.

Hutter prize, AIXI had a competition to compress world information . Lossless, not lossy compression was the key to intelligence [Hutter prize](http://prize.hutter1.net/hfaq.htm#about) .  Think about the XOR gate mentioned for a moment,   

${\displaystyle A\cdot {\overline {B}}+{\overline {A}}\cdot B}$



| A  | B | Output |
| ------------- | ------------- |----------|
| 0  | 0 | 0|
| 0  | 1 | 1|
| 1  | 0 | 1 |
| 1  | 1 |  0 |

XOR gate truth table.




Here, if the inputs A,B is compressed to one OUT, the xor gate waits for input A and also for input B and gives an equivalent output thus helping in compressing the path to a single equivalent path.

For a Xor gate NN, we can use quantization as a compression technique.


<p align="center">
  <img src="https://www.researchgate.net/profile/A-Salameh-Walid-2/publication/267839030/figure/fig1/AS:295357566210049@1447430139845/nitial-Weights-for-XOR-2-2-1-problem-Table-63-shows-the-parameters-to-solve-the-XOR.png"/>
</p>

<p style="text-align:center;">XOR-gate weights, source:- Research gate</p>


The error in the XOR problem , varies as a function of a single weight. In larger networks, any single weight has a relatively low contribution to the output. In low dimensional spaces, this can be an issue, but when we consider higher dimensions, the network becomes less likely to get trapped in a global minima and . If the error is low, then the model is likely converge quickly to a non-global minima.




### Day 2:-

Attention mechanisms in Transformer models.

When we observe the closeness of words in the context of what we say,  there lies an assumed relationship between what was just said and the subsequent word. For instance, when we say the word "eating", we can automatically assume that "food" is what is supposed to appear in its context. Hence, it is necessary to focus on the "important"  weights  and how it relates to that particular context [[23]](https://www.google.com/url?q=https%3A%2F%2Flilianweng.github.io%2Fposts%2F2018-06-24-attention%2F). A context vector can therefore be used to estimate how important is the correlation.  


### Day 3:-

E/acc

In my twitter journey , I found the concept of AI acceleration really fascinating and was thus introduced to the concept of Effective Accelerationism. I will try to explain the philosophy in very simple terms.  Let us suppose that all of humanity is in a car and there are two schools of thought. The  Accelerationists and the Existentialists [[10]](https://www.google.com/url?q=https%3A%2F%2Fforum.effectivealtruism.org%2Fposts%2FhkKJF5qkJABRhGEgF%2Fhelp-me-find-the-crux-between-ea-xr-and-progress-studies).

Both the progressives and the accelerationists agree that the trip is good, and that as long as we don't crash, faster would be better. But:

- The Existentialists think that the car is out of control and that we need a better grip on the steering wheel. We should not accelerate until we can steer better, and maybe we should even slow down in order to avoid crashing.
- The  accelerationists thinks we're already slowing down, and so wants to put significant attention into re-accelerating. Sure, we probably need better steering too, but that's secondary.

I can elaborate more on the moral standing of the two and the cost benefits and risks, but that is not within the scope of this piece.

The Existentialist school of thought is closer to the probability of doom,  p(doom).  Meanwhile, the Accelerationist school of thought is aiming to get the optimal probability as soon as possible. p(doom) < p(optimal)

These two schools of thought can be categorized into two questions:-

1) Can AGIs be able to solve the prisoner's dilemma, the accelerationists seem to think yes, I do too. (More on that later)

2) If not, will the AI agents be able to co-operate or play defect until we are able to Align with the optimal strategy [[7]](https://www.google.com/url?q=https%3A%2F%2Fgeohot.github.io%2Fblog%2Fjekyll%2Fupdate%2F2023%2F08%2F16%2Fp-doom.html).

Let me explain,

Game theory suggests that we have two prisoners, A and B. Now, given a choice they can either choose to confess or choose to defect. The strategy here is that both choose to co-operate - in which case both will get a reduced sentencing or they both choose to defect- in which case they both receive a longer sentencing. If both are trying to choose a dominant winning strategy over the other, both will receive a greater sentencing.

As a proponent for e/acc , I would want both the AI agents to align itself into finding the optimal strategy and thus solve the prisoner's dilemma.  But at the same time, also considering the point of view for the existentialists, anytime there is a higher probability for doom (both defecting), the agents need to radically search for solutions that would lead us to get to the optimal strategy.

My solution initially was to set the exploiter up to maximum and have the agents search the environment , because the other agents are other human beings and LLMs...!!. Counterfactual learning or reinforcement learning on the entire world. This will have the search time offset any type of negative growth encountered when we are going through a less than optimal growth phase.  Since then muzero has made great strides in dynamic learning.


### Day 4:

Particles are just field excitations acting as operators on  a 2D Hilbert space.

I would like to have a few more significant days like these.




## **A refresher on Quantum mechanics and Quantum computing**

Let's think about a quantum mechanical system, and how to measure the state of a Quantum system. We can guess a wave function which is evolving with time, in this case a time dependent schrodinger equation according to some energy operator/ hamiltonian.


$\mathrm{i}\hbar{\frac{d}{dt}}\mid\psi(t)\rangle={\hat{H}}\mid\psi(t)\rangle$

Suppose , we have two  energy states, in that case, we can represent the wave function as linear combination of two states. There may not be a definite value  for either 0 or 1 at a given point in time, but we can consider a general superposition of the two states, where $\alpha$ and $\beta$ are two complex numbers.


$\mid\psi\rangle = \alpha\mid\psi _{1}\rangle + \beta{\mid\psi _{2}\rangle}$

and the states are:-



$\mid\psi _{1}\rangle  and  {\mid\psi _{2}\rangle}$






Subsequently for  2 Qubits:

$|\psi⟩ = \alpha\mid 00⟩ + \beta\mid 01⟩ + \gamma\mid 10⟩ + ...$


where, we basically mean, that if we want to know the probability of the 1st particle , we don't need to know what the other particles are doing . When we have separable states, measuring one Qbit says nothing about the next Qbit.

Now, let's think about a classical computer that works on a binary 0 or 1 bits. So, if we wanted to represent N number of bits,  ${0, 1}^N$  will be the word length represented by the processor. Let's say we want to build a system where we take a string of binary bits 0011 and apply an inverter NOT gate to it, we will get 1100. The word length will be 16 bits.

If we want to build a quantum system to represent N number of quantum bits or Qubits, we have to represent $2^N$ complex numbers.  So, if we wanted to simulate the quantum configuration of a system which has N number of bits on a classical system , it presents a frustrating reality  in terms of memory , because we have to represent $2^N$ complex numbers. Richard Feynman in the 1980s had addressed this specific  problem [22]
But, we shouldn't be using quantum computers to simulate classical machine learning, it is just inefficient. Quantum computers are great at representing quantum interference patterns, entanglement, superposition but not much for probabilistic algorithms.


