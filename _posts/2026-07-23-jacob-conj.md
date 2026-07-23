## The Jacobian Conjecture - (CounterExample)

**Note:** The following concerns a counterexample to the Jacobian Conjecture announced by Levent Alpöge on [X](https://x.com/__alpoge__/status/2079028340955197566?s=46) on July 19–20, 2026. 
As of July 23, 2026, no formal paper , apart from [this](https://www.ulam.ai/research/jacobian.pdf) has appeared, 
The arithmetic has been checked by multiple independent [sources](https://community.wolfram.com/groups/-/m/t/3766129). I have provided
one hand-checked verification below.
The $n=2$  case remains open. This note presents the announced example and its structure; 
it will be revised as formal publications appear.
This post is also part of a detailed post about how Frontier models have contributed to solving Open Problems across disciplines.

Formulated by German mathematician Ott-Heinrich Keller in 1939, the Jacobian Conjecture stood as an open problem for 87 years. 

**Conjecture 1:-**  Let $F: \mathbb{C}^n \longrightarrow \mathbb{C}^n$ ,
be a polynomial map in $n$ complex variables , whose Jacobian $\det(J_F)$ is a non-zero constant, 
then $F$ is invertible (with polynomial inverse)

The condition here is that the Jacobian is non-zero. 
From fundamental algebra, it is known that once the Jacobian polynomial $\det(J_F)$ is non-zero, 
it is a constant.  This means that the map is locally invertible at every point by the inverse function theorem.
This begs the question that if the map is perfectly regular everywhere, does it guarantee that 
it is globally one-to-one and its inverse is also a polynomial ?.  Let's try and understand the basic definition.

So, let , $F: \mathbb{C}^n \longrightarrow \mathbb{C}^n$ be a polynomial mapping which is determined by $n$ polynomials in $n$ variables:-

$$\begin{align}
F(x_1, x_2,..., x_n) = (f_1(x_1,x_2,....x_n), f_2(x_1,x_2,....x_n), \ldots, f_n(x_1,x_2,....x_n))
\end{align}$$

The Jacobian matrix $J_F$ of this mapping is the $n \times n$ matrix of the first order partial derivatives.  

$$\begin{align}J_F = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}
{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ 
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\frac{\partial f_4}{\partial x_1} & \frac{\partial f_n}{\partial x_2} & \cdots & \frac{\partial f_n}{\partial x_n}
\end{pmatrix}\end{align}$$


The Jacobian determinant is denoted as $\det(J_F)$. 
Now, in calculus, the [Inverse Function Theorem](https://en.wikipedia.org/wiki/Inverse_function_theorem) states 
that if a continuously differentiable function has a Jacobian determinant
which is non-zero at a point $p$, it is then locally  invertible around a point $p$.  
The Jacobian conjecture basically asks  whether or not, the global property holds true.  

This , on the surface might sound simple enough to be true, 
but the global inversion of polynomial maps behaves in surprisingly rigid ways as
compared to smooth or analytic functions. 

The conjecture is trivially true for $n = 1$. But for $n = 2$ , it has remained a 
deceptively open problem despite continuous efforts.  
It was listed as Problem 16 in Stephen Smale's 1998 [list of problems](https://en.wikipedia.org/wiki/Smale%27s_problems#cite_note-1) for the 21st century. 

For $n =1$ it is well understood. For example, suppose:-  $f(x) = x^3 + x$. 
The derivative which acts as a $1D$ Jacobian is $f^{'}(x)$, which is $3x^2 + 1$.  
Notice that for $f^{'}(x)$, it is never zero for any real 
number $x$(it has a minimum value which is greater than $1$). 
However, $f(x)$ is *not invertible* over any real number , 
since it isn't a bijection ($f(x) = y$ can yield three roots for certain $y$). 
Over the complex numbers $\mathbb{C}$, $3x^2 + 1 = 0$ _does_ have roots ($x = \pm \frac{i}{\sqrt{3}}$).
Because the derivative vanishes in $\mathbb{C}$, the 1D version fails the global condition over the complex numbers. 
For the conjecture to hold in higher dimensions,  
the Jacobian determinant must be a _constant_ ( meaning free of variables entirely), 
preventing the determinant from ever vanishing anywhere in $\mathbb{C}^n$.

Mathematicians had tried to prove counterexamples for $n \ge 2$ , but solutions remained elusive. 
The Bass-Connell-Wright (BCW) reduction ---  published in a 1982 [paper](https://projecteuclid.org/journalArticle/Download?urlId=bams%2F1183549636) which had 
significantly simplified the Jacobian Conjecture by proving that to establish the 
conjecture for polynomials of any degree, it is sufficient to prove it only for maps of the form

$$F(x) = x + H(x)$$

where $H(x)$ is a homogeneous vector of degree 3, 
and the Jacobian matrix of $H$ is nilpotent ($J_H^2 = 0$). 
This drastically simplified the structure of polynomials that need to be tested.


It was [recently shown](https://www.newscientist.com/article/2580374/ais-solution-to-87-year-old-riddle-takes-mathematicians-by-surprise/) (using Claude Fable $5$)
that the conjecture is false in three dimensions (and thus in higher dimensions as well). 
The breakthrough came to light when Anthropic researcher and mathematician Levent Alpöge 
used the model to tackle the 87-year-old problem. Rather than generating a massive, impenetrable proof, Fable 5 
constructed a concise polynomial map from three-dimensional 
complex space to itself ($\mathbb{C}^3 \to \mathbb{C}^3$). 

**Theorem 2**:-  (Counterexample to conjecture). There exists a polynomial  $F: \mathbb{C}^3 \longrightarrow \mathbb{C}^3$ which has
non-zero constant Jacobian, but is not invertible.

The example can be stated explicitly From the counter example [paper](https://www.ulam.ai/research/jacobian.pdf). 
Let $(x, y, z)$ denote coordinates on $\mathbb{C}^3$. Define $F = (P, Q, R)$ where:

$$\begin{align}
P &= (1 + xy)^3 z + y^2(1 + xy)(4 + 3xy) \\
Q &= y + 3x(1 + xy)^2 z + 3xy^2(4 + 3xy) \\
R &= 2x - 3x^2y - x^3z
\end{align}$$

and after a brief calculation, it can be found that when the partial derivatives
of this map are arranged into a Jacobian matrix and evaluated, 
its determinant $\det(J_F)$ simplifies to a clean non-zero constant $-2$ everywhere. 
This satisfies Keller's core condition for the conjecture. 
Now, despite having a non-vanishing constant determinant ,
the map is non-injective. Specifically, three distinct output 
points collapse into a single $(-1/4, 0, 0)$, where -  $F(0, 0, -1/4)$ ,  $F(1, -3/2, 13/2)$,  $F(-1, 3/2, 13/2)$. 
Let's understand the calculation:- 

Expanding each component:

$$\begin{align}
P &= x^3y^3z + 3x^2y^4 + 3x^2y^2z + 7xy^3 + 3xyz + 4y^2 + z \\
Q &= 3x^3y^2z + 9x^2y^3 + 6x^2yz + 12xy^2 + 3xz + y \\
R &= -x^3z - 3x^2y + 2x
\end{align}$$

The Jacobian matrix $J_F$ is the $3 \times 3$ matrix of partial derivatives. Computing entry by entry:

$$\begin{align}
\partial_x P &= 6xy^4 + 7y^3 + 3yz(1 + xy)^2 \\
\partial_y P &= 3x^3y^2z + 12x^2y^3 + 6x^2yz + 21xy^2 + 3xz + 8y \\
\partial_z P &= (1 + xy)^3 \\
\partial_x Q &= 9x^2y^2z + 18xy^3 + 12xyz + 12y^2 + 3z \\
\partial_y Q &= 6x^3yz + 27x^2y^2 + 6x^2z + 24xy + 1 \\
\partial_z Q &= 3x(1 + xy)^2 \\
\partial_x R &= -3x^2z - 6xy + 2 \\
\partial_y R &= -3x^2 \\
\partial_z R &= -x^3
\end{align}$$

Computing the determinant and simplifying, one finds:

$$\det(J_F) = -2$$

This is a non-zero constant, independent of $(x, y, z)$. The non-constant terms cancel exactly --- a massive
algebraic cancellation that is the heart of the construction.

To verify this directly, one may substitute any point. For example, at $(0, 0, -\frac{1}{4})$:

$$\det(J_F)\big|_{(0,0,-\frac{1}{4})} = -2$$

At $(1, -\frac{3}{2}, \frac{13}{2})$:

$$\det(J_F)\big|_{(1,-\frac{3}{2},\frac{13}{2})} = -2$$

At $(-1, \frac{3}{2}, \frac{13}{2})$:

$$\det(J_F)\big|_{(-1,\frac{3}{2},\frac{13}{2})} = -2$$

This satisfies Keller's core condition for the [conjecture](https://en.wikipedia.org/wiki/Jacobian_conjecture).
Now, despite having a non-vanishing constant determinant, 
the map is non-injective. Specifically, three distinct input 
points collapse to a single output $(-\frac{1}{4}, 0, 0)$. We verify this by direct substitution.

At $(0, 0, -\frac{1}{4})$:

$$\begin{align}
P &= 0 + 0 + 0 + 0 + 0 + 0 + (-\tfrac{1}{4}) = -\tfrac{1}{4} \\
Q &= 0 + 0 + 0 + 0 + 0 + 0 = 0 \\
R &= 0 - 0 + 0 = 0
\end{align}$$

At $(1, -\frac{3}{2}, \frac{13}{2})$, let $u = 1 + xy = 1 - \frac{3}{2} = -\frac{1}{2}$. Then:

$$\begin{align}
P &= u^3 z + y^2 u (4 + 3xy) \\
  &= (-\tfrac{1}{2})^3(\tfrac{13}{2}) + (\tfrac{9}{4})(-\tfrac{1}{2})(4 - \tfrac{9}{2}) \\
  &= -\tfrac{13}{16} + (\tfrac{9}{4})(-\tfrac{1}{2})(-\tfrac{1}{2}) \\
  &= -\tfrac{13}{16} + \tfrac{9}{16} = -\tfrac{4}{16} = -\tfrac{1}{4} \\
Q &= y + 3xu^2 z + 3xy^2(4 + 3xy) \\
  &= -\tfrac{3}{2} + 3(1)(\tfrac{1}{4})(\tfrac{13}{2}) + 3(1)(\tfrac{9}{4})(-\tfrac{1}{2}) \\
  &= -\tfrac{3}{2} + \tfrac{39}{8} - \tfrac{27}{8} \\
  &= -\tfrac{12}{8} + \tfrac{39}{8} - \tfrac{27}{8} = 0 \\
R &= 2(1) - 3(1)^2(-\tfrac{3}{2}) - (1)^3(\tfrac{13}{2}) \\
  &= 2 + \tfrac{9}{2} - \tfrac{13}{2} \\
  &= \tfrac{4}{2} + \tfrac{9}{2} - \tfrac{13}{2} = 0
\end{align}$$

At $(-1, \frac{3}{2}, \frac{13}{2})$, let $u = 1 + xy = 1 - \frac{3}{2} = -\frac{1}{2}$. Then:

$$\begin{align}
P &= u^3 z + y^2 u (4 + 3xy) \\
  &= (-\tfrac{1}{2})^3(\tfrac{13}{2}) + (\tfrac{9}{4})(-\tfrac{1}{2})(4 - \tfrac{9}{2}) \\
  &= -\tfrac{13}{16} + \tfrac{9}{16} = -\tfrac{1}{4} \\
Q &= \tfrac{3}{2} + 3(-1)(\tfrac{1}{4})(\tfrac{13}{2}) + 3(-1)(\tfrac{9}{4})(-\tfrac{1}{2}) \\
  &= \tfrac{3}{2} - \tfrac{39}{8} + \tfrac{27}{8} \\
  &= \tfrac{12}{8} - \tfrac{39}{8} + \tfrac{27}{8} = 0 \\
R &= 2(-1) - 3(-1)^2(\tfrac{3}{2}) - (-1)^3(\tfrac{13}{2}) \\
  &= -2 - \tfrac{9}{2} + \tfrac{13}{2} \\
  &= -\tfrac{4}{2} - \tfrac{9}{2} + \tfrac{13}{2} = 0
\end{align}$$

Thus:

$$F(0, 0, -\tfrac{1}{4}) = F(1, -\tfrac{3}{2}, \tfrac{13}{2}) = F(-1, \tfrac{3}{2}, \tfrac{13}{2}) = (-\tfrac{1}{4}, 0, 0)$$

The polynomial $F$ has seven degrees of freedom (upon expanding, the first part $((1 + z_1z_2)^3)z_3$ gives terms up to degree 7, and so on for each part), 
so a priori the Jacobian $\det(J_F)$ ought to be a polynomial in three variables of degree as large as $3 \times 6 = 18$, 
so the fact that all non-constant coefficients of this polynomial vanish looks like a massive cancellation involving 

$$\binom{18+3}{3} - 1 = \binom{21}{3} - 1 = \frac{21 \times 20 \times 19}{3 \times 2 \times 1} - 1 = 1330 - 1 = 1329$$

In the context of polynomial maps, this formula $\binom{n+d}{d} - 1$ calculates the total number of **possible non-constant monomial terms** 
of degree up to $d$ in a system with $n$ variables. This is much larger than $\binom{7+3}{3} = 120$.  
The example has since been [explained in more geometric terms](https://x.com/davikrehalt/status/2079175065695035442).

Now, because three separate input vectors given by $F$ yield a single identical output vector, 
the function cannot be run backward uniquely. This destroys the central claim of the conjecture, 
as a non-injective map cannot possess an inverse. Through standard algebraic padding 
(adding independent, untouched variables), this 3D counterexample 
automatically invalidates the conjecture for all higher dimensions as well. 
In short, since the map sends multiple inputs to the same output, 
it cannot have an inverse---polynomial or otherwise. 
This directly violates the conjecture's conclusions.

Despite this massive development, the historical two-variable version ($n = 2$) remains strictly open, as low-dimensional
constraints prevent 3D counterexamples from being compressed into a plane. 
The counterexample isn't just a random assortment of numbers; it 
is constructed using algebraic geometry and symmetry groups (specifically mapping spaces of homogeneous polynomials 
and fiber products, such as symmetric products of projective lines like $\operatorname{Sym}^3(\mathbb{P}^1)$). 
This is exactly what Terrance Tao elaborates upon in his [blog post](https://terrytao.wordpress.com/2026/07/21/a-digestion-of-the-jacobian-conjecture-counterexample/). 
For $120$ degrees of freedom for a seven degree polynomial , 
Tao  highlights how  a brute force search over the entirety 
of the search space would be an exercise in futility, 
which was the probable reason why it had remained an open challenge for such a long time.

Navigating a landscape where you have to simultaneously satisfy 
over a thousand massive algebraic constraints while avoiding 
triviality is why human mathematicians spent decades missing it. 
Fable had successfully  navigated this high-dimensional 
combinatorial maze to land on a compact, working solution. 

So a few takeaway points from this post will be:-

- The **general Jacobian Conjecture** (for all dimensions $n \ge 3$ ) appears to be **false**.A counterexample in 3 variables immediately extends to any higher dimension by adding untouched coordinates.

- The **2-variable case** remains open and unaffected.

- Alpöge credits "Fable" (Claude Fable, Anthropic's AI model) with the discovery, but the full prompt history, model version, and extent of human guidance have not been publicly released. This is credible first-party testimony but not independently auditable.

- **Peer review**: No formal paper has been published or peer-reviewed as of this writing on  July 23, 2026. However, many mathematicians have independently verified the algebra on social media and in blog posts.

- **Formal verification**: A Lean formalization of the counterexample has been written and submitted to Google's "Formal Conjectures" [repository](https://github.com/google-deepmind/formal-conjectures/pull/4474).
