# Witten Index


Supersymmetric quantum mechanics possesses a remarkably rigid structure:
the energy spectrum is non-negative, and non-zero energy states come in
boson--fermion pairs. This pairing suggests a natural topological
invariant --- the *Witten index* --- which counts the difference between
the number of bosonic and fermionic ground states and, crucially, does
not change under continuous deformations of the theory. In these notes,
we build up to the definition of the Witten index starting from the
supersymmetry algebra, explain why it is a robust diagnostic for
spontaneous supersymmetry breaking, illustrate it with a concrete
example, and finally sketch its path-integral representation.

# 1. The SUSY Algebra and the Positive-Definite Spectrum 

The supercharges $Q$ and $Q^{\dagger}$ are fermionic operators that
generate supersymmetry transformations, and they satisfy the
anti-commutation relation

$$\\{ Q, Q^{\dagger}\\} = {QQ^{\dagger} + Q^{\dagger}} = 2H$$

Along with the nilpotency conditions $\{Q, Q\} = 0$ and
$\{Q^{\dagger}, Q^{\dagger}\} = 0$, this tells us that the Hamiltonian
is the "square" of the supersymmetry generator, ensuring that all energy
values are non-negative, $E \geq 0$. Meanwhile, the commutation
$[H, Q] = 0$ guarantees that supercharges map energy eigenstates into
other states at the same energy.

Furthermore, we see that the energy $E$ is only zero for states
$|\psi\rangle$ that are annihilated by both the supercharge and its
adjoint. Considering the usual expectation value in a state
$|\psi\rangle$:

$$2\langle \psi | H | \psi \rangle
  = \langle \psi | Q^{\dagger}Q + QQ^{\dagger} | \psi \rangle
  = |Q|\psi\rangle|^{2} + |Q^{\dagger}|\psi\rangle|^{2} \geq 0.$$

This follows directly from the anti-commutation relations. For $E = 0$:

$$Q|\psi\rangle = Q^{\dagger}|\psi\rangle = 0.$$

Already, the idea that we have a positive-definite spectrum is somewhat
surprising. Usually in quantum mechanics, we do not really care about
the overall energy of states, since we can always add a constant to the
Hamiltonian without changing the physics. But that is not the case for
supersymmetric quantum mechanics. The requirement that $E \geq 0$ also
rules out some very familiar quantum mechanical potentials, like
$V = -1/r$ of the hydrogen atom. The potential in supersymmetric QM must
always be positive definite. \[Note:- *The positivity of the spectrum
should not be confused with positivity of the potential itself. In
supersymmetric quantum mechanics the Hamiltonian is positive
semi-definite because it is constructed from the supercharges, even
though the potential may become negative in some regions of
configuration space*\]

Considering the set of states with some fixed energy $E$, we have

$$H|\psi\rangle = E|\psi\rangle.$$

It is simple to check from the supersymmetry algebra that
$[H, Q] = [H, Q^{\dagger}] = 0$, which requires us to also use
$Q^{2} = {Q^{\dagger}}^{2} = 0$. This means that the operators $Q$ and
$Q^{\dagger}$ act within an energy eigenspace. If the energy $E \neq 0$,
we have

$$\\{Q, Q^{\dagger}\\} = 2E \implies \\{c, c^{\dagger}\\} = 1 \quad \text{with} \quad c = \frac{Q}{\sqrt{2E}}.$$

We also have $c^{2} = {c^{\dagger}}^{2} = 0$. This is the same algebra
formed by fermionic creation and annihilation operators. The algebra has
a two-dimensional irreducible representation spanned by the states
$|0\rangle$ and $|1\rangle$ with the properties $c|0\rangle = 0$ and
$c^{\dagger}|0\rangle = |1\rangle$. The implication of this algebra is
that energy states where $E \neq 0$ must come in pairs.

There could be a still bigger degeneracy with several pairs all having
the same energy. But at each level, the number of states must be even.

The only exception is $E = 0$. If such a state exists, it is a ground
state. It is possible to have a lone state; in that case, any such
ground state will obey

$$Q|\psi\rangle = Q^{\dagger}|\psi\rangle = 0.$$

It is also possible to have more than one ground state. But if that is
the case, they are not related by the action of $Q$ and $Q^{\dagger}$.

# 2. The Fermion Number Operator and the $\mathbb{Z}_2$ Grading 

There is a more formal way of viewing the above story. Inspired by the
connection to fermionic creation operators, we can define the "fermion
number operator" $F \equiv c^{\dagger}c$. This obeys

$$[F, Q] = -Q, \quad [F, Q^{\dagger}] = Q^{\dagger}, \quad [F, H] = 0.$$

It is important to note that this operator is well defined only on
states with energy $E \neq 0$, where it acts as $F|0\rangle = 0$ and
$F|1\rangle = |1\rangle$. The $\mathbb{Z}_2$ grading can be extended to
the full Hilbert space as a definition, with the convention that
zero-energy states are assigned to either the bosonic or fermionic
sector depending on whether they are annihilated by $Q$ or $Q^{\dagger}$
in a way consistent with their transformation properties under rotation.

Correspondingly, the Hilbert space decomposes into "bosonic states" with
$F = 0$, and "fermionic states" with $F = 1$:

$$\mathcal{H} = \mathcal{H}_{B} \oplus \mathcal{H}_{F}.$$

This is a $\mathbb{Z}_{2}$ grading of the Hilbert space. 

The $E \neq 0$ pairs have one state in  $\mathcal{H}_{B}$

and one in $\mathcal{H}_{F}$.

One essential piece of terminology here is that, if a ground state with energy $E = 0$ exists, then we say that supersymmetry is **unbroken**.

If the ground state has energy $E > 0$, then we say that supersymmetry is **broken**. 
This is higher-dimensional language where symmetries that do not leave the vacuum invariant are said to be "spontaneously broken" [3].

So given a theory defined with a Hilbert space $\mathcal{H}$, the main
concern is whether there exist theories in $\mathcal{H}$ with
zero-energy states. In supersymmetric theories, the energy $E$ is
greater than or equal to the magnitude of the momentum $|P|$ for any
state. Zero-energy states must therefore have $P = 0$ \[5\].

In the subspace of states with zero momentum, the supersymmetry algebra
is particularly simple. In a basis of properly normalized supersymmetry
(super)charges $Q_{1}, Q_{2}, \ldots, Q_{k}$ ($k = 4$ for supersymmetry
in four dimensions), the algebra is:

$$Q_{1}^{2} = Q_{2}^{2} = \cdots = Q_{k}^{2} = H,$$
$$Q_{i}Q_{j} + Q_{j}Q_{i} = 0, \quad \text{for } i \neq j.$$

These supersymmetry generators map fermions into bosons and bosons into
fermions. What is going to be central here, as mentioned previously in
connection with the fermion number operator, is $\mathrm{Tr}\,(-1)^{F}$
(the parity operator). $F$ here has eigenvalue either $0$ or $1$ \[3\].
A bosonic state $|b\rangle$ satisfies
$e^{2\pi J_z}|b\rangle = |b\rangle$, and a fermionic state $|f\rangle$
satisfies $e^{2\pi J_z}|f\rangle = -|f\rangle$, where $J_z$ is the
infinitesimal rotation generator. A crucial observation here is that the
states of non-zero energy $E$ are paired by the action of the
supercharges. Let $|b\rangle$ be any bosonic state with non-zero energy
$E$. The action of the supercharges on $|b\rangle$ and $|f\rangle$ is:

$$Q^{\dagger}|b\rangle = \sqrt{2E}\,|f\rangle, \quad Q|f\rangle = \sqrt{2E}\,|b\rangle.$$

This can be checked from the supersymmetry algebra and the
creation/annihilation operator formalism above, using $[Q, H] = 0$. All
the other states of non-zero energy are paired in two-dimensional
supermultiplets with this structure. This gives us a much more rigorous
description of why $E \geq 0$. On the other hand, zero-energy states are
not paired this way. With $Q^2 = H$, each state annihilated by $H$ is
also annihilated by $Q$. Any bosonic or fermionic state of zero energy
satisfies $Q|b\rangle = 0$ or $Q|f\rangle = 0$. They form trivial,
one-dimensional supersymmetry multiplets.

# 3. Parameter Variation and the Witten Index

The question now is: what happens when we vary the parameters of the
theory? As we vary the parameters, the states of non-zero energy move
around in energy. They move in boson--fermion pairs. So, as the
parameters are varied, a state with $E > 0$ may move down to $E = 0$. In
this case, $n^{E = 0}_B$ and $n^{E = 0}_F$ both increase by $1$. As the
parameters are varied, some states of zero energy may gain non-zero
energy. However, it is not possible for a single zero-energy state to
acquire a non-zero energy: as soon as it has non-zero energy, it must
have a supersymmetric partner. A pair of states can migrate from $E = 0$
to $E \neq 0$. In this case, both $n^{E = 0}_B$ and $n^{E = 0}_F$
decrease by $1$. The quantity $n^{E = 0}_B - n^{E = 0}_F$ is therefore
invariant under parameter variation, and is useful primarily because of
two properties:

-   It can be calculated reliably.

-   If it is not zero, supersymmetry is not spontaneously broken.

Now, if $n^{E = 0}_B - n^{E = 0}_F \neq 0$, then obviously
$n^{E = 0}_B \neq 0$ or $n^{E = 0}_F \neq 0$ or both. In any case, there
will be some states of zero energy, so supersymmetry is unbroken.

So, what if $n^{E = 0}_B - n^{E = 0}_F = 0$? In this case, we cannot
distinguish between two possibilities:

-   $n^{E =0}_B = n^{E=0}_F = 0$: supersymmetry is broken.

-   $n^{E =0}_B$ and $n^{E=0}_F$ are equal but non-zero: supersymmetry
    is unbroken.

If it is (A), then supersymmetry is spontaneously broken, and there is a
massless Goldstone fermion. If it is (B), there are no Goldstone
fermions, but there are zero-energy fermionic states which can be
interpreted as massless fermions. Hence, the quantity
$n^{E = 0}_B - n^{E = 0}_F$ can be regarded as the trace of the operator
$(-1)^F$ --- this was introduced previously. States of non-zero energy
do not contribute to the trace of $(-1)^F$ because for every bosonic
state of non-zero energy that contributes $+1$ to the trace, there is a
fermionic state of non-zero energy that contributes $-1$ and cancels the
boson contribution.

Therefore, $\mathrm{Tr}\,(-1)^F$ can be evaluated among the zero-energy
states only, and equals $n^{E = 0}_B - n^{E = 0}_F$.

We can thus write,

$$\mathrm{Tr}\,(-1)^F = n^{E = 0}_B - n^{E = 0}_F \simeq  I_{W}$$

This is the **Witten index**. Now, one could regularize
$\mathrm{Tr}\,(-1)^F$ by writing instead,
$\mathrm{Tr}\,(-1)^F \, e^{-\beta H}$; for arbitrary positive $\beta$.
The parameter $\beta$ plays a role analogous to inverse temperature
$\beta = 1/(k_B T)$ in statistical mechanics. The Witten index differs
from the usual statistical mechanics partition function by the signs
$(-1)^F$.

In supersymmetric theories, the Witten index is actually independent of
$\beta$:

$$\frac{d I_W}{d \beta} = 0.$$

Why is $dI_W/d\beta = 0$?    To see this, expand

$\mathrm{Tr}\,(-1)^F e^{-\beta H}$ in the energy eigenbasis.

Each bosonic state of energy $E > 0$ contributes $+e^{-\beta E}$, while its
fermionic partner contributes $-e^{-\beta E}$. These cancel exactly,
leaving only the contributions from the $E = 0$ states, which are
independent of $\beta$. Hence $I_W = n^{E=0}_B - n^{E=0}_F$ does not
depend on $\beta$ at all.

Formally, there is an isomorphism between $\mathcal{H}_{B}$ and

$\mathcal{H}_{F}$ on the $E \neq 0$ subspace due to the boson--fermion
pairing and cancellation. 

In other words, the Witten index really counts
the difference in the number of ground states in each sector.

A comment that is necessary to make is that since $I_W$ does not depend
on $\beta$, one might wonder why we do not just set $\beta \to 0$ and
consider only the trace, $\mathrm{Tr}\,(-1)^F$. We need to remember here
that $\mathrm{Tr}\,(-1)^F$ is an infinite series of $+1$ and $-1$, and
by pairing terms together in various ways, one can obtain any answer one
likes. Including $e^{-\beta H}$ in the definition acts as a regulator,
rendering the trace finite and unambiguous.

# 4. A Concrete Realization: SUSY QM on a Line 

Now, as we noted at the outset, the Hilbert space decomposes into
bosonic and fermionic states:  

$$\mathcal{H} = \mathcal{H}_{B} \oplus \mathcal{H}_{F}.$$


A $\mathbb{Z}_{2}$-graded Hilbert space 

(often called a super Hilbert
space) is a Hilbert space decomposed into even and odd sectors. The
anticommutation relation of the supercharges $\\{Q, Q^{\dagger}\\} = 2H$,
or equivalently 
$H = \frac{1}{2}\\{Q, Q^{\dagger}\\}$, 

is elegant, but we need to build intuition from a concrete example that
realises this algebra \[3\].

Let us consider the quantum mechanics of a particle moving on a line.
The Hilbert space is
$\mathcal{H} = L^2(\mathbb{R}) \otimes \mathbb{C}^2$, where
$L^2(\mathbb{R})$ denotes the normalisable functions on the real line
$\mathbb{R}$ (the usual Hilbert space for a particle on a line), and
$\mathbb{C}^2$ accounts for the internal degree of freedom.

The Hilbert space can then be decomposed into "fermionic" and "bosonic"
sectors:

$$\mathcal{H} = L^2(\mathbb{R}) |0\rangle \oplus L^2(\mathbb{R})|1\rangle = \mathcal{H}_B \oplus \mathcal{H}_F.$$

In this context, $|0\rangle$ and $|1\rangle$ can be thought of as the
spin degree of freedom, with $\mathcal{H}_B$ and $\mathcal{H}_F$ being
the "spin down" and "spin up" components of the Hilbert space. For our
supercharge $Q$, we take:

$$Q = \left(p - ih'(x)\right) \otimes\begin{pmatrix}
0 &  0\\
1 & 0
\end{pmatrix}$$

where $p = -i\,\partial/\partial x = -i\nabla$ is the momentum operator
and $h(x)$ is a real function. We have $Q^2 = 0$ because the
$2 \times 2$ matrix squares to zero. The conjugate gives us,

$$Q^{\dagger} = \left(p + ih'(x)\right) \otimes \begin{pmatrix}
0 & 1 \\
0 & 0
\end{pmatrix}$$

# 5. The $S^1$ Example: $I_W = 0$ with Unbroken SUSY

Recall that if $n^{E = 0}_B - n^{E = 0}_F \neq 0$, SUSY is not
spontaneously broken. However, it is not difficult to exhibit examples
where $I_W = 0$, yet there exists a pair of bosonic and fermionic
$E = 0$ states --- so that SUSY is actually unbroken. This is precisely
the ambiguity (A) vs. (B) discussed earlier. A particularly simple
example arises from a particle moving on an $S^1$ circle of radius $R$.

The supercharge $Q$ and Hamiltonian $H$ take the same form as above and
are characterized by a periodic function $h(x) = h(x + 2\pi R)$. We can
figure out the family of ground states, and the wave functions can be
constructed from them.

The Hamiltonian from the two supercharges was
$H = \frac{1}{2}\\{Q, Q^{\dagger}\\}$, which gives us:

$$H = \frac{1}{2}\\{QQ^{\dagger} + Q^{\dagger}Q\\} = \frac{1}{2}\bigl(p^2 + (h')^2\bigr)\,\mathbb{I} - \frac{1}{2}h'' \sigma_3$$

where $(h')^2$ denotes $\bigl(h'(x)\bigr)^2$. The first factor is the
Hamiltonian for a particle with unit mass moving on a line with
potential $V(x) = \frac{1}{2}(h')^2$. This term comes with a
$2 \times 2$ unit matrix $\mathbb{I}$. The second term contains the
Pauli matrix 

$$\sigma^3 = \begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix}$$

Now, for the ground states we require
$V(x) = 0 \Leftrightarrow h'(x) = 0$. If we Taylor expand around a
critical point $x = x_0$, we have

$$h(x) \simeq h(x_0) + \tfrac{1}{2}\,\omega\,(x - x_{0})^2 + \cdots$$

While the classical ground state energy of a harmonic oscillator
vanishes, quantum mechanically we have $E = \frac{1}{2}|\omega|$. But
the supersymmetric system also gets a contribution from the second term
in the Hamiltonian, which at leading order is
$\Delta E = \pm\frac{1}{2}|\omega|$.

If we take the negative sign, this precisely cancels the contribution
from the harmonic oscillator ground state energy, giving us a total
semi-classical energy $E = 0$. Now, to determine whether an $E = 0$
ground state exists, consider a general state of the form:

$$\Psi(x) = \begin{pmatrix}
\psi(x) \\
\phi(x)
\end{pmatrix}$$

In order to qualify as an $E = 0$ ground state, this must be annihilated
by both supercharges: $Q\Psi = Q^{\dagger}\Psi = 0$. The equations are
straightforward: $\psi(x) = e^{-h}$ and $\phi(x) = e^{+h}$.

If $h \to +\infty$, we must have $\phi(x) = 0$, so the state lives in
$\mathcal{H}_B$. If $h \to -\infty$, we must have $\psi(x) = 0$, so the
state lives in $\mathcal{H}_F$. Now, if $h$ has neither of these
properties (i.e. $h$ is not bounded in either direction in a way that
makes one component normalisable), then there is no $E = 0$ ground state
and SUSY is broken. In this case, the ground state energy is
non-degenerate (there is a unique ground state at $E > 0$ in each
sector, paired by supersymmetry).

Therefore, the wave function we discussed can be constructed from two
linearly independent parameters $\alpha$ and $\beta$ (a two-parameter
family of ground states), where $\alpha, \beta \in \mathbb{C}$:

$$\Psi = \alpha\begin{pmatrix}
e^{-h} \\
0
\end{pmatrix}  + \beta \begin{pmatrix}
0 \\
e^{+h}
\end{pmatrix}$$

Yet, because one ground state lives in $\mathcal{H}_B$ and the other in
$\mathcal{H}_F$, the Witten index of this system is $I_W = 0$. **This is
a concrete illustration of case (B) above**: $I_W = 0$, but SUSY is
unbroken because there exist both a bosonic and a fermionic zero-energy
state. This is precisely the ambiguity that the Witten index alone
cannot resolve.

We need to note that understanding whether SUSY is broken or not in a
given theory is important for hypothetical phenomenological studies. In
fact, this is the main motivation for the Witten index.

# 6. The Path Integral Representation {#the-path-integral-representation .unnumbered}

From the equation $\mathrm{Tr}\,(-1)^F e^{-\beta H} \simeq I_W$ for an
arbitrary $\beta$, we can expand as a functional integral. For
four-dimensional theories, the Witten index $I_W$ can be represented as
a supersymmetric partition function because, as we have already proven,
$E \geq 0$ energy states always exist in pairs. Only the $E = 0$ states
contribute, thus making the index independent of continuous parameters
like the coupling constant and temperature. While evaluating 4D
theories, one can capture this index via path integrals by placing the
theory in a specific curved spacetime background \[1\].

So, to sum it up, the non-zero energy states cancel in pairs under
parameter variation. To see how non-zero energy states cancel in the
trace, consider any eigenstate $|\psi\rangle$ with $E > 0$. Since
$[H, Q] = 0$, the state $Q|\psi\rangle$ (if non-zero) has the same
energy $E$, but the fermion number $F$ changes by one unit because $Q$
is fermionic. Thus, the bosonic and fermionic states at any energy
$E > 0$ come in pairs, related by the action of $Q$, and these pairs
contribute with opposite signs in the trace, cancelling exactly. The
only states that can survive this pairing are zero-energy states that
are annihilated by $Q$, since $Q|0\rangle = 0$ implies that the state
$|0\rangle$ is unpaired. However, as we have seen, boson--fermion pairs
can exist at $E = 0$ for specific instances (such as the $S^1$ example
above), and these also cancel in the index.

The Witten index therefore reduces to the difference of the number of
unpaired bosonic and fermionic zero-energy states, and more importantly,
this difference is an integer which cannot change under any continuous
changes in potential, coupling constants, or even the topology of the
target manifold in field theory generalizations \[1, 2\].

Upon expanding the right side of $\mathrm{Tr}\,(-1)^F e^{-\beta H}$, it
can be represented as a functional integral:

$$I_W \propto \int \prod_{j} dq_j(\tau) \prod_{\alpha} d\psi_{\alpha}(\tau)\, d\overline{\psi}_{\alpha}(\tau)\;\exp\\{ -\int_{0}^{\beta} L_E [q_j(\tau), \psi_{\alpha}(\tau), \overline{\psi}_{\alpha}(\tau)]\, d\tau\\}$$

where $L_E$ is the Euclidean Lagrangian depending on the bosonic ($q_j$)
and fermionic $(\psi_\alpha, \overline{\psi}_{\alpha})$ dynamic
variables in the reduced mechanical system, which satisfies periodic
boundary conditions (PBC) in the imaginary time $\tau$:

$$q_j(\beta) = q_j(0); \quad \psi_\alpha(\beta) = \psi_\alpha(0); \quad \overline{\psi}_{\alpha}(\beta) = \overline{\psi}_{\alpha}(0).$$

For many systems, the higher Fourier harmonics in the expansion,

$$q_j(\tau) = \sum_{n} q_j^{(n)}\, e^{2\pi i n \tau/\beta}, \quad \psi_\alpha(\tau) = \sum_{n} \psi_\alpha^{(n)}\, e^{2\pi i n \tau/\beta},$$

can, for small $\beta$, be effectively integrated out. The functional
integral is then reduced to an ordinary phase-space integral over the
constant (zero-mode) field configurations:

$$I_W = \lim_{\beta \to 0} \int \prod_{j} \frac{dp_j\, dq_j}{2\pi} \prod_{\alpha} d\psi_{\alpha}\, d\overline{\psi}_{\alpha}\; e^{-\beta H(p_j, q_j;\, \psi_\alpha, \overline{\psi}_{\alpha})}$$

Calculating this integral allows one to evaluate the index in the
original theory, or simply \[4\]:

$$\mathrm{Tr}\,(-1)^F e^{-\beta H} = \int_{\mathrm{PBC}} d\phi(t)\, d\psi(t)\;\exp\\{ -S_E(\phi, \psi) \\}$$

where PBC, as mentioned, denotes periodic boundary conditions with
period $\beta$, and $S_E$ is the Euclidean action of the theory.
Expanding eq. (3) yields eq. (1), and eq. (2) is the reduced functional
integral after integrating out the higher Fourier modes.

This path-integral representation is one of the reasons the Witten index
is powerful. It provides a bridge between supersymmetric quantum field
theory and topology, ultimately leading to results such as the
Atiyah--Singer Index Theorem and modern localization techniques.

# References

1.  V. Pestun *et al.*, "Witten index in 4d supersymmetric gauge
    theories," arXiv:2308.1294v4.

2.  D. Tong, "Supersymmetry," lecture notes, University of Cambridge.

3.  D. Tong, "Supersymmetric Quantum Mechanics," lecture notes,
    University of Cambridge.

4.  L. Alvarez-Gaumé, "Supersymmetry and the Atiyah--Singer Index
    Theorem," *Commun. Math. Phys.* **90** (1983) 161.

5.  E. Witten, "Constraints on Supersymmetry Breaking," *Nucl. Phys. B*
    **202** (1982) 253.
