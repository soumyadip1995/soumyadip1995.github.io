### QFT Formalism

#### Introduction


Fields, where and how do we begin with fields ?.  


When one mentions the word fields, a green pasture over a curvature on a hill , is the first thing to come to mind. I know it was for me when I had heard of it. However, in the context of mathematics and physics, similar curvatures   in space are formed by our brains when we are told to visualize it. Fields in this very context can be thought of as having dynamic values at every point in space and time.
As far as the physical implications of fields goes by virtue of a source object, it can be traced back to the ancient Greeks.  Empedocles, the greek philosopher had tried to provide an explanation of magnetism by introducing the concept of  "effluences" which are emitted from  iron.  Later on during the renaissance period, Descartes mentioned that these "effluences" moved around in closed loops.
In the early years of the nineteenth century, Coulomb and Ampere had  developed mostly Newtonian definitions of electric force and electric currents. But we still needed a solid understanding between the relationship between electric and magnetic fields.
As our scientific understanding began to evolve and our knowledge became more precise, it was Michael Faraday who demonstrated that a change in the magnetic field produced an electric field.
In his experiment, Faraday wrapped two insulated coils of wire around an iron ring and upon passing current through one coil, a momentary magnetic field was induced  in the second coil , which is called a solenoid. This phenomenon is known today as inductance.  Then, as he moved  the smaller coil around through the larger coil, something incredible happened ! . An electric current started to flow.  The Galvanometer showed a voltage !
This relation became one of the foundational equations of Maxwell, who named it Faraday's law. It is one of the four equations of Maxwell's equations.  Faraday's experiment remains as one of the most consequential works for the development of field theory.  

Physicists, think about fields slightly differently. From here on, moving forward we need to think about fields as dynamical objects in space  whose constituents evolve depending on locality, scale, energy, time , invariance etc. They cannot be seen , but their effects are very real.
Over the years, field theory has developed our understanding of how we look at our universe. In our context of Quantum fields, the particles which make up our universe are merely excitations of such fields.  We are going to get to it down the line.


### Concept of locality

The laws of Sir Isaac Newton and Coulomb, was dependent on "action at a distance". The effect a planet(or electron) experiences, changes immediately if a distant object moves. This is experimentally  misleading when we consider many particles, hence another approach was necessary when studying fields, especially when the particles are in close proximity with one another. The concept of locality in its most basic form helps us  in understanding the behavior of nearby objects without referencing distant objects.
Think about this, for a moment.

Two electrons anywhere in the Universe, whatever their origin or history, are observed to have exactly the same properties.  Therefore, there must be an existing electron field everywhere in the universe. The question that arises
So, is the particle fundamental or the field ?

Let's think about it intuitively,

Consider the wave-particle duality of light, it tells us that fundamentally, the properties of both electrons and photons are very similar. But classically speaking they are very different, when we consider charge , mass etc.
Electrons and other matter particles are fundamental to nature whereas Maxwell told us that light arises as a ripple effect  of an  electromagnetic field. We had to figure out a way to treat them equally, since under the right circumstances they can be both wave-like as well as particle-like.  By applying quantum mechanics to the EM field, we find that light isn't just continuous waves but "photons". Hence, the introduction of fields.  So, across all of our universe , the excitation of these "fields" gets tied up into packets of energy called "particles".


And fields exist not just for electrons , but for every other fundamental particle that appears in nature. The existence of such particles depends directly on the treatment of fields.
Hence, the answer is that the field here,  is considered to be primary and the particles are the consequence of the field after quantization. The quantization of an electron field is the electron and the quantization of an electromagnetic field is the photon.

The concept of locality being introduced in the study of quantum fields is in complete contrast with quantum mechanics , which in theory is non-local !. We will see to it.


### Why quantum field theory was formalized ?


In quantum mechanics, we describe a free particle by the Schrodinger Equation.
The wave function of a free particle evolves with respect to energy and time.  
Hence, the quantum mechanical  configuration of a free particle changes.  
One of the drawbacks of the Schrodinger equation is that it is a second order derivative and it will 
lead to negative probability densities. Especially when we try to add relativity to the equation, it breaks 
down.  Since it works for a free particle, that are no rules for the wave function in Quantum mechanics when
we consider many particle states or even different particles and anti-particles occurring at short distances. 
This failure tells us that we need a new regime, when we try to enter the relativistic scale. 
It helps us to treat the states with unverified number of particles (where particle number has not been conserved) and has been formalized as Quantum Field theory.
It is absolutely essential to understand why we cannot directly perform canonical
quantization of particles at relativistic levels just like we can quantize a classical particle and 
why the Schrodinger equation breaks when we try to quantize particles at short distances and high energies 
(fixed number of particles or otherwise) in order to realize why QFT formalism was necessary.


### Relativistic field equations

A field is a dynamic quantity in space-time. It can be denoted by $\phi(x, t)$. The spatial variables x have joined the temporal time
variable t. The concept of “Spacetime” has come into being as the 4-dimensional
aspect upon which field theories are written.

Field theories as a matter of fact have actually have predisposed relativity

From the  hamiltonian formulation of the least action principle, 
we can have the field theory counterpart to the classical formalism, 
where $\mathcal{L}$  is the lagrangian density.


$\delta\int_{time-interval}^{}Ldt = 0$


$\delta\int_{space-time} \mathcal{L}dxdydzdt = 0$

  

  

Considering a real scalar field, we can consider its lagrangian. 
That is the change in action at a distance and thus derive the euler Lagrange equations of motion
(from a classical point of view). 
We can consider several local fields in interaction with one another, 
giving rise to complex scalar fields.
From a real scalar field, we can develop its lagrangian density


$L(t) = \int_{} d^3x \mathcal{L}(\phi, \partial_\mu \phi )$


Therefore, the action will be denoted by S:

$S = \int_{t_{2}}^{t_{1}} dt \int \ d^3 x \mathcal{L} = \int_{} d^4 x  \mathcal{L}$......eq(1)

Recall that in particle mechanics L depends on $q$ and its derivative $q\cdot$.. 
Similarly, here in field theory it depends on $\phi$ and its derivative  $\phi^{.}$


We can determine the equations of motion by the principle of least action. We vary
the path, keeping the end points fixed and require $\delta_S$= 0

We can consider  the change in action at a distance and thus derive the euler lagrange equations of motion.
 


The equations of motion when we expand on eq (1). is given by $\delta S$


$\delta S  =  \int \partial^4x [\frac{\partial\mathcal{L}}{\partial\phi}.\delta\phi  + \frac{\delta\mathcal{L}}{\partial(\partial_\mu\phi))}. \delta(\partial_\mu\phi)]$

Requiring   $\delta_S = 0$, we get the Euler lagrange equations of motions for the path


$\frac{d}{dt}\left(\frac{\partial L}{\partial(\partial_\mu\phi)}\right) - \frac{\partial L}{\partial \phi} = 0$

We can derive our first relativistic field equation from the lagrangian density called the Klein Gordon equation.

From the lagrangian density

$\mathcal{L} = \frac{1}{2}\partial_\mu\phi\partial^\mu\phi - \frac{1}{2}m^2 \phi^2$


(You can derive it from $E = mc^2 + p^2c^4$)
we can determine the klein gordon equation by subsituting the value of the lagrangian density in the euler lagrange equation of motion

We are going to get,

$-m^2\phi - \partial_\mu\phi\partial^\mu\phi = 0$
Therefore,

= $\phi(m^2 + \partial_\mu\phi\partial^\mu\phi) =0$

The quantization of which gives us spin zero particles

 But there are certain disadvantages too with the Klein-Gordon equation like:-
 negative energy solutions, negative probability densities because it is a second order 
 derivatives and only works for spin 0 particles.

Therefore we needed the Dirac equation for the first order derivative, which also works for 
spin 1/2 particles and has positive probability densities.

$\mathcal{L}_{Dirac} = \bar{\psi}(i\gamma^\mu \partial\mu - m)\psi$


where $\psi$ and $\bar{\psi}$ is the fermionic field and its conjugate.
$\gamma$ represents the gamma matrices

The quantization of which gives us spin 1/2 particles like electrons.

The Derivation of the dirac equation is out of scope for this piece.


To construct such field theories we need to prove that these theories remain
invariant under lorentz transformation.


### Interacting fields


Quantum fields, no surprise can also interact with one another, we can add couplings between two fields, think of two tiles coupled by a string. A minimum amount of energy is needed to pull on the string so that it creates a vibration in the fields. When, scientists say that an electron has jumped in energy levels , this is what they mean. The vibrations between the fields can also be transferred. When an electron has dropped down in energy levels, this means that the field is vibrating less.  This lost energy can be transferred to an electromagnetic field , 
creating an excitation called the photon. By virtue of particle-antiparticle annihilation , 
two photons are exchanged , particularly at low energies.  
and thus we can devise a new interaction theory of light and matter called Quantum Electrodynamics or QED , 
this was introduced by Paul Dirac in the 1930s. 
The lagrangian has to remain local as there aren't any terms in the Lagrangian coupling.

For two fields $\phi(x, t)$ and $\phi(y,t)$ ,  $x \neq y$



### Vacuum state

Intuitively, when we talk about space and fields existing in space, it is commonplace to consider the 
ever changing dynamics of fields with regards to symmetry changes, scale, invariance etc. 
When the question is asked about vacuum states in quantum mechanics, we automatically consider it to be empty,
which has been proven to be  experimentally false.

Empty space is filled with fluctuations of quantum fields which  allows virtual
particles to come in and out of existence, particles and antiparticles annihilating with each other.
It is counter-intuitive in the sense that we consider vacuum to be devoid of any activity.  
It is in fact because of such fluctuations , if we held two conducting plates 
incredibly closely to one another, we can experience a measurable attractive or a repulsive force. 
This phenomenon is called the Casimir effect.  At the smallest scale which is the Planck scale , 
the fluctuations in space tie due to virtual particles is known as quantum foam.


### Symmetries and Noether's theorem

Moving on, another essential aspect of QFT formalism is symmetries as well as Noether's theorem.
One of the most beautiful and elegant theorems ever devised in my opinion and one on which gauge 
theory and particle physics is resting upon is Noether's theorem.
Noether's theorem states that for every symmetry out there, there needs to be a conserved quantity.
For example:- for translational symmetry, energy is conserved, for rotational symmetry, 
angular momentum is conserved etc. Similarly, for continuous symmetry of the wave function - 
electric charge is conserved.

From Maxwell's equation we can prove that charge is conserved under Noether's theorem.

$\frac{\partial\rho}{\partial t} + \nabla. j = 0$.... eq(1)


We can show the continuity equation preserves charge under Noether's theorem.

$J = \rho.v$ , where $\rho$ = free electric charge density and $v$ = velocity of charges. 

Considering the four currents,

$J = (c\rho, j_{x}, j_{y}, j_{z})$



Therefore,

$J^{\mu} = (\rho, \overrightarrow j)$

So,



$\frac{\partial}{\partial x}J^{\mu} = \frac{\partial \rho}{\partial t} + \nabla. \overrightarrow J = 0$ from eq(1) and therefore,

$\partial ^{\mu} J ^{\mu} = 0$

This has implications when we try to construct a lagrangian for QED.


Coming to symmetries, in quantum field there are two kinds of symmetry, 
local symmetry and global symmetry.

A global symmetry is a transformation that acts uniformly on all points in spacetime.
Mathematically, a global symmetry transformation is represented by an operator $U$.

U that commutes with the field operators of the theory. For a scalar field theory, 
a global symmetry transformation can be expressed as the transformation of the field by 
using an Unitary operator $U$:


$\psi(x) \rightarrow \psi^{'}(x) = U\psi(x)\phi(x)$

where $\psi(x)$ is the field operator at spacetime point x, and U is the global symmetry operator.
Global symmetries lead to conservation laws through Noether's theorem which states that for every symmetry
there is a conserved quantity, whether it be momentum, angular momentum or energy.

For example, a global U(1) symmetry leads to the conservation of electric charge in QED.

U(1) is the simplest unitary group that represents the symmetry of electromagnetic force :- usually represented by $U^{-ei\alpha}$ where $\alpha$ is an infinitesmal rotation.


Both the Klein-Gordon equation and the Dirac Equation satisfy the continuity equation 
(conservation of charge).

 Overall, the Lagrangian formalism provides a
 powerful framework for describing the dynamics of physical systems, and its
 invariance under various transformations plays a crucial role in understanding
 the underlying symmetries and conservation laws of nature.

### Degrees of freedom

To understand the general principles underlying Quantum field theory  and its consequences requires us to understand how degrees of freedom work.
In standard quantum mechanics, we are taught to take the classical 
degrees of freedom and then promote them to operators acting on a Hilbert space.

For example:- classical degrees of freedom of fields promoted to operators

$\phi(x,t)\rightarrow\hat{\phi}(x,t)$

The rules for the quantization of fields are no different. 
Thus the basic degrees  of freedom in FT are operator valued functions in 
space and time. This means counting an infinite degrees of freedom - 
at least one for every point in space. These infinities come back to bite us at a lot of different instances. 
The potential pitfalls associated with the existence of an infinite number of degrees 
of freedom first showed up in connection with the problem which led to the birth of quantum theory,
that is the ultraviolet catastrophe of blackbody radiation theory. 
One way to handle these degrees of freedom is to use Gauge fixing,  
where we can consider a massless vector field $A_{\mu}$ which transforms under 
the covariance of an arbitrary scalar function $\lambda$

$A_{\mu}\rightarrow A_{\mu} + \partial_\mu\lambda$

We can fix certain gauge conditions, to reduce the degrees of freedom.

$\partial_\mu A^\mu\lambda = 0$

This is known as Lorentz gauge.

A very good example of using gauge fixing to reduce degrees of 
freedom is when we have to quantize an electromagnetic field. 
Upon quantization the photon has two degrees of freedom corresponding to its two polarization states.

### Conclusion


So, in conclusion we can safely say that QFT on a 
fundamental level can be used to develop theories as a whole.
This includes both abelian and non-abelian gauge theories.
On a mathematical level, QFT serves as a tool to understand interactions, 
multi-particle behaviour across scales, symmetries as well as energy levels. 
One of the issues that remains , especially when working with the vacuum is the emergence of infinities.
As a matter of fact, Quantum Field Theory is riddled with infinities.
One such infinity is the ultraviolet divergence - this happens when the space is large.
To reconcile this we often develop boundary conditions and other renormalization techniques. Renormalization can be another piece in itself. Just keep in mind , to not be skeptical while encountering infinities. It's incredibly common.

It is  worth mentioning  that  gravity and the standard model of particle 
physics are incompatible. Although, general relativity which exists as a geometric representation of 
gravity can also exist as a subset of quantum field theory , 
if we can experimentally determine that two particles can exchange a massless spin-2 particle
called the graviton. This has yet to be proven.

So, all in all there a lot of things one can accomplish with QFT in many ways. 
I have only mentioned a few.

Mathematically as well as experimentally,
it is one of the most beautiful and elegant tools we have ever developed to
study the universe and its most basic underlying elements.
70 years after Einstein it remains one of the fundamnental theories that provide us with a 
framework on how a unified theory can be developed. 
String theory being an extension of it. Will we find a way, the search continues...

### References

- David tong - cambridge lectures- qft.
- peskin qft
- Anthony zee - qft
- [The need for gauge field theories](https://soumyadip1995.github.io/2024/11/06/gauge_theory.html)
- qft - Frank wilczek
- wiki - faraday, charge conservation.
