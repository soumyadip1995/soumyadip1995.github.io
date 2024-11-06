### The need for Gauge field theories


(Written:- July, 2024)


## **Introduction**

Let's begin by trying to understand the necessity of studying a field
theory approach to quantum mechanics and as the title suggests why we
need gauge field theories. We will start off by understanding how gauge
theory was formalized, invariance and symmetry and from there we will
work our way up to how we can use gauge theory to construct strong and
weak interactions (the strong force and the weak force). One thing that
needs to be considered while studying field theories , we are trying to
understand processes that occur at very small i.e. (quantum-mechanical)
scales and very large (relativistic) energies. One might ask why we must
study the quantization of fields. Just like the classical formalism
approach,~ why can't we just quantize relativistic particles the way we
quantized non relativistic particles? . The answer to that question is
simple. When we try to quantize relativistic particles at high energies
and shorter distances, the wave equation breaks down. In other words,
the Schrodinger equation works for a free particle , but when you try to
add relativity to it, it breaks down. We also run into other problems
like negative energy solutions, a disorder in first and second order
derivatives, negative probability densities etc.~ Therefore, a 
totally new framework is needed to deal with particles (fixed number of particles
or otherwise) at relativistic levels. This is how quantum field theory
was formalized and As we have been hearing since high school, particles
are just field excitations acting as operators on a 2D Hilbert space. I
am assuming that you as the reader are familiar with concepts like
principle of least action, four vectors, Lagrangian and Hamiltonian
formalism. If not, check out any MIT OCW playlist. 

##  **How gauge theory was formalized**


If we consider the dynamics of a particle as Sir Issac Newton had formulated in classical mechanics, it was carried into perfection by both Lagrange and Hamilton. The space-time distinction still remained vivid. In field theories, if we consider a field $ϕ(x, t)$ the spatial variables x have joined the temporal time variable t. The concept of “Spacetime” has come into being as the 4-dimensional
aspect upon which field theories are written. Field theories as a matter of fact have actually have predisposed relativity


for eg:- the Hamiltonian formulation of the least action principle:- which suggests that-

$\delta\int_{time-interval}^{}Ldt = 0$

  
 The field theory counterpart will be:

$\delta\int_{space-time} \mathcal{L}dxdydzdt = 0$
 
 where $\mathcal{L}$  is the  Lagrangian density.


Considering a real scalar field, we can consider its Lagrangian. That is the change in action at a distance and thus derive the Euler Lagrange equations of motion. We can consider several local fields in interaction with one another, giving rise

to complex scalar fields.  More on that later.

Now, coming to the question of how gauge theory was formalized , if we consider classical electrodynamics and we try to add relativity to it, we get thrown into certain contradictions. In classical electrodynamics , particles are treated as well defined points, unlike in quantum electrodynamics where there is a combination approach. Considering a point charge, the field produced by a point charge is inversely proportional to the square of the distance from the charge. This as we know is called the coulomb's law. Thus the potential of the field $\phi = \frac{e}{R}$. If we have a system of charges, then the field produced by the system is equal to the superposition of the sum of each field individually. Given the formula,

  
$R$  is the distance from the charge $e$.

$\phi_a = \sum_{}^{} \frac{e_b}{R_{ab}}$

where $R_{ab}$  is the distance between the charges $e_a$ , $e_b$

  
But, we know that in relativity , every elementary particle must be considered as point-like. So, at $ R-> 0$, the potential becomes infinity. Thus, according to electrodynamics, the electron would have to have "infinite" self energy and mass. This violates the fundamental framework of classical electrodynamics. Thus special relativity working conjointly   with the principle of superposition throws us into contradictions when it comes to shorter distances.

So, we needed a new "theory of interaction" to design free fields and thus, we came up with gauge theory where the dynamics remain invariant under certain changes/ transformations,

  

$\mathcal{L}  \rightarrow \mathcal{L}^{'}  = \mathcal{L} + \partial_\mu A_\mu$


where  $A_{\mu}$ is the gauge field representing interactions and $A_\mu$ can be anything.

let's talk about invariance for a second:-


## **Invariance**

  
1. Definition:

   - Invariance is a broader concept and refers to the property of remaining unchanged under a specific transformation or set of transformations.

   - In physics, a system or an  equation is said to be invariant under a particular transformation if applying that transformation does not alter the physical laws or properties described by the system or equation.

1. Examples:

  

   - In classical mechanics, Newton's laws are invariant under Galilean transformations.
   - In special relativity, physical laws are invariant under Lorentz transformations.

  
2. Mathematical Representation:

   - Invariance is often expressed mathematically using transformation rules or equations that remain unchanged after a specified transformation.



## **Lagrangian Invariance:**



The Lagrangian formalism in physics is based on the principle of least action, where the dynamics of a system are described by minimizing the action integral. The action ( S ) of a system is defined as the integral of the Lagrangian $( \mathcal{L} \$) over time (\( t \)):

$S = \int_{t_1}^{t_2} \mathcal{L}(q, \dot{q}, t) \, dt$

where:
- $\mathcal{L}$ is the Lagrangian function, which depends on the generalized coordinates $q$, their time derivatives $\dot{q}$ , and possibly time $t$.


- $t_1$ and  $t_2$
 are the initial and final times of the system's motion.

The principle of least action states that the true trajectory of a system between two points in configuration space is the one that minimizes the action integral.

The Lagrangian formalism is invariant under certain transformations, such as:

1. **Time Translation Invariance**:
   If the Lagrangian $\mathcal{L}$   does not depend explicitly on time (\( t \)), i.e., $\frac{\partial \mathcal{L}}{\partial t} = 0 $, then the system is invariant under time translations. This implies that the equations of motion derived from the Lagrangian are unchanged if the system's initial time is shifted by a constant.

2. **Generalized Coordinate Transformations**:
   The Lagrangian formalism is invariant under transformations of the generalized coordinates $q$. If the Lagrangian remains invariant under such transformations, the resulting equations of motion are equivalent. This property is related to the principle of relativity in physics.

3. **Symmetry Transformations**:
   If the Lagrangian remains unchanged under certain symmetry transformations, such as translations, rotations, or gauge transformations, then the resulting equations of motion are invariant under those transformations. This leads to conservation laws, such as conservation of momentum or energy, arising from Noether's theorem.

Mathematically, the invariance of the Lagrangian under a transformation can be expressed as:

$[ \delta S = \int_{t_1}^{t_2} \delta \mathcal{L} \, dt = 0 ]$

where $\delta \mathcal{L}$ represents the variation of the Lagrangian under the transformation. This condition leads to the Euler-Lagrange equations of motion, which govern the dynamics of the system.

Overall, the Lagrangian formalism provides a powerful framework for describing the dynamics of physical systems, and its invariance under various transformations plays a crucial role in understanding the underlying symmetries and conservation laws of nature.  

But, since we are interested in field theory, we are going to take a look at lagrangian invariance from a field theory perspective.



### **Invariance for relativistic field equations**



From a real scalar field, we can develop its lagrangian density


$L(t) = \int_{} d^3x \mathcal{L}(\phi, \partial_\mu \phi )$


Therefore, the action will be:

$S = \int_{t_{2}}^{t_{1}} dt \int \ d^3 x \mathcal{L} = \int_{} d^4 x  \mathcal{L}$......eq(1)

Recall that in particle mechanics L depends on $q$ and  $q\cdot$.. Similarly, here in field theory it depends on $\phi$ and $\phi^{.}$


We can determine the equations of motion by the principle of least action. We vary
the path, keeping the end points fixed and require $\delta_S$= 0




Considering a real scalar field, we can consider its lagrangian. That is the change in action at a distance and thus derive the euler lagrange equations of motion.


The equations of motion when we expand on eq (1).


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

 But there are certain disadvantages like:- negative energy solutions, negative probability densities because it is a second order derivatives and only works for spin 0 particles.

Therefore we needed the Dirac equation for the first order derivative, which also works for spin 1/2 particles and has positive probability densities.


$\mathcal{L}_{\text{Dirac}} = \bar{\psi}(i\gamma^\mu \partial_\mu - m)\psi$



To construct such field theories we need to prove that these theories remain invariant under lorentz transformation.

But, the dirac lagrangian density needs to remain invariant, which we will get to in a moment




The laws of Nature are relativistic, and one of the main motivations to develop quantum field theory is to reconcile quantum mechanics with special relativity. To this end, we
want to construct field theories in which space and time are placed on an equal footing
and the theory is invariant under Lorentz transformations.

$x^{µ }\rightarrow (x^{'})^{\mu}  = \lambda$

where $\lambda$ is a function of space time.



The Lorentz transformations have a representation on the fields. The simplest example is the scalar field which, under the Lorentz transformation .

$\phi(x) \rightarrow \phi^{'}(x) = \phi(\lambda^{-1}x)$

The inverse  appears in the argument because we are dealing with an active transformation in which the field is truly shifted.

## Mathematical formalism for gauge principle


The gauge principle is a fundamental concept in theoretical physics that states that the laws of physics should be invariant under local transformations of a certain group. In the context of gauge theories, such as electromagnetism and the weak and strong nuclear forces, the gauge principle underlies the symmetries and interactions of elementary particles.

### Mathematical Formalism:

1. **Gauge Transformations:**
   Let's consider a complex scalar field $\psi(x)$ as an example. Under a gauge transformation, the field  $\psi(x)$ undergoes a local phase transformation:

   $\psi(x) \rightarrow \psi'(x) = e^{i\alpha(x)} \psi(x)$

   Here, $\alpha(x)$ is an arbitrary real-valued function of spacetime $\(x\)$.

2. **Gauge Invariance:**
   The gauge principle demands that the physical predictions of the theory remain unchanged under such local gauge transformations. Mathematically, this can be expressed as:

   $\mathcal{L}(\psi, \partial_\mu \psi, A_\mu) = \mathcal{L}(\psi', \partial_\mu \psi', A_\mu)$
  where the gauge field   $A_\mu$ representing the interaction.

3. **Introduction of Gauge Field:**
   To ensure gauge invariance, we introduce a gauge field  $A_\mu(x)$
    
   that transforms under gauge transformations such that the gauge-invariant derivative is preserved. This is done by replacing ordinary derivatives with covariant derivatives:

   $D_\mu = \partial_\mu - iqA_\mu $

   where \(q\) is a coupling constant associated with the interaction.

4. **Covariant Derivative:**
   Under a gauge transformation, the gauge field  $A_\mu$  transforms as:

   $A_\mu \rightarrow A_\mu' = A_\mu - \frac{1}{q}\partial_\mu \alpha(x)$

   which can be derived from

   $\partial_\mu\psi  = \frac{1}{e}(\psi(x +  \epsilon. n) -  \psi(x)) $

   (the two fields are subtracted because of different transformations and $n^\mu$  is the direction vector)
   
   where $\epsilon$  is an infinitesmal change which tends to zero. This can  be transformed under an Unitary transformation, which gives us the covariant derivative as,

    $D_\mu\psi ( x ) = d_\mu\psi( x ) - iqA_\mu\psi(x)$
   
   We derive the covariant derivative to get the above  gauge field $A_\mu$ transformation  as the above.



   The covariant derivative  ensures  gauge invariance of the Lagrangian.



5. **Gauge Symmetry Group:**
   The gauge principle is associated with a gauge symmetry group, such as \(U(1)\) for electromagnetism or \(SU(2)\) for the weak force. The choice of gauge group depends on the specific theory being considered.


The gauge principle is a fundamental concept, underlying the formulation of gauge theories and the understanding of fundamental interactions between elementary particles. It ensures the consistency and invariance of physical laws under local transformations, leading to the introduction of gauge fields and the covariant derivatives that preserve gauge invariance.
