#  Level 2 vs Level 4 Autonomous Driving Systems (Featuring :- Tesla, Waymo):- Perception Systems and Neural Network vs Tesla AutoPilot Part 2

![alt_text](https://miro.medium.com/max/1000/1*paN5hOlhUTKwyFYAkLdxRg.gif)

*Image from Perception Projects for Self driving car [Source](https://miro.medium.com/max/1000/1*paN5hOlhUTKwyFYAkLdxRg.gif)*


## **Table of Contents**

1. TOC
{:toc}

> - Updates:- Feb 25,2020.
> - Updates:- Credits and Citations,Feb 26, 2020

## **Perception Based Systems for Level 2**

In the previous Blog post we looked into the Various levels of autonomous driving put forth by the SAE. We, also saw a few examples that cover level 2 as well as an overview of the Tesla Autopilot.

Now, let us look into something known as perception based systems. How an autonomous vehicle sees the external world and its surrounding environment depends on Perception. Perception in an autonomous driving system is integral for both Level 2 and Level 4  systems. The purpose of Perceptual Systems can range in anything from Driving Perception i.e, Object detection, driving scene, Trajectory Generation to Localization and Mapping. Any system that improves the robustness of the overall perception of an autonomous vehicle , can be consituted as a perception system or a perception model. Hence, the Tesla Autopilot as we had talked about in the previous Blog Post, it can also be constituted as A perception system.


 But, since we are still continuing with Level 2, the rest (Level 4) will be covered later on. 

One of the things mentioned in the previous Blog post is that for Tesla (Level 2) , Deep Learning is the cake and not the icing on the cake. Hence, the main focus for Perception Systems for Level 2 is going to be Deep Learning. 

Around October of 2019, Tesla acquired a company called DeepScale, which specializes in the development of Perceptual systems for Autonomous vehicles. 

> "Tesla has reportedly acquired the four-year-old startup DeepScale, which provides interesting insight into the state of Artificial Intelligence in assisted and automated driving.  Operating on $18M in venture funding, DeepScale described themselves as developers of perceptual systems for semi-autonomous and autonomous vehicles, focusing on low-wattage processors used in mass-market automotive crash avoidance systems to power more accurate perception.  This is an important niche in the intelligent vehicle eco-system"- [Forbes, 2019](https://www.forbes.com/sites/richardbishop1/2019/10/04/what-teslas-grab-of-deepscale-is-all-about/#28dde90b1d3c)"

Visit [DeepScale](http://deepscale.ai/)

![alt_text](https://i.ytimg.com/vi/MzHv7i5L65w/maxresdefault.jpg)

*Image from Youtube:- Tesla acquires Deepscale*

### **Perception Systems using Deep Neural Networks**

**Deep Neural Networks for Vision and Perception**

One of the main challenges for developing a more robust perception system is to take into consideration, the amount of power neural networks are going to use while making real time predictions.

 Although , the rise of Neural Networks has significantly improved semantic segmentation (*Semantic segmentation refers to the process of linking each pixel in an image to a class label. These labels could include a person, car, flower, piece of furniture, etc., just to mention a few.*), 3D reconstruction, it still needs a lot of power. (8 GPUs, 250 watts each and then about 500 watts of CPUs, hard drives etc)

According to Tesla's estimation of how much power they use on the highway and a Model S if you have five kilowatts of compute in the trunk running you  actually diminish your 
electric vehicle range by at least 25 percent. 

So, there is a need to Build smaller DNNs with fewer parametres that are easily updatable in real time. 

### **Example**

One example of a model that uses less parametres that can easily fit into a computer memory and can be easily transmitted over a network is SqueezeNet. SqueezeNet was developed by researchers at DeepScale, University of California, Berkeley, and Stanford University.

[SqueezeNet paper](https://arxiv.org/abs/1602.07360)

>"For a given accuracy level, it is typically possible to identify multiple CNN architectures that achieve that accuracy level. With
equivalent accuracy, smaller CNN architectures offer at least three advantages: (1) Smaller CNNs require less communication across servers during distributed training. (2) Smaller CNNs require less bandwidth to export a new model from the cloud to an autonomous car. (3) Smaller CNNs are more feasible to deploy on FPGAs and other hardware with limited memory. To provide all of these advantages,
we propose a small CNN architecture called SqueezeNet. SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters." - From the abstract of the SqueezeNet Paper.

**A few tips to build smaller DNNs**

- Replace Fully Connected Layers with Convolutions. Because in models like Alexnet or VGG-Net, the majority of parametres are in the FC layer
- Kernel Reduction:- Figuring out ways of Reducing the height and Width of the Filters, while retaining information.


#### **Deep Learning based Perception
Driving Scene Understanding using vision Techniques**

An autonomous car should be able to detect trafﬁc participants and drivable areas, particularly in urban areas where a wide variety of object appearances and occlusions may appear. Deep learning based perception, in particular Convolutional Neural Networks (CNNs), became the standard in Object Detection. AlexNet shifted the focus towards object Detection.

1) Image based Object Detection

State-of-the-art methods that can be applied to Autonomous Driving Systems rely generally on DCNNs, there currently exist a clear distinction between them:

1) Single stage detection frameworks that use a single network to produce object detection locations . SSD (single shot multibox detector) Wei Liu et.al,2015 and YOLO (You Only Look Once) Redmon et.al, 2016 are mainly used.

2) Double stage Object detection frameworks:- R-CNN Ross Girshick et.al, 2013 , Faster R-CNN Shaoqing Ren et.al, 2015 are mainly used.

In general, single stage detectors do not provide the same performances as double stage detectors, but are signiﬁcantly faster.

2) Semantic and Instance Segmentation

"Instance Segmentation: Identify each object instance of each pixel for every known object within an image"

Mask-RCNN can be applied for instance segmentation.

The advantages of using Mask-RCNN are:-

It is simple, flexible, and a general framework for object instance segmentation.
Efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance.
"Driving scene understanding can also be achieved using semantic segmentation, representing the categorical labeling of each pixelin an image. In the autonomous driving context,pixels can be marked with categorical labels representing drivable area,pedestrians,trafﬁc participants,buildings,etc. It is one of the high-level tasks that paves the way towards complete scene understanding, being used in applications such as autonomous driving"- from section 4.2, A survey of Deep Learning Techniques for Autonomous Driving

So, we talked about The Tesla Autopilot in the Previous Blog Post. As you can see the above ideas corressponds to Level 2.

In the next Section , we will see what happens in a competition between the Tesla AutoPilot and a Neural Network..!!


### **Neural Network vs the Tesla Autopilot**

This Section is Based on the Work of Lex Fridman:- [Lex Fridman. et.al,2017](https://arxiv.org/abs/1710.04459)

One of the primary challenges of an AI system is to achieve greater perfection when making Life critical decisions is in question. The challenge lies in reducing the overall system error.

In the paper, the authors consider the paradigms of a black box AI system that can make life critical decisions. 

> "We propose an “arguing machines” framework that pairs the primary AI system with a secondary one that is independently trained to perform the same task....We demonstrate this system in two applications: (1) an illustrative example of image classification and
(2) on large-scale real-world semi-autonomous driving data"- from the abstract of the paper.


The primary idea of the “arguing machines” framework is that it adds a secondary system to a primary “black box” AI system which makes life-critical decisions and uses disagreement between the two primary and secondary systems as a signal to seek human supervision. 

For this section, we are going to be looking into the 2nd application as suggested in the paper.- large scale real-world semi autonomous driving data.

Before, diving into the details let's take a look into this figure, this should provide clarity

![alt text](https://hcai.mit.edu/wordpress/wp-content/uploads/2018/09/arguing_machines_1200.png)

*Figure 1:- The arguing Machines frmaework that adds a secondary system to a primary blackbox system. The authors demonstrate that this can
be a powerful way to reduce overall system error. Image from the paper*


#### **Methodology**

The study at MIT for semi- autonomous vehicles includes perception on both sides of the windshield. Inward facing cameras for driver face state etc and outward facing cameras for driving scene perception and other perception controls. Now, in this paper the authors device a technique to evaluate something known as the disagreement function. 

- The main perception control system is the Tesla Autopilot. 

- On the dashboard of the vehicle (Tesla S) the authors set up a monocular camera (Hardware 1), an NVIDIA Jetson TX2 which is equipped with an end to end neural network. The purpose of this is to capture sequence of images from the outside environment in real time as one drives and produce steering commands. This is the 2nd Perception control.

The camera feeds the video stream as one drives in real time to the Jetson TX2 and the neural network predicts the Steering commands.

[insert figure 8 from paper]

So, now we have two perception control systems, the Tesla AutoPilot vs the Neural Network. We will see the Autopliot arguing against the Neural Network.

#### **How the Disagreement is detected**

[insert picture from screenshot ]

There is an LCD display inside the car that shows the steering commands from both the control systems, the temporal difference input to the neural network, and (in red text) a notice to the driver when a disagreement is detected. If you Look closely there are two lines. A pink line and a cyan line. If the two control systems disagree, the LCD display indicates that there is a disagreement. The magnitude of the disagreement varies with increasing or decreasing levels of disagreement between the two systems.

The pink line is the steering commands from the neural network and the cyan line is the Tesla Autopilot. 


#### **Why is the disagreement interesting**


Watch as Prof Fridman takes the Tesla S model down the Highway and explains the reasons behind the disagreements in the video below.

[insert Fridman Video]



The reason why the disagreement is interesting can be summed up in two parts:-

- manual interference, and
- vision perpection.
- driving scene interpretation

1) **In the case of manual interference**, if one tries to operate the vehicle i.e, try and manoveur the vehicle which is already on Autopilot, the neural network disagrees, which means that the driver's steering commands are divergent to the neural network's steering commands and hence the driver should pay extra attention to his own steering decision. This happens around 3:42 and 6:33 in the video where Prof Fridman tries to take control of the vehicle on Autopilot, the neural network disagrees with his steering decision and the Disgreement is displayed on the LCD display in the front. 

2) **Vision Perception**:- From a computer Vision Standpoint, it is interesting because a divergence between the steering commands of the Neural Network and the Tesla Autopilot means that there is a chance of detecting edge cases on which the neural networks can be further trained on in order to improve the secondary system  so as to reduce overall system error (Look at the Figure 1). (Prof Fridman explains this around the 5:00 mark). 

3) **Driving scene interpretation**:- Let's say that from a driving scene perception the neural network and the AutoPilot are disgreeing a lot or disagreeing at a greater magnitude , then the indication might be that it is the driver's turn to retreive control from the Autopilot and take charge . (Prof Fridman explains this around the 4:00 mark) 


**Dataset**:- For the 2nd application, the video stream is fed to the Jetson TX2 in real time and steering commands are predicted by the end to end Neural Network.

> "We perform two evaluations in our application of arguing
machines to semi-autonomous driving. First, we evaluate
the ability of the end-to-end network to predict steering
angles commensurate with real-world steering angles that
were used to keep the car in its lane. For this, we use distinct periods of automated lane-keeping during Autopilot
engagement as the training and evaluation datasets. Second,
we evaluate the ability of an argument arbitrator (termed
“disagreement function”) to estimate, based on a short time
window, the likelihood that a transfer of control is initiated,
whether by the human driver (termed “human-initiated”) or the Autopilot system itself (termed “machine-initiated”). We
have 6,500 total disengagements in our dataset. All disengagements (whether human-initiated or machine-initiated)
are considered to be representative of cases where the visual characteristics of the scene (e.g., poor lane markings,
complex lane mergers, light variations) were better handled
by a human operator. Therefore, we chose to evaluate the
disagreement function by its ability to predict these disengagements, which it is able to do with 90.4% accuracy" - From the paper, Section 4.Arguing Machines for Semi Autonomus driving.

[insert screenshot of graph from fridman video]

This result shows that there is a lot of signal in this disagreement even when the disagreement
is based on a simple threshold.-[Arguing Machines](https://hcai.mit.edu/arguing-machines/)

## Credits/Citations

- [DeepScale Video- Perception Systems for Autonomous Vehicles using Efficient Deep Neural Networks](https://www.youtube.com/watch?v=Knvl-vHzlUc&t=1s)
- [SqueezeNet Paper](https://arxiv.org/abs/1602.07360). [	arXiv:1602.07360 [cs.CV]](https://arxiv.org/abs/1602.07360)
- [Lex Fridman Video on Arguing Machines](https://www.youtube.com/watch?v=YBvcKtLKNAw)
- [A Survey of Deep Learning Techniques for Autonomous Driving](https://arxiv.org/abs/1910.07738). [	arXiv:1910.07738 [cs.LG]](https://arxiv.org/abs/1910.07738)
- [Self Driving cars:- A survey](https://arxiv.org/abs/1901.04407). [	arXiv:1901.04407 [cs.RO]](https://arxiv.org/abs/1901.04407)
- [Arguing Machines: Human Supervision of Black Box
AI Systems That Make Life-Critical Decisions](https://arxiv.org/pdf/1710.04459.pdf). [	arXiv:1710.04459 [cs.AI]](https://arxiv.org/abs/1710.04459)
- [Arguing Machines:- MIT Human Centred Autonomy](https://hcai.mit.edu/arguing-machines/)
