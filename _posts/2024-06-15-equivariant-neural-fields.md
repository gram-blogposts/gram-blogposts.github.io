---
layout: distill
title: Equivariant Neural Fields - continuous representations grounded in geometry
description: An intro to geometry-grounded continuous signal representations and their use in modelling spatio-temporal dynamics.
tags: equivariance geometry neural-fields
giscus_comments: true
date: 2024-06-01
featured: true

authors:
  - name: David R. Wessels*
    url: "https://www.linkedin.com/in/david-wessels-b24299122/?originalSubdomain=nl"
    affiliations:
      name: University of Amsterdam
  - name: David M. Knigge*
    url: "https://davidmknigge.nl"
    affiliations:
      name: University of Amsterdam

bibliography: 2024-06-15-equivariant-neural-fields.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
    subsections:
      - name: The Evolution of Neural Fields
      - name: Introducing Equivariant Neural Fields
      - name: Key Properties of ENFs
      - name: Use of Neural Fields in downstream tasks
  - name: Methodology
    subsections:
      - name: Requirements for Equivariant Neural Fields
      - name: Equivariance through Bi-invariant Cross-Attention
      - name: Enforcing Locality in Equivariant Neural Fields 
  - name: Experimental Validation
    subsections:
      - name: Image and Shape Reconstruction and Classification
      - name: Latent Space Editing
      - name: Spatiotemporal Dynamics Modelling
  - name: Conclusion 
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---
## Introduction
Neural fields (NeFs) <d-cite key="xie2022neural"></d-cite> have emerged as a promising paradigm for representing continuous signals in a variety of domains. 
Recently, they have been used as a continuous alternative for classical discrete signal representations - showing promising results especially in higher dimensional settings where traditional grid-based methods often fall short <d-cite key="dupont2022data"></d-cite>. 

A major limitation of NeFs as representation is their lack of interpretibility and preservation of geometric information. In this blog post, we delve into the recent advancements presented in the paper "Grounding Continuous Representations in Geometry: Equivariant Neural Fields" <d-cite key="wessels2024ENF"></d-cite>, and explore how Equivariant Neural Fields (ENFs) enhance the capabilities of NeFs through geometric grounding and equivariance properties. We then elaborate upon their use as a representation by discussing the paper "Space-Time Continuous PDE Forecasting using Equivariant Neural Fields" <d-cite key="knigge2024pde"></d-cite>, which demonstrates the use of ENFs in modelling spatiotemporal dynamics. An important upcomming field of research, in which the geometric grounding of NeFs is crucial.

### The Evolution of Neural Fields

Neural fields are functions that map spatial coordinates to feature representations. For instance, a neural field $$ f_{\theta}: \mathbb{R}^d \rightarrow \mathbb{R}^c $$ can map pixel coordinates $$ x $$ to RGB values to represent images. These fields are typically parameterized by neural networks, which are optimized to approximate a target signal $$f_\theta$$ within a reconstruction task. Although this gives rise to continuous representations, for multiple signals, the weights $$\theta_f$$ are optimized separately for each signal $$ f $$, leading to a lack of shared structure across different signals and the need to train different seperate models.

Neural fields are functions that map spatial coordinates to feature representations. Specifically, a neural field $$ f_{\theta}: \mathbb{R}^d \rightarrow \mathbb{R}^c $$ can map pixel coordinates $$ x $$ to RGB values, thereby representing images. These fields are typically parameterized by neural networks and optimized to approximate a target signal $$f_\theta$$ within a reconstruction task.

While this approach results in continuous representations, it also presents a significant drawback. For multiple signals, the weights $$\theta_f$$ must be optimized separately for each signal $$ f $$ This leads to a lack of shared structure across different signals and necessitates training separate models for each individual signal.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-15-equivariant-neural-fields/nf.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Neural Fields. when applied to images, a neural field $f_\theta$ maps coordinates $x \in \mathbb{R}^2$ to pixel values $I(x) \in \mathbb{R}^3$.
</div>

Conditional Neural Fields (CNFs) extend this concept of neural fields by introducing a conditioning variable $$ z_f $$ that modulates the neural field for a specific signal $$ f $$. This enhancement allows CNFs to effectively represent an entire dataset of signals $$ f \in \mathcal{D} $$ using a single, shared set of weights $$ \theta $$ along with a set of unique conditioning variables $$ z_f $$. Since these representations are signal-specific, they latents can be used as a representation in downstream tasks. This approach has been successful in various tasks, including classification <d-cite key="dupont2022data"></d-cite>, segmentation <d-cite key="de2023deep"></d-cite>, generation <d-cite key="zhang20233dshape2vecset"></d-cite> and even solving partial differential equations<d-cite key="yin2022continuous"></d-cite> <d-cite key="knigge2024pde"></d-cite>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-15-equivariant-neural-fields/cnf.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Conditional Neural Fields. Conditional neural fields extend neural fields by introducing a conditioning variable $z$ that modulates the shared base field $f_\theta$.
</div>

However, conventional CNFs often lack geometric interpretability, they are able to capture textures and appearances which is shown by their performance in reconstruction. However, they do struggle to encode explicit geometric information necessary for tasks requiring spatial reasoning. Think for example of simple geometric transformations like rotations or translations, which are not inherently captured by CNFs; it is unclear how these transformations would manifest in the latent space.

### Introducing Equivariant Neural Fields

Equivariant Neural Fields (ENFs) address this limitation by grounding neural field representations in geometry. ENFs use latent point clouds as conditioning variables, where each point is a tuple consisting of a pose and a context vector. This grounding ensures that transformations in the field correspond to transformations in the latent space, a property known as equivariance.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-15-equivariant-neural-fields/ENF_latents.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Equivariant Neural Fields parameterize the conditioning variable $z$ as an attributed point-cloud of poses $p_i$ and corresponding context vectors $\mathbf{c}_i$: $z = \{ (p_i, \mathbf{c}_i \}_{i=0}^N$, explicitly grounding the latent space in geometry. 
</div>

### Key Properties of ENFs

- **Equivariance**: If the field transforms, the latent representation transforms accordingly. This property ensures that the latent space preserves geometric patterns, enabling better geometric reasoning.
- **Weight Sharing**: ENFs utilize shared weights over similar local patterns, leading to more efficient learning.
- **Localized Representations**: The latent point sets in ENFs enable localized cross-attention mechanisms, enhancing interpretability and allowing unique field editing capabilities.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-15-equivariant-neural-fields/enf-properties.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Illustration of key properties of Equivariant Neural Fields. ENFs exhibit equivariance by weight-sharing over local patterns through a latent set of poses and context vectors. This enables localized representations and geometric reasoning in the latent space.
</div>

### Use of Neural Fields in downstream tasks
As brief interjection, we provide some background on how NeFs are used in downstream tasks. As (a subset of) model NeF parameters are optimized reconstruct specific samples, these parameters may be used as a representation of their corresponding signals. These representations serve as input to downstream models for tasks such as classification, segmentation or even solving partial differential equations.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-15-equivariant-neural-fields/downstream-example.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Using NeFs in downstream tasks. For "conventional" NeFs, the weights $\theta_j$ are used as input to a downstream model that can operate on the computational graph of the neural field. For CNFs, the latent vectors $z_j$ are used as representation instead, allowing the use of simple MLPs.
    In ENFs instead the latent point sets $z_j$ are used as input to the downstream model, allowing for preservation of geometric information in the downstream task through the use of equivariant graph models.
</div>

## Methodology
We now delve into the technical details of ENF, focusing on the key components that enable the model to ground continuous representations in geometry.

### Requirements for Equivariant Neural Fields
The equivariance or steerability property of ENFs can be formally defined as:

$$ 
\forall g \in G : f_{\theta}(g^{-1}x, z) = f_{\theta}(x, gz). 
$$

This property ensures that if the field as a whole transforms, the latent representation will transform in a consistent manner. This is crucial for maintaining geometric coherence in the latent space.
In order reason about the application of a group action on $z$, the authors equip the latent space with a group action by defining $z$ as a set of tuples $$(p_i, \mathbf{c}_i)$$, where $G$ acts on $z$ by transforming poses $$p_i: gz = \{(gp_i, \mathbf{c}_i)\}_{i=1}^N$$.

For a neural field to satisfy the steerability property, the authors show it must be bi-invariant with respect to both coordinates and latents. 
This means that the field $ f_{\theta} $ must remain unchanged under group transformations applied to both the input coordinates and the latent point cloud, i.e.:

$$
\forall g \in G: f_\theta(gx, gz) = f_\theta(x, z).
$$

This observation is leveraged to define the architecture of ENFs, ensuring that the model is equivariant by design.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-15-equivariant-neural-fields/enf-comm-diag.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Commutative diagram illustrating the steerability of ENFs. 
</div>

### Equivariance through Bi-invariant Cross-Attention
ENFs utilize a bi-invariant cross-attention mechanism to parametrize the neural fields in order to achieve the aforementioned steerability property.
The cross-attention operation is defined as:
$ f_{\theta}(x, z) = \sum_{i=1}^{N} \text{att}(x, z) v(a(x, p_i), c_i) $
where $ a(x, p_i) $ is an invariant pair-wise attribute that captures the geometric relationship between the coordinate $ x $ and the latent pose $ p_i $, 
ensuring that the cross-attention operation respects the aforementioned bi-invariance condition.

Note that the choice of bi-invariant is related to the choice of group $$G$$ and the specific application domain. For example,
in natural images $$G$$ may be the group of 2D Euclidean transformations, while in 3D shape representations $$G$$ may be the group of 3D rigid transformations,
leading to different choices of bi-invariant $$a(x, p_i)$$. For a better understanding of bi-invariant properties we refer to <d-cite key="bekkers2023fast"></d-cite> which shows optimal invariant attributes in terms of expressivity for different groups.

### Enforcing Locality in Equivariant Neural Fields
To enforce locality, ENFs incorporate a Gaussian window into the attention mechanism. 
This ensures that each coordinate receives attention primarily from nearby latents, akin to the localized kernels in convolutional networks. 
This locality improves the interpretability of the latent representations, as specific features can be related to specific latent points $(p_i, \mathbf{c}_i)$.
Moreover, locality also improves parameter-efficiency by allowing for weight sharing over similar patterns. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-15-equivariant-neural-fields/enf-summ.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fitting different signals with ENFs. A field $f_j$ denoting a specific signal is represented by a set of localized latent points $z_j=\{ (p_i, \mathbf{c}_i)\}_{i=1}^N$. In the case of images (left), the latent points are distributed over the image plane. In the case of shapes (right), the latent points are distributed over 3D space.
</div>

## Experimental Validation
The authors validate the properties of ENFs through various experiments on image and shape datasets, providing 
metrics for reconstruction and downstream classification. Moreover, authors play around with the ENFs latent space
to demonstrate the benefits of having a geometrically grounded latent space. A
separate study by Knigge et al. demonstrates the use of ENFs in modelling spatiotemporal dynamics.

### Image and Shape Reconstruction and Classification
ENFs were evaluated on several image datasets, including CIFAR-10, CelebA, and STL-10. 
The results show that ENFs achieve higher peak signal-to-noise ratio (PSNR) in image reconstruction tasks compared to CNFs. 
This improvement is attributed to the geometric grounding and weight-sharing properties of ENFs.

| **Model**                                           | **Symmetry** | **Cifar10** | **CelebA** | **STL-10** |
|-----------------------------------------------------|--------------|-------------|------------|------------|
| Functa       <d-cite key="dupont2022data"></d-cite> | x            | 31.9        | 28.0       | 20.7       |
| **ENF - abs pos**                                   | x            | 31.5        | 16.8       | 22.8       |
| **ENF - rel pos**                                   | $\mathbb{R}^2$  | **34.8**   | **34.6**   | **26.8**   |
| **ENF - abs rel pos**                               | SE(2)        | 32.8        | 32.4       | 23.9       |
| **ENF - ponita**                                    | $\rm SE(2)$     | 33.9       | 32.9       | 25.4       |

<div class="caption">
    Reconstruction accuracy for ENFs compared to CNFs on CIFAR-10, CelebA and STL10 for different choices of bi-invariant $a$.
</div>


For classification, the authors used the latent point sets extracted from the trained ENF models. 
The classification accuracy on CIFAR-10 shows a significant improvement over conventional CNFs, 
highlighting the superior representation capabilities of ENFs.

| **Model**                   | **Symmetry** | **Cifar10** |
|-----------------------------|--------------|-------------|
| Functa <d-cite key="dupont2022data"></d-cite> | x            | 68.3        |
| **ENF - abs pos**           | x            | 68.7        |
| **ENF - rel pos**           | $\mathbb{R}^2$ | **82.1**    |
| **ENF - abs rel pos**       | SE(2)        | 70.9        |
| **ENF - ponita**            | SE(2)        | 81.5        |

<div class="caption">
    Classification accuracy for ENFs compared to CNFs on CIFAR-10 for different choices of bi-invariant $a$.
</div>

The authors also tested ENFs on shape datasets using Signed Distance Functions (SDFs). 
The results indicate that ENFs can effectively represent geometric shapes with high fidelity, 
further validating the geometric interpretability of the latent representations.

| **Model**                  | **Reconstruction IoU (voxel)** | **Reconstruction IoU (SDF)** | **Classification** |
|----------------------------|--------------------------------|------------------------------|--------------------|
| Functa <d-cite key="dupont2022data"></d-cite> | 99.44                          | -                            | 93.6               |
| **ENF**                    | -                              | 55                           | 89                 |

<div class="caption">
    Shape reconstruction and classification metrics for ENFs compared to CNFs on ShapeNet.
</div>

### Latent Space Editing
The authors demonstrate the benefits of the geometrically grounded latent space in ENFs by performing latent space editing.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-15-equivariant-neural-fields/enf-carduck.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Latent space editing with ENFs. By construction, ENF representations can be stitched together to create new fields. Here, the authors demonstrate the ability to create a "car-duck" by combining the latent representations of reconstructions of a car and a duck.
</div>


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-15-equivariant-neural-fields/interpolation.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Localized latent space interpolation. The authors demonstrate the ability to interpolate between two neural fields by interpolating between their latent point sets. This allows for localized editing of the fields.
</div>

### Spatiotemporal Dynamics Modelling

Another usecase for ENFs is highlighted in the paper "Space-Time Continuous PDE Forecasting using Equivariant Neural Fields" <d-cite key="knigge2024pde"></d-cite>.
Authors use the ENF as a continuous state representation for solving partial differential equations; learning to forecast dynamics by modelling them with a Neural ODE as a
flow in the latent space of the ENF. Since PDEs are often defined over continuous domains in terms of local differential operators, ENFs are well-suited to model these dynamics, as
they provide localized continuous representations.
This approach allows for symmetry-preserving continuous forecasting of spatiotemporal dynamics,
showing promising results on a variety of PDEs defined over different geometries. An initial state $$\nu_0:\mathcal{X}\rightarrow \mathbb{R}$$ representing the current state of the PDE
is fit with a corresponding latent $$z_0$$, which is unrolled in latent space with an equivariant graph-based neural ODE $$F_\psi$$.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-15-equivariant-neural-fields/fig-enf-pde.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    In Equivariant Neural Fields $f_\theta$, a field $\nu_t$ is represented by a set of latents 
$z^\nu_t = \{(p_{i}^\nu,\mathbf{c}_{i}^\nu)\}_{i=1}^N$ consisting of a pose $p_{i}$ and context vector $\mathbf{c}_{i}$. 
Using meta-learning, the initial latent $z^\nu_0$ is fit in only 3 SGD steps, after which an equivariant neural ODE $F_\psi$ models the solution as a latent flow.
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-15-equivariant-neural-fields/shallow-water.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Due to its continuous nature, the ENF forecasting model is able to natively handle zero-shot super-resolution, as demonstrated on the shallow water equations. 
Top: low resolution test sample at train resolution. Middle: high resolution test sample at test resolution. Bottom: ENF forecast at test resolution. 
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-15-equivariant-neural-fields/internally-heated-convection.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Because of its geometric grounding, the model is able to handle complicated geometries, as demonstrated on internally heated convection equations in the ball.
</div>

<iframe src="{{ '/assets/plotly/2024-06-15-equivariant-neural-fields/navier-stokes.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
<div class="caption">
    The authors demonstrate the use of ENFs in modelling spatiotemporal dynamics by solving the Navier-Stokes 
equations over a 2D domain with periodic boundary conditions. The ENF respects the corresponding translational symmetries. Left: ground truth dynamics. Middle: ENF forecast. Right: Absolute forecast error. A test-sample is visualized, i.e. the model is unrolled from the initial state $\nu_0$. During training the model is supervised for 10 timesteps.
</div>


## Conclusion

Equivariant Neural Fields leverage geometric grounding and equivariance properties to provide continuous signal representations preserving geometric information.
This approach opens up new possibilities for tasks that require geometric reasoning and localized representations, such as image and shape analysis, and shows promising results in 
forecasting spatiotemporal dynamics.

This blog post has explored the foundational concepts and the significant advancements brought forward by Equivariant Neural Fields. By grounding neural fields in geometry and incorporating equivariance properties, ENFs pave the way for more robust and interpretable continuous signal representations. As research in this area progresses, we can expect further innovations that leverage the geometric and localized nature of these fields, unlocking new potentials across diverse applications.
