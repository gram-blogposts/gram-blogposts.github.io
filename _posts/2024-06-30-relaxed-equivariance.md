---
layout: distill
title: Effect of equivariance on training dynamics
description: Can relaxing equivariance help in finding better minima?
tags: distill formatting
giscus_comments: true
date: 2024-07-20
featured: true

authors:
  - name: Diego Canez
    url: "https://github.com/dgcnz"
    affiliations:
      name: University of Amsterdam
  - name: Nesta Midavaine
    url: "https://github.com/Nesta-gitU"
    affiliations:
      name: University of Amsterdam
  - name: Thijs Stessen
    url: "https://github.com/MeneerTS"
    affiliations:
      name: University of Amsterdam
  - name: Jiapeng Fan
    url: "https://github.com/JiapengFan"
    affiliations:
      name: University of Amsterdam
  - name: Sebastian Arias
    url: "https://github.com/SebastianUrielArias"
    affiliations:
      name: University of Amsterdam
  - name: Alejandro Garcia (supervisor)
    url: "https://github.com/AGarciaCast"
    affiliations:
      name: University of Amsterdam


bibliography: 2024-06-30-relaxed-equivariance.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Background
    subsections:
      - name: Regular GCNN
      - name: Steerable GCNN
      - name: Relaxed regular GCNN
      - name: Relaxed steerable GCNN
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Methodology
    subsections:
      - name: Datasets
      - name: Training Dynamics Evaluation
  - name: Results
    subsections:
      - name: Smoke Plume with Full Equivariance
      - name: Super Resolution
  - name: Concluding Remarks
  - name: References

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


*Group Equivariant Convolutional Network* (G-CNN) has gained significant traction in recent years owing to their ability to generalize the property of CNNs being equivariant to translations in convolutional layers. With equivariance, the network is able to exploit groups of symmetries and a direct consequence of this is that it generally needs less data to perform well. However, incorporating such knowledge into the network may not always be advantageous, especially when the data itself does not exhibit full equivariance. To address this issue, the G-CNN was modified, introducing *relaxed group equivariant CNNs* (RG-CNN). Such modified networks adaptively learn the degree of equivariance imposed on the network, i.e. enabling it to operate on a level between full equivariance and no equivariance.
Surprisingly, for rotational symmetries on fully equivariant data, <d-cite key="wang2023relaxed"></d-cite> found that a G-CNN exhibits poorer performance compared to a RG-CNN. This is a surprising result because a G-CNN, i.e. a fully equivariant network, is designed to perform well on fully equivariant data. Possibly the training dynamics benefit from relaxing of the equivariance constraint. To investigate this, we use the framework described in <d-cite key="park2022visiontransformerswork"></d-cite> for measuring convexity and flatness using the Hessian spectra.

Inspired by the aforementioned observations, this blog post aims to answer the question: **How does the equivariance imposed on a network affect its training dynamics?** We identify the following subquestions:

1. How does equivariance imposed on a network influence generalization?
2. How does equivariance imposed on a network influence the convexity of the loss landscape?   

We tackle these subquestions by analyzing trained models to investigate their training dynamics.

In view of space constraint, in this blogpost, we omit our reproducibility study and refer the readers to [our extended blog post](https://github.com/dgcnz/relaxed-equivariance-dynamics/blob/main/blogpost.md). Nevertheless, our reproducibility studies corroborated the following claims:
1. Relaxed steerable G-CNN outperforms steerable G-CNN (fully equivariant network) on fully rotationally equivariant data as shown in the experiment on the super resolution dataset in <d-cite key="wang2023relaxed"></d-cite>.
2. Relaxed G-CNN outperforms G-CNN on non-fully rotationally equivariant data as shown in the experiment on the smoke plume dataset in <d-cite key="wang2022approximatelyequivariantnetworksimperfectly"></d-cite>.

## Background

### Regular G-CNN

Consider the segmentation task depicted in the picture below.

<div style="text-align: center;">
  <img src="https://analyticsindiamag.com/wp-content/uploads/2020/07/u-net-segmentation-e1542978983391.png" alt="Figure 1" style="max-width: 100%;">
  <p>Annotated segmented image taken from <d-cite key="cordts2016cityscapesdatasetsemanticurban"></d-cite></p>
</div>

Naturally, applying segmentation on a rotated 2D image should give the same segmented image as applying such rotation after segmentation. Mathematically, for a neural network $NN$ to be equivariant w.r.t. the group $(G,\cdot)$, such as 2D rotations, then the following property needs to be satisfied: 


$$\begin{align*} 
NN (T_g x) = T'_g NN (x) & \qquad \qquad \forall g \in G. \\
\end{align*}
$$


To build such a network, it is sufficient that each of its layers is equivariant in the same sense. Recall that a CNN achieves equivariance to translations by sharing weights in kernels that are translated across the input in each of its convolution layers. Hence, a G-CNN extends this concept of weight sharing to achieve equivariance w.r.t an arbitrary locally-compact group $G$.  

For now on we will focus on affine groups, i.e., let $G := \mathbb{Z}^n \rtimes H$, where $H$ can be, for example, the rotation subgroup $SO(n)$ and $\mathbb{Z}^n$, the discrete translation group.

Furthermore, we'll consider an input signal of $c_0$ channels on an $n$-dimensional grid $f_0: \mathbb{Z}^n \rightarrow \mathbb{R}^{c_0}$, e.g. RGB images ($f: \mathbb{Z}^2 \rightarrow \mathbb{R}^3$ ). 

#### Lifting convolution

The first layer of a G-CNN lifts the input signal $f_0$ to the group $G$ using the kernel $\psi : \mathbb{Z}^n \rightarrow \mathbb{R}^{c_1 \times c_0}$ as follows:

$$
(f_0 \star \psi)(\mathbf{x}, h) = \sum_{\mathbf{y} \in \mathbb{Z}^n} f_0(\mathbf{y}) \psi(h^{-1}(\mathbf{y} - \mathbf{x}))
$$
 
where $\mathbf{x} \in \mathbb{Z}^n$ and $h \in H$. This yields $f_1: \mathbb{Z}^n \times H \rightarrow \mathbb{R}^{c_1}$ which is fed to the next layer.

#### $G$-equivariant convolution

Then, $f_1$ undergoes $G$-equivariant convolution with a kernel $\Psi: G \rightarrow \mathbb{R}^{c_2 \times c_1}$:

$$
(f_1 \star \Psi)(\mathbf{x}, h) = \sum_{\mathbf{y} \in \mathbb{Z}^n} \sum_{h' \in H} f_1(\mathbf{y}, h') \Psi(h^{-1}(\mathbf{y} - \mathbf{x}), h^{-1}h')
$$

where $\mathbf{x} \in \mathbb{Z}^n$ and $h \in H$. This outputs the signal $f_2: \mathbb{Z}^n \times H \rightarrow \mathbb{R}^{c_2}$. This way of convolving is repeated for all subsequent layers until the final aggregation layer, e.g. linear layer, if there is one.

Note that for *regular* group convolution to be practically feasible, $G$ has to be **finite** or addecuatly subsampled. Some of these limitations can be solved by *steerable* group convolutions.

#### Steerable G-CNN

First, consider the group representations $\rho_{in}: H \rightarrow \mathbb{R}^{c_\text{in} \times c_\text{in}}$ and $\rho_{out}: H \rightarrow \mathbb{R}^{c_\text{out} \times c_\text{out}}$. To address the aforementioned equivariance problem, $G$-steerable convolution modifies $G$-equivariant convolution with the following three changes:

- The input signal becomes $f: \mathbb{Z}^n \rightarrow \mathbb{R}^{c_\text{in}}$.
- The kernel $\psi: \mathbb{Z}^n \rightarrow \mathbb{R}^{c_\text{out} \times c_\text{in}}$ used must satisfy the following constraint for all $h \in H$: $$\psi(h\mathbf{x}) = \rho_{out}(h) \psi(\mathbf{x}) \rho_{in}(h^{-1})$$
- Standard convolution only over $\mathbb{Z}^n$ and not $G := \mathbb{Z}^n \rtimes H$ is performed.

To secure kernel $\psi$ has the mentioned property, we precompute a set of non-learnable basis kernels $(\psi_l)_{l=1}^L$ which do have it, and define all other kernels as weighted combinations of the basis kernels, using learnable weights with the same shape as the kernels.

Therefore, the convolution is of the form:

$$
(f \star_{\mathbb{Z}^n} \psi) (\mathbf{x}) = \sum_{\mathbf{y} \in \mathbb{Z}^n} \sum_{l=1}^L (w_l ⊙ \psi_l(\mathbf{y}))f(\mathbf{x}+\mathbf{y})
$$

Whenever both $\rho_{in}$ and $\rho_{out}$ can be decomposed into smaller building blocks called **irreducible representations**, equivariance w.r.t. infinite group $G$ is achieved (see Appendix A.1 of <d-cite key="bekkers2024fastexpressivesenequivariant"></d-cite>).


#### Relaxed G-CNN

The desirability of equivariance in a network depends on the amount of equivariance possessed by the data of interest. To this end, *relaxed* G-CNN is built on top of a regular G-CNN using a modified (relaxed) kernel consisting of a linear combination of standard G-CNN kernels $ \\{\Psi_l \\}_1^{L} $. Consider $G := \mathbb{Z}^n \rtimes H$. Then, *relaxed* G-equivariant group convolution is defined as:

$$
(f\;\tilde{\star}\;\Psi)(\mathbf{x}, h) = \sum_{\mathbf{y} \in \mathbb{Z}^n}\sum_{h' \in H} f(\mathbf{y}, h') \sum_{l=1}^L w_l(h) \Psi_l(h^{-1}(\mathbf{y} - \mathbf{x}), h^{-1} h')
$$

or equivalently as a linear combination of regular group convolutions with different kernels:

$$
\begin{aligned}
(f\;\tilde{\star}\;\Psi)(\mathbf{x}, h) &= \sum_{l=1}^L w_l(h) \sum_{\mathbf{y} \in \mathbb{Z}^n}\sum_{h' \in H} f(\mathbf{y}, h')  \Psi_l(h^{-1}(\mathbf{y} - \mathbf{x}), h^{-1} h')\\
 &= \sum_{l=1}^L w_l(h) [(f \star \Psi_l)(\mathbf{x}, h)]
\end{aligned}
$$

This second formulation makes for a more interpretable visualization, as one can see in the following figure. There, one can observe how a network might learn to downweight the feature maps corresponding to 180 degree rotations, thus breaking rotational equivariance and allowing for different processing of images picturing 6s and 9s.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-30-relaxed-equivariance/rgconv.png" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
Visualization of relaxed lifting convolutions ($L=1$) as template matching.
An input image $f_\text{in}$ contains a pattern $e$ in different orientations, each of which is weighted differently by the model.
</div>


<!---
 $G$-equivariance of the group convolution arises from kernel $\psi$'s dependence on the composite variable $g^{-1}h$, rather than on both variables $g$ and $h$ separately. This property is broken in relaxed kernels, leading to a loss of equivariance.

Therefore, using relaxed group convolutions allows the network to relax strict symmetry constraints, offering greater flexibility at the cost of reduced equivariance.
-->

#### Relaxed steerable G-CNN

Relaxed steerable G-CNN modified steerable G-CNN in a similar manner. Again, let the kernel in convolution be a linear combination of other kernels, such that the weights used depend on the variable of integration, leading to loss of equivariance.

$$(f \;\tilde{\star}_{\mathbb{Z}^n} \;\psi) (\mathbf{x}) = \sum_{\mathbf{y} \in \mathbb{Z}^n} \sum_{l=1}^L (w_l(\mathbf{y}) ⊙ \psi_l(\mathbf{y}))f(\mathbf{x}+\mathbf{y})$$


Furthermore, <d-cite key="wang2022approximatelyequivariantnetworksimperfectly"></d-cite> introduces a regularization term to impose equivariance on both relaxed models mentioned above. In our experiments, however, the best-performing models were those without this term.

## Methodology
### Datasets
#### Super-Resolution

The data consists of liquid flowing in 3D space and is produced by a high-resolution state-of-the-art simulation hosted by the John Hopkins University <d-cite key="jhtdb"></d-cite> . Importantly, this dataset is forced to be isotropic, i.e. fully equivariant to rotations, by design. 

For the experiment, a subset of 50 timesteps are taken, each downsampled from $1024^3$ to $64^3$ and processed into a task suitable for learning. The model is given an input of 3 consecutive timesteps, $t, t+1, t+2$ (which are first downsampled to $16^3$), and is tasked to upsample timestep $t+1$ to $64^3$, see Figure 1 for a visualization.

We use the following $3$ models from <d-cite key="wang2023relaxed"></d-cite>'s experiment on the same dataset in [Results](#Results):

- CNN.
- Regular G-CNN.
- Relaxed G-CNN.

<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/r1WCqrL4A.png" alt="Figure 1" style="max-width: 100%;">
  <p>Figure 1: Super Resolution architecture, taken from [1].</p>
</div>

#### Smoke Plume
<!-- For this experiment in [Wang et al. (2022)](#References), a specialized 2D smoke simulation was generated using PhiFlow [(Holl et al., 2020)](#References). -->
This is a synthetic $64 \times 64$ 2D smoke simulation dataset generated by PhiFlow <d-cite key="phiflow"></d-cite>, where dispersion of smoke in a scene starting from an inflow position with a buoyant force is simulated (Figure 2). 

The dataset we used has a fixed inflow with buoyant force only pointing in one of the following $4$ directions: upwards, downwards, left, or right. For our experiments we keep the buoyant force the same in all directions such that the data is fully equivariant w.r.t. $90$ degree rotations. 

<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/S1RILeLE0.png" alt="Figure 2" style="max-width: 100%;">
  <p>Figure 2: Example of a Smoke Plume sequence generated by PhiFlow.</p>
</div>

The models trained on this dataset are tasked with predicting the upcoming frame based on the current one. We use the following $2$ models in [Results](#Results):

- Relaxed steerable G-CNN from <d-cite key="wang2022approximatelyequivariantnetworksimperfectly"></d-cite> with relaxed equivariance w.r.t the C4 group. 
- Steerable G-CNN from <d-cite key="weiler2021generale2equivariantsteerablecnns"></d-cite> with full equivariance w.r.t the C4 group.


### Training Dynamics Evaluation

To assess the training dynamics of a network, we are interested in the final performance and the generalizability of the learned parameters, which are quantified by the final RMSE, and the sharpness of the loss landscape near the final weight-point proposed in <d-cite key="zhao2024improvingconvergencegeneralizationusing"></d-cite>. 

#### Sharpness

To measure the sharpness of the loss landscape after training, we consider changes in the loss averaged over random directions. Let $D$ denote a set of vectors randomly drawn from the unit sphere, and $T$ a set of displacements, i.e. real numbers. Then, the sharpness of the loss $\mathcal{L}$ at a point $w$ is: 

$$ \phi(w,D,T) = \frac{1}{|D||T|} \sum_{t \in T} \sum_{d \in D} |\mathcal{L}(w+dt)-\mathcal{L}(w)| 
$$

This definition is an adaptation from the one in <d-cite key="zhao2024improvingconvergencegeneralizationusing"></d-cite>. A sharper loss landscape around the model's final weights usually implies a greater generalization gap.

#### Hessian Eigenvalue 

Finally, the Hessian eigenvalue spectrum <d-cite key="park2022visiontransformerswork"></d-cite> sheds light on both the efficiency and efficacy of neural network training. Negative Hessian eigenvalues indicate a non-convex loss landscape, which can disturb the optimization process, whereas very large eigenvalues indicate training instability, sharp minima and consequently poor generalization.

## Results
In this section, we study how equivariance imposed on a network influences the convexity of the loss landscape and generalization, answering all the subquestions posed in [Introduction](#Introduction). 

### Smoke Plume with full Equivariance

First, we examine the training, validation and test RMSE for the Steerable G-CNN (E2CNN) <d-cite key="wang2022approximatelyequivariantnetworksimperfectly"></d-cite> and Relaxed Steerable G-CNN (Rsteer) <d-cite key="weiler2021generale2equivariantsteerablecnns"></d-cite> models on the fully equivariant Smoke Plume dataset.

<table>
  <tr>
    <td>
      <img src="https://hackmd.io/_uploads/ByqLXIUEA.png" alt="Figure 5" style="max-width: 100%;">
      <p align="center">Figure 5: Train RMSE curve for rsteer and E2CNN models</p>
    </td>
    <td>
      <img src="https://hackmd.io/_uploads/rJ58Q8LVC.png" alt="Figure 6" style="max-width: 100%;">
      <p align="center">Figure 6: Validation RMSE curve for rsteer and E2CNN models</p>
    </td>
  </tr>
</table>


<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/HyqIXLIEA.png" alt="Figure 7" style="max-width: 100%;">
  <p>Figure 7: Test RMSE for best models, averaged over five seeds</p>
</div>


Figures 5 and 6 show the train and validation RMSE curves. While rsteer and E2CNN perform similarly on the training data, rsteer has lower RMSE on the validation data, indicating better generalization. Figure 7 confirms that rsteer performs best on the test set, consistent with results on the Isotropic Flow dataset in <d-cite key="wang2023relaxed"></d-cite>.

To understand why relaxed equivariant models outperform fully equivariant ones, we examine the sharpness of the loss and the Hessian spectra.

<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/S14l5XZ4A.png" alt="Figure 10" style="max-width: 100%;">
  <p>Figure 8: Sharpness at early and best epochs for rsteer and E2CNN models. On the equivariant Smokeplume dataset</p>
</div>

Figure 10 shows that the rsteer model has much lower sharpness of the loss landscape compared to E2CNN for both checkpoints. This indicates a lower generalization gap, and thus more effective learning. This matches the lower validation RMSE curve we saw earlier.

<!---
<table>
  <tr>
    <td>
      <img src="assets/img/2024-06-30-relaxed-equivariance/correct_hession3.png" alt="Epoch 3" style="max-width: 100%;">
      <p align="center">Figure 9: Hessian spectra at an early epoch for rsteer and E2CNN models</p>
    </td>
    <td>
      <img src="assets/img/2024-06-30-relaxed-equivariance/correct_hessian50.png" alt="Epoch best" style="max-width: 100%;">
      <p align="center">Figure 10: Hessian spectra at the best epoch for rsteer and E2CNN models</p>
    </td>
  </tr>
</table>
-->

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-30-relaxed-equivariance/correct_hession3.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-30-relaxed-equivariance/correct_hessian50.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A simple, elegant caption looks good between image rows, after each row, or doesn't have to be there at all.
</div>

Figures 9 and 10 show Hessian spectra for the same checkpoints as the previous analysis. Regarding loss landscape flatness, both plots indicate that E2CNN has much larger eigenvalues than rsteer, potentially leading to training instability, less flat minima, and poor generalization for E2CNN.

To evaluate the convexity of the loss landscape, we examine the negative eigenvalues in the Hessian spectra. Neither model shows any negative eigenvalues, suggesting that both E2CNN and rsteer encounter convex loss landscapes. Therefore, convexity does not seem to significantly impact performance in this case.


### Super Resolution

Similarly, we also analyze the training dynamics of the superresolution models on the isotropic Super-Resolution dataset.

First, we examine the training and validation MAE curves for the Relaxed Equivariant (RGCNN), Fully Equivariant (GCNN), and non-equivariant (CNN) models (run on 6 different seeds).

<table>
  <tr>
    <td>
      <img src="https://hackmd.io/_uploads/HJrVE8IEA.png" alt="Figure 8" style="max-width: 100%;">
      <p align="center">Figure 11: Training MAE curve for RGCNN, GCNN and CNN models</p>
    </td>
    <td>
      <img src="https://hackmd.io/_uploads/HJrV48UN0.png" alt="Figure 9" style="max-width: 100%;">
      <p align="center">Figure 12: Validation MAE curve for RGCNN, GCNN and CNN models</p>
    </td>
  </tr>
</table>



Here, we observe that early in the training (around epoch $3$), RGCNN starts outperforming the other two models and keeps this lead until its saturation at around $0.1$ MAE. For this reason, we take a checkpoint for each model on epoch $3$ (early) and on its best epoch (Best), to examine the corresponding sharpness values.  

<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/SkpgREzV0.png" alt="Figure 10" style="max-width: 100%;">
  <p>Figure 13: Sharpness of the loss landscape on the super resolution dataset. Ran over 6 seeds, error bars represent the standard deviation. For early, the third epoch was chosen, while for best the epoch with the best validation loss was chosen.</p>
    
</div>

Figure 13 shows that the relaxed model has the lowest sharpeness in both cases. This indicates that the relaxed steerable GCNN has better generalisability during its training and at its convergence, matching our findings on the previous dataset.


## Concluding Remarks

We reproduced and extended the relevant findings in <d-cite key="wang2023relaxed"></d-cite> reaffirming the effectiveness of relaxed equivariant models and demonstrating that they are able to outperform fully equivariant models even on perfectly equivariant datasets. 

We furthermore investigated the authors' speculation that this superior performance could be due to relaxed models having enhanced training dynamics. Our experiments empirically support this hypothesis, showing that relaxed models exhibit lower validation error, a flatter loss landscape around the final weights, and smaller Hessian eigenvalues, all of which are indicators of improved training dynamics and better generalization.

Our results suggest that replacing fully equivariant networks with relaxed equivariant networks could be advantageous in all application domains where some level of model equivariance is desired, including those where full equivariance is beneficial. For future research, we should investigate different versions of the relaxed model to find out which hyperparameters, like the number of filter banks, correlate with sharpness. Additionally, the method should be applied to different types of data to see if the same observations can be made there.

## Code

- [Code and experiments for this blog](https://github.com/dgcnz/relaxed-equivariance-dynamics)
- [`gconv`, a PyTorch library for (relaxed) regular GCNNs](https://github.com/dgcnz/gconv)
- [JHTDB 🤗 HuggingFace Dataset](https://huggingface.co/datasets/dl2-g32/jhtdb)