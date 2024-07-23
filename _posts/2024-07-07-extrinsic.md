---
layout: distill
title: Correct, Incorrect and Extrinsic Equivariance
description: What happens if we use an equivariant network when there is a mismatch between the model symmetry and the problem symmetry?
tags: distill formatting
giscus_comments: false
date: 2024-07-07
featured: true

authors:
  - name: Dian Wang
    url: "https://pointw.github.io"
    affiliations:
      name: Northeastern University
  - name: Jung Yeon Park
    url: "https://jypark0.github.io"
    affiliations:
      name: Northeastern University
  - name: Xupeng Zhu
    url: "https://zxp-s-works.github.io"
    affiliations:
      name: Northeastern University
  - name: Neel Sortur
    url: "https://nsortur.github.io"
    affiliations:
      name: Northeastern University
  - name: Mingxi Jia
    url: "https://saulbatman.github.io"
    affiliations:
      name: Brown University
  - name: Guanang Su
    url: "https://xxs90.github.io"
    affiliations:
      name: University of Minnesota
  - name: Lawson L.S. Wong
    url: "https://www.khoury.northeastern.edu/home/lsw/"
    affiliations:
      name: Northeastern University
  - name: Robert Platt
    url: "https://www2.ccs.neu.edu/research/helpinghands/people/"
    affiliations:
      name: Northeastern University
  - name: Robin Walters
    url: "http://www.robinwalters.com"
    affiliations:
      name: Northeastern University


bibliography: 2024-07-07-extrinsic.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Abstract
  - name: Introduction
  - name: Correct, Incorrect, and Extrinsic Equivariance
  - name: Extrinsic Equivairance Helps Learning
  - name: Lower Bound of Incorrect Equivariance
  - name: Conclusion

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

## Abstract

Although equivariant machine learning has proven effective at many tasks, success depends heavily on the assumption that the ground truth function is symmetric over the entire domain matching the symmetry in an equivariant neural network. A missing piece in the equivariant learning literature is the analysis of equivariant networks when symmetry exists only partially or implicitly in the domain. We propose the definitions of correct, incorrect, and extrinsic equivariance, which describe the relationship between model symmetry and problem symmetry. We show that imposing extrinsic equivariance can improve the model's performance. We also provide the lower error bound analysis of incorrect equivariance, quantitatively showing the degree to which the symmetry mismatch will impede learning.

<!-- In this work, we
present a general theory for such a situation. We propose pointwise definitions of
correct, incorrect, and extrinsic equivariance, which allow us to quantify continuously the degree of each type of equivariance a function displays. We then study
the impact of various degrees of incorrect or extrinsic symmetry on model error.
We prove error lower bounds for invariant or equivariant networks in classification
or regression settings with partially incorrect symmetry. We also analyze the potentially harmful effects of extrinsic equivariance. Experiments validate these results
in three different environments.

Extensive work has demonstrated that equivariant neural networks can significantly improve sample efficiency and generalization by enforcing an inductive bias in the network architecture. These applications typically assume that the domain symmetry is fully described by explicit transformations of the model inputs and outputs. However, many real-life applications contain only latent or partial symmetries which cannot be easily described by simple transformations of the input. In these cases, it is necessary to learn symmetry in the environment instead of imposing it mathematically on the network architecture. We discover, surprisingly, that imposing equivariance constraints that do not exactly match the domain symmetry is very helpful in learning the true symmetry in the environment. We differentiate between extrinsic and incorrect symmetry constraints and show that while imposing incorrect symmetry can impede the model’s performance, imposing extrinsic symmetry can actually improve performance. We demonstrate that an equivariant model can significantly outperform non-equivariant methods on domains with latent symmetries both in supervised learning and in reinforcement learning for robotic manipulation and control problems. -->

## Introduction

Equivariant Networks have shown great benefit for improving sample efficiency.  

<!-- <p align="center">
  <img src="assets/img/2024-07-07-extrinsic/equi.gif" width="450px">
</p> -->

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/equi.gif" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

For example, consider the above position estimation task. We can use a rotationally equivariant network which will automatically generalize to different rotations of the same input. However, a **perfect top-down image** is normally required in order to model the problem symmetry as transformations of the input image.  
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/non_equi.gif" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Such assumption can be easily violated in the real world where there could be a fixed background or a tilted view angle.  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/obj_trans_vs_img_trans.gif" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<!-- <p align="center">
  <img src="img/obj_trans_vs_img_trans.gif" width="480px">
</p> -->

In these cases, the transformation of the object will be different from that of the image

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/obj_trans.gif" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<!-- <p align="center">
  <img src="img/obj_trans.gif" width="520">
</p> -->

Such object transformation will be hard to model and an equivariant network will not directly apply. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/img_trans.gif" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<!-- <p align="center">
  <img src="img/img_trans.gif" width="520">
</p> -->

However, we can still use an equivariant network that encodes the image-wise symmetry instead. In this work, we study what will happen if we use equivariant networks under such symmetry-mismatch scenarios.

<!-- we propose to use an equivariant network that encodes the image-wise symmetry instead to help modeling the object-wise symmetry. We call this **extrinsic equivariance**. -->

## Correct, Incorrect, and Extrinsic Equivariance

We first define **correct**, **incorrect**, and **extrinsic** equivariance, three different relationships between model symmetry and problem symmetry.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/task.gif" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
    </div>
</div>

<!-- <p align="center">
  <img src="assets/img/2024-07-07-extrinsic/task.gif" width="200">
</p> -->

Consider a classification task where the model needs to classify the blue and orange points in the plane.  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/correct.gif" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
          If we enforce reflection symmetry across the horizontal axis, the transformed data under reflection will have the same color as the original data, so the model preserves the problem symmetry, and we call it correct equivariance.
        </div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/incorrect.gif" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
          If we enforce rotation symmetry by pi, the transformed data under the rotation will have different color as the original data, so the model will be forced to generate wrong answers, and we call it incorrect equivariance.
        </div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/extrinsic.gif" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
          If we enforce scale symmetry, the transformed data under scaling will be outside of the input distribution shown in the gray ring, so we call it extrinsic equivariance.
        </div>
    </div>
</div>

## Extrinsic Equivairance Helps Learning

We show that extrinsic equivariance can helps learning. Our hypothesis is that extrinsic equivariance can makes it easier for the network to generate the decision boundary. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/decision_boundary.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
    </div>
</div>


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/exp.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<!-- <p align="center">
  <img src="img/exp.png" width="800">
</p> -->

We test our proposal in robotic manipulation (and in other domains, please see the paper <d-cite key="iclr23"></d-cite>) using SO(2)-Equivariant SAC <d-cite key="iclr22"></d-cite>, but the observation is taken from a camera with a tilted view angle. This tilted view angle makes the symmetry extrinsic because a rotated image will be out-of-distribution. We show that the extrinsic equivariant methods <span style="color: #3b3bff">(blue)</span> significantly outperform the unconstrained baselines.

## Lower Bound of Incorrect Equivariance

We futher analyze the lower bound of error caused by incorrect equivariance. Consider a digit classification task where we use a $$D_2$$-invariant network to classify digits.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/mixed_equi.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

For digit 2, a $$\pi$$-rotation symmetry is correct equivariance, while a vertical flip symmetry is incorrect (as it transforms 2 into a 5). For digit 3, a $$\pi$$-rotation symmetry is extrinsic (the rotated digit is out-of-distribution), while vertical flip symmetry is correct equivariance.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/orbit.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Let us focus on an image $$x$$ of digit 2 first, we can get the orbit $$Gx$$ with respect to the group $$G=D_2$$.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/majority_label.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Inside the orbit, two elements will have correct equivariance, and two other elements will have incorrect equivariance. If we assume the probability $$p$$ of all four images are identical, we can calculate the minimum error of a $$D_2$$-invariant network inside the orbit $$Gx$$ as $$k(Gx)=\frac{1}{2}$$.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/all_digits.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Doing the same calculation for all digits, we can calculate the minimum error as the mean of all $$k(Gx)$$.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-07-extrinsic/theory.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Formally, we define $$k(Gx)$$ as the **total dissent** of the orbit $$Gx$$, which is the integrated probability density of the elements in the orbit having a different label than the majority label. The invariant classification is then lower bounded by the integral of rotal dissent over the fundamental domain $$F$$.

We also analyze the lower bounds of invariant and equivariant regression, please see them in the paper <d-cite key="neurips23"></d-cite>.

## Conclusion
In this work, we both theoretically and empirically study the use of equivariant networks under a mismatch between the model symmetry and the problem symmetry. We define correct, incorrect, and extrinsic equivariance, and show that while incorrect equivariance will create an error lower bound, extrinsic equivariance can aid learning. For more information, please checkout our full papers <d-cite key="iclr23"></d-cite> and <d-cite key="neurips23"></d-cite>.

