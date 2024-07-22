---
layout: distill
title: Learning Embedding Spaces with Metrics via Contrastive Learning
description: 
tags: distill formatting
giscus_comments: false
date: 2024-06-13
featured: true

authors:
  - name: Patrick Feeney
    url: "https://patrickfeeney.github.io/"
    affiliations:
      name: Tufts University

toc:
  - name: Similarity Functions
    subsections:
      - name: Cosine Similarity
      - name: Negative Arc Length
      - name: Negative Euclidean Distance
  - name: Contrastive Losses
    subsections:
      - name: Margin Losses
      - name: Cross Entropy Losses
  - name: Case Study with Contrastive Learning on a Hypersphere
    subsections:
      - name: Supervised Learning Accuracy
      - name: Supervised Learning Embedding Space
  - name: Discussion

bibliography: 2024-06-13-contrast-learning.bib
---

Contrastive learning encompasses a variety of methods that learn a constrained embedding space to solve a task. The embedding space is constrained such that a chosen metric, a function that measures the distance between two embeddings, satisfies some desired properties, usually that small distances imply a shared class. Contrastive learning underlies many self-supervised methods, such as MoCo <d-cite key="he_momentum_2020"></d-cite>, <d-cite key="chen_empirical_2021"></d-cite>, SimCLR <d-cite key="chen_simple_2020"></d-cite>, <d-cite key="chen_big_2020"></d-cite>, and BYOL <d-cite key="grill_bootstrap_2020"></d-cite>, as well as supervised methods such as SupCon <d-cite key="khosla_supervised_2020"></d-cite> and SINCERE <d-cite key="feeney_sincere_2024"></d-cite>.

In contrastive learning, there are two components that determine the constraints on the learned embedding space: the similarity function and the contrastive loss. The similarity function takes a pair of embedding vectors and quantifies how similar they are as a scalar. The contrastive loss determines which pairs of embeddings have similarity evaluated and how the resulting set of similarity values are used to measure error with respect to a task, such as classification. Backpropagating to minimize this error causes a model to learn embeddings that best satisfy the constraints induced by the similarity function and contrastive loss.

This blog post examines how similarity functions and contrastive losses affect the learned embedding spaces. We first examine the different choices for similarity functions and contrastive losses. Then we conclude with a brief case study investigating the effects of different similarity functions on supervised contrastive learning.


## Similarity Functions

A similarity function $$s(z_1, z_2): \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}$$ maps a pair of $$d$$-dimensional embedding vectors $$z_1$$ and $$z_2$$ to a real similarity value, with greater values indicating greater similarity. A temperature hyperparameter $$0 < \tau \leq 1$$ is often included, via $$\frac{s(z_1, z_2)}{\tau}$$, to scale a similarity function. If the similarity function has a range that is a subset of $$\mathbb{R}$$, then $$\tau$$ can increase that range. $$\tau$$ is omitted for simplicity here.

### Cosine Similarity

A common similarity function is cosine similarity:

$$s(z_1, z_2) = \frac{z_1 \cdot z_2}{||z_1|| \cdot ||z_2||}$$

This function measures the cosine of the angle between $$z_1$$ and $$z_2$$ as a scalar in $$[-1, 1]$$. Cosine similarity violates the triangle inequality, making it the only similarity function discussed here that is not derived from a distance metric.

### Negative Arc Length

The recently proposed negative arc length similarity function <d-cite key="koishekenov_geometric_2023"></d-cite> provides an analogue for cosine similarity that is a distance metric:

$$s(z_1, z_2) = 1 - \frac{\text{arccos}(z_1 \cdot z_2)}{\pi}$$

This function assumes that 
$$||z_1|| = ||z_2|| = 1$$
which is a common normalization <d-cite key="le-khac_contrastive_2020"></d-cite> that restricts the embeddings to a hypersphere. The arc length $$\text{arccos}(z_1 \cdot z_2)$$ is a natural choice for comparing such vectors as it is the geodesic distance, or the length of the shortest path between $$z_1$$ and $$z_2$$ on the hypersphere. Subtracting the arc length converts the distance metric into a similarity function with range $$[0, 1]$$. Koishekenov et al. <d-cite key="koishekenov_geometric_2023"></d-cite> recently reported improved downstream performance by replacing cosine similarity with negative arc length for two self-supervised cross entropy losses.

### Negative Euclidean Distance

The negative Euclidean distance similarity function is simply:

$$s(z_1, z_2) = -||z_1 - z_2||_2$$

Euclidean distance measures the shortest path in Euclidean space, making it the geodesic distance when $$z_1$$ and $$z_2$$ can take any value in $$\mathbb{R}^d$$. In this case the similarity function has range $$[-\infty, 0]$$.

The negative Euclidean distance can also be used with embeddings restricted to a hypersphere, resulting in range $$[-2, 0]$$. However, this is not the geodesic distance for the hypersphere as the path being measured is inside the sphere. The Euclidean distance will be less than the arc length unless $$z_1 = z_2$$, in which case they both equal 0.

## Contrastive Losses

A contrastive loss function maps a set of embeddings and a similarity function to a scalar value. Losses are written such that derivatives for backpropagation are taken with respect to the embedding $$z$$.

### Margin Losses

The original contrastive loss <d-cite key="chopra_learning_2005"></d-cite> maximizes similarity for examples $$z^+$$ and minimizes similarity for examples $$z^-$$ until the similarity is below margin hyperparameter $$m$$:

$$L(z, z^+) = s(z, z^+); L(z, z^-) = \max( 0, m - s(z, z^-) )$$

The structure of this loss implies that $$z_1$$ and $$z_2$$ share a class if $$s(z_1, z_2) < m$$ and otherwise they do not share a class. This margin hyperparameter can be challenging to tune for efficiency throughout the training process because it needs to be satisfiable but also provide $$z^-$$ samples within the margin in order to backpropagate the error.

The triplet loss <d-cite key="schroff_facenet_2015"></d-cite> avoids this by using a margin between similarity values:

$$L(z, z^+, z^-) = \max( 0, s(z, z^+) - s(z, z^-) + m)$$

The triplet loss only updates a network when its loss is positive, so finding triplets satisfying that condition are important for learning efficiency.

Lifted Structured Loss <d-cite key="oh_song_deep_2016"></d-cite> handles this by precomputing similarities for all pairs in a batch then selecting the $$z^-$$ with maximal similarity:

$$L(z, z^+) = \max( 0, s(z, z^+) + m - \max [ \max_{z^-} s(z, z^-), \max_{z^-} s(z^+, z^-) ] )$$

The Batch Hard loss <d-cite key="hermans_defense_2017"></d-cite> takes this even further by selecting $$z^+$$ with minimal similarity:

$$L(z, z^+) = \max( 0, \min_{z^+} [ s(z, z^+) ] + m - \max_{z^-} [ s(z, z^-) ] )$$

The decision to compute the loss based on comparisons between $$z$$, a single $$z^+$$, and a single $$z^-$$ comes with advantages and disadvantages. These methods can be easier to adapt for learning with varying levels of supervision because complete knowledge of whether similarity should be maximized or minimized for each pair in the dataset is not required. However, these methods also make training efficiently difficult and provide relatively loose constraints on the embedding space.

### Cross Entropy Losses

A common contrastive loss is the Information Noise Contrastive Estimation (InfoNCE) <d-cite key="oord_representation_2019"></d-cite> loss:

$$L(z, z^+, z^-_1, z^-_2, \ldots, z^-_n) = -\log \frac{ e^{s(z, z^+)} }{ e^{s(z, z^+)} + \sum_{i=1}^n e^{s(z, z^-_i)} }$$

InfoNCE is a cross entropy loss whose logits are similarities for $$z$$. $$z^+$$ is a single embedding whose similarity with $$z$$ should be maximized while $$z^-_1, z^-_2, \ldots, z^-_n$$ are a set of $$n$$ embeddings whose similarity with $$z$$ should be minimized. The structure of this loss implies that $$z_1$$ shares a class with $$z_2$$ if no other embedding has greater similarity with $$z_1$$.

The choice of $$z^+$$ and $$z^-$$ sets varies across methods. The self-supervised InfoNCE loss chooses $$z^+$$ to be an embedding of an augmentation of the input that produced $$z$$ and $$z^-$$ to be the other inputs and augmentations in the batch. This is called instance discrimination because only augmentations of the same input instance have their similarity maximized.

Supervised methods expand the definition of $$z^+$$ to also include embeddings which share a class with $$z$$. The expectation of InfoNCE loss over choices of $$z^+$$ is used to jointly maximize their similarity to $$z$$. The Supervised Contrastive (SupCon) loss <d-cite key="khosla_supervised_2020"></d-cite> uses all embeddings not currently set as $$z$$ or $$z^+$$ as $$z^-$$, including embeddings that share a class with $$z$$ and therefore will also be used as $$z^+$$. This creates loss terms that would minimize similarity between embeddings that share a class. Supervised Information Noise-Contrastive Estimation REvisited (SINCERE) loss <d-cite key="feeney_sincere_2024"></d-cite> removes embeddings that share a class with $$z$$ from $$z^-$$, leaving only embeddings with different classes. An additional margin hyperparameter can also be added to these losses <d-cite key="barbano_unbiased_2022"></d-cite>, which allows for interpolation between the original losses and losses with the $$e^{s(z, z^+)}$$ term removed from the denominator.

Considering a set of similarities during loss calculation allows the loss to implicitly perform hard negative mining <d-cite key="khosla_supervised_2020"></d-cite>, avoiding the challenge of selecting triplets required by a margin loss. The lack of a margin places strict constraints on the embedding space, as similarities are always being pushed towards the maximum or minimum. This enables analysis of embedding spaces that minimize the loss. For example, InfoNCE and SINCERE losses with cosine similarity are minimized by embedding spaces with clusters of inputs mapped to single points (maximizing similarity) that are uniformly distributed on the unit sphere (minimizing similarity) <d-cite key="wang_understanding_nodate"></d-cite>.

## Case Study: Contrastive Learning on a Hypersphere

Many modern contrastive learning techniques build off of the combination of cosine similarity and cross entropy losses. However, few papers have explored changing similarity functions and losses outside of the context of a more complex model.

Koishekenov et al. <d-cite key="koishekenov_geometric_2023"></d-cite> recently reported improved downstream performance by replacing cosine similarity with negative arc length for two self-supervised cross entropy losses. This change is motivated by the desire to use the geodesic distance on the embedding space, which in this case is a unit hypersphere. We investigate whether replacing cosine similarity with negative arc length similarity can improve performance with the SINCERE loss, which is supervised, and how each similarity affects the learned embedding space.

### Supervised Learning Accuracy

We utilize the methodology of Feeney and Hughes <d-cite key="feeney_sincere_2024"></d-cite> to evaluate if the results of Koishekenov et al. <d-cite key="koishekenov_geometric_2023"></d-cite> generalize to supervised cross entropy losses. Specifically, we train models with SINCERE loss and each similarity function then evaluate the models with nearest neighbor classifiers on the test set.

<table>
  <tr>
   <td>
   </td>
   <td colspan="2" ><strong>CIFAR-10</strong>
   </td>
   <td colspan="2" ><strong>CIFAR-100</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Similarity</strong>
   </td>
   <td><strong>1NN</strong>
   </td>
   <td><strong>5NN</strong>
   </td>
   <td><strong>1NN</strong>
   </td>
   <td><strong>5NN</strong>
   </td>
  </tr>
  <tr>
   <td>Cosine
   </td>
   <td>95.88
   </td>
   <td>95.91
   </td>
   <td>76.23
   </td>
   <td>76.13
   </td>
  </tr>
  <tr>
   <td>Negative Arc Length
   </td>
   <td>95.66
   </td>
   <td>95.65
   </td>
   <td>75.81
   </td>
   <td>76.41
   </td>
  </tr>
</table>

We find no statistically significant difference based on the 95% confidence interval of the accuracy difference <d-cite key="foody_classification_2009"></d-cite> from 1,000 iterations of test set bootstrapping. This aligns with the results in Feeney and Hughes <d-cite key="feeney_sincere_2024"></d-cite>, which used different loss functions with cosine similarity and found a similar lack of statistically significant results across choices of supervised contrastive cross entropy losses. This suggests that supervised learning accuracy is similar across choices of reasonable similarity functions and contrastive losses.

### Supervised Learning Embedding Space

We also visualize the learned embedding space for each CIFAR-10 model. For each test set image, the similarity value is plotted for the closest training set image that shares a class (“Target”) and that does not share a class (“Noise”). This visualizes the 1-nearest neighbor decision process. Both similarity functions are plotted for each model, with the title denoting the similarity function used during training.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-13-contrast-learning/cos_cifar10_all_cos.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-13-contrast-learning/cos_cifar10_all_arccos.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

The model trained with cosine similarity maximizes the similarity to target images well. There are a small number of noise images with near maximal similarity, but the majority are below 0.3 cosine similarity. Interestingly, the peaks seen in the noise similarity reflects the fact that individual classes will have different modes of their noise histograms.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-13-contrast-learning/arccos_cifar10_all_cos.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-13-contrast-learning/arccos_cifar10_all_arccos.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

The model trained with negative arc length similarity does a better job of forcing target similarity values very close to 1 negative arc length similarity, but also has a notable number of target similarities near 0.5 negative arc length similarity. The noise distribution also reflects the fact that individual classes have different modes for their noise histograms, but in this case the modes are spread across more similarity values. Notably the peak for the horse class is very close to the max similarity due to a high similarity to the dog class, although they are still separated enough from the target similarities to not have an impact on accuracy.

### Discussion

The choice of similarity function clearly has an effect on the learned embedding space despite a lack of statistically significant changes in accuracy. The cosine similarity histogram most cleanly aligns with the intuition that contrastive losses should be maximizing and minimizing similarities. In contrast, the negative arc length similarity histogram suggests similarity minimization is sacrificed for very consistent maximization, producing small differences in similarity between some target classes and noise examples. I hypothesize that this change in behavior arises from the difference in similarity function behavior with small angles described in Koishekenov et al. <d-cite key="koishekenov_geometric_2023"></d-cite>.

These differences in the learned embedding spaces could affect performance on downstream tasks such as transfer learning. I hypothesize that the larger difference between target and noise similarities seen in the cosine similarity model would improve transfer learning performance, similar to the improvement of SINCERE over SupCon loss reported in Feeney and Hughes <d-cite key="feeney_sincere_2024"></d-cite>.
