---
layout: distill
title: Towards Equivariant Adaptation of Large Pretrained Models
description: How do you make your foundation model <ins>equivariant and robust</ins> to known transformations without re-training from scratch? 
tags: distill formatting
giscus_comments: true
date: 2024-07-21
featured: true

authors:
  - name: Siba Smarak Panigrahi
    url: "https://sibasmarak.github.io/"
    affiliations:
      name: McGill University and Mila
  - name: Arnab Kumar Mondal
    url: "https://arnab39.github.io/"
    affiliations:
      name: McGill University and Mila
  - name: Sékou-Oumar Kaba
    url: "https://oumarkaba.github.io/"
    affiliations:
      name: McGill University and Mila


bibliography: 2024-07-11-equivariant-adaptation.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: What is Equivariance?
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Decoupling Equivariance from Architecture with Canonicalization
  - name: Learning to Predict the Correct Orientation for the Pretrained Network
    subsections:
        - name: Enter the Canonicalization Prior
  - name: Results at a Glance
    subsections:
        - name: Image Classification
        - name: Instance Segmentation
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
  .fixed-size-img2 {
    width: 100%;
    height: 240px; /* Adjust the height as needed */
    object-fit: cover; /* Ensures the image covers the specified area */
  }
  .fixed-size-img3 {
    width: 100%;
    height: 180px; /* Adjust the height as needed */
  }
---

Deep learning has witnessed tremendous growth in the past decade. Still, as we strive for more nuanced understanding and performance improvements, one challenge emerges clearly: how do we ensure our models understand data transformations? Enter equivariance, an idea that can help our networks maintain consistent behaviour with data transformations. But with the rise of large pretrained models, how do we make them equivariant without changing their architecture or retraining the model from scratch with data augmentation? In this blogpost, we delve into ideas presented
in the paper "Equivariant Adaptation of Large Pretrained Models"<d-cite key="mondal2023equivariant"></d-cite> to answer this question.

## What is Equivariance?
Equivariant networks <d-cite key="cohen2016group,worrall2019deep,bronstein2021geometric"></d-cite> are deep neural networks that maintain consistent behaviour when input data undergo transformations like rotation, scaling, or translation. In simpler terms, if we rotate an image of a cat, an equivariant network would still recognize it as a cat! Another example of this would be segmentation maps on images. If we rotate an image, the segmentation map should rotate in the same way to maintain consistency.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-11-towards-equivariant-adaptation/image1.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-11-towards-equivariant-adaptation/image2.png" class="img-fluid rounded z-depth-1 fixed-size-img2" %}
    </div>
</div>
<div class="caption">
    Equivariant tasks. Instance segmentation requires segmentation maps to be consistent with the input image transformations and 
    classification requires the network to recognize the same object in different orientations.
</div>


The beauty of this is that such networks lead to more accurate, robust predictions and need fewer samples to train – this is great in theory but hard to implement in practice, especially for large pretrained models whose equivariant counterparts are not trivial to design or are very expensive to re-train from scratch. These massive models pretrained on the entire internet are extremely good at solving and reasoning about different tasks and are called `foundation models` <d-cite key="bommasani2021opportunities"></d-cite>. Despite having such capabilities, foundation models are not naturally equivariant and usually don’t handle transformations well. (see the GPT-4 example below) Our goal is to incorporate the benefits of equivariance in existing foundation models.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-11-towards-equivariant-adaptation/image3.png" class="img-fluid rounded z-depth-1 fixed-size-img3" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-11-towards-equivariant-adaptation/image4.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    ChatGPT-4 for image parsing. The model is accurate in extracting text from 'straight' images, but it fails to do so for 'inverted' images.
</div>

## Decoupling Equivariance from Architecture with Canonicalization

A recent alternative to designing equivariant networks was proposed by Kaba et al. <d-cite key="kaba2023equivariance"></d-cite>.  It suggests that instead of changing the network architecture to incorporate equivariance, why not first learn to transform the input data into a ‘standard’ format (or orientation), also known as `canonical form`. This way, our task prediction network can work on this standardized format, ensuring consistency. This process involves adding an additional inexpensive network called the `canonicalization network` or $c$, which learns to standardize the input. In our formulation, for an input $x$, the output from canonicalization network is $c(x) = g$, where $g$ denotes the group element corresponding to the orientation of $x$. The primary network that learns to solve the task based on the standardized input is called the `prediction network` or $\phi$. In this particular formulation, achieving equivariance requires only ensuring that the canonicalization process is invariant to the transformation of the input. This means no matter which orientation you see the input, the canonicalization process should always bring it back to the same canonical orientation. This is achieved by using a shallow and cheap equivariant architecture for the canonicalization network. (see <d-cite key="kaba2023equivariance"></d-cite> for more details)

Finally, the combination of the canonicalization network and the prediction network can be represented as $\Phi$:


$$\Phi(x) = c(x) \circ \phi(c(x)^{-1}. x)$$

$$\Rightarrow \Phi(g. x) = c(g. x) \circ \phi(c(g. x)^{-1}. g. x)$$


$$\Rightarrow \Phi(g. x) = g.c(x) \circ \phi(c(x)^{-1}. x) = g \circ \Phi(x)$$

The beauty of this approach lies in how the canonicalization network separates the equivariance requirement from the core prediction network architecture. This means that you have the flexibility to employ any powerful pretrained large neural network for the main prediction task.

Sounds straightforward? Well, it has a hitch.

The **main challenge** is ensuring the canonicalization network ‘plays nice’ with the prediction network. For example, the canonicalization network can output orientations that hurt the training of the prediction network, leading to poor task performance. This becomes more important when the prediction network is pretrained on a certain dataset. For instance, if the canonicalization network transforms all images to be upside-down, but our pretrained prediction network wasn’t trained on upside-down images, the whole system falls apart. So, it’s vital that the canonicalization network outputs orientations of the data that is in-distribution for the pretrained prediction network.

## Learning to Predict the Correct Orientation for the Pretrained Network

The magic lies in designing our canonicalization function not just to transform data but to do so while being aware of how our prediction model was initially trained. The key is ensuring that the data being transformed (or standardized) is done to align with what the pretrained prediction model expects. Mathematically, the goal is to bring the predicted out-of-distribution orientations to the distribution of orientations the pretrained prediction network has seen.

# Enter the Canonicalization Prior

In simple terms, it’s a guiding force ensuring that our canonicalization function behaves and produces output that the pretrained prediction network would expect and appreciate. 
We leverage the idea that our data can provide hints on the ‘typical’ transformations it undergoes. By encoding this into a prior, one can guide our canonicalization function to produce transformed data that’s not just standardized but also aligned with what the prediction network was trained on.

While mathematical and intricate, this entire process can be boiled down to ensuring that the large pretrained prediction network always looks at in-distribution samples. 
This results in a highly robust model that can confidently handle varied transformations in the input data, giving accurate predictions every time.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-11-towards-equivariant-adaptation/image5.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Training and inference with canonicalization prior. The canonicalization function learns to output the canonical orientations seen in the dataset during training by minimising KL between the orientation distributions of predicted and pretraing dataset (prior regularization). During inference, transformed data is brought back to the canonical orientation by the canonicalization process.
</div>

## Results at a Glance

This section highlights the effectiveness of the approach for image classification and instance segmentation tasks. Additional results and experiments including point cloud classification and part segmentation are detailed in <d-cite key="mondal2023equivariant"></d-cite>.

# Image Classification

The authors select Vision Transformer (ViT) <d-cite key="dosovitskiy2020image"></d-cite> and ResNet-50 <d-cite key="he2016deep"></d-cite> as pretrained<d-footnote>Pretrained on ImageNet<d-cite key="deng2009imagenet"></d-cite>.</d-footnote> prediction network for image classification and $$C_8$$ group, i.e., eight discrete rotations (multiples of 45$$^\circ$$) as the set of known transformations. The objective is to make the prediction networks equivariant and robust to these transformations, as an example, on CIFAR-100 dataset <d-cite key="krizhevsky2009learning"></d-cite>.

The authors compare different fine-tuning setups. First, **Vanilla** indicates the standard fine-tuning on the downstream dataset. **C8-Aug.** indicates fine-tuning on the downstream
dataset and $$C_8$$ group data augmentations. **LC** is the learned canonicalization approach proposed in Kaba et. al. <d-cite key="kaba2023equivariance"></d-cite>. 
**Prior-Regularized LC** is the learned canonicalization approach with prior regularization (as described in above sections, proposed in <d-cite key="mondal2023equivariant"></d-cite>). The evaluation includes reporting the performance of the models on the test set of **CIFAR-100** and an augmented version, **CIFAR-100 \[C8\]**, where each sample is augmented with every transformation of $$C_8$$ group. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-11-towards-equivariant-adaptation/image6.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fine-tuning performance of ResNet-50 and ViT on CIFAR-100 dataset. Vanilla has the highest performance on CIFAR-100, but it is the worst on CIFAR-100 [C8] which indicates poor
    robustness. Prior-Regularized LC preserves the performance on CIFAR-100 and outperforms other baselines on CIFAR-100 [C8].
</div>

# Instance Segmentation
Furthermore, the authors scale this idea to large foundation models like the Segment Anything Model (SAM) <d-cite key="kirillov2023segment"></d-cite> and make it robust to rotations while having a nominal increase in the number of parameters and inference speed.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-07-11-towards-equivariant-adaptation/image7.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Predicted masks from the Segment Anything Model (SAM) <d-cite key="kirillov2023segment"></d-cite> showcasing both the original model and prior-regularized equivariant adaptation 
    for 90-degrees counter-clockwise rotated input images taken from the COCO 2017 dataset <d-cite key="lin2014microsoft"></d-cite>. The approach makes SAM equivariant to the
    group of 90-degrees rotations while only requiring 0.3% extra parameters and modestly increasing the inference time by 7.3%.
</div>

Finally, to facilitate the ideas discussed on equivariant adaptation of large-scale models, an open-source package [Equiadapt](https://github.com/arnab39/equiadapt) is available from the authors.

## Conclusion

In the ever-evolving world of AI and deep learning, it is critical to ensure models are robust and aware of symmetries. By learning to smartly transform our input data so that they are in the correct orientation for the pretrained models, we can create large-scale models that are powerful and aware of data transformations, bringing us a step closer to AI systems that understand the world as we do. As research into scaling continues, the fusion of large foundational models with equivariant adaptation techniques such as the one presented in this blogpost has the potential to emerge as a fundamental approach in enhancing the consistency and reliability of AI systems.