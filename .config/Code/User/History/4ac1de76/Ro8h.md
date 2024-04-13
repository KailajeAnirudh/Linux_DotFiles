# Observations
## 12-03-2023

Debugging thre transformer initial model. Noticed a bunch of peculiarities: 

 - The positional encoding requires the data to be sent as (seq_len, batch_size, embedding_dim/channels)
 - x.mean(dim = 1) in line 81 is actually across the samples, which is incorrect. 

 ## 12-06-2023

 Exploring the dataset
 
 Explored the individual label inbalance. The structure of the labels is as follows:

1. Diagnostic: 44 total labels
    Main Classes, each with sub-labels
    - Normal
    - MI: Myocardial Infarction
    - CD: Conduction Disturbance
    - STTC: ST/T-Changes
    - HYP: Hypertrophy

2. Form: 19 form, 4 common with diagnostic

    - 

3. Rhythm: 12 labels:

    - Related to changes such as Arhythmia and Atrial Fibrillation

![Absolute Label Distribution](./LabelDistribution.png)
![Data Structure](./DataStructure.png)

Distribution of Gender: 52% male and 48% female.

Benchmarking Our Models through macro-averaged ROC-AUC. Why? Two Reasons.

- There's a lot of class imbalance The same metric is used in the PTB-XL benchmarks. [1]


# 12-09-2023

## New Design

![Proposed Architecture](./ModelDesign.svg)

### What is done until now?

Tried two different architectures to classify labels(at a given category depth)
 
 - Resnet 101 1d: Smaller number of learnable parameters, give good performance.
 - Transformer based: A lot higher learnable parameteres

Trained on two tasks: 

- Classifying the abnormality super class: (Between Diagnostic, Form and Rhythm)
- Classifying the diagnostic sub-class: (Between MI, HYP, CD, STTC, and NORM)







## References

1. Deep Learning for ECG Analysis: Benchmarks
and Insights from PTB-XL