# Tree_Lead_Sheet
Structured music generation with a three-layer LSTM tree

## Overview
* Tree Lead Sheet is a deep learning model for symbolic music generation with macro structure reflecting repetition and variation properties. 

* A three-layer recurrent neural network architecture is proposed to model the section-chord-note structure in human music composition. 

* The encoder-decoder model is used to incorporate the concept of motif, the smallest structural unit of an integral music piece. 

* For a more flexible rhythm in generated samples, we directly encode duration of every note event instead of doing time quantization in note representation

* The resulting model can generate convincing structured jazz lead sheets with monophonic melody and harmony.

* Objective evaluation is performed to measure macro structure using Information Rate.

## An Example of Generated Score

![generated_score_example](./generated_score_example.png =100x100)

## Model Structure

![model_structure](./model_structure.jpg =100x100)

[Model Demonstration](./Demo_Demonstration_Jazz_Generation.pdf)

## Datasets
* [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html)
* The Real Bebop Book and The Charlie Parker Real Book in musicXML format 

## Objective Evaluation Results

![eval](./avg_ir_box.png 100x100)

The balance of repetition and variation is a critical property in generated music with a self-similarity structure. We design our objective evaluation based on this ratio, which can be formulated as Information Rate(IR). For a given sequence, it is defined as the mutual information between the present and the past observations.

It is clear that our proposed model performs better than the baseline model on the average IR value, which indicates more distinct macro structures in music pieces. Nevertheless, the IR of the training samples is the highest.

The baseline model only consists of a single-layer notes decoder and an encoder using notes as seed sequences.
