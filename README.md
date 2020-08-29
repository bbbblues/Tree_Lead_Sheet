# Tree_Lead_Sheet
Structured music generation with a three-layer LSTM tree

## Overview
* Tree Lead Sheet is a deep learning model for symbolic music generation with macro structure reflecting repetition and variation properties. 

* A three-layer recurrent neural network architecture is proposed to model the section-chord-note structure in human music composition. 

* The encoder-decoder model is used to incorporate the concept of motif, the smallest structural unit of an integral music piece. 

* For a more flexible rhythm in generated samples, we directly encode duration of every note event instead of doing time quantization in note representation

* The resulting model can generate convincing structured jazz lead sheets with monophonic melody and harmony.

* Objective evaluation is performed to measure macro structure using Information Rate.
