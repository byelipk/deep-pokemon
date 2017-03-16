# Deep POKEMON
### I'm using TensorFlow to train neural networks on a Kaggle dataset of Pokemon characters and documenting my results.

* First tried classifying *Type 1* and that didn't get anywhere. I was able to overfit the dataset, but wasn't able to achieve a test accuracy above 27%. I think the main problem with classifying *Type 1* is that there are 18 target labels and only 800 examples in the entire dataset. Thus, the proportions of most of the classes were too small to get a generalizable model. I think the dataset would need to grow by several magnitudes in order for classification on this attribute to work.

* I then tried classifying *Generation*. This class only had 6 labels, but once again I didn't have much luck. My guess is for the same reasons.

* My latest attempt was to classify *Legendary*. I'm having much more success with this, achieving accuracy levels in the low to mid 90s. This is a binary classification problem.
