# silse
Similar Image In Large Scale Environment

### Abstract

#### Challenge
Online spam accounts often utlize the same avatar, simple hashing method on similarity like md5sum(it's more like exactness instead of similarity) can be evaded easily.

Some similarity algorithm compare the similarity between the activations of images, often end up compare vectors. Such solution require all such vectors loaded into memory.Which puts a lot of engineering headaches when the project swells up in size.

#### Solution
Here we provide a neural network model to hash the image into strings, hence the easiness of storing and engineering. And it can be applied to really large scale. The way we deal with similiar image is simply to perfom SQL operation apply to string in database.

The main point of this technique, is to conquer **variations** created on purpose by the malicious users, with unlabeled dataset.

We can put in the variation mode we choose, the similarity detection can be color/shift/shear/rescale tolerable, but still pin point nothing but the similar images.

For Details, see the [explaining notebook](https://github.com/raynardj/silse/blob/master/sim_city2_explain.ipynb) here.
![Training Structure](https://github.com/raynardj/silse/blob/master/img/training.png?raw=true)
![Production Structure](https://github.com/raynardj/silse/blob/master/img/production.png?raw=true)
![Model Structure](https://github.com/raynardj/silse/blob/master/img/structure.png?raw=true)