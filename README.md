# Similar Image @ Large Scale Environment

![Production Structure](https://github.com/raynardj/silse/blob/master/img/production.png?raw=true)

#### Challenge for data scientists with image dataset

Of course, you work isn't always going to be using well prepared dataset like imagenet and coco.

In some cases, you collected images from your own platform, or crawled them from the internet. In most of these cases, duplicate/ similar image will occur, even in large quantity. The diversity of image feature will be less than we intend to count on. The problem will be worse when sample size is small. Also, correlation between certain features and certain labels will be over stressed. 

For example, in porn detection, if one photo repeated too many times, the bed sheet pattern will be assoiated with positive prediction. The inference will return positive if such bed sheet pattern appear, without any human in the picture.

#### Challenge for any social platform with users

Online spam accounts often use the same avatar, simple hashing method on exactness/similarity like md5sum can be evaded easily.

Some similarity algorithm compare the similarity between the activations of images, often end up compare vectors. Such solution require all such vectors loaded into memory.Which puts a lot of engineering headaches when the project swells up in size.

#### Solution

Here we provide a neural network model to hash the image into strings, hence the easiness of storing and engineering. And it can be applied to really large scale. The way we deal with similar image is simply to perfom SQL operation apply to string in database.

For Details, see the [explaining notebook](https://github.com/raynardj/silse/blob/master/sim_city2_explain.ipynb) here.

![Training Structure](https://github.com/raynardj/silse/blob/master/img/training.png?raw=true)

![Model Structure](https://github.com/raynardj/silse/blob/master/img/structure.png?raw=true)