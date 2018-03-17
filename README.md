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

### Data input
$\large X{\small \alpha}$ is the original input, $\large X{\small \beta}$ is the slightly transformed image. The variations are manually created based on the original input. 

The following list of variation, does not intend to, or should not be definitive or conclusive:

* Rotation (upto 10 or -10 degrees)
* Shift in height
* Shift in width
* Recaling in each element(in every channel), in 10% or 10%

$ \large \{( X_{\alpha},X_{\beta}) \in R^3 \}$

### Hashing the image
To train a convolutional neural network model $\large f(X)$

$\large W_{\alpha}=f(X_{\alpha})$

$\large W_{\beta}=f(X_{\beta})$

Now we get $\large W{\small \alpha}$ and $\large W{\small \beta}$

$\large W= \{ (w _{1},w _{2},...,w_{48})\in R;w_{i} \in (0,1) \}$ are vectors of length 48, during inference, it will later be transfromed to hexidecimal string like "8d04a2e4068" of length 12, I call it as the "twin value".

### Loss function
If $\large X{\small \alpha},X{\small \beta}$ are similiar images, $\large W_{\alpha}$ and $\large W_{\beta}$ show have look-alike distribution. Elsewise, the distribution should be as different as possible.

At here, we define a loss function manually:

* $\large L_{mae}(W_{\alpha},W_{\beta})$ is the mean absolute error of $\large W_{\alpha}$ and $\large W_{\beta}$.

* $\large L_{sim}=s(1-\log(L_{mae}(W_{\alpha},W_{\beta})))+(s-1)(-L_{mae}(W_{\alpha},W_{\beta}))$

$s$ is the input indicating the if $ \large X_{\alpha} {\small \&} X_{\beta}$ look alike:

* $s=0$ meaning: $ \large X_{\alpha}  {\small \&}  X_{\beta}$ look alike;
* $s=1$ meaning: $ \large X_{\alpha}  {\small \&}  X_{\beta}$ do not look alike;

The rest is good old Adam optimization, with label purposely set to all zero

### The structure of  $\large f(X)$
$\large f(X)$ use 108,108 as the input size, rgb as color channels.

The preprocessing function normalize each picture, let $ \large \{( X_{\alpha},X_{\beta}) \in R^3 ; X_{\alpha i j c}\in (-1,1),X_{\beta i j c}\in (-1,1)\}$.

For down-sampling, we set convolution stride to (2,2) on conv2d_3,conv2d_6,conv2d_9 to drop the least information. 

Poolings are to be tried for comparison.

The outcome is surprisingly better then other hashing technique, so the following structure wasn't even the result of constant hyper parametering and fine-tuning.

I suspect certain upsampling(deconvolution) first then downsampling will make the model more capable of dealing with height/width/resize variations.

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
_________________________________________________________________
universal_input (InputLayer) (None, 108, 108, 3)       0         
_________________________________________________________________
color_preprocessing (Lambda) (None, 108, 108, 3)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 108, 108, 64)      1792      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 108, 108, 64)      36928     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 54, 54, 64)        36928     
_________________________________________________________________
batch_normalization_1 (Batch (None, 54, 54, 64)        256       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 54, 54, 128)       73856     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 54, 54, 128)       147584    
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 27, 27, 128)       147584    
_________________________________________________________________
batch_normalization_2 (Batch (None, 27, 27, 128)       512       
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 27, 27, 128)       147584    
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 27, 27, 128)       147584    
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 14, 14, 128)       147584    
_________________________________________________________________
batch_normalization_3 (Batch (None, 14, 14, 128)       512       
_________________________________________________________________
flatten_layer (Flatten)      (None, 25088)             0         
_________________________________________________________________
fc2_160 (Dense)              (None, 48)                1204272   
_________________________________________________________________
Total params: 2,092,976

Trainable params: 2,092,336

Non-trainable params: 640
_________________________________________________________________
