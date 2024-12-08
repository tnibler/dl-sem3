All models are trained for ~20 epochs with learning rates from $10^{-3}$ to $10^{-4}$. Most runs used a scheduler that halves the learning rates after three epochs without loss improvement.

Minibatch size: 64

## Very basic model

The noise level/time step is not used at all.

![](images/loss_most_basic.png)

The loss curve maybe shows some suboptimal initialization.

## Conditioning on timestep/noise level

The BatchNorms use affine parameters learned from the noise embedding through a linear layer.

![](images/loss_noise_cond1.png)

![](images/noise_cond1.png)
![](images/noise_cond2.png)

## Spatial Encoding

We add cartesian X/Y and polar coordinates to the model input, which maybe improves coherence of the generated images.

![](images/spatial1.png)
![](images/spatial2.png)

## Add class embeddings to generate specific classes

A 4-dimensional class embedding is trained and concatenated with the input image.

![](images/class_cond1.png)
![](images/class_cond2.png)
![](images/class_cond3.png)

## Generation process visualized

![](images/process.png)

![](images/process2.png)

# Unet


Class conditioning only concatenated to input once: only generates two classes regardless of what is asked for.

Adding it as an input to the conditional batchnorm improves things.

Good bugs:

 - tensors are CoW, so you need to copy them to keep residuals
 - Pooling as first step in a Unet block did not work, as you're throwing away a lot of information before any convolutios

I don't know what was happening here, but some Unet setups worked okay for the first few epochs, but then only ever produced the same blob regardless of sampling or class label:

Epoch 5:
![](images/weird_unet/epoch5-class0.png)
![](images/weird_unet/epoch5-class3.png)

Epoch 10:
![](images/weird_unet/epoch10-class0.png)
![](images/weird_unet/epoch10-class3.png)

Epoch 15:
![](images/weird_unet/epoch15-class3.png)

Epoch 20:
![](images/weird_unet/epoch20-class3.png)

Epoch 25 and after:
![](images/weird_unet/epoch25-class3.png)