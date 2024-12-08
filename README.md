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