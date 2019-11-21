# VAEGAN-Face
Variational Autoencoder with Generative Adversarial Networks for face compression and generation

It's a Variational Adversarial Convolutional Autoencoder for faces


(ps: The GAN part is in WIP phase)

### Tensorflow 2 required to run

Preferred instalation method is via conda:

`conda create -c conda-forge -n vaegan tensorflow opencv dlib pydot graphviz`

Replace `tensorflow` by `tensorflow-gpu` for training and running models on GPU

## Training:
It is expected that the data for training lives inside a folder named `data`


`python model.py -w save_file.tf`


## Evaluating:
`python model.py -w save_file.tf -f image.png`

The model accepts a wide variety of image formats, like png, jpg, jpeg etc.


The face will be extracted from the given image, resized to 128x128, encoded to 64 features in latent space and decoded into a 128x128 final image


## Compressing
`python model.py -w save_file.tf -c image.png`

## Decompressing
`python model.py -w save_file.tf -d image.compressed`