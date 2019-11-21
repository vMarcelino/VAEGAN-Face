# VAEGAN-Face
Variational Autoencoder with Generative Adversarial Networks for face compression and generation

It's a Variational Adversarial Autoencoder for faces


### Tensorflow 2 required to run

Preferred instalation method is via conda:

`conda create -n vaegan tensorflow opencv dlib`

Replace `tensorflow` by `tensorflow-gpu` for training and running models on GPU

## Training:
`python model.py -w save_file.tf`


## Evaluating:
`python model.py -w save_file.tf -f image.png`

The model accepts a wide variety of image formats, like png, jpg, jpeg etc.


The face will be extracted from the given image, resized to 128x128, encoded to 64 features in latent space and decoded into a 128x128 final image
