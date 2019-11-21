import tensorflow as tf

K = tf.keras.backend

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import generator

# network parameters
batch_size = 500
epochs = 100
image_size = 128
input_shape = (image_size, image_size, 3)
latent_dim = 64
id = '11'
max_samples = 0
split = 0.8


def make_graph(layers):
    result = layers[0]
    for i in range(1, len(layers)):
        result = layers[i](result)

    return result


def make_models():

    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    def sampling(z_mean_z_log_var):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            z_mean_z_log_var (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = z_mean_z_log_var
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # VAE model = encoder + decoder

    #------------ENC------------

    # build encoder model
    main_inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')  # 128x128x3
    encoder_graph = make_graph([
        main_inputs,
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="SAME", activation='relu'),  # 64x64x32
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="SAME", activation='relu'),  # 32x32x64
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="SAME", activation='relu'),  # 16x16x64
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="SAME", activation='relu'),  # 8x8x64
        tf.keras.layers.Flatten(),  # 4096
        tf.keras.layers.Dense(512, activation='relu')
    ])
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(encoder_graph)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(encoder_graph)

    # use reparameterization trick to push the sampling out as input
    z = tf.keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder_model = tf.keras.models.Model(main_inputs, [z, z_mean, z_log_var], name='encoder')
    encoder_model.summary()
    tf.keras.utils.plot_model(encoder_model, to_file='vae_mlp_encoder.png', show_shapes=True)

    #------------DEC------------

    # build decoder model
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim, ), name='z_sampling')
    decoder_graph = make_graph([
        latent_inputs,
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Reshape(target_shape=(8, 8, 64)),
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="SAME", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding="SAME", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="SAME", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding="SAME", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="SAME", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="SAME", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding="SAME")
    ])

    # instantiate decoder model
    decoder_model = tf.keras.models.Model(latent_inputs, decoder_graph, name='decoder')
    decoder_model.summary()
    tf.keras.utils.plot_model(decoder_model, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    main_outputs = decoder_model(encoder_model(main_inputs)[0])  # 0 refers to Z in encoder model outputs
    vae = tf.keras.models.Model(main_inputs, main_outputs, name='vae_mlp')

    # Make loss function
    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = tf.keras.losses.mse(K.flatten(main_inputs), K.flatten(main_outputs))
    else:
        reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(main_inputs), K.flatten(main_outputs))

    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    #vae.compile(optimizer='adam', loss=vae_loss)
    vae.summary()
    tf.keras.utils.plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

    return vae, encoder_model, decoder_model


if __name__ == '__main__':
    save_file_name = os.path.join('saves', id, f'weights_{id}_{latent_dim}_{epochs}_{max_samples*split}_{batch_size}')
    save_file_path = save_file_name + '.tf'
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", "--weights", help="Load h5 or tf model trained weights", default=save_file_path)
    parser.add_argument('-f', '--file', help='File to predict. Works only when -w is used.')
    parser.add_argument('-c', '--compress', help='File to compress. Works only when -w is used.')
    parser.add_argument('-d', '--decompress', help='File to decompress. Works only when -w is used.')
    parser.add_argument('-F',
                        '--force',
                        help='force image to enter the network, regardless of the face preprocessing',
                        action='store_true')
    parser.add_argument('-D',
                        '--double',
                        help='To compress or decompress using float32 instead of float16.',
                        action='store_true')
    parser.add_argument("-m",
                        "--mse",
                        help="Use mse loss instead of binary cross entropy (default)",
                        action='store_true')
    args = parser.parse_args()

    vae, encoder, decoder = make_models()
    print('looking for', args.weights)
    if args.weights and os.path.isfile(args.weights + '.index'):
        print('found')
        vae.load_weights(args.weights)
        if args.file:
            import cv2
            if args.force:
                img = cv2.resize(cv2.imread(args.file), (128, 128))
            else:
                import face
                img = face.get_face(args.file, resize=(128, 128))

            cv2.imshow('compressed image', img)
            encoded = encoder.predict((img / 255).reshape(1, 128, 128, 3))[0]
            print(encoded)
            with open(args.file + '.compressed', 'wb') as f:
                if not args.double:
                    encoded[0].astype('float16').tofile(f)
                else:
                    encoded[0].tofile(f)

            #encoded = np.linspace(1, -1, 64).reshape(1, 64)
            prediction = decoder.predict(encoded)
            prediction = prediction[0]
            cv2.imshow('result', prediction)
            cv2.imwrite(args.file + '.compressed.png', (prediction * 255).astype(int))

        elif args.compress:
            import cv2
            if args.force:
                img = cv2.resize(cv2.imread(args.file), (128, 128))
            else:
                import face
                img = face.get_face(args.file, resize=(128, 128))

            cv2.imshow('compressed image', img)
            encoded = encoder.predict((img / 255).reshape(1, 128, 128, 3))[0]
            print(encoded)
            with open(args.compress + '.compressed', 'wb') as f:
                if not args.double:
                    encoded[0].astype('float16').tofile(f)
                else:
                    encoded[0].tofile(f)

        elif args.decompress:
            import cv2

            with open(args.decompress, 'rb') as f:
                if not args.double:
                    encoded = np.fromfile(f, dtype='float16')
                else:
                    encoded = np.fromfile(f, dtype='float32')

            #encoded = np.linspace(1, -1, 64).reshape(1, 64)
            prediction = decoder.predict(encoded.reshape(1, 64))
            prediction = prediction[0]
            cv2.imshow('decompressed image', prediction)
            cv2.imwrite(args.decompress + '.png', (prediction * 255).astype(int))

        cv2.waitKey()

    else:
        print('not found')
        # train the autoencoder
        train_generator = generator.DataGenerator('data', batch_size, test=False, split=split, max_samples=max_samples)
        test_generator = generator.DataGenerator('data', batch_size, test=True, split=split, max_samples=max_samples)
        ckpt = tf.keras.callbacks.ModelCheckpoint(save_file_name + '.{epoch:02d}.tf',
                                                  verbose=0,
                                                  monitor='val_loss',
                                                  save_best_only=False,
                                                  save_weights_only=False,
                                                  save_freq=1)
        vae.fit(train_generator,
                validation_data=test_generator,
                epochs=epochs,
                shuffle=False,
                callbacks=[ckpt],
                workers=8,
                use_multiprocessing=True)
        vae.save_weights(save_file_name)
