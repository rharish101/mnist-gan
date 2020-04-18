"""Class for generation using the GAN."""
import os

import tensorflow as tf
from tqdm import tqdm


class BiGANImgGenHelper:
    """Class to generate images using an MNIST BiGAN."""

    def __init__(self, generator, noise_dims):
        """Store the required objects and info.

        Args:
            generator (`tf.keras.Model`): The generator model to be trained
            noise_dims (int): The dimensions for the inputs to the generator

        """
        self.generator = generator
        self.noise_dims = noise_dims

    def generate(self, imgs_per_digit, output_dir):
        """Generate the digits and save them to disk.

        Args:
            imgs_per_digit (int): The number of images to generate per digit
            output_dir (str): Where to save the generated images

        """
        with tqdm(total=10 * imgs_per_digit, desc="Saving") as pbar:
            for digit in range(10):
                for instance in range(imgs_per_digit):
                    noise = tf.random.normal([1, self.noise_dims])
                    label = tf.convert_to_tensor([digit])  # adding batch-axis
                    generated = self.generator([noise, label])

                    # Convert 4D [-1, 1] float32 to 3D [0, 255] uint8
                    output = generated / 2 + 0.5
                    output = tf.image.convert_image_dtype(output, tf.uint8)
                    output = output[0]

                    img_str = tf.image.encode_jpeg(output)
                    img_name = os.path.join(
                        output_dir, f"{digit}_{instance}.jpg"
                    )
                    with open(img_name, "wb") as img_file:
                        img_file.write(img_str.numpy())

                    pbar.update()
