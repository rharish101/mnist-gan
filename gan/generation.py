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

    @tf.function
    def _gen_img(self, digits):
        """Generate images of the requested digits."""
        noise = tf.random.normal([digits.shape[0], self.noise_dims])
        label = tf.convert_to_tensor(digits)
        generated = self.generator([noise, label])

        # Convert [-1, 1] float to uint8
        outputs = generated / 2 + 0.5
        outputs = tf.image.convert_image_dtype(outputs, tf.uint8)

        # XXX: Use CPU as a workaround for this bug:
        # https://github.com/tensorflow/tensorflow/issues/28007
        with tf.device("/cpu:0"):
            return tf.map_fn(tf.io.encode_jpeg, outputs, dtype=tf.string)

    def generate(self, imgs_per_digit, batch_size, output_dir):
        """Generate the digits and save them to disk.

        Args:
            total_imgs (int): The total number of images to generate
            batch_size (int): The number of images in each batch of generation
            output_dir (str): Where to save the generated images

        """
        total_imgs = 10 * imgs_per_digit

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with tqdm(total=total_imgs, desc="Saving") as pbar:
            for start in range(0, total_imgs, batch_size):
                end = min(total_imgs, start + batch_size)
                digits = tf.convert_to_tensor(
                    [i % 10 for i in range(start, end)]
                )
                imgs = self._gen_img(digits).numpy()
                for i, img in enumerate(imgs):
                    idx = start + i
                    img_name = os.path.join(
                        output_dir, f"{idx % 10}-{1 + idx // 10}.jpg"
                    )
                    with open(img_name, "wb") as img_file:
                        img_file.write(img)
                    pbar.update()
