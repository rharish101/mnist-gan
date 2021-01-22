"""Class for generation using the GAN."""
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tqdm import tqdm


class GANEvaluator:
    """Class to generate images using a GAN.

    Attributes:
        generator: The generator model to be evaluated
        noise_dims: The dimensions for the inputs to the generator
    """

    def __init__(self, generator: Model, noise_dims: int):
        """Store the required objects and info.

        Args:
            generator: The generator model to be evaluated
            noise_dims: The dimensions for the inputs to the generator
        """
        self.generator = generator
        self.noise_dims = noise_dims

    @tf.function
    def _gen_img(self, digits: tf.Tensor) -> tf.Tensor:
        """Generate images of the requested digits."""
        noise = tf.random.normal([digits.shape[0], self.noise_dims])
        label = tf.convert_to_tensor(digits)
        generated = self.generator([noise, label])

        # Convert [-1, 1] float to uint8
        outputs = generated / 2 + 0.5
        outputs = tf.image.convert_image_dtype(outputs, tf.uint8)

        # Convert images into strings having JPEG content
        return tf.map_fn(
            tf.io.encode_jpeg, outputs, fn_output_signature=tf.string
        )

    def generate(
        self, imgs_per_digit: int, batch_size: int, output_dir: Path
    ) -> None:
        """Generate the digits and save them to disk.

        Args:
            imgs_per_digit: The total number of images to generate per digit
            batch_size: The number of images in each batch of generation
            output_dir: Where to save the generated images
        """
        total_imgs = 10 * imgs_per_digit

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        with tqdm(total=total_imgs, desc="Saving") as pbar:
            for start in range(0, total_imgs, batch_size):
                end = min(total_imgs, start + batch_size)
                digits = tf.convert_to_tensor(
                    [i % 10 for i in range(start, end)]
                )
                imgs = self._gen_img(digits).numpy()
                for i, img in enumerate(imgs):
                    idx = start + i
                    img_name = output_dir / f"{idx % 10}-{1 + idx // 10}.jpg"
                    with open(img_name, "wb") as img_file:
                        img_file.write(img)
                    pbar.update()
