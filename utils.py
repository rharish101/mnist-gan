"""Utilities for the MNIST GAN."""
import tensorflow as tf


def wasserstein_gradient_penalty(inputs, outputs, tape):
    """Return the Wasserstein Gradient Penalty loss.

    The original paper can be found at: https://arxiv.org/abs/1704.00028

    Args:
        inputs (`tf.Tensor`): The inputs for the generator
        outputs (`tf.Tensor`): The discriminator's outputs for the images
            generated from the above inputs
        tape (`tf.GradientTape`): The persistent gradient tape for calculating
            gradients

    Returns:
        float: The gradient penalty loss

    """
    grads = tape.gradient(outputs, inputs)
    norm = tf.sqrt(tf.reduce_sum(grads ** 2, axis=1))
    penalty = tf.reduce_mean((norm - 1) ** 2)
    return penalty


@tf.function
def get_grid(img):
    """Convert a batch of float images from [-1, 1] to a uint8 image grid.

    The returned image will contain a batch size of 1.
    """
    img = img / 2 + 0.5  # [-1, 1] to [0, 1]
    img = tf.image.convert_image_dtype(img, tf.uint8)

    batch_size = img.shape[0]
    if batch_size >= 16:
        grid = img[:16]
        grid_shape = (4, 4)
    elif batch_size >= 9:
        grid = img[:9]
        grid_shape = (3, 3)
    elif batch_size >= 4:
        grid = img[:4]
        grid_shape = (2, 2)
    else:
        return img[:1]  # retain the batch axis

    # Reshape to (grid_height, grid_width, img_height, img_width, channels)
    grid = tf.reshape(grid, grid_shape + img.shape[1:])
    # Permute axes to (grid_height, grid_width, img_width, img_height,
    # channels) for convenient combination of widths.
    grid = tf.transpose(grid, perm=[0, 1, 3, 2, 4])
    # Combine grid width and image width
    grid = tf.reshape(grid, (grid_shape[0], -1, img.shape[1], img.shape[3]))
    # Permute axes to (grid_height, img_height, combined_width, channels) for
    # convenient combination of heights.
    grid = tf.transpose(grid, perm=[0, 2, 1, 3])
    # Combine grid height and image height, with a batch axis
    grid = tf.reshape(grid, (1, -1, grid.shape[2], img.shape[3]))

    return grid
