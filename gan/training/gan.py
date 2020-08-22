"""Class for training the GAN."""
import os
from typing import Dict, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.distribute import ReduceOp, Strategy
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from typing_extensions import Final

from ..evaluation import RunningFID
from ..utils import get_grid, iterator_product, reduce_concat


class GANTrainer:
    """Class to train an MNIST GAN.

    Attributes:
        GEN_PATH: The prefix for the file name of the generator's saved weights
        CRIT_PATH: The prefix for the file name of the critic's saved weights
        generator: The generator model being trained
        critic: The critic model being trained
        train_dataset: The dataset of real images and labels for training
        val_dataset: The dataset of real images and labels for validation
        gen_optim: The optimizer for the generator
        crit_optim: The optimizer for the critic
        evaluator: The object that calculates the running FID
        writer: The summary writer to log TensorBoard summaries
        batch_size: The global batch size
        crit_steps: The number of critic steps per generator step
        noise_dims: The dimensions for the inputs to the generator
        gp_weight: Weights for the critic's gradient penalty
        save_dir: Directory where to store model weights
    """

    GEN_PATH: Final = "generator.ckpt"
    CRIT_PATH: Final = "critic.ckpt"

    def __init__(
        self,
        generator: Model,
        critic: Model,
        classifier: Model,
        strategy: Strategy,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        crit_steps: int,
        noise_dims: int,
        gen_lr: float,
        crit_lr: float,
        gp_weight: float,
        log_dir: str,
        save_dir: str,
    ):
        """Store main models and info required for training.

        Args:
            generator: The generator model to be trained
            critic: The critic model to be trained
            classifier: The trained classifier model for FID
            strategy: The distribution strategy for training the GAN
            train_dataset: The dataset of real images and labels for training
            val_dataset: The dataset of real images and labels for validation
            batch_size: The global batch size
            crit_steps: The number of critic steps per generator step
            noise_dims: The dimensions for the inputs to the generator
            gen_lr: The learning rate for the generator's optimizer
            crit_lr: The learning rate for the critic's optimizer
            gp_weight: Weights for the critic's gradient penalty
            log_dir: Directory where to write event logs
            save_dir: Directory where to store model weights
        """
        self.generator = generator
        self.critic = critic

        self.strategy = strategy

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        with strategy.scope():
            self.gen_optim = Adam(gen_lr, 0.5)
            self.crit_optim = Adam(crit_lr, 0.5)

        self.evaluator = RunningFID(classifier)
        self.writer = tf.summary.create_file_writer(log_dir)

        self.batch_size = batch_size
        self.crit_steps = crit_steps
        self.noise_dims = noise_dims
        self.gp_weight = gp_weight

        self.save_dir = save_dir

    @tf.function
    def _init_optim(self) -> None:
        """Initialize the optimizer variables.

        This is needed because TensorFlow doesn't allow variable creation after
        `tf.function`'s graph has been traced once. This is a workaround for
        the TensorFlow issue here:
        https://github.com/tensorflow/tensorflow/issues/27120
        """

        def create_vars():
            for model, optim in [
                (self.generator, self.gen_optim),
                (self.critic, self.crit_optim),
            ]:
                # The optimizer will initialize its variables only on applying
                # gradients. Therefore, we use zero grads.
                grads_and_vars = [
                    (tf.zeros_like(var), var)
                    for var in model.trainable_variables
                ]
                optim.apply_gradients(grads_and_vars)

        self.strategy.run(create_vars, args=tuple())

    def _gradient_penalty(
        self, real: Tensor, generated: Tensor, labels: Tensor
    ) -> Tensor:
        """Return the Wasserstein Gradient Penalty loss.

        The original paper can be found at: https://arxiv.org/abs/1704.00028

        Args:
            real: The input real images
            generated: The corresponding generated images
            labels: The corresponding input labels

        Returns:
            The gradient penalty loss
        """
        with tf.GradientTape() as tape:
            # U[0, 1] random value used for linear interpolation
            gp_rand = tf.random.uniform(())
            gp_inputs = real * gp_rand + generated * (1 - gp_rand)

            # Forces the tape to track the inputs, which is needed for
            # calculating gradients in the gradient penalty.
            tape.watch(gp_inputs)
            crit_gp_out = self.critic([gp_inputs, labels], training=True)

        grads = tape.gradient(crit_gp_out, gp_inputs)
        flat_grads = tf.reshape(grads, (grads.shape[0], -1))
        norm = tf.norm(flat_grads, axis=1)
        gp_batch = (norm - 1) ** 2
        return tf.nn.compute_average_loss(
            gp_batch, global_batch_size=self.batch_size
        )

    def _train_step(
        self, real: Tensor, labels: Tensor, train_gen: bool
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Run a single training step on a single GPU.

        Args:
            real: The input real images
            labels: The corresponding input labels
            train_gen: Whether to train the generator or train the critic

        Returns:
            The generated images
            The dictionary of losses, as required by `log_summaries`
        """
        noise = tf.random.normal((real.get_shape()[0], self.noise_dims))

        with tf.GradientTape(persistent=True) as tape:
            crit_real_out = self.critic([real, labels], training=True)

            generated = self.generator([noise, labels], training=True)
            crit_fake_out = self.critic([generated, labels], training=True)

            # Wasserstein distance
            wass_loss = tf.nn.compute_average_loss(
                crit_real_out - crit_fake_out,
                global_batch_size=self.batch_size,
            )
            # Wasserstein Gradient Penalty
            grad_pen = self._gradient_penalty(real, generated, labels)

            # Regularization losses
            # NOTE: Regularization needs to be scaled by the number of GPUs in
            # the strategy, as gradients will be added.
            gen_reg = tf.nn.scale_regularization_loss(
                sum(self.generator.losses)
            )
            crit_reg = tf.nn.scale_regularization_loss(sum(self.critic.losses))

            gen_loss = wass_loss + gen_reg
            crit_loss = -wass_loss + crit_reg + self.gp_weight * grad_pen

        if train_gen:
            train_vars = self.generator.trainable_variables
            loss = gen_loss
            optim = self.gen_optim
        else:
            train_vars = self.critic.trainable_variables
            loss = crit_loss
            optim = self.crit_optim

        grads = tape.gradient(loss, train_vars)
        optim.apply_gradients(zip(grads, train_vars))

        # Losses required for logging summaries
        losses = {
            "wass": wass_loss,
            "grad_pen": grad_pen,
            "gen_reg": gen_reg,
            "crit_reg": crit_reg,
        }

        # Returned values are used for logging summaries
        return generated, losses

    @tf.function
    def train_step(
        self, real: Tensor, labels: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Run a single training step, distributing across all GPUs.

        This training step will train the critic for the required number of
        steps as well as train the generator for a single step. The returned
        dict of losses are used for logging summaries.

        Args:
            real: The input real images
            labels: The corresponding input labels

        Returns:
            The generated images
            The dictionary of losses, as required by `log_summaries`
        """
        for _ in range(self.crit_steps):
            self.strategy.run(self._train_step, args=(real, labels, False))
        gen, losses = self.strategy.run(
            self._train_step, args=(real, labels, True)
        )

        gen = reduce_concat(self.strategy, gen)
        # Sum losses across all GPUs
        losses = {
            key: self.strategy.reduce(ReduceOp.SUM, value, axis=None)
            for key, value in losses.items()
        }

        return gen, losses

    def _get_fid(self) -> Tensor:
        """Calculate FID over the validation dataset."""
        self.evaluator.reset()

        for real, lbls in self.val_dataset:
            inputs = tf.random.normal((real.shape[0], self.noise_dims))
            generated = self.generator([inputs, lbls])
            self.evaluator.update(real, generated)

        return self.evaluator.get_fid()

    def log_summaries(
        self,
        real: Tensor,
        generated: Tensor,
        losses: Dict[str, Tensor],
        global_step: int,
    ) -> None:
        """Log summaries to disk.

        The dict of losses should have the following key-value pairs:
            wass: The Wasserstein loss
            grad_pen: The Wasserstein gradient penalty
            gen_reg: The L2 regularization loss for the generator
            crit_reg: The L2 regularization loss for the critic

        Args:
            real: The input real images
            generated: The generated images
            losses: The dictionary of losses
            global_step: The current global training step
        """
        with self.writer.as_default():
            with tf.name_scope("losses"):
                tf.summary.scalar(
                    "wasserstein_loss", losses["wass"], step=global_step
                )
                tf.summary.scalar(
                    "gradient_penalty", losses["grad_pen"], step=global_step
                )
                tf.summary.scalar(
                    "generator_regularization",
                    losses["gen_reg"],
                    step=global_step,
                )
                tf.summary.scalar(
                    "critic_regularization",
                    losses["crit_reg"],
                    step=global_step,
                )

            with tf.name_scope("metrics"):
                fid = self._get_fid()
                tf.summary.scalar("FID", fid, step=global_step)

            # Save generated and real images in a square grid
            with tf.name_scope("images"):
                real_grid = get_grid(real)
                gen_grid = get_grid(generated)
                tf.summary.image("real", real_grid, step=global_step)
                tf.summary.image("generated", gen_grid, step=global_step)

            with tf.name_scope("generator"):
                for var in self.generator.trainable_variables:
                    tf.summary.histogram(var.name, var, step=global_step)

            with tf.name_scope("critic"):
                for var in self.critic.trainable_variables:
                    tf.summary.histogram(var.name, var, step=global_step)

    def save_models(self) -> None:
        """Save the models to disk."""
        for model, file_name in [
            (self.generator, self.GEN_PATH),
            (self.critic, self.CRIT_PATH),
        ]:
            model.save_weights(os.path.join(self.save_dir, file_name))

    def train(
        self,
        epochs: int,
        record_steps: int,
        save_steps: int,
        log_graph: bool = False,
    ) -> None:
        """Execute the training loops for the GAN.

        Args:
            epochs: Number of epochs to train the GAN
            record_steps: Step interval for recording summaries
            save_steps: Step interval for saving the model
            log_graph: Whether to log the graph of the model
        """
        # Total no. of batches in the training dataset
        total_batches = self.train_dataset.cardinality().numpy()
        dataset = self.strategy.experimental_distribute_dataset(
            self.train_dataset
        )
        # Iterate over dataset in epochs
        data_in_epochs = iterator_product(range(epochs), dataset)

        # Initialize all optimizer variables
        self._init_optim()

        for global_step, (_, (real, lbls)) in tqdm(
            enumerate(data_in_epochs, 1),
            total=epochs * total_batches,
            desc="Training",
        ):
            # The graph must be exported the first time the tf.function is run,
            # otherwise the graph is empty.
            if global_step == 1 and log_graph:
                tf.summary.trace_on()

            gen, losses = self.train_step(real, lbls)

            if global_step == 1 and log_graph:
                with self.writer.as_default():
                    tf.summary.trace_export("gan_step", step=global_step)

            if global_step % record_steps == 0:
                self.log_summaries(
                    reduce_concat(self.strategy, real),
                    gen,
                    losses,
                    global_step,
                )

            if global_step % save_steps == 0:
                self.save_models()

        self.save_models()

    @classmethod
    def load_generator_weights(cls, generator: Model, load_dir: str) -> None:
        """Load the generator weights from disk.

        This replaces the generator's weights with the loaded ones, in place.

        Args:
            generator: The generator model whose weights are to be loaded
            load_dir: Directory from where to load model weights
        """
        generator.load_weights(os.path.join(load_dir, cls.GEN_PATH))
