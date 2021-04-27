# SPDX-FileCopyrightText: 2019 Harish Rajagopal <harish.rajagopals@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Class for training the GAN."""
from pathlib import Path
from typing import Final, List, NamedTuple, Tuple

import tensorflow as tf
from tensorflow import Tensor, Variable
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow_addons.optimizers import AdamW
from tqdm import tqdm

from ..evaluation import RunningFID
from ..utils import Config, get_grid, reduce_concat


class _Losses(NamedTuple):
    """Holds all the losses for logging.

    Attributes:
        wass: The Wasserstein loss
        grad_pen: The Wasserstein gradient penalty
    """

    wass: Tensor
    grad_pen: Tensor


class GANTrainer:
    """Class to train a GAN.

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
        config: The hyper-param config
        save_dir: Directory where to store model weights
    """

    GEN_PATH: Final = "generator.ckpt"
    CRIT_PATH: Final = "critic.ckpt"

    def __init__(
        self,
        generator: Model,
        critic: Model,
        classifier: Model,
        strategy: tf.distribute.Strategy,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: Config,
        log_dir: Path,
        save_dir: Path,
    ):
        """Store main models and info required for training.

        Args:
            generator: The generator model to be trained
            critic: The critic model to be trained
            classifier: The trained classifier model for FID
            strategy: The distribution strategy for training the GAN
            train_dataset: The dataset of real images and labels for training
            val_dataset: The dataset of real images and labels for validation
            config: The hyper-param config
            log_dir: Directory where to write event logs
            save_dir: Directory where to store model weights
        """
        self.generator = generator
        self.critic = critic

        self.strategy = strategy
        self.mixed_precision = config.mixed_precision

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        with strategy.scope():

            def get_lr_sched(lr):
                return ExponentialDecay(
                    lr, config.decay_steps, config.decay_rate
                )

            self.gen_optim = AdamW(
                get_lr_sched(config.gen_weight_decay),
                get_lr_sched(config.gen_lr),
                0.5,
            )
            self.crit_optim = AdamW(
                get_lr_sched(config.crit_weight_decay),
                get_lr_sched(config.crit_lr),
                0.5,
            )

        if config.mixed_precision:
            self.gen_optim = LossScaleOptimizer(self.gen_optim)
            self.crit_optim = LossScaleOptimizer(self.crit_optim)

        self.evaluator = RunningFID(classifier)
        self.writer = tf.summary.create_file_writer(str(log_dir))

        self.config = config
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

    def _get_losses(
        self,
        real: Tensor,
        generated: Tensor,
        labels: Tensor,
        crit_real_out: Tensor,
        crit_fake_out: Tensor,
    ) -> _Losses:
        """Calculate the losses."""
        # Wasserstein distance
        wass = tf.nn.compute_average_loss(
            crit_real_out - crit_fake_out,
            global_batch_size=self.config.gan_batch_size,
        )
        # Wasserstein Gradient Penalty
        grad_pen = self._gradient_penalty(real, generated, labels)

        return _Losses(wass, grad_pen)

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
            gp_batch, global_batch_size=self.config.gan_batch_size
        )

    def _optimize(
        self,
        train_vars: List[Variable],
        loss: Tensor,
        optim: Optimizer,
        tape: tf.GradientTape,
    ) -> None:
        """Optimize the variables."""
        if self.mixed_precision:
            loss = optim.get_scaled_loss(loss)
        grads = tape.gradient(loss, train_vars)
        if self.mixed_precision:
            grads = optim.get_unscaled_gradients(grads)
        optim.apply_gradients(zip(grads, train_vars))

    def _train_step_critic(
        self, real: Tensor, generated: Tensor, labels: Tensor
    ) -> None:
        """Run a single training step for the critic on a single GPU."""
        with tf.GradientTape() as crit_tape:
            crit_real_out = self.critic([real, labels], training=True)
            crit_fake_out = self.critic([generated, labels], training=True)

            losses = self._get_losses(
                real, generated, labels, crit_real_out, crit_fake_out
            )
            crit_loss = -losses.wass + self.config.gp_weight * losses.grad_pen

        self._optimize(
            self.critic.trainable_variables,
            crit_loss,
            self.crit_optim,
            crit_tape,
        )

    def _train_step(
        self, real: Tensor, labels: Tensor
    ) -> Tuple[Tensor, _Losses]:
        """Run a single training step on a single GPU.

        This training step will train the critic for the required number of
        steps as well as train the generator for a single step.

        Args:
            real: The input real images
            labels: The corresponding input labels

        Returns:
            The generated images
            The losses object
        """
        noise = tf.random.normal((real.get_shape()[0], self.config.noise_dims))

        with tf.GradientTape() as gen_tape:
            generated = self.generator([noise, labels], training=True)

            # No need to calculate gradients of critic optimization
            with gen_tape.stop_recording():
                for _ in range(self.config.crit_steps):
                    self._train_step_critic(real, generated, labels)

            crit_real_out = self.critic([real, labels], training=True)
            crit_fake_out = self.critic([generated, labels], training=True)

            losses = self._get_losses(
                real, generated, labels, crit_real_out, crit_fake_out
            )
            gen_loss = losses.wass

        self._optimize(
            self.generator.trainable_variables,
            gen_loss,
            self.gen_optim,
            gen_tape,
        )

        # Returned values are used for logging summaries
        return generated, losses

    @tf.function
    def train_step(
        self, real: Tensor, labels: Tensor
    ) -> Tuple[Tensor, _Losses]:
        """Run a single training step, distributing across all GPUs.

        Args:
            real: The input real images
            labels: The corresponding input labels

        Returns:
            The generated images
            The losses object
        """
        gen, losses = self.strategy.run(self._train_step, args=(real, labels))

        gen = reduce_concat(self.strategy, gen)
        # Sum losses across all GPUs
        losses = [
            self.strategy.reduce(tf.distribute.ReduceOp.SUM, value, axis=None)
            for value in losses
        ]

        return gen, _Losses(*losses)

    def _get_fid(self) -> Tensor:
        """Calculate FID over the validation dataset."""
        self.evaluator.reset()

        for real, lbls in self.val_dataset:
            inputs = tf.random.normal((real.shape[0], self.config.noise_dims))
            generated = self.generator([inputs, lbls])
            self.evaluator.update(real, generated)

        return self.evaluator.get_fid()

    @staticmethod
    def _get_l2_reg(model: Model) -> Tensor:
        """Get the L2 regularization loss."""
        loss = 0.0
        for param in model.trainable_variables:
            loss += tf.reduce_sum(param**2)
        return loss

    def log_summaries(
        self,
        real: Tensor,
        generated: Tensor,
        losses: _Losses,
        global_step: int,
    ) -> None:
        """Log summaries to disk.

        Args:
            real: The input real images
            generated: The generated images
            losses: The losses object
            global_step: The current global training step
        """
        gen_reg = self._get_l2_reg(self.generator)
        crit_reg = self._get_l2_reg(self.critic)

        with self.writer.as_default():
            with tf.name_scope("losses"):
                tf.summary.scalar(
                    "wasserstein_loss", losses.wass, step=global_step
                )
                tf.summary.scalar(
                    "gradient_penalty", losses.grad_pen, step=global_step
                )
                tf.summary.scalar(
                    "generator_regularization",
                    gen_reg,
                    step=global_step,
                )
                tf.summary.scalar(
                    "critic_regularization",
                    crit_reg,
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
            model.save_weights(self.save_dir / file_name)

    def train(
        self, record_steps: int, save_steps: int, log_graph: bool = False
    ) -> None:
        """Execute the training loops for the GAN.

        Args:
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
        data_in_epochs = (
            item for epoch in range(self.config.gan_epochs) for item in dataset
        )

        # Initialize all optimizer variables
        self._init_optim()

        for global_step, (real, lbls) in tqdm(
            enumerate(data_in_epochs, 1),
            total=self.config.gan_epochs * total_batches,
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
    def load_generator_weights(cls, generator: Model, load_dir: Path) -> None:
        """Load the generator weights from disk.

        This replaces the generator's weights with the loaded ones, in place.

        Args:
            generator: The generator model whose weights are to be loaded
            load_dir: Directory from where to load model weights
        """
        generator.load_weights(load_dir / cls.GEN_PATH)
