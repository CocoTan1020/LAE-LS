"""Tweaked version of corresponding AllenNLP file"""
import datetime
import logging
import math
import os
import time
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any
import torch
import torch.optim.lr_scheduler
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, gpu_memory_mb, peak_memory_mb, lazy_groups_of
from allennlp.data.fields import LabelField
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer_base import TrainerBase
from torch import nn

from utils.helpers import metadata2senteces, write_json_data, wordlistlist2sentence, write_json_for
from gector.seq2similarityloss import Remainloss, Semanticloss, MarginLoss, ContrastLoss

logger = logging.getLogger(__name__)
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

class Trainer(TrainerBase):
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        iterator: DataIterator,
        train_dataset: Iterable[Instance],
        validation_dataset: Optional[Iterable[Instance]] = None,
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_iterator: DataIterator = None,
        shuffle: bool = True,
        num_epochs: int = 20,
        accumulated_batch_count: int = 1,
        serialization_dir: Optional[str] = None,
        num_serialized_models_to_keep: int = 20,
        keep_serialized_model_every_num_seconds: int = None,
        checkpointer: Checkpointer = None,
        model_save_interval: float = None,
        cuda_device: Union[int, List] = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        summary_interval: int = 100,
        histogram_interval: int = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        log_batch_size_period: Optional[int] = None,
        moving_average: Optional[MovingAverage] = None,
        cold_step_count: int = 0,
        cold_lr: float = 1e-3,
        cuda_verbose_step=None,
    ) -> None:
        """
        A trainer for doing supervised learning. It just takes a labeled dataset
        and a ``DataIterator``, and uses the supplied ``Optimizer`` to learn the weights
        for your model over some fixed number of epochs. You can also pass in a validation
        dataset and enable early stopping. There are many other bells and whistles as well.

        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.

            If you are training your model using GPUs, your model should already be
            on the correct device. (If you use `Trainer.from_params` this will be
            handled for you.)
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_iterator : ``DataIterator``, optional (default=None)
            An iterator to use for the validation set.  If ``None``, then
            use the training `iterator`.
        shuffle: ``bool``, optional (default=True)
            Whether to shuffle the instances in the iterator or not.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : ``int``, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : ``int``, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        checkpointer : ``Checkpointer``, optional (default=None)
            An instance of class Checkpointer to use instead of the default. If a checkpointer is specified,
            the arguments num_serialized_models_to_keep and keep_serialized_model_every_num_seconds should
            not be specified. The caller is responsible for initializing the checkpointer so that it is
            consistent with serialization_dir.
        model_save_interval : ``float``, optional (default=None)
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        cuda_device : ``Union[int, List[int]]``, optional (default = -1)
            An integer or list of integers specifying the CUDA device(s) to use. If -1, the CPU is used.
        grad_norm : ``float``, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : ``LearningRateScheduler``, optional (default = None)
            If specified, the learning rate will be decayed with respect to
            this schedule at the end of each epoch (or batch, if the scheduler implements
            the ``step_batch`` method). If you use :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`,
            this will use the ``validation_metric`` provided to determine if learning has plateaued.
            To support updating the learning rate on every batch, this can optionally implement
            ``step_batch(batch_num_total)`` which updates the learning rate given the batch number.
        momentum_scheduler : ``MomentumScheduler``, optional (default = None)
            If specified, the momentum will be updated at the end of each batch or epoch
            according to the schedule.
        summary_interval: ``int``, optional, (default = 100)
            Number of batches between logging scalars to tensorboard
        histogram_interval : ``int``, optional, (default = ``None``)
            If not None, then log histograms to tensorboard every ``histogram_interval`` batches.
            When this parameter is specified, the following additional logging is enabled:
                * Histograms of model parameters
                * The ratio of parameter update norm to parameter norm
                * Histogram of layer activations
            We log histograms of the parameters returned by
            ``model.get_parameters_for_histogram_tensorboard_logging``.
            The layer activations are logged for any modules in the ``Model`` that have
            the attribute ``should_log_activations`` set to ``True``.  Logging
            histograms requires a number of GPU-CPU copies during training and is typically
            slow, so we recommend logging histograms relatively infrequently.
            Note: only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.
        should_log_parameter_statistics : ``bool``, optional, (default = True)
            Whether to send parameter statistics (mean and standard deviation
            of parameters and gradients) to tensorboard.
        should_log_learning_rate : ``bool``, optional, (default = False)
            Whether to send parameter specific learning rate to tensorboard.
        log_batch_size_period : ``int``, optional, (default = ``None``)
            If defined, how often to log the average batch size.
        moving_average: ``MovingAverage``, optional, (default = None)
            If provided, we will maintain moving averages for all parameters. During training, we
            employ a shadow variable for each parameter, which maintains the moving average. During
            evaluation, we backup the original parameters and assign the moving averages to corresponding
            parameters. Be careful that when saving the checkpoint, we will save the moving averages of
            parameters. This is necessary because we want the saved model to perform as well as the validated
            model if we load it later. But this may cause problems if you restart the training from checkpoint.
        """
        super().__init__(serialization_dir, cuda_device)

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model

        self.iterator = iterator
        self._validation_iterator = validation_iterator
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_dataset
        self._validation_data = validation_dataset
        self.accumulated_batch_count = accumulated_batch_count
        self.cold_step_count = cold_step_count
        self.cold_lr = cold_lr
        self.cuda_verbose_step = cuda_verbose_step

        if patience is None:  # no early stopping
            if validation_dataset:
                logger.warning(
                    "You provided a validation dataset but patience was set to None, "
                    "meaning that early stopping is disabled"
                )
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError(
                '{} is an invalid value for "patience": it must be a positive integer '
                "or None (if you want to disable early stopping)".format(patience)
            )

        # For tracking is_best_so_far and should_stop_early
        self._metric_tracker = MetricTracker(patience, validation_metric)
        # Get rid of + or -
        self._validation_metric = validation_metric[1:]

        self._num_epochs = num_epochs

        if checkpointer is not None:
            # We can't easily check if these parameters were passed in, so check against their default values.
            # We don't check against serialization_dir since it is also used by the parent class.
            if num_serialized_models_to_keep != 20 \
                    or keep_serialized_model_every_num_seconds is not None:
                raise ConfigurationError(
                    "When passing a custom Checkpointer, you may not also pass in separate checkpointer "
                    "args 'num_serialized_models_to_keep' or 'keep_serialized_model_every_num_seconds'."
                )
            self._checkpointer = checkpointer
        else:
            self._checkpointer = Checkpointer(
                serialization_dir,
                keep_serialized_model_every_num_seconds,
                num_serialized_models_to_keep,
            )

        self._model_save_interval = model_save_interval

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping

        self._learning_rate_scheduler = learning_rate_scheduler
        self._momentum_scheduler = momentum_scheduler
        self._moving_average = moving_average

        # We keep the total batch number as an instance variable because it
        # is used inside a closure for the hook which logs activations in
        # ``_enable_activation_logging``.
        self._batch_num_total = 0

        self._tensorboard = TensorboardWriter(
            get_batch_num_total=lambda: self._batch_num_total,
            serialization_dir=serialization_dir,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
        )

        self._log_batch_size_period = log_batch_size_period

        self._last_log = 0.0  # time of last logging

        # Enable activation logging.
        if histogram_interval is not None:
            self._tensorboard.enable_activation_logging(self.model)

    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.model, self._grad_norm)

    def batch_loss(self, batch_group: List[TensorDict], for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = training_util.data_parallel(batch_group, self.model, self._cuda_devices)
        else:
            assert len(batch_group) == 1
            batch = batch_group[0]
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            output_dict = self.model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
                # print('self.model.get_regularization_penalty()', self.model.get_regularization_penalty())  # 0.0
        except KeyError:
            if for_training:
                raise RuntimeError(
                    "The model you are trying to optimize does not contain a"
                    " 'loss' key in the output of model.forward(inputs)."
                )
            loss = None

        return loss

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        print("Epoch ", epoch, "/", self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        print((f"Peak CPU memory usage MB: " + str(peak_cpu_usage)))
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")
            print((f"GPU memory usage MB: " + str(memory)))

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()

        num_gpus = len(self._cuda_devices)

        # Get tqdm for the training batches
        raw_train_generator = self.iterator(self.train_data, num_epochs=1, shuffle=self.shuffle)

        # print('self.train_data')
        # temp = 0
        # for i in self.train_data:
        #     temp += 1
        #     print('temp', temp)
        #     if temp == 59 or temp == 60:
        #         print(i)

        # print('raw_train_generator')
        # temp = 0
        # for i in raw_train_generator:
        #     temp += 1
        #     print('temp', temp)
        #     print(i)
        train_generator = lazy_groups_of(raw_train_generator, num_gpus)
        num_training_batches = math.ceil(self.iterator.get_num_batches(self.train_data) / num_gpus)
        residue = num_training_batches % self.accumulated_batch_count
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        print('Training')
        # print('num_training_batches', num_training_batches)
        # print('train_generator', train_generator)
        # print(len(train_generator))
        # temp = 0
        # for i in train_generator:
        #     temp += 1
        #     print('temp', temp)
        #     print(i)
        train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_training_batches)
        cumulative_batch_size = 0
        self.optimizer.zero_grad()
        for batch_group in train_generator_tqdm:
            # print('batch_group', batch_group) # {tokens{bert, bert-offsets, mask}, metadata, labels, d_tags}
            # print(batch_group[0]['tokens']['bert'].shape)
            # print(batch_group[0]['tokens']['bert-offsets'].shape)
            # print(batch_group[0]['tokens']['mask'].shape)
            # print(len(batch_group[0]['metadata']))
            # print(batch_group[0]['labels'].shape)
            # print(batch_group[0]['d_tags'].shape)
            # eg.
            # torch.Size([32, 49])
            # torch.Size([32, 40])
            # torch.Size([32, 40])
            # 32
            # torch.Size([32, 40])
            # torch.Size([32, 40])
            # for x in batch_group:
            #     print("x['tokens']", x['tokens'])
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            iter_len = self.accumulated_batch_count \
                if batches_this_epoch <= (num_training_batches - residue) else residue

            if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
                print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')
            try:
                loss = self.batch_loss(batch_group, for_training=True) / iter_len
            except RuntimeError as e:
                print(e)
                for x in batch_group:
                    all_words = [len(y['words']) for y in x['metadata']]
                    print(f"Total sents: {len(all_words)}. "
                          f"Min {min(all_words)}. Max {max(all_words)}")
                    for elem in ['labels', 'd_tags']:
                        tt = x[elem]
                        print(
                            f"{elem} shape {list(tt.shape)} and min {tt.min().item()} and {tt.max().item()}")
                    for elem in ["bert", "mask", "bert-offsets"]:
                        tt = x['tokens'][elem]
                        print(
                            f"{elem} shape {list(tt.shape)} and min {tt.min().item()} and {tt.max().item()}")
                raise e

            if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
                print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            loss.backward()

            if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
                print(f'After backprop - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After backprop - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            train_loss += loss.item() * iter_len

            del batch_group, loss
            torch.cuda.empty_cache()

            if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
                print(f'After collecting garbage - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After collecting garbage - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            if self._tensorboard.should_log_histograms_this_batch():
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {
                    name: param.detach().cpu().clone()
                    for name, param in self.model.named_parameters()
                }
                if batches_this_epoch % self.accumulated_batch_count == 0 or \
                        batches_this_epoch == num_training_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1))
                    param_norm = torch.norm(param.view(-1)).cpu()
                    self._tensorboard.add_train_scalar(
                        "gradient_update/" + name, update_norm / (param_norm + 1e-7)
                    )
            else:
                if batches_this_epoch % self.accumulated_batch_count == 0 or \
                        batches_this_epoch == num_training_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch)
            description = training_util.description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if self._tensorboard.should_log_this_batch():
                self._tensorboard.log_parameter_and_gradient_statistics(self.model, batch_grad_norm)
                self._tensorboard.log_learning_rates(self.model, self.optimizer)

                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                self._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._tensorboard.should_log_histograms_this_batch():
                self._tensorboard.log_histograms(self.model, histogram_parameters)

            if self._log_batch_size_period:
                cur_batch = sum([training_util.get_batch_size(batch) for batch in batch_group])
                cumulative_batch_size += cur_batch
                if (batches_this_epoch - 1) % self._log_batch_size_period == 0:
                    average = cumulative_batch_size / batches_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    self._tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    self._tensorboard.add_train_scalar("mean_batch_size", average)

            # Save model if needed.
            if self._model_save_interval is not None and (
                time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                    "{0}.{1}".format(epoch, training_util.time_to_str(int(last_save_time)))
                )

        metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch, reset=True)
        metrics["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        return metrics

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self.model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)

        raw_val_generator = val_iterator(self._validation_data, num_epochs=1, shuffle=False)
        val_generator = lazy_groups_of(raw_val_generator, num_gpus)
        num_validation_batches = math.ceil(
            val_iterator.get_num_batches(self._validation_data) / num_gpus
        )
        val_generator_tqdm = Tqdm.tqdm(val_generator, total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch_group in val_generator_tqdm:

            loss = self.batch_loss(batch_group, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()

        return val_loss, batches_this_epoch

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        if self.cold_step_count > 0:
            base_lr = self.optimizer.param_groups[0]['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cold_lr
            self.model.text_field_embedder._token_embedders['bert'].set_weights(freeze=True)

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for epoch in range(epoch_counter, self._num_epochs):
            if epoch == self.cold_step_count and epoch != 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = base_lr
                self.model.text_field_embedder._token_embedders['bert'].set_weights(freeze=False)

            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if "cpu_memory_MB" in train_metrics:
                metrics["peak_cpu_memory_MB"] = max(
                    metrics.get("peak_cpu_memory_MB", 0), train_metrics["cpu_memory_MB"]
                )
            for key, value in train_metrics.items():
                if key.startswith("gpu_"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            # clear cache before validation
            torch.cuda.empty_cache()
            if self._validation_data is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = training_util.get_metrics(
                        self.model, val_loss, num_batches, reset=True
                    )

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            self._tensorboard.log_metrics(
                train_metrics, val_metrics=val_metrics, log_to_console=True, epoch=epoch + 1
            )  # +1 because tensorboard doesn't like 0

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            # if self.cold_step_count <= epoch:
            self.scheduler.step(metrics['validation_loss'])

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir:
                dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
                )

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric, epoch)

            self._save_checkpoint(epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (
                    (self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1
                )
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # make sure pending events are flushed to disk and files are closed properly
        # self._tensorboard.close()

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

    def _save_checkpoint(self, epoch: Union[int, str]) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        """
        # If moving averages are used for parameters, we save
        # the moving average values into checkpoint, instead of the current values.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        # These are the training states we need to persist.
        training_states = {
            "metric_tracker": self._metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "batch_num_total": self._batch_num_total,
        }

        # If we have a learning rate or momentum scheduler, we should persist them too.
        if self._learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()
        if self._momentum_scheduler is not None:
            training_states["momentum_scheduler"] = self._momentum_scheduler.state_dict()

        self._checkpointer.save_checkpoint(
            model_state=self.model.state_dict(),
            epoch=epoch,
            training_states=training_states,
            is_best_so_far=self._metric_tracker.is_best_so_far(),
        )

        # Restore the original values for parameters so that training will not be affected.
        if self._moving_average is not None:
            self._moving_average.restore()

    def _restore_checkpoint(self) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        Returns
        -------
        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        model_state, training_state = self._checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self._learning_rate_scheduler is not None \
                and "learning_rate_scheduler" in training_state:
            self._learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        if self._momentum_scheduler is not None and "momentum_scheduler" in training_state:
            self._momentum_scheduler.load_state_dict(training_state["momentum_scheduler"])
        training_util.move_optimizer_to_cuda(self.optimizer)

        # Currently the ``training_state`` contains a serialized ``MetricTracker``.
        if "metric_tracker" in training_state:
            self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked ``val_metric_per_epoch``.
        elif "val_metric_per_epoch" in training_state:
            self._metric_tracker.clear()
            self._metric_tracker.add_metrics(training_state["val_metric_per_epoch"])
        # And before that we didn't track anything.
        else:
            self._metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split(".")[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get("batch_num_total")
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return

    # Requires custom from_params.
    @classmethod
    def from_params(  # type: ignore
        cls,
        model: Model,
        serialization_dir: str,
        iterator: DataIterator,
        train_data: Iterable[Instance],
        validation_data: Optional[Iterable[Instance]],
        params: Params,
        validation_iterator: DataIterator = None,
    ) -> "Trainer":

        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        momentum_scheduler_params = params.pop("momentum_scheduler", None)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
        if "moving_average" in params:
            moving_average = MovingAverage.from_params(
                params.pop("moving_average"), parameters=parameters
            )
        else:
            moving_average = None

        if lr_scheduler_params:
            lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            lr_scheduler = None
        if momentum_scheduler_params:
            momentum_scheduler = MomentumScheduler.from_params(optimizer, momentum_scheduler_params)
        else:
            momentum_scheduler = None

        if "checkpointer" in params:
            if "keep_serialized_model_every_num_seconds" in params \
                    or "num_serialized_models_to_keep" in params:
                raise ConfigurationError(
                    "Checkpointer may be initialized either from the 'checkpointer' key or from the "
                    "keys 'num_serialized_models_to_keep' and 'keep_serialized_model_every_num_seconds'"
                    " but the passed config uses both methods."
                )
            checkpointer = Checkpointer.from_params(params.pop("checkpointer"))
        else:
            num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
            keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None
            )
            checkpointer = Checkpointer(
                serialization_dir=serialization_dir,
                num_serialized_models_to_keep=num_serialized_models_to_keep,
                keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
            )
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)

        params.assert_empty(cls.__name__)
        return cls(
            model,
            optimizer,
            iterator,
            train_data,
            validation_data,
            patience=patience,
            validation_metric=validation_metric,
            validation_iterator=validation_iterator,
            shuffle=shuffle,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=lr_scheduler,
            momentum_scheduler=momentum_scheduler,
            checkpointer=checkpointer,
            model_save_interval=model_save_interval,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
            log_batch_size_period=log_batch_size_period,
            moving_average=moving_average,
        )


class Trainer_Discriminator(TrainerBase):
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        iterator: DataIterator,
        train_dataset: Iterable[Instance],
        validation_dataset: Optional[Iterable[Instance]] = None,
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_iterator: DataIterator = None,
        shuffle: bool = True,
        num_epochs: int = 20,
        accumulated_batch_count: int = 1,
        serialization_dir: Optional[str] = None,
        num_serialized_models_to_keep: int = 20,
        keep_serialized_model_every_num_seconds: int = None,
        checkpointer: Checkpointer = None,
        model_save_interval: float = None,
        cuda_device: Union[int, List] = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        summary_interval: int = 100,
        histogram_interval: int = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        log_batch_size_period: Optional[int] = None,
        moving_average: Optional[MovingAverage] = None,
        cold_step_count: int = 0,
        cold_lr: float = 1e-3,
        cuda_verbose_step=None,
    ) -> None:
        """
        A trainer for doing supervised learning. It just takes a labeled dataset
        and a ``DataIterator``, and uses the supplied ``Optimizer`` to learn the weights
        for your model over some fixed number of epochs. You can also pass in a validation
        dataset and enable early stopping. There are many other bells and whistles as well.

        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.

            If you are training your model using GPUs, your model should already be
            on the correct device. (If you use `Trainer.from_params` this will be
            handled for you.)
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_iterator : ``DataIterator``, optional (default=None)
            An iterator to use for the validation set.  If ``None``, then
            use the training `iterator`.
        shuffle: ``bool``, optional (default=True)
            Whether to shuffle the instances in the iterator or not.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : ``int``, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : ``int``, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        checkpointer : ``Checkpointer``, optional (default=None)
            An instance of class Checkpointer to use instead of the default. If a checkpointer is specified,
            the arguments num_serialized_models_to_keep and keep_serialized_model_every_num_seconds should
            not be specified. The caller is responsible for initializing the checkpointer so that it is
            consistent with serialization_dir.
        model_save_interval : ``float``, optional (default=None)
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        cuda_device : ``Union[int, List[int]]``, optional (default = -1)
            An integer or list of integers specifying the CUDA device(s) to use. If -1, the CPU is used.
        grad_norm : ``float``, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : ``LearningRateScheduler``, optional (default = None)
            If specified, the learning rate will be decayed with respect to
            this schedule at the end of each epoch (or batch, if the scheduler implements
            the ``step_batch`` method). If you use :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`,
            this will use the ``validation_metric`` provided to determine if learning has plateaued.
            To support updating the learning rate on every batch, this can optionally implement
            ``step_batch(batch_num_total)`` which updates the learning rate given the batch number.
        momentum_scheduler : ``MomentumScheduler``, optional (default = None)
            If specified, the momentum will be updated at the end of each batch or epoch
            according to the schedule.
        summary_interval: ``int``, optional, (default = 100)
            Number of batches between logging scalars to tensorboard
        histogram_interval : ``int``, optional, (default = ``None``)
            If not None, then log histograms to tensorboard every ``histogram_interval`` batches.
            When this parameter is specified, the following additional logging is enabled:
                * Histograms of model parameters
                * The ratio of parameter update norm to parameter norm
                * Histogram of layer activations
            We log histograms of the parameters returned by
            ``model.get_parameters_for_histogram_tensorboard_logging``.
            The layer activations are logged for any modules in the ``Model`` that have
            the attribute ``should_log_activations`` set to ``True``.  Logging
            histograms requires a number of GPU-CPU copies during training and is typically
            slow, so we recommend logging histograms relatively infrequently.
            Note: only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.
        should_log_parameter_statistics : ``bool``, optional, (default = True)
            Whether to send parameter statistics (mean and standard deviation
            of parameters and gradients) to tensorboard.
        should_log_learning_rate : ``bool``, optional, (default = False)
            Whether to send parameter specific learning rate to tensorboard.
        log_batch_size_period : ``int``, optional, (default = ``None``)
            If defined, how often to log the average batch size.
        moving_average: ``MovingAverage``, optional, (default = None)
            If provided, we will maintain moving averages for all parameters. During training, we
            employ a shadow variable for each parameter, which maintains the moving average. During
            evaluation, we backup the original parameters and assign the moving averages to corresponding
            parameters. Be careful that when saving the checkpoint, we will save the moving averages of
            parameters. This is necessary because we want the saved model to perform as well as the validated
            model if we load it later. But this may cause problems if you restart the training from checkpoint.
        """
        super().__init__(serialization_dir, cuda_device)

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model

        self.iterator = iterator
        self._validation_iterator = validation_iterator
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_dataset
        self._validation_data = validation_dataset
        self.accumulated_batch_count = accumulated_batch_count
        self.cold_step_count = cold_step_count
        self.cold_lr = cold_lr
        self.cuda_verbose_step = cuda_verbose_step

        if patience is None:  # no early stopping
            if validation_dataset:
                logger.warning(
                    "You provided a validation dataset but patience was set to None, "
                    "meaning that early stopping is disabled"
                )
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError(
                '{} is an invalid value for "patience": it must be a positive integer '
                "or None (if you want to disable early stopping)".format(patience)
            )

        # For tracking is_best_so_far and should_stop_early
        self._metric_tracker = MetricTracker(patience, validation_metric)
        # Get rid of + or -
        self._validation_metric = validation_metric[1:]

        self._num_epochs = num_epochs

        if checkpointer is not None:
            # We can't easily check if these parameters were passed in, so check against their default values.
            # We don't check against serialization_dir since it is also used by the parent class.
            if num_serialized_models_to_keep != 20 \
                    or keep_serialized_model_every_num_seconds is not None:
                raise ConfigurationError(
                    "When passing a custom Checkpointer, you may not also pass in separate checkpointer "
                    "args 'num_serialized_models_to_keep' or 'keep_serialized_model_every_num_seconds'."
                )
            self._checkpointer = checkpointer
        else:
            self._checkpointer = Checkpointer(
                serialization_dir,
                keep_serialized_model_every_num_seconds,
                num_serialized_models_to_keep,
            )

        self._model_save_interval = model_save_interval

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping

        self._learning_rate_scheduler = learning_rate_scheduler
        self._momentum_scheduler = momentum_scheduler
        self._moving_average = moving_average

        # We keep the total batch number as an instance variable because it
        # is used inside a closure for the hook which logs activations in
        # ``_enable_activation_logging``.
        self._batch_num_total = 0

        self._tensorboard = TensorboardWriter(
            get_batch_num_total=lambda: self._batch_num_total,
            serialization_dir=serialization_dir,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
        )

        self._log_batch_size_period = log_batch_size_period

        self._last_log = 0.0  # time of last logging

        # Enable activation logging.
        if histogram_interval is not None:
            self._tensorboard.enable_activation_logging(self.model)

    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.model, self._grad_norm)

    def batch_loss(self, batch_group: List[TensorDict], for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = training_util.data_parallel(batch_group, self.model, self._cuda_devices)
        else:
            assert len(batch_group) == 1
            batch = batch_group[0]
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            # print('batch', batch)
            output_dict = self.model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()

        except KeyError:
            if for_training:
                raise RuntimeError(
                    "The model you are trying to optimize does not contain a"
                    " 'loss' key in the output of model.forward(inputs)."
                )
            loss = None

        return loss

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        print("Epoch ", epoch, "/", self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        print((f"Peak CPU memory usage MB: " + str(peak_cpu_usage)))
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")
            print((f"GPU memory usage MB: " + str(memory)))

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()

        num_gpus = len(self._cuda_devices)

        # Get tqdm for the training batches
        raw_train_generator = self.iterator(self.train_data, num_epochs=1, shuffle=self.shuffle)

        # print('self.train_data', self.train_data)
        # for i in self.train_data:
        #     print(i)
        # Instance with fields: tokens, metadata, simplabel

        train_generator = lazy_groups_of(raw_train_generator, num_gpus)
        num_training_batches = math.ceil(self.iterator.get_num_batches(self.train_data) / num_gpus)
        residue = num_training_batches % self.accumulated_batch_count
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        print('Training')
        # print('num_training_batches', num_training_batches)
        train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_training_batches)
        cumulative_batch_size = 0
        self.optimizer.zero_grad()
        for batch_group in train_generator_tqdm:
            # print('batch_group', batch_group) # {tokens{bert, bert-offsets, mask}, metadata, simplabels}
            # print(batch_group[0]['tokens']['bert'].shape)
            # print(batch_group[0]['tokens']['bert-offsets'].shape)
            # print(batch_group[0]['tokens']['mask'].shape)
            # print(len(batch_group[0]['metadata']))
            # print(batch_group[0]['simplabels'].shape)
            # eg.
            # torch.Size([32, 49])
            # torch.Size([32, 40])
            # torch.Size([32, 40])
            # 32
            # torch.Size([32])
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            iter_len = self.accumulated_batch_count \
                if batches_this_epoch <= (num_training_batches - residue) else residue

            if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
                print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')
            try:
                loss = self.batch_loss(batch_group, for_training=True) / iter_len
            except RuntimeError as e:
                print(e)
                for x in batch_group:
                    all_words = [len(y['words']) for y in x['metadata']]
                    print(f"Total sents: {len(all_words)}. "
                          f"Min {min(all_words)}. Max {max(all_words)}")
                    for elem in ['simplabels']:
                        tt = x[elem]
                        print(
                            f"{elem} shape {list(tt.shape)} and min {tt.min().item()} and {tt.max().item()}")
                    for elem in ["bert", "mask", "bert-offsets"]:
                        tt = x['tokens'][elem]
                        print(
                            f"{elem} shape {list(tt.shape)} and min {tt.min().item()} and {tt.max().item()}")
                raise e

            if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
                print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            loss.backward()

            if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
                print(f'After backprop - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After backprop - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            train_loss += loss.item() * iter_len

            del batch_group, loss
            torch.cuda.empty_cache()

            if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
                print(f'After collecting garbage - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After collecting garbage - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            if self._tensorboard.should_log_histograms_this_batch():
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {
                    name: param.detach().cpu().clone()
                    for name, param in self.model.named_parameters()
                }
                if batches_this_epoch % self.accumulated_batch_count == 0 or \
                        batches_this_epoch == num_training_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1))
                    param_norm = torch.norm(param.view(-1)).cpu()
                    self._tensorboard.add_train_scalar(
                        "gradient_update/" + name, update_norm / (param_norm + 1e-7)
                    )
            else:
                if batches_this_epoch % self.accumulated_batch_count == 0 or \
                        batches_this_epoch == num_training_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch)
            description = training_util.description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if self._tensorboard.should_log_this_batch():
                self._tensorboard.log_parameter_and_gradient_statistics(self.model, batch_grad_norm)
                self._tensorboard.log_learning_rates(self.model, self.optimizer)

                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                self._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._tensorboard.should_log_histograms_this_batch():
                self._tensorboard.log_histograms(self.model, histogram_parameters)

            if self._log_batch_size_period:
                cur_batch = sum([training_util.get_batch_size(batch) for batch in batch_group])
                cumulative_batch_size += cur_batch
                if (batches_this_epoch - 1) % self._log_batch_size_period == 0:
                    average = cumulative_batch_size / batches_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    self._tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    self._tensorboard.add_train_scalar("mean_batch_size", average)

            # Save model if needed.
            if self._model_save_interval is not None and (
                time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                    "{0}.{1}".format(epoch, training_util.time_to_str(int(last_save_time)))
                )

        metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch, reset=True)
        metrics["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        return metrics

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self.model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)

        raw_val_generator = val_iterator(self._validation_data, num_epochs=1, shuffle=False)
        val_generator = lazy_groups_of(raw_val_generator, num_gpus)
        num_validation_batches = math.ceil(
            val_iterator.get_num_batches(self._validation_data) / num_gpus
        )
        val_generator_tqdm = Tqdm.tqdm(val_generator, total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch_group in val_generator_tqdm:

            loss = self.batch_loss(batch_group, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch)
            description = training_util.description_from_metrics(val_metrics)
            print('description', description)
            val_generator_tqdm.set_description(description, refresh=False)

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()

        return val_loss, batches_this_epoch

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        if self.cold_step_count > 0:
            base_lr = self.optimizer.param_groups[0]['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cold_lr
            self.model.text_field_embedder._token_embedders['bert'].set_weights(freeze=True)

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value


        for epoch in range(epoch_counter, self._num_epochs):
            if epoch == self.cold_step_count and epoch != 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = base_lr
                self.model.text_field_embedder._token_embedders['bert'].set_weights(freeze=False)

            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if "cpu_memory_MB" in train_metrics:
                metrics["peak_cpu_memory_MB"] = max(
                    metrics.get("peak_cpu_memory_MB", 0), train_metrics["cpu_memory_MB"]
                )
            for key, value in train_metrics.items():
                if key.startswith("gpu_"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            # clear cache before validation
            torch.cuda.empty_cache()
            if self._validation_data is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = training_util.get_metrics(
                        self.model, val_loss, num_batches, reset=True
                    )

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            self._tensorboard.log_metrics(
                train_metrics, val_metrics=val_metrics, log_to_console=True, epoch=epoch + 1
            )  # +1 because tensorboard doesn't like 0

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            # if self.cold_step_count <= epoch:
            self.scheduler.step(metrics['validation_loss'])

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir:
                dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
                )

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric, epoch)

            self._save_checkpoint(epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (
                    (self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1
                )
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # make sure pending events are flushed to disk and files are closed properly
        # self._tensorboard.close()

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

    def _save_checkpoint(self, epoch: Union[int, str]) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        """
        # If moving averages are used for parameters, we save
        # the moving average values into checkpoint, instead of the current values.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        # These are the training states we need to persist.
        training_states = {
            "metric_tracker": self._metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "batch_num_total": self._batch_num_total,
        }

        # If we have a learning rate or momentum scheduler, we should persist them too.
        if self._learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()
        if self._momentum_scheduler is not None:
            training_states["momentum_scheduler"] = self._momentum_scheduler.state_dict()

        self._checkpointer.save_checkpoint(
            model_state=self.model.state_dict(),
            epoch=epoch,
            training_states=training_states,
            is_best_so_far=self._metric_tracker.is_best_so_far(),
        )

        # Restore the original values for parameters so that training will not be affected.
        if self._moving_average is not None:
            self._moving_average.restore()

    def _restore_checkpoint(self) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        Returns
        -------
        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        model_state, training_state = self._checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self._learning_rate_scheduler is not None \
                and "learning_rate_scheduler" in training_state:
            self._learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        if self._momentum_scheduler is not None and "momentum_scheduler" in training_state:
            self._momentum_scheduler.load_state_dict(training_state["momentum_scheduler"])
        training_util.move_optimizer_to_cuda(self.optimizer)

        # Currently the ``training_state`` contains a serialized ``MetricTracker``.
        if "metric_tracker" in training_state:
            self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked ``val_metric_per_epoch``.
        elif "val_metric_per_epoch" in training_state:
            self._metric_tracker.clear()
            self._metric_tracker.add_metrics(training_state["val_metric_per_epoch"])
        # And before that we didn't track anything.
        else:
            self._metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split(".")[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get("batch_num_total")
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return

    # Requires custom from_params.
    @classmethod
    def from_params(  # type: ignore
        cls,
        model: Model,
        serialization_dir: str,
        iterator: DataIterator,
        train_data: Iterable[Instance],
        validation_data: Optional[Iterable[Instance]],
        params: Params,
        validation_iterator: DataIterator = None,
    ) -> "Trainer":

        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        momentum_scheduler_params = params.pop("momentum_scheduler", None)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
        if "moving_average" in params:
            moving_average = MovingAverage.from_params(
                params.pop("moving_average"), parameters=parameters
            )
        else:
            moving_average = None

        if lr_scheduler_params:
            lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            lr_scheduler = None
        if momentum_scheduler_params:
            momentum_scheduler = MomentumScheduler.from_params(optimizer, momentum_scheduler_params)
        else:
            momentum_scheduler = None

        if "checkpointer" in params:
            if "keep_serialized_model_every_num_seconds" in params \
                    or "num_serialized_models_to_keep" in params:
                raise ConfigurationError(
                    "Checkpointer may be initialized either from the 'checkpointer' key or from the "
                    "keys 'num_serialized_models_to_keep' and 'keep_serialized_model_every_num_seconds'"
                    " but the passed config uses both methods."
                )
            checkpointer = Checkpointer.from_params(params.pop("checkpointer"))
        else:
            num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
            keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None
            )
            checkpointer = Checkpointer(
                serialization_dir=serialization_dir,
                num_serialized_models_to_keep=num_serialized_models_to_keep,
                keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
            )
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)

        params.assert_empty(cls.__name__)
        return cls(
            model,
            optimizer,
            iterator,
            train_data,
            validation_data,
            patience=patience,
            validation_metric=validation_metric,
            validation_iterator=validation_iterator,
            shuffle=shuffle,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=lr_scheduler,
            momentum_scheduler=momentum_scheduler,
            checkpointer=checkpointer,
            model_save_interval=model_save_interval,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
            log_batch_size_period=log_batch_size_period,
            moving_average=moving_average,
        )


class Trainer_Generator(TrainerBase):
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        iterator: DataIterator,
        train_dataset: Iterable[Instance],
        validation_dataset: Optional[Iterable[Instance]] = None,
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_iterator: DataIterator = None,
        shuffle: bool = True,
        num_epochs: int = 20,
        accumulated_batch_count: int = 1,
        serialization_dir: Optional[str] = None,
        num_serialized_models_to_keep: int = 20,
        keep_serialized_model_every_num_seconds: int = None,
        checkpointer: Checkpointer = None,
        model_save_interval: float = None,
        cuda_device: Union[int, List] = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        summary_interval: int = 100,
        histogram_interval: int = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        log_batch_size_period: Optional[int] = None,
        moving_average: Optional[MovingAverage] = None,
        cold_step_count: int = 0,
        cold_lr: float = 1e-3,
        cuda_verbose_step=None,
    ) -> None:
        """
        A trainer for doing supervised learning. It just takes a labeled dataset
        and a ``DataIterator``, and uses the supplied ``Optimizer`` to learn the weights
        for your model over some fixed number of epochs. You can also pass in a validation
        dataset and enable early stopping. There are many other bells and whistles as well.

        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.

            If you are training your model using GPUs, your model should already be
            on the correct device. (If you use `Trainer.from_params` this will be
            handled for you.)
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_iterator : ``DataIterator``, optional (default=None)
            An iterator to use for the validation set.  If ``None``, then
            use the training `iterator`.
        shuffle: ``bool``, optional (default=True)
            Whether to shuffle the instances in the iterator or not.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : ``int``, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : ``int``, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        checkpointer : ``Checkpointer``, optional (default=None)
            An instance of class Checkpointer to use instead of the default. If a checkpointer is specified,
            the arguments num_serialized_models_to_keep and keep_serialized_model_every_num_seconds should
            not be specified. The caller is responsible for initializing the checkpointer so that it is
            consistent with serialization_dir.
        model_save_interval : ``float``, optional (default=None)
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        cuda_device : ``Union[int, List[int]]``, optional (default = -1)
            An integer or list of integers specifying the CUDA device(s) to use. If -1, the CPU is used.
        grad_norm : ``float``, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : ``LearningRateScheduler``, optional (default = None)
            If specified, the learning rate will be decayed with respect to
            this schedule at the end of each epoch (or batch, if the scheduler implements
            the ``step_batch`` method). If you use :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`,
            this will use the ``validation_metric`` provided to determine if learning has plateaued.
            To support updating the learning rate on every batch, this can optionally implement
            ``step_batch(batch_num_total)`` which updates the learning rate given the batch number.
        momentum_scheduler : ``MomentumScheduler``, optional (default = None)
            If specified, the momentum will be updated at the end of each batch or epoch
            according to the schedule.
        summary_interval: ``int``, optional, (default = 100)
            Number of batches between logging scalars to tensorboard
        histogram_interval : ``int``, optional, (default = ``None``)
            If not None, then log histograms to tensorboard every ``histogram_interval`` batches.
            When this parameter is specified, the following additional logging is enabled:
                * Histograms of model parameters
                * The ratio of parameter update norm to parameter norm
                * Histogram of layer activations
            We log histograms of the parameters returned by
            ``model.get_parameters_for_histogram_tensorboard_logging``.
            The layer activations are logged for any modules in the ``Model`` that have
            the attribute ``should_log_activations`` set to ``True``.  Logging
            histograms requires a number of GPU-CPU copies during training and is typically
            slow, so we recommend logging histograms relatively infrequently.
            Note: only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.
        should_log_parameter_statistics : ``bool``, optional, (default = True)
            Whether to send parameter statistics (mean and standard deviation
            of parameters and gradients) to tensorboard.
        should_log_learning_rate : ``bool``, optional, (default = False)
            Whether to send parameter specific learning rate to tensorboard.
        log_batch_size_period : ``int``, optional, (default = ``None``)
            If defined, how often to log the average batch size.
        moving_average: ``MovingAverage``, optional, (default = None)
            If provided, we will maintain moving averages for all parameters. During training, we
            employ a shadow variable for each parameter, which maintains the moving average. During
            evaluation, we backup the original parameters and assign the moving averages to corresponding
            parameters. Be careful that when saving the checkpoint, we will save the moving averages of
            parameters. This is necessary because we want the saved model to perform as well as the validated
            model if we load it later. But this may cause problems if you restart the training from checkpoint.
        """
        super().__init__(serialization_dir, cuda_device)

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model

        self.iterator = iterator
        self._validation_iterator = validation_iterator
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_dataset
        self._validation_data = validation_dataset
        self.accumulated_batch_count = accumulated_batch_count
        self.cold_step_count = cold_step_count
        self.cold_lr = cold_lr
        self.cuda_verbose_step = cuda_verbose_step

        if patience is None:  # no early stopping
            if validation_dataset:
                logger.warning(
                    "You provided a validation dataset but patience was set to None, "
                    "meaning that early stopping is disabled"
                )
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError(
                '{} is an invalid value for "patience": it must be a positive integer '
                "or None (if you want to disable early stopping)".format(patience)
            )

        # For tracking is_best_so_far and should_stop_early
        self._metric_tracker = MetricTracker(patience, validation_metric)
        # Get rid of + or -
        self._validation_metric = validation_metric[1:]

        self._num_epochs = num_epochs

        if checkpointer is not None:
            # We can't easily check if these parameters were passed in, so check against their default values.
            # We don't check against serialization_dir since it is also used by the parent class.
            if num_serialized_models_to_keep != 20 \
                    or keep_serialized_model_every_num_seconds is not None:
                raise ConfigurationError(
                    "When passing a custom Checkpointer, you may not also pass in separate checkpointer "
                    "args 'num_serialized_models_to_keep' or 'keep_serialized_model_every_num_seconds'."
                )
            self._checkpointer = checkpointer
        else:
            self._checkpointer = Checkpointer(
                serialization_dir,
                keep_serialized_model_every_num_seconds,
                num_serialized_models_to_keep,
            )

        self._model_save_interval = model_save_interval

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping

        self._learning_rate_scheduler = learning_rate_scheduler
        self._momentum_scheduler = momentum_scheduler
        self._moving_average = moving_average

        # We keep the total batch number as an instance variable because it
        # is used inside a closure for the hook which logs activations in
        # ``_enable_activation_logging``.
        self._batch_num_total = 0

        self._tensorboard = TensorboardWriter(
            get_batch_num_total=lambda: self._batch_num_total,
            serialization_dir=serialization_dir,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
        )

        self._log_batch_size_period = log_batch_size_period

        self._last_log = 0.0  # time of last logging

        # Enable activation logging.
        if histogram_interval is not None:
            self._tensorboard.enable_activation_logging(self.model)

    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.model, self._grad_norm)

    def batch_loss(self, batch_group: List[TensorDict], for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = training_util.data_parallel(batch_group, self.model, self._cuda_devices)
        else:
            assert len(batch_group) == 1
            batch = batch_group[0]
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            # print('batch', batch)
            output_dict, posed_tag_batch = self.model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError(
                    "The model you are trying to optimize does not contain a"
                    " 'loss' key in the output of model.forward(inputs)."
                )
            loss = None

        return loss

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        print("Epoch ", epoch, "/", self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        print((f"Peak CPU memory usage MB: " + str(peak_cpu_usage)))
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")
            print((f"GPU memory usage MB: " + str(memory)))

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()

        num_gpus = len(self._cuda_devices)

        # Get tqdm for the training batches
        raw_train_generator = self.iterator(self.train_data, num_epochs=1, shuffle=self.shuffle)

        # print('self.train_data', self.train_data)
        # for i in self.train_data:
        #     print(i)
        # Instance with fields: tokens, metadata, simplabel

        train_generator = lazy_groups_of(raw_train_generator, num_gpus)
        num_training_batches = math.ceil(self.iterator.get_num_batches(self.train_data) / num_gpus)
        residue = num_training_batches % self.accumulated_batch_count
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        print('Training')
        # print('num_training_batches', num_training_batches)
        train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_training_batches)
        cumulative_batch_size = 0
        self.optimizer.zero_grad()
        for batch_group in train_generator_tqdm:
            # print('batch_group', batch_group) # {tokens{bert, bert-offsets, mask}, metadata, simplabels}
            # print(batch_group[0]['tokens']['bert'].shape)
            # print(batch_group[0]['tokens']['bert-offsets'].shape)
            # print(batch_group[0]['tokens']['mask'].shape)
            # print(len(batch_group[0]['metadata']))
            # print(batch_group[0]['simplabels'].shape)
            # eg.
            # torch.Size([32, 49])
            # torch.Size([32, 40])
            # torch.Size([32, 40])
            # 32
            # torch.Size([32])
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            iter_len = self.accumulated_batch_count \
                if batches_this_epoch <= (num_training_batches - residue) else residue

            if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
                print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')
            try:
                loss = self.batch_loss(batch_group, for_training=True) / iter_len
            except RuntimeError as e:
                print(e)
                for x in batch_group:
                    all_words = [len(y['words']) for y in x['metadata']]
                    print(f"Total sents: {len(all_words)}. "
                          f"Min {min(all_words)}. Max {max(all_words)}")
                    for elem in ['simplabels']:
                        tt = x[elem]
                        print(
                            f"{elem} shape {list(tt.shape)} and min {tt.min().item()} and {tt.max().item()}")
                    for elem in ["bert", "mask", "bert-offsets"]:
                        tt = x['tokens'][elem]
                        print(
                            f"{elem} shape {list(tt.shape)} and min {tt.min().item()} and {tt.max().item()}")
                raise e

            if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
                print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            loss.backward()

            if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
                print(f'After backprop - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After backprop - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            train_loss += loss.item() * iter_len

            del batch_group, loss
            torch.cuda.empty_cache()

            if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
                print(f'After collecting garbage - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After collecting garbage - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            if self._tensorboard.should_log_histograms_this_batch():
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {
                    name: param.detach().cpu().clone()
                    for name, param in self.model.named_parameters()
                }
                if batches_this_epoch % self.accumulated_batch_count == 0 or \
                        batches_this_epoch == num_training_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1))
                    param_norm = torch.norm(param.view(-1)).cpu()
                    self._tensorboard.add_train_scalar(
                        "gradient_update/" + name, update_norm / (param_norm + 1e-7)
                    )
            else:
                if batches_this_epoch % self.accumulated_batch_count == 0 or \
                        batches_this_epoch == num_training_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch)
            description = training_util.description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if self._tensorboard.should_log_this_batch():
                self._tensorboard.log_parameter_and_gradient_statistics(self.model, batch_grad_norm)
                self._tensorboard.log_learning_rates(self.model, self.optimizer)

                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                self._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._tensorboard.should_log_histograms_this_batch():
                self._tensorboard.log_histograms(self.model, histogram_parameters)

            if self._log_batch_size_period:
                cur_batch = sum([training_util.get_batch_size(batch) for batch in batch_group])
                cumulative_batch_size += cur_batch
                if (batches_this_epoch - 1) % self._log_batch_size_period == 0:
                    average = cumulative_batch_size / batches_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    self._tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    self._tensorboard.add_train_scalar("mean_batch_size", average)

            # Save model if needed.
            if self._model_save_interval is not None and (
                time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                    "{0}.{1}".format(epoch, training_util.time_to_str(int(last_save_time)))
                )

        metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch, reset=True)
        metrics["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        return metrics

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self.model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)

        raw_val_generator = val_iterator(self._validation_data, num_epochs=1, shuffle=False)
        val_generator = lazy_groups_of(raw_val_generator, num_gpus)
        num_validation_batches = math.ceil(
            val_iterator.get_num_batches(self._validation_data) / num_gpus
        )
        val_generator_tqdm = Tqdm.tqdm(val_generator, total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch_group in val_generator_tqdm:

            loss = self.batch_loss(batch_group, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()

        return val_loss, batches_this_epoch

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        if self.cold_step_count > 0:
            base_lr = self.optimizer.param_groups[0]['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cold_lr
            self.model.text_field_embedder._token_embedders['bert'].set_weights(freeze=True)

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for epoch in range(epoch_counter, self._num_epochs):
            if epoch == self.cold_step_count and epoch != 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = base_lr
                self.model.text_field_embedder._token_embedders['bert'].set_weights(freeze=False)

            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if "cpu_memory_MB" in train_metrics:
                metrics["peak_cpu_memory_MB"] = max(
                    metrics.get("peak_cpu_memory_MB", 0), train_metrics["cpu_memory_MB"]
                )
            for key, value in train_metrics.items():
                if key.startswith("gpu_"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            # clear cache before validation
            torch.cuda.empty_cache()
            if self._validation_data is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = training_util.get_metrics(
                        self.model, val_loss, num_batches, reset=True
                    )

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            self._tensorboard.log_metrics(
                train_metrics, val_metrics=val_metrics, log_to_console=True, epoch=epoch + 1
            )  # +1 because tensorboard doesn't like 0

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            # if self.cold_step_count <= epoch:
            self.scheduler.step(metrics['validation_loss'])

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir:
                dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
                )

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric, epoch)

            self._save_checkpoint(epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (
                    (self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1
                )
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # make sure pending events are flushed to disk and files are closed properly
        # self._tensorboard.close()

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

    def _save_checkpoint(self, epoch: Union[int, str]) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        """
        # If moving averages are used for parameters, we save
        # the moving average values into checkpoint, instead of the current values.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        # These are the training states we need to persist.
        training_states = {
            "metric_tracker": self._metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "batch_num_total": self._batch_num_total,
        }

        # If we have a learning rate or momentum scheduler, we should persist them too.
        if self._learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()
        if self._momentum_scheduler is not None:
            training_states["momentum_scheduler"] = self._momentum_scheduler.state_dict()

        self._checkpointer.save_checkpoint(
            model_state=self.model.state_dict(),
            epoch=epoch,
            training_states=training_states,
            is_best_so_far=self._metric_tracker.is_best_so_far(),
        )

        # Restore the original values for parameters so that training will not be affected.
        if self._moving_average is not None:
            self._moving_average.restore()

    def _restore_checkpoint(self) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        Returns
        -------
        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        model_state, training_state = self._checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self._learning_rate_scheduler is not None \
                and "learning_rate_scheduler" in training_state:
            self._learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        if self._momentum_scheduler is not None and "momentum_scheduler" in training_state:
            self._momentum_scheduler.load_state_dict(training_state["momentum_scheduler"])
        training_util.move_optimizer_to_cuda(self.optimizer)

        # Currently the ``training_state`` contains a serialized ``MetricTracker``.
        if "metric_tracker" in training_state:
            self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked ``val_metric_per_epoch``.
        elif "val_metric_per_epoch" in training_state:
            self._metric_tracker.clear()
            self._metric_tracker.add_metrics(training_state["val_metric_per_epoch"])
        # And before that we didn't track anything.
        else:
            self._metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split(".")[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get("batch_num_total")
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return

    # Requires custom from_params.
    @classmethod
    def from_params(  # type: ignore
        cls,
        model: Model,
        serialization_dir: str,
        iterator: DataIterator,
        train_data: Iterable[Instance],
        validation_data: Optional[Iterable[Instance]],
        params: Params,
        validation_iterator: DataIterator = None,
    ) -> "Trainer":

        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        momentum_scheduler_params = params.pop("momentum_scheduler", None)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
        if "moving_average" in params:
            moving_average = MovingAverage.from_params(
                params.pop("moving_average"), parameters=parameters
            )
        else:
            moving_average = None

        if lr_scheduler_params:
            lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            lr_scheduler = None
        if momentum_scheduler_params:
            momentum_scheduler = MomentumScheduler.from_params(optimizer, momentum_scheduler_params)
        else:
            momentum_scheduler = None

        if "checkpointer" in params:
            if "keep_serialized_model_every_num_seconds" in params \
                    or "num_serialized_models_to_keep" in params:
                raise ConfigurationError(
                    "Checkpointer may be initialized either from the 'checkpointer' key or from the "
                    "keys 'num_serialized_models_to_keep' and 'keep_serialized_model_every_num_seconds'"
                    " but the passed config uses both methods."
                )
            checkpointer = Checkpointer.from_params(params.pop("checkpointer"))
        else:
            num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
            keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None
            )
            checkpointer = Checkpointer(
                serialization_dir=serialization_dir,
                num_serialized_models_to_keep=num_serialized_models_to_keep,
                keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
            )
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)

        params.assert_empty(cls.__name__)
        return cls(
            model,
            optimizer,
            iterator,
            train_data,
            validation_data,
            patience=patience,
            validation_metric=validation_metric,
            validation_iterator=validation_iterator,
            shuffle=shuffle,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=lr_scheduler,
            momentum_scheduler=momentum_scheduler,
            checkpointer=checkpointer,
            model_save_interval=model_save_interval,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
            log_batch_size_period=log_batch_size_period,
            moving_average=moving_average,
        )


class Trainer_Adversarial(TrainerBase):
    def __init__(
        self,
        generator: Model,
        discriminator: Model,
        reader_discriminator,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer,
        schedule_generatorr: torch.optim.lr_scheduler,
        scheduler_discriminator: torch.optim.lr_scheduler,
        iterator: DataIterator,
        train_source: Iterable[Instance],
        train_target: Iterable[Instance],
        valid_source: Iterable[Instance],
        valid_target: Iterable[Instance],
        test_source: Iterable[Instance],
        test_target: Iterable[Instance],
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_iterator: DataIterator = None,
        shuffle: bool = True,
        num_epochs: int = 20,
        accumulated_batch_count: int = 1,
        serialization_dir: Optional[str] = None,
        num_serialized_models_to_keep: int = 20,
        keep_serialized_model_every_num_seconds: int = None,
        checkpointer: Checkpointer = None,
        model_save_interval: float = None,
        cuda_device: Union[int, List] = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        summary_interval: int = 100,
        histogram_interval: int = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        generator_update_batch: int = 1,
        discriminator_update_batch: int = 1,
        log_batch_size_period: Optional[int] = None,
        moving_average: Optional[MovingAverage] = None,
        cold_step_count: int = 0,
        cold_lr: float = 1e-3,
        cuda_verbose_step=None,
        g_loss1_hp=1,
        g_loss2_hp=1,
        g_loss3_hp=1,
    ) -> None:
        super().__init__(serialization_dir, cuda_device)

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.generator = generator
        self.discriminator = discriminator
        self.reader_discriminator = reader_discriminator
        self.iterator = iterator
        self._validation_iterator = validation_iterator
        self.shuffle = shuffle
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.schedule_generatorr = schedule_generatorr
        self.scheduler_discriminator = scheduler_discriminator
        self.train_source = train_source
        self.train_target = train_target
        self.valid_source = valid_source
        self.valid_target = valid_target
        self.test_source = test_source
        self.test_target = test_target
        self.accumulated_batch_count = accumulated_batch_count
        self.cold_step_count = cold_step_count
        self.cold_lr = cold_lr
        self.cuda_verbose_step = cuda_verbose_step
        self._num_epochs = num_epochs
        self._model_save_interval = model_save_interval
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._learning_rate_scheduler = learning_rate_scheduler
        self._momentum_scheduler = momentum_scheduler
        self._moving_average = moving_average
        self.generator_update_batch = generator_update_batch
        self.discriminator_update_batch = discriminator_update_batch

        # We keep the total batch number as an instance variable because it
        # is used inside a closure for the hook which logs activations in
        # ``_enable_activation_logging``.
        self._batch_num_total = 0
        self._log_batch_size_period = log_batch_size_period
        self._last_log = 0.0  # time of last logging

        self.remainloss = Remainloss()
        self.semanticloss = Semanticloss()

        self.g_loss1_hp = g_loss1_hp
        self.g_loss2_hp = g_loss2_hp
        self.g_loss3_hp = g_loss3_hp


    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.generator, self._grad_norm)

    def logits2embedding2dis(self, logits, ground_truth):
        logits_embedding = torch.matmul(logits, self.discriminator.text_field_embedder.token_embedder_bert.
                                        bert_model.embeddings.word_embeddings.weight)
        if ground_truth == 'simple':
            # .to(self._cuda_devices[0])
            simplabels = torch.zeros(logits.shape[0]).long().to(self._cuda_devices[0])
        elif ground_truth == 'unsimple':
            simplabels = torch.ones(logits.shape[0]).long().to(self._cuda_devices[0])
        out_dic = self.discriminator(simplabels=simplabels, logits_embedding=logits_embedding)
        return out_dic

    def logits2embedding2similarity(self, logits, cls_embedding):
        # print('logits', logits.shape)
        # print('cls_embedding', cls_embedding.shape)
        logits_ = torch.matmul(logits, self.generator.text_field_embedder.token_embedder_bert.
                                        bert_model.embeddings.word_embeddings.weight)
        # print('logits_', logits_.shape)
        logits_embedding = self.generator(logits_embedding=logits_)
        # print('logits_embedding', logits_embedding.shape)
        out_dic = self.semanticloss(logits_embedding.to(self._cuda_devices[0]), cls_embedding.to(self._cuda_devices[0]),
                                    device=self._cuda_devices[0])
        return out_dic


    def batch_loss(self, batch_group: List[TensorDict], target_group: List[TensorDict], for_training: bool, need_post_tag_batch=False) -> Tuple[
        Union[int, Any], Union[int, Any], Any, Union[float, Any]]:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """

        assert len(batch_group) == 1
        # unsimple batch
        batch = batch_group[0]
        batch = nn_util.move_to_device(batch, self._cuda_devices[0])
        # simple batch (non-parallel with unsimple batch)
        batch_target = target_group[0]
        batch_target = nn_util.move_to_device(batch_target, self._cuda_devices[0])
        # print('batch', batch)
        # print(batch['tokens']['bert'])
        # print('batch_target', batch_target)
        # exit()
        output_dict_generator, post_tag_batch, tags, cls_embedding = self.generator(**batch, need_post_tag_batch=need_post_tag_batch, return_cls=True)
        # exit()

        # print('batch_target', batch_target)
        # print('post_tag_batch', post_tag_batch)
        # print('output_dict_generator', output_dict_generator['logits_labels'])
        # print(output_dict_generator['logits_labels'].shape)  # torch.Size([32, 40, 5002]) -> torch.Size([32, 40, 30522])
        # print('discriminator', self.discriminator)
        # print('nnembedding', self.discriminator.text_field_embedder.token_embedder_bert.bert_model.embeddings.word_embeddings.weight.shape)  # torch.Size([30523, 768])



        # --------------------Calculate generator loss--------------------
        # loss1: simp_loss
        # 输入：使用generator输出的logits_labels，simplabels=simple
        # 输出：discriminator的loss和accuracy
        out_dic = self.logits2embedding2dis(output_dict_generator['logits_labels'], ground_truth='simple')
        g_loss1 = out_dic['loss']
        g_acc1 = out_dic['accuracy']
        print()
        print('g_loss1', g_loss1, end='  ')
        print('g_acc1', g_acc1)

        # loss2: remain_loss
        # 输入：source_batch的token_ids 和 generator生成的sentence的token_ids的logits
        # 输出：CrossEntropyLoss
        out_dic2 = self.remainloss(output_dict_generator['logits_labels'], batch['tokens']['bert'])
        g_loss2 = out_dic2['loss']
        print('g_loss2', g_loss2)
        # exit()

        # loss3: semantic_loss
        # 输入：source_batch的sentence_embedding 和 generator生成的sentences的source_batch的sentence_embedding
        # 输出：CosineEmbeddingLoss
        out_dic3 = self.logits2embedding2similarity(output_dict_generator['logits_labels'], cls_embedding)
        g_loss3 = out_dic3['loss']
        print('g_loss3', g_loss3)
        # exit()


        # --------------------Calculate discriminator loss--------------------
        # loss1:
        # input: [batch, simplabels=unsimple]
        # model: discriminator
        # output: loss, accuracy
        _output_dict_discriminator1 = self.discriminator(**batch)
        d_loss1 = _output_dict_discriminator1['loss']
        d_acc1 = _output_dict_discriminator1["accuracy"]
        print('d_loss1', d_loss1, end='  ')
        print('d_acc1', d_acc1)

        # loss2:
        # input: [batch_target, simplabels=simple]
        # model: discriminator
        # output: loss, accuracy
        # print('batch_target', batch_target)
        _output_dict_discriminator2 = self.discriminator(**batch_target)
        d_loss2 = _output_dict_discriminator2['loss']
        d_acc2 = _output_dict_discriminator2['accuracy']
        print('d_loss2', d_loss2, end='  ')
        print('d_acc2', d_acc2)

        # loss3:
        # 输入：使用generator输出的logits_labels，simplabels=unsimple
        # 输出：discriminator的loss和accuracy
        _out_dic = self.logits2embedding2dis(output_dict_generator['logits_labels'], ground_truth='unsimple')
        d_loss3 = _out_dic['loss']
        d_acc3 = _out_dic['accuracy']
        print('d_loss3', d_loss3, end='  ')
        print('d_acc3', d_acc3)


        loss_generator = self.g_loss1_hp * g_loss1 + self.g_loss2_hp * g_loss2 + self.g_loss3_hp * g_loss3
        loss_discriminator = d_loss1 + d_loss2 + d_loss3
        accuracy_generator = g_acc1
        accuracy_discriminator = (d_acc1 + d_acc2 + d_acc3) / 3
        if need_post_tag_batch:
            return loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator, post_tag_batch, tags
        else:
            return loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        print("Epoch ", epoch, "/", self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        print((f"Peak CPU memory usage MB: " + str(peak_cpu_usage)))
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")
            print((f"GPU memory usage MB: " + str(memory)))

        train_generator_loss = 0.0
        train_discriminator_loss = 0.0
        # Set the model to "train" mode.
        self.generator.train()
        self.discriminator.train()

        num_gpus = len(self._cuda_devices)  # 0

        # Get tqdm for the training batches
        raw_train_source_generator = self.iterator(self.train_source, num_epochs=1, shuffle=self.shuffle)
        raw_train_target_generator = self.iterator(self.train_target, num_epochs=1, shuffle=self.shuffle)

        train_source_generator = lazy_groups_of(raw_train_source_generator, num_gpus)
        train_target_generator = lazy_groups_of(raw_train_target_generator, num_gpus)
        num_training_batches_source = math.ceil(self.iterator.get_num_batches(self.train_source) / num_gpus)  # 1
        residue_source = num_training_batches_source % self.accumulated_batch_count  # 1
        num_training_batches_target = math.ceil(self.iterator.get_num_batches(self.train_target) / num_gpus)  # 1
        residue_target = num_training_batches_target % self.accumulated_batch_count  # 1
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        train_loss_generator_all = 0.0
        train_loss_discriminator_all = 0.0
        train_accuracy_generator_all = 0.0
        train_accuracy_discriminator_all = 0.0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        logger.info("Training")
        print('Training')
        train_generator_tqdm_source = Tqdm.tqdm(train_source_generator, total=num_training_batches_source)
        self.optimizer_generator.zero_grad()
        self.optimizer_discriminator.zero_grad()

        # for count whether update G or D or both
        batch_num = 0
        for batch_group, target_group in zip(train_generator_tqdm_source, train_target_generator):
            batch_num += 1
            # print(batch_group)
            # print(batch_group[0]['metadata'])
            # sentences = metadata2senteces(batch_group[0]['metadata'])
            # print('sentences', sentences)
            # temp = wordlistlist2sentence(sentences)
            # print(temp)
            # write_json_data(sentences, sentences, '0.json')
            # exit()
            # print(batch_group[0]['tokens']['bert'].shape)
            # print(target_group)
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            iter_len = self.accumulated_batch_count \
                if batches_this_epoch <= (num_training_batches_source - residue_source) else residue_source  # 1

            loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator = \
                self.batch_loss(batch_group, target_group, for_training=True)
            assert iter_len == 1
            # exit()
            # loss_generator = loss_generator / iter_len
            # loss_discriminator = loss_discriminator / iter_len
            # print()
            # print('loss_generator', loss_generator)
            # print('loss_discriminator', loss_discriminator)
            # print('accuracy_generator_', accuracy_generator)
            # print('accuracy_discriminator_', accuracy_discriminator)
            train_loss_generator_all += loss_generator.detach().cpu().numpy()
            train_loss_discriminator_all += loss_discriminator.detach().cpu().numpy()
            train_accuracy_generator_all += accuracy_generator
            train_accuracy_discriminator_all += accuracy_discriminator

            # print('batch_num', batch_num)
            # print('self.generator_update_batch', self.generator_update_batch)
            # print('self.discriminator_update_batch', self.discriminator_update_batch)
            # print('batch_num % self.generator_update_batch', batch_num % self.generator_update_batch)
            # print('batch_num % self.discriminator_update_batch', batch_num % self.discriminator_update_batch)
            # for generator to update
            if batch_num % self.generator_update_batch == 0:
                # for generator to update
                print('Update generator.')
                loss_generator.backward()
                # torch.cuda.empty_cache()
                self.optimizer_generator.step()
                self.optimizer_generator.zero_grad()

            # for discriminator to update
            if batch_num % self.discriminator_update_batch == 1:
                print('Update discriminator.')
                loss_discriminator.backward()
                # torch.cuda.empty_cache()
                self.optimizer_discriminator.step()
                self.optimizer_discriminator.zero_grad()

            del batch_group, target_group, loss_generator, loss_discriminator
            # self.optimizer_generator.zero_grad()
            # self.optimizer_discriminator.zero_grad()
            torch.cuda.empty_cache()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            # metrics_generator = training_util.get_metrics(self.generator, train_generator_loss, batches_this_epoch)
            # metrics_discriminator = training_util.get_metrics(self.discriminator, train_discriminator_loss, batches_this_epoch)
            metrics_all = {
                'generator_loss': train_loss_generator_all / batches_this_epoch,
                'generator_accuracy': train_accuracy_generator_all / batches_this_epoch,
                'discriminator_loss': train_loss_discriminator_all / batches_this_epoch,
                'discriminator_accuracy': train_accuracy_discriminator_all / batches_this_epoch
            }
            description = training_util.description_from_metrics(metrics_all)
            print(description)
            train_generator_tqdm_source.set_description(description, refresh=False)

        metrics_generator = {}
        metrics_discriminator = {}
        metrics_generator['loss'] = train_loss_generator_all / batches_this_epoch
        metrics_generator['accuracy'] = train_accuracy_generator_all / batches_this_epoch
        metrics_discriminator['loss'] = train_loss_discriminator_all / batches_this_epoch
        metrics_discriminator['accuracy'] = train_accuracy_discriminator_all / batches_this_epoch

        metrics_generator["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics_generator["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        metrics_discriminator["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics_discriminator["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        print('metrics_generator', metrics_generator)
        print('metrics_discriminator', metrics_discriminator)
        return metrics_generator, metrics_discriminator

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")
        print('Validating')

        self.generator.eval()
        self.discriminator.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)  # 1

        raw_val_generator_source = val_iterator(self.valid_source, num_epochs=1, shuffle=False)
        val_generator_source = lazy_groups_of(raw_val_generator_source, num_gpus)
        num_validation_batches_source = math.ceil(
            val_iterator.get_num_batches(self.valid_source) / num_gpus
        )
        val_generator_tqdm_source = Tqdm.tqdm(val_generator_source, total=num_validation_batches_source)

        raw_val_generator_target = val_iterator(self.valid_target, num_epochs=1, shuffle=False)
        val_generator_target = lazy_groups_of(raw_val_generator_target, num_gpus)
        num_validation_batches_source = math.ceil(
            val_iterator.get_num_batches(self.valid_source) / num_gpus
        )
        # val_generator_tqdm_target = Tqdm.tqdm(val_generator_target, total=num_validation_batches_source)

        batches_this_epoch = 0
        val_loss_generator_all = 0.0
        val_loss_discriminator_all = 0.0
        val_accuracy_generator_all = 0.0
        val_accuracy_discriminator_all = 0.0
        for source_group, target_group in zip(val_generator_tqdm_source, val_generator_target):
            iter_len, _ = source_group[0]['tokens']['bert'].shape
            loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator = \
                self.batch_loss(source_group, target_group, for_training=False)
            if loss_generator is not None and loss_discriminator is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss_generator_all += loss_generator.detach().cpu().numpy()
                val_loss_discriminator_all += loss_discriminator.detach().cpu().numpy()
                val_accuracy_generator_all += accuracy_generator
                val_accuracy_discriminator_all += accuracy_discriminator

            # Update the description with the latest metrics
            metrics_all = {
                'generator_loss': val_loss_generator_all / batches_this_epoch,
                'generator_accuracy': val_accuracy_generator_all / batches_this_epoch,
                'discriminator_loss': val_loss_discriminator_all / batches_this_epoch,
                'discriminator_accuracy': val_accuracy_discriminator_all / batches_this_epoch
            }
            description = training_util.description_from_metrics(metrics_all)
            print(description)
            val_generator_tqdm_source.set_description(description, refresh=False)


        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()
        metrics_generator = {}
        metrics_discriminator = {}
        metrics_generator['loss'] = val_loss_generator_all / batches_this_epoch
        metrics_generator['accuracy'] = val_accuracy_generator_all / batches_this_epoch
        metrics_discriminator['loss'] = val_loss_discriminator_all / batches_this_epoch
        metrics_discriminator['accuracy'] = val_accuracy_discriminator_all / batches_this_epoch
        print('metrics_generator', metrics_generator)
        print('metrics_discriminator', metrics_discriminator)
        return metrics_generator, metrics_discriminator

    def _test_predict(self) -> Tuple[float, int]:
        logger.info("Testing")
        print('Testing')

        self.generator.eval()
        self.discriminator.eval()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)  # 1

        raw_val_generator_source = val_iterator(self.test_source, num_epochs=1, shuffle=False)
        val_generator_source = lazy_groups_of(raw_val_generator_source, num_gpus)
        num_validation_batches_source = math.ceil(
            val_iterator.get_num_batches(self.test_source) / num_gpus
        )

        val_generator_tqdm_source = Tqdm.tqdm(val_generator_source, total=num_validation_batches_source)

        raw_val_generator_target = val_iterator(self.valid_target, num_epochs=1, shuffle=False)
        val_generator_target = lazy_groups_of(raw_val_generator_target, num_gpus)
        num_validation_batches_source = math.ceil(
            val_iterator.get_num_batches(self.test_source) / num_gpus
        )
        # val_generator_tqdm_target = Tqdm.tqdm(val_generator_target, total=num_validation_batches_source)

        batches_this_epoch = 0
        test_loss_generator_all = 0.0
        test_loss_discriminator_all = 0.0
        test_accuracy_generator_all = 0.0
        test_accuracy_discriminator_all = 0.0

        source_batch_all = []
        post_tag_batch_all = []
        tags_all = []


        for source_group, target_group in zip(val_generator_tqdm_source, val_generator_target):

            iter_len, _ = source_group[0]['tokens']['bert'].shape
            loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator, post_tag_batch, tags = \
                self.batch_loss(source_group, target_group, for_training=False, need_post_tag_batch=True)

            # add to all list
            for i in metadata2senteces(source_group[0]['metadata']):
                source_batch_all.append(i)
            for i in post_tag_batch:
                post_tag_batch_all.append(i)
            for i in tags:
                tags_all.append(i)

            print('source_bath_all', source_batch_all)
            # print(len(source_batch_all))
            print('post_tag_batch_all', post_tag_batch_all)
            # print(len(post_tag_batch_all))
            print('tags_all', tags_all)
            #
            # write_json_data(source_sentences=source_batch_all, predict_sentences=post_tag_batch_all, filename='0.json')
            # exit()

            if loss_generator is not None and loss_discriminator is not None:
                batches_this_epoch += 1
                test_loss_generator_all += loss_generator.detach().cpu().numpy()
                test_loss_discriminator_all += loss_discriminator.detach().cpu().numpy()
                test_accuracy_generator_all += accuracy_generator
                test_accuracy_discriminator_all += accuracy_discriminator

            # Update the description with the latest metrics
            metrics_all = {
                'generator_loss': test_loss_generator_all / batches_this_epoch,
                'generator_accuracy': test_accuracy_generator_all / batches_this_epoch,
                'discriminator_loss': test_loss_discriminator_all / batches_this_epoch,
                'discriminator_accuracy': test_accuracy_discriminator_all / batches_this_epoch
            }
            description = training_util.description_from_metrics(metrics_all)
            # print(description)
            val_generator_tqdm_source.set_description(description, refresh=False)


        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()
        metrics_generator = {}
        metrics_discriminator = {}
        metrics_generator['loss'] = test_loss_generator_all / batches_this_epoch
        metrics_generator['accuracy'] = test_accuracy_generator_all / batches_this_epoch
        metrics_discriminator['loss'] = test_loss_discriminator_all / batches_this_epoch
        metrics_discriminator['accuracy'] = test_accuracy_discriminator_all / batches_this_epoch
        print('metrics_generator', metrics_generator)
        print('metrics_discriminator', metrics_discriminator)
        return metrics_generator, metrics_discriminator, source_batch_all, post_tag_batch_all, tags_all

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()


        for epoch in range(self._num_epochs):
            epoch_start_time = time.time()
            metrics_generator, metrics_discriminator = self._train_epoch(epoch)

            # clear cache before validation
            torch.cuda.empty_cache()
            source_batch_all_all = []
            post_tag_batch_all_all = []
            tags_all_all = []
            if self.valid_target is not None and self.valid_source is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    metrics_generator_val, metrics_discriminator_val = self._validation_loss()
                    metrics_generator_test, metrics_discriminator_test, source_batch_all, post_tag_batch_all, tags_all = self._test_predict()
                    source_batch_all_all += source_batch_all
                    post_tag_batch_all_all += post_tag_batch_all
                    tags_all_all += tags_all
            train_metrics = {
                'generator': metrics_generator,
                'discriminator': metrics_discriminator
            }
            val_metrics = {
                'generator': metrics_generator_val,
                'discriminator': metrics_discriminator_val
            }
            test_metrics = {
                'generator': metrics_generator_test,
                'discriminator': metrics_discriminator_test
            }
            print('train_metrics', train_metrics)
            print('val_metrics', val_metrics)
            print('test_metrics', test_metrics)

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value
            for key, value in test_metrics.items():
                metrics["test_" + key] = value


            if self._serialization_dir:
                dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
                )

            print('Saving test with predict.')
            print('source_batch_all_all', source_batch_all_all)
            print('post_tag_batch_all_all', post_tag_batch_all_all)
            print('tags_all_all', tags_all_all)
            filename = os.path.join(self._serialization_dir, f"test_predict_{epoch}.json")
            write_json_data(source_sentences=source_batch_all_all, predict_sentences=post_tag_batch_all_all, tags=tags_all_all, filename=filename)
            # exit()

            # self._save_checkpoint(epoch)
            print('Saving model.')
            out_model = os.path.join(self._serialization_dir, f'generator_{epoch}.th', )
            with open(out_model, 'wb') as f:
                torch.save(self.generator.state_dict(), f)
            print("Generator is dumped")

            out_model = os.path.join(self._serialization_dir, f'discriminator_{epoch}.th')
            with open(out_model, 'wb') as f:
                torch.save(self.discriminator.state_dict(), f)
            print("Discriminator is dumped")

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))
            print("Epoch duration: ", datetime.timedelta(seconds=epoch_elapsed_time))

            epochs_trained += 1

        print('metrics', metrics)
        return metrics


class Trainer_Adversarial_pipeline(TrainerBase):
    def __init__(
        self,
        generator: Model,
        discriminator: Model,
        reader_discriminator,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer,
        schedule_generatorr: torch.optim.lr_scheduler,
        scheduler_discriminator: torch.optim.lr_scheduler,
        iterator: DataIterator,
        train_source: Iterable[Instance],
        train_target: Iterable[Instance],
        valid_source: Iterable[Instance],
        valid_target: Iterable[Instance],
        test_source: Iterable[Instance],
        test_target: Iterable[Instance],
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_iterator: DataIterator = None,
        shuffle: bool = True,
        num_epochs: int = 20,
        accumulated_batch_count: int = 1,
        serialization_dir: Optional[str] = None,
        num_serialized_models_to_keep: int = 20,
        keep_serialized_model_every_num_seconds: int = None,
        checkpointer: Checkpointer = None,
        model_save_interval: float = None,
        cuda_device: Union[int, List] = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        summary_interval: int = 100,
        histogram_interval: int = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        generator_update_batch: int = 1,
        discriminator_update_batch: int = 1,
        log_batch_size_period: Optional[int] = None,
        moving_average: Optional[MovingAverage] = None,
        cold_step_count: int = 0,
        cold_lr: float = 1e-3,
        cuda_verbose_step=None,
        g_loss1_hp=1,
        g_loss2_hp=1,
        g_loss3_hp=1,
        g_loss4_hp=1,
        g_loss5_hp=1,
        train_temp_path='train_temp'
    ) -> None:
        super().__init__(serialization_dir, cuda_device)

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.generator = generator
        self.discriminator = discriminator
        self.reader_discriminator = reader_discriminator
        self.iterator = iterator
        self._validation_iterator = validation_iterator
        self.shuffle = shuffle
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.schedule_generatorr = schedule_generatorr
        self.scheduler_discriminator = scheduler_discriminator
        self.train_source = train_source
        self.train_target = train_target
        self.valid_source = valid_source
        self.valid_target = valid_target
        self.test_source = test_source
        self.test_target = test_target
        self.accumulated_batch_count = accumulated_batch_count
        self.cold_step_count = cold_step_count
        self.cold_lr = cold_lr
        self.cuda_verbose_step = cuda_verbose_step
        self._num_epochs = num_epochs
        self._model_save_interval = model_save_interval
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._learning_rate_scheduler = learning_rate_scheduler
        self._momentum_scheduler = momentum_scheduler
        self._moving_average = moving_average
        self.generator_update_batch = generator_update_batch
        self.discriminator_update_batch = discriminator_update_batch

        # We keep the total batch number as an instance variable because it
        # is used inside a closure for the hook which logs activations in
        # ``_enable_activation_logging``.
        self._batch_num_total = 0
        self._log_batch_size_period = log_batch_size_period
        self._last_log = 0.0  # time of last logging

        self.remainloss = Remainloss()
        self.semanticloss = Semanticloss()

        self.g_loss1_hp = g_loss1_hp
        self.g_loss2_hp = g_loss2_hp
        self.g_loss3_hp = g_loss3_hp
        self.g_loss4_hp = g_loss4_hp
        self.g_loss5_hp = g_loss5_hp

        self.train_temp_path = train_temp_path


    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.generator, self._grad_norm)

    def logits2embedding2dis(self, logits, orig_tokens, ground_truth='simple'):
        # print('orig_tokens', orig_tokens, orig_tokens.shape)
        equal = torch.zeros_like(orig_tokens)
        orig_mask = (1 - torch.eq(orig_tokens, equal).long()).long()
        # print('orig_mask', orig_mask, orig_mask.shape)
        logits = logits[:,:,1].unsqueeze(2)
        # print('logits', logits.shape)  # torch.Size([32, 30, 1])

        word_embedding = self.discriminator.text_field_embedder.token_embedder_bert.bert_model.embeddings.word_embeddings
        word_embed = word_embedding(orig_tokens)
        # print('word_embed', word_embed.shape)  # torch.Size([32, 30, 768])

        logits_embedding = logits * word_embed
        # print('logits_embedding', logits_embedding.shape)  # torch.Size([32, 30, 768])

        if ground_truth == 'simple':
            # .to(self._cuda_devices[0])
            simplabels = torch.zeros(logits.shape[0]).long().to(self._cuda_devices[0])
        elif ground_truth == 'unsimple':
            simplabels = torch.ones(logits.shape[0]).long().to(self._cuda_devices[0])
        out_dic = self.discriminator(simplabels=simplabels, logits_embedding=logits_embedding, mask=orig_mask)
        return out_dic

    def logits2embedding2similarity(self, logits, orig_tokens):
        equal = torch.zeros_like(orig_tokens)
        pad = (1 - torch.eq(orig_tokens, equal).long()).unsqueeze(2).repeat(1,1,768)
        logits = logits[:, :, 1].unsqueeze(2)
        # print('logits', logits.shape)  # torch.Size([2, 30, 1])
        word_embedding = self.generator.text_field_embedder.token_embedder_bert.bert_model.embeddings.word_embeddings
        word_embed = word_embedding(orig_tokens) * pad
        # print('word_embed', word_embed.shape)  # torch.Size([32, 30, 768])
        source_embed = torch.sum(word_embed, dim=1)
        # print('source_embed', source_embed.shape)  # torch.Size([32, 768])
        logits_embed = torch.sum(logits * word_embed, dim=1)
        # print('logits_embed', logits_embed.shape)  # torch.Size([32, 768])
        out_dic = self.semanticloss(logits_embed, source_embed, device=self._cuda_devices[0])
        return out_dic

    def keep_loss_calculate(self, logits):
        b, s, e = logits.shape
        logits = logits.view(b*s, e)
        # print('logits', logits, logits.shape)
        equal = torch.ones(b*s).long()
        # print('equal', equal, equal.shape)
        crossentropyloss = nn.CrossEntropyLoss()
        loss = crossentropyloss(logits.to(self._cuda_devices[0]), equal.to(self._cuda_devices[0]))
        out_dic = {"loss": loss}
        return out_dic

    def confusion_loss_calculate(self, probabilities, orig_tokens):
        out_dic = self.logits2embedding2dis(probabilities, orig_tokens)
        probabilities = out_dic['class_probabilities_simplabel']
        # print('probabilities', probabilities.shape, probabilities)  # torch.Size([32, 2])
        b, e = probabilities.shape
        input = probabilities.view(b * e)
        # print('input', input, input.shape)
        target = torch.ones_like(input) * 0.5
        # print('target', target, target.shape)
        # 用于回归任务的loss函数：'L1Loss', 'MSELoss', 'HuberLoss'
        # L1Loss = nn.L1Loss()
        # MSELoss = nn.MSELoss()
        # HuberLoss = nn.HuberLoss()
        # for i in ['L1Loss', 'MSELoss', 'HuberLoss']:
        #     loss_function = eval(i)
        #     loss = loss_function(input, target)
        #     print(loss)
        smoothl1loss = nn.SmoothL1Loss()
        loss = smoothl1loss(input, target)
        out_dic = {"loss": loss}
        return out_dic


    def batch_loss(self, batch_group: List[TensorDict], target_group: List[TensorDict], for_training: bool,
                   need_post_tag_batch=False, batch_num=0, epoch_num=0) -> Tuple[Union[int, Any], Union[int, Any], Any, Union[float, Any]]:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """

        assert len(batch_group) == 1
        # unsimple batch
        batch = batch_group[0]
        batch = nn_util.move_to_device(batch, self._cuda_devices[0])
        # simple batch (non-parallel with unsimple batch)
        batch_target = target_group[0]
        batch_target = nn_util.move_to_device(batch_target, self._cuda_devices[0])
        print('batch', batch)
        print(batch['tokens']['bert'])
        print('batch_target', batch_target)
        exit()
        output_dict_generator, post_tag_batch, tags, cls_embedding, orig_batch = self.generator(**batch, need_post_tag_batch=True, return_cls=True)
        # print('output_dict_generator', output_dict_generator)
        # print('post_tag_batch', post_tag_batch)
        # print('tags', tags)
        # print('cls_embedding', cls_embedding)
        # exit()

        # print('batch_target', batch_target)
        # print('post_tag_batch', post_tag_batch)
        # print('output_dict_generator', output_dict_generator['logits_labels'])
        # print(output_dict_generator['logits_labels'].shape)  # torch.Size([32, 40, 5002]) -> torch.Size([32, 40, 30522])
        # print('discriminator', self.discriminator)
        # print('nnembedding', self.discriminator.text_field_embedder.token_embedder_bert.bert_model.embeddings.word_embeddings.weight.shape)  # torch.Size([30523, 768])



        # --------------------Calculate generator loss--------------------
        # g_loss1: simp_loss
        # 输入：使用generator输出的logits_labels，simplabels=simple
        # 输出：discriminator的loss和accuracy
        out_dic = self.logits2embedding2dis(output_dict_generator['class_probabilities_labels'], batch['tokens']['bert'],
                                            ground_truth='simple')
        g_loss1 = out_dic['loss']
        g_acc1 = out_dic['accuracy']
        print()
        print('g_loss1', self.g_loss1_hp * g_loss1, end='  ')
        print('g_acc1', g_acc1)
        # exit()

        # g_loss2: remain_loss
        # 输入：source_batch的token_ids 和 generator生成的sentence的token_ids的logits
        # 输出：CrossEntropyLoss
        # out_dic2 = self.remainloss(output_dict_generator['logits_labels'], batch['tokens']['bert'])
        # g_loss2 = out_dic2['loss']
        # print('g_loss2', g_loss2)
        # exit()

        # g_loss3: semantic_loss
        # 输入：output_dict_generator['class_probabilities_labels']代表每一个token的[keep]的程度 & 原始的token
        # 输出：CosineEmbeddingLoss
        out_dic3 = self.logits2embedding2similarity(output_dict_generator['class_probabilities_labels'], batch['tokens']['bert'])
        g_loss3 = out_dic3['loss']
        print('g_loss3', self.g_loss3_hp * g_loss3)
        # exit()

        # g_loss4: keep_loss
        # 输入：output_dict_generator['logits_labels']，希望更多的keep，以保留原句的语义信息
        # 输出：crossentropyloss（定义正确的label是[keep]）
        out_dic4 = self.keep_loss_calculate(output_dict_generator['logits_labels'])
        g_loss4 = out_dic4['loss']
        print('g_loss4', self.g_loss4_hp * g_loss4)
        # exit()

        # g_loss5: confusion_loss
        # 输入：output_dict_generator['class_probabilities_labels'], batch['tokens']['bert']
        # 模型：Discriminator
        # 输出：Discriminator的困惑度
        # 困惑度的计算方式：转换为回归问题，期望discriminator输出的class_probabilities_labels的每一个数字都接近0.5
        out_dic5 = self.confusion_loss_calculate(output_dict_generator['class_probabilities_labels'], batch['tokens']['bert'])
        g_loss5 = out_dic5['loss']
        print('g_loss5', self.g_loss5_hp * g_loss5)
        # exit()



        # --------------------Calculate discriminator loss--------------------
        # d_loss1: true_unsimple_loss
        # input: [batch, simplabels=unsimple]
        # model: discriminator
        # output: loss, accuracy
        _output_dict_discriminator1 = self.discriminator(**batch)
        d_loss1 = _output_dict_discriminator1['loss']
        d_acc1 = _output_dict_discriminator1["accuracy"]
        print('d_loss1', d_loss1, end='  ')
        print('d_acc1', d_acc1)

        # d_loss2: true_simple_loss
        # input: [batch_target, simplabels=simple]
        # model: discriminator
        # output: loss, accuracy
        # print('batch_target', batch_target)
        _output_dict_discriminator2 = self.discriminator(**batch_target)
        d_loss2 = _output_dict_discriminator2['loss']
        d_acc2 = _output_dict_discriminator2['accuracy']
        print('d_loss2', d_loss2, end='  ')
        print('d_acc2', d_acc2)

        # d_loss3: generated_loss
        # 输入：使用generator输出的logits_labels，simplabels=unsimple
        # 输出：discriminator的loss和accuracy
        _out_dic = self.logits2embedding2dis(output_dict_generator['class_probabilities_labels'], batch['tokens']['bert'],
                                            ground_truth='unsimple')
        d_loss3 = _out_dic['loss']
        d_acc3 = _out_dic['accuracy']
        print('d_loss3', d_loss3, end='  ')
        print('d_acc3', d_acc3)


        # loss_generator = self.g_loss1_hp * g_loss1 + self.g_loss2_hp * g_loss2 + self.g_loss3_hp * g_loss3
        # loss_discriminator = d_loss1 + d_loss2 + d_loss3
        # accuracy_generator = g_acc1
        # accuracy_discriminator = (d_acc1 + d_acc2 + d_acc3) / 3

        loss_generator = self.g_loss1_hp * g_loss1 + self.g_loss3_hp * g_loss3 + self.g_loss4_hp * g_loss4 + self.g_loss5_hp * g_loss5
        loss_discriminator = d_loss1 + d_loss2 + d_loss3
        accuracy_generator = g_acc1
        accuracy_discriminator = (d_acc1 + d_acc2 + d_acc3) / 3

        # loss_generator = self.g_loss1_hp * g_loss1 + self.g_loss3_hp * g_loss3
        # loss_discriminator = torch.tensor(0)
        # accuracy_generator = g_acc1
        # accuracy_discriminator = 1

        # 提供一些example供训练的时候参考
        if batch_num % 1000 == 0:
            original_sentence_all = []
            post_tag_batch_all = []
            tags_all = []
            for iii in range(len(orig_batch)):
                orig = []
                for i in orig_batch[iii]:
                    p = self.generator.vocab.get_token_from_index(int(i), namespace='labels')
                    orig.append(p)
                original_sentence_all.append(orig)
                post_tag_batch_all.append(post_tag_batch[iii])
                tags_all.append(tags[iii])

            dic = {
                "original_sentence_all": original_sentence_all,
                "post_tag_batch_all": post_tag_batch_all,
                "tags_all": tags_all
            }
            file_name = self.train_temp_path + '/train_' + str(epoch_num) + '_' + str(batch_num) + '.json'
            write_json_for(dic, file_name)

        if need_post_tag_batch:
            return loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator, post_tag_batch, tags
        else:
            return loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        print("Epoch ", epoch, "/", self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        print((f"Peak CPU memory usage MB: " + str(peak_cpu_usage)))
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")
            print((f"GPU memory usage MB: " + str(memory)))

        train_generator_loss = 0.0
        train_discriminator_loss = 0.0
        # Set the model to "train" mode.
        self.generator.train()
        self.discriminator.train()

        num_gpus = len(self._cuda_devices)  # 0

        # Get tqdm for the training batches
        raw_train_source_generator = self.iterator(self.train_source, num_epochs=1, shuffle=self.shuffle)
        raw_train_target_generator = self.iterator(self.train_target, num_epochs=1, shuffle=self.shuffle)

        train_source_generator = lazy_groups_of(raw_train_source_generator, num_gpus)
        train_target_generator = lazy_groups_of(raw_train_target_generator, num_gpus)
        num_training_batches_source = math.ceil(self.iterator.get_num_batches(self.train_source) / num_gpus)  # 1
        residue_source = num_training_batches_source % self.accumulated_batch_count  # 1
        num_training_batches_target = math.ceil(self.iterator.get_num_batches(self.train_target) / num_gpus)  # 1
        residue_target = num_training_batches_target % self.accumulated_batch_count  # 1
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        train_loss_generator_all = 0.0
        train_loss_discriminator_all = 0.0
        train_accuracy_generator_all = 0.0
        train_accuracy_discriminator_all = 0.0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        logger.info("Training")
        print('Training')
        train_generator_tqdm_source = Tqdm.tqdm(train_source_generator, total=num_training_batches_source)
        self.optimizer_generator.zero_grad()
        self.optimizer_discriminator.zero_grad()

        # for count whether update G or D or both
        batch_num = 0
        for batch_group, target_group in zip(train_generator_tqdm_source, train_target_generator):

            # print(batch_group)
            # print(batch_group[0]['metadata'])
            # sentences = metadata2senteces(batch_group[0]['metadata'])
            # print('sentences', sentences)
            # temp = wordlistlist2sentence(sentences)
            # print(temp)
            # write_json_data(sentences, sentences, '0.json')
            # exit()
            # print(batch_group[0]['tokens']['bert'].shape)
            # print(target_group)
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            iter_len = self.accumulated_batch_count \
                if batches_this_epoch <= (num_training_batches_source - residue_source) else residue_source  # 1

            loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator = \
                self.batch_loss(batch_group, target_group, for_training=True, batch_num=batch_num, epoch_num=epoch)
            assert iter_len == 1
            batch_num += 1
            # exit()
            # loss_generator = loss_generator / iter_len
            # loss_discriminator = loss_discriminator / iter_len
            # print()
            # print('loss_generator', loss_generator)
            # print('loss_discriminator', loss_discriminator)
            # print('accuracy_generator_', accuracy_generator)
            # print('accuracy_discriminator_', accuracy_discriminator)
            train_loss_generator_all += loss_generator.detach().cpu().numpy()
            train_loss_discriminator_all += loss_discriminator.detach().cpu().numpy()
            train_accuracy_generator_all += accuracy_generator
            train_accuracy_discriminator_all += accuracy_discriminator

            # print('batch_num', batch_num)
            # print('self.generator_update_batch', self.generator_update_batch)
            # print('self.discriminator_update_batch', self.discriminator_update_batch)
            # print('batch_num % self.generator_update_batch', batch_num % self.generator_update_batch)
            # print('batch_num % self.discriminator_update_batch', batch_num % self.discriminator_update_batch)
            # for generator to update
            if batch_num % self.discriminator_update_batch != 0:
                # for generator to update
                print('Update generator.')
                loss_generator.backward()
                # torch.cuda.empty_cache()
                self.optimizer_generator.step()
                self.optimizer_generator.zero_grad()

            # for discriminator to update
            # if batch_num % self.generator_update_batch != 0 and batch_num % self.discriminator_update_batch == 0:
            if batch_num % self.discriminator_update_batch == 0:
                print('Update discriminator.')
                loss_discriminator.backward()
                # torch.cuda.empty_cache()
                self.optimizer_discriminator.step()
                self.optimizer_discriminator.zero_grad()

            del batch_group, target_group, loss_generator, loss_discriminator
            # self.optimizer_generator.zero_grad()
            # self.optimizer_discriminator.zero_grad()
            torch.cuda.empty_cache()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            # metrics_generator = training_util.get_metrics(self.generator, train_generator_loss, batches_this_epoch)
            # metrics_discriminator = training_util.get_metrics(self.discriminator, train_discriminator_loss, batches_this_epoch)
            metrics_all = {
                'generator_loss': train_loss_generator_all / batches_this_epoch,
                'generator_accuracy': train_accuracy_generator_all / batches_this_epoch,
                'discriminator_loss': train_loss_discriminator_all / batches_this_epoch,
                'discriminator_accuracy': train_accuracy_discriminator_all / batches_this_epoch
            }
            description = training_util.description_from_metrics(metrics_all)
            print(description)
            train_generator_tqdm_source.set_description(description, refresh=False)

        metrics_generator = {}
        metrics_discriminator = {}
        metrics_generator['loss'] = train_loss_generator_all / batches_this_epoch
        metrics_generator['accuracy'] = train_accuracy_generator_all / batches_this_epoch
        metrics_discriminator['loss'] = train_loss_discriminator_all / batches_this_epoch
        metrics_discriminator['accuracy'] = train_accuracy_discriminator_all / batches_this_epoch

        metrics_generator["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics_generator["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        metrics_discriminator["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics_discriminator["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        print('metrics_generator', metrics_generator)
        print('metrics_discriminator', metrics_discriminator)
        return metrics_generator, metrics_discriminator

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")
        print('Validating')

        self.generator.eval()
        self.discriminator.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)  # 1

        raw_val_generator_source = val_iterator(self.valid_source, num_epochs=1, shuffle=False)
        val_generator_source = lazy_groups_of(raw_val_generator_source, num_gpus)
        num_validation_batches_source = math.ceil(
            val_iterator.get_num_batches(self.valid_source) / num_gpus
        )
        val_generator_tqdm_source = Tqdm.tqdm(val_generator_source, total=num_validation_batches_source)

        raw_val_generator_target = val_iterator(self.valid_target, num_epochs=1, shuffle=False)
        val_generator_target = lazy_groups_of(raw_val_generator_target, num_gpus)
        num_validation_batches_source = math.ceil(
            val_iterator.get_num_batches(self.valid_source) / num_gpus
        )
        # val_generator_tqdm_target = Tqdm.tqdm(val_generator_target, total=num_validation_batches_source)

        batches_this_epoch = 0
        val_loss_generator_all = 0.0
        val_loss_discriminator_all = 0.0
        val_accuracy_generator_all = 0.0
        val_accuracy_discriminator_all = 0.0
        for source_group, target_group in zip(val_generator_tqdm_source, val_generator_target):
            iter_len, _ = source_group[0]['tokens']['bert'].shape
            loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator = \
                self.batch_loss(source_group, target_group, for_training=False)
            if loss_generator is not None and loss_discriminator is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss_generator_all += loss_generator.detach().cpu().numpy()
                val_loss_discriminator_all += loss_discriminator.detach().cpu().numpy()
                val_accuracy_generator_all += accuracy_generator
                val_accuracy_discriminator_all += accuracy_discriminator

            # Update the description with the latest metrics
            metrics_all = {
                'generator_loss': val_loss_generator_all / batches_this_epoch,
                'generator_accuracy': val_accuracy_generator_all / batches_this_epoch,
                'discriminator_loss': val_loss_discriminator_all / batches_this_epoch,
                'discriminator_accuracy': val_accuracy_discriminator_all / batches_this_epoch
            }
            description = training_util.description_from_metrics(metrics_all)
            print(description)
            val_generator_tqdm_source.set_description(description, refresh=False)


        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()
        metrics_generator = {}
        metrics_discriminator = {}
        metrics_generator['loss'] = val_loss_generator_all / batches_this_epoch
        metrics_generator['accuracy'] = val_accuracy_generator_all / batches_this_epoch
        metrics_discriminator['loss'] = val_loss_discriminator_all / batches_this_epoch
        metrics_discriminator['accuracy'] = val_accuracy_discriminator_all / batches_this_epoch
        print('metrics_generator', metrics_generator)
        print('metrics_discriminator', metrics_discriminator)
        return metrics_generator, metrics_discriminator

    def _test_predict(self) -> Tuple[float, int]:
        logger.info("Testing")
        print('Testing')

        self.generator.eval()
        self.discriminator.eval()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)  # 1

        raw_val_generator_source = val_iterator(self.test_source, num_epochs=1, shuffle=False)
        val_generator_source = lazy_groups_of(raw_val_generator_source, num_gpus)
        num_validation_batches_source = math.ceil(
            val_iterator.get_num_batches(self.test_source) / num_gpus
        )

        val_generator_tqdm_source = Tqdm.tqdm(val_generator_source, total=num_validation_batches_source)

        raw_val_generator_target = val_iterator(self.valid_target, num_epochs=1, shuffle=False)
        val_generator_target = lazy_groups_of(raw_val_generator_target, num_gpus)
        num_validation_batches_source = math.ceil(
            val_iterator.get_num_batches(self.test_source) / num_gpus
        )
        # val_generator_tqdm_target = Tqdm.tqdm(val_generator_target, total=num_validation_batches_source)

        batches_this_epoch = 0
        test_loss_generator_all = 0.0
        test_loss_discriminator_all = 0.0
        test_accuracy_generator_all = 0.0
        test_accuracy_discriminator_all = 0.0

        source_batch_all = []
        post_tag_batch_all = []
        tags_all = []


        for source_group, target_group in zip(val_generator_tqdm_source, val_generator_target):

            iter_len, _ = source_group[0]['tokens']['bert'].shape
            loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator, post_tag_batch, tags = \
                self.batch_loss(source_group, target_group, for_training=False, need_post_tag_batch=True)

            # add to all list
            for i in metadata2senteces(source_group[0]['metadata']):
                source_batch_all.append(i)
            for i in post_tag_batch:
                post_tag_batch_all.append(i)
            for i in tags:
                tags_all.append(i)

            print('source_bath_all', source_batch_all)
            # print(len(source_batch_all))
            print('post_tag_batch_all', post_tag_batch_all)
            # print(len(post_tag_batch_all))
            print('tags_all', tags_all)
            #
            # write_json_data(source_sentences=source_batch_all, predict_sentences=post_tag_batch_all, filename='0.json')
            # exit()

            if loss_generator is not None and loss_discriminator is not None:
                batches_this_epoch += 1
                test_loss_generator_all += loss_generator.detach().cpu().numpy()
                test_loss_discriminator_all += loss_discriminator.detach().cpu().numpy()
                test_accuracy_generator_all += accuracy_generator
                test_accuracy_discriminator_all += accuracy_discriminator

            # Update the description with the latest metrics
            metrics_all = {
                'generator_loss': test_loss_generator_all / batches_this_epoch,
                'generator_accuracy': test_accuracy_generator_all / batches_this_epoch,
                'discriminator_loss': test_loss_discriminator_all / batches_this_epoch,
                'discriminator_accuracy': test_accuracy_discriminator_all / batches_this_epoch
            }
            description = training_util.description_from_metrics(metrics_all)
            # print(description)
            val_generator_tqdm_source.set_description(description, refresh=False)


        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()
        metrics_generator = {}
        metrics_discriminator = {}
        metrics_generator['loss'] = test_loss_generator_all / batches_this_epoch
        metrics_generator['accuracy'] = test_accuracy_generator_all / batches_this_epoch
        metrics_discriminator['loss'] = test_loss_discriminator_all / batches_this_epoch
        metrics_discriminator['accuracy'] = test_accuracy_discriminator_all / batches_this_epoch
        print('metrics_generator', metrics_generator)
        print('metrics_discriminator', metrics_discriminator)
        return metrics_generator, metrics_discriminator, source_batch_all, post_tag_batch_all, tags_all

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()


        for epoch in range(self._num_epochs):
            epoch_start_time = time.time()
            metrics_generator, metrics_discriminator = self._train_epoch(epoch)

            # clear cache before validation
            torch.cuda.empty_cache()
            source_batch_all_all = []
            post_tag_batch_all_all = []
            tags_all_all = []
            if self.valid_target is not None and self.valid_source is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    metrics_generator_val, metrics_discriminator_val = self._validation_loss()
                    metrics_generator_test, metrics_discriminator_test, source_batch_all, post_tag_batch_all, tags_all = self._test_predict()
                    source_batch_all_all += source_batch_all
                    post_tag_batch_all_all += post_tag_batch_all
                    tags_all_all += tags_all
            train_metrics = {
                'generator': metrics_generator,
                'discriminator': metrics_discriminator
            }
            val_metrics = {
                'generator': metrics_generator_val,
                'discriminator': metrics_discriminator_val
            }
            test_metrics = {
                'generator': metrics_generator_test,
                'discriminator': metrics_discriminator_test
            }
            print('train_metrics', train_metrics)
            print('val_metrics', val_metrics)
            print('test_metrics', test_metrics)

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value
            for key, value in test_metrics.items():
                metrics["test_" + key] = value


            if self._serialization_dir:
                dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
                )

            print('Saving test with predict.')
            print('source_batch_all_all', source_batch_all_all)
            print('post_tag_batch_all_all', post_tag_batch_all_all)
            print('tags_all_all', tags_all_all)
            filename = os.path.join(self._serialization_dir, f"test_predict_{epoch}.json")
            write_json_data(source_sentences=source_batch_all_all, predict_sentences=post_tag_batch_all_all, tags=tags_all_all, filename=filename)
            # exit()

            # self._save_checkpoint(epoch)
            print('Saving model.')
            out_model = os.path.join(self._serialization_dir, f'generator_{epoch}.th', )
            with open(out_model, 'wb') as f:
                torch.save(self.generator.state_dict(), f)
            print("Generator is dumped")

            out_model = os.path.join(self._serialization_dir, f'discriminator_{epoch}.th')
            with open(out_model, 'wb') as f:
                torch.save(self.discriminator.state_dict(), f)
            print("Discriminator is dumped")

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))
            print("Epoch duration: ", datetime.timedelta(seconds=epoch_elapsed_time))

            epochs_trained += 1

        print('metrics', metrics)
        return metrics


class Trainer_Adversarial_pipeline_llm(TrainerBase):
    def __init__(
        self,
        generator: Model,
        discriminator: Model,
        reader_discriminator,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer,
        schedule_generatorr: torch.optim.lr_scheduler,
        scheduler_discriminator: torch.optim.lr_scheduler,
        iterator: DataIterator,
        train_source: Iterable[Instance],
        train_target: Iterable[Instance],
        valid_source: Iterable[Instance],
        valid_target: Iterable[Instance],
        test_source: Iterable[Instance],
        test_target: Iterable[Instance],
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_iterator: DataIterator = None,
        shuffle: bool = True,
        num_epochs: int = 20,
        accumulated_batch_count: int = 1,
        serialization_dir: Optional[str] = None,
        num_serialized_models_to_keep: int = 20,
        keep_serialized_model_every_num_seconds: int = None,
        checkpointer: Checkpointer = None,
        model_save_interval: float = None,
        cuda_device: Union[int, List] = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        summary_interval: int = 100,
        histogram_interval: int = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        generator_update_batch: int = 1,
        discriminator_update_batch: int = 1,
        log_batch_size_period: Optional[int] = None,
        moving_average: Optional[MovingAverage] = None,
        cold_step_count: int = 0,
        cold_lr: float = 1e-3,
        cuda_verbose_step=None,
        g_loss1_hp=0,
        g_loss2_hp=0,
        g_loss3_hp=0,
        g_loss4_hp=0,
        g_loss5_hp=0,
        g_loss6_hp=0,
        g_loss7_hp=0,
        g_loss8_hp=0,

        d_loss1_hp=0,
        d_loss2_hp=0,
        d_loss3_hp=0,
        train_temp_path='train_temp'
    ) -> None:
        super().__init__(serialization_dir, cuda_device)

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.generator = generator
        self.discriminator = discriminator
        self.reader_discriminator = reader_discriminator
        self.iterator = iterator
        self._validation_iterator = validation_iterator
        self.shuffle = shuffle
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.schedule_generatorr = schedule_generatorr
        self.scheduler_discriminator = scheduler_discriminator
        self.train_source = train_source
        self.train_target = train_target
        self.valid_source = valid_source
        self.valid_target = valid_target
        self.test_source = test_source
        self.test_target = test_target
        self.accumulated_batch_count = accumulated_batch_count
        self.cold_step_count = cold_step_count
        self.cold_lr = cold_lr
        self.cuda_verbose_step = cuda_verbose_step
        self._num_epochs = num_epochs
        self._model_save_interval = model_save_interval
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._learning_rate_scheduler = learning_rate_scheduler
        self._momentum_scheduler = momentum_scheduler
        self._moving_average = moving_average
        self.generator_update_batch = generator_update_batch
        self.discriminator_update_batch = discriminator_update_batch

        # We keep the total batch number as an instance variable because it
        # is used inside a closure for the hook which logs activations in
        # ``_enable_activation_logging``.
        self._batch_num_total = 0
        self._log_batch_size_period = log_batch_size_period
        self._last_log = 0.0  # time of last logging

        self.remainloss = Remainloss()
        self.semanticloss = Semanticloss()
        self.marginLoss = MarginLoss(margin=0.1)
        self.contrast_loss = ContrastLoss(scale=5.0)

        self.g_loss1_hp = g_loss1_hp
        self.g_loss2_hp = g_loss2_hp
        self.g_loss3_hp = g_loss3_hp
        self.g_loss4_hp = g_loss4_hp
        self.g_loss5_hp = g_loss5_hp
        self.g_loss6_hp = g_loss6_hp
        self.g_loss7_hp = g_loss7_hp
        self.g_loss8_hp = g_loss8_hp

        self.d_loss1_hp = d_loss1_hp
        self.d_loss2_hp = d_loss2_hp
        self.d_loss3_hp = d_loss3_hp

        self.train_temp_path = train_temp_path


    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.generator, self._grad_norm)

    def logits2embedding2dis(self, logits, orig_tokens, ground_truth='simple'):
        # print('orig_tokens', orig_tokens, orig_tokens.shape)
        equal = torch.zeros_like(orig_tokens)
        orig_mask = (1 - torch.eq(orig_tokens, equal).long()).long()
        # print('orig_mask', orig_mask, orig_mask.shape)
        logits = logits[:,:,1].unsqueeze(2)
        # print('logits', logits.shape)  # torch.Size([32, 30, 1])

        word_embedding = self.discriminator.text_field_embedder.token_embedder_bert.bert_model.embeddings.word_embeddings
        word_embed = word_embedding(orig_tokens)
        # print('word_embed', word_embed.shape)  # torch.Size([32, 30, 768])

        logits_embedding = logits * word_embed
        # print('logits_embedding', logits_embedding.shape)  # torch.Size([32, 30, 768])

        if ground_truth == 'simple':
            # .to(self._cuda_devices[0])
            simplabels = torch.zeros(logits.shape[0]).long().to(self._cuda_devices[0])
        elif ground_truth == 'unsimple':
            simplabels = torch.ones(logits.shape[0]).long().to(self._cuda_devices[0])
        out_dic = self.discriminator(simplabels=simplabels, logits_embedding=logits_embedding, mask=orig_mask)
        return out_dic

    def logits2embedding2similarity(self, logits, orig_tokens):
        equal = torch.zeros_like(orig_tokens)
        pad = (1 - torch.eq(orig_tokens, equal).long()).unsqueeze(2).repeat(1,1,768)
        logits = logits[:, :, 1].unsqueeze(2)
        # print('logits', logits.shape)  # torch.Size([2, 30, 1])
        word_embedding = self.generator.text_field_embedder.token_embedder_bert.bert_model.embeddings.word_embeddings
        word_embed = word_embedding(orig_tokens) * pad
        # print('word_embed', word_embed.shape)  # torch.Size([32, 30, 768])
        source_embed = torch.sum(word_embed, dim=1)
        # print('source_embed', source_embed.shape)  # torch.Size([32, 768])
        logits_embed = torch.sum(logits * word_embed, dim=1)
        # print('logits_embed', logits_embed.shape)  # torch.Size([32, 768])
        out_dic = self.semanticloss(logits_embed, source_embed, device=self._cuda_devices[0])
        return out_dic

    def keep_loss_calculate(self, logits):
        b, s, e = logits.shape
        logits = logits.view(b*s, e)
        # print('logits', logits, logits.shape)
        equal = torch.ones(b*s).long()
        # print('equal', equal, equal.shape)
        crossentropyloss = nn.CrossEntropyLoss()
        loss = crossentropyloss(logits.to(self._cuda_devices[0]), equal.to(self._cuda_devices[0]))
        out_dic = {"loss": loss}
        return out_dic

    def confusion_loss_calculate(self, probabilities, orig_tokens):
        out_dic = self.logits2embedding2dis(probabilities, orig_tokens)
        probabilities = out_dic['class_probabilities_simplabel']
        # print('probabilities', probabilities.shape, probabilities)  # torch.Size([32, 2])
        b, e = probabilities.shape
        input = probabilities.view(b * e)
        # print('input', input, input.shape)
        target = torch.ones_like(input) * 0.5
        # print('target', target, target.shape)
        # 用于回归任务的loss函数：'L1Loss', 'MSELoss', 'HuberLoss'
        # L1Loss = nn.L1Loss()
        # MSELoss = nn.MSELoss()
        # HuberLoss = nn.HuberLoss()
        # for i in ['L1Loss', 'MSELoss', 'HuberLoss']:
        #     loss_function = eval(i)
        #     loss = loss_function(input, target)
        #     print(loss)
        smoothl1loss = nn.SmoothL1Loss()
        loss = smoothl1loss(input, target)
        out_dic = {"loss": loss}
        return out_dic

    def offset_labels(self, labels, offsets, max_len):
        # print('labels', labels.shape)  # [batch_size, sequence_length]
        # print('offsets', offsets.shape)  # [batch_size, sequence_length]
        labels_offset_all = []
        for b in range(labels.size(0)):
            labels_offset = []
            for i in range(len(offsets[b]) - 1):
                index1 = offsets[b][i]
                index2 = offsets[b][i + 1]
                if index2 <= index1:
                    labels_offset.append(0)
                else:
                    for j in range(index1, index2):
                        labels_offset.append(int(labels[b][i]))
            labels_offset.append(int(labels[b][-1]))
            labels_offset_all.append(labels_offset)
        # print('labels_offset_all', labels_offset_all)
        padded_list = [sublist + [0] * (max_len - len(sublist)) for sublist in labels_offset_all]
        padded_list = [sublist[:max_len] for sublist in padded_list]
        # print('padded_list', padded_list)
        # print(len(padded_list[0]))
        # print(len(padded_list[1]))
        return torch.tensor(padded_list, dtype=torch.long)

    def llmcwi_loss_calculate(self, logits, llmcwi_labels, tokens):
        # print('bert-offsets', tokens['bert-offsets'], tokens['bert-offsets'].shape)
        b, s, e = logits.shape
        logits = logits.view(b * s, e)
        # print('logits', logits, logits.shape)
        equal = self.offset_labels(llmcwi_labels, tokens['bert-offsets'], s)
        # print('equal', equal, equal.shape)
        equal = equal.view(b*s)
        # print('equal', equal, equal.shape)
        crossentropyloss = nn.CrossEntropyLoss()
        loss = crossentropyloss(logits.to(self._cuda_devices[0]), equal.to(self._cuda_devices[0]))
        # loss = crossentropyloss(logits, equal)
        out_dic = {"loss": loss}
        return out_dic

    def margin_comfusion_loss(self, probabilities, orig_tokens):
        out_dic = self.logits2embedding2dis(probabilities, orig_tokens)
        probabilities = out_dic['class_probabilities_simplabel']
        b, e = probabilities.shape
        input = probabilities.view(b * e)
        loss = self.marginLoss(input)
        out_dic = {"loss": loss}
        return out_dic


    def contrast_loss_calculate(self, batch, probabilities):
        # 得到原句的Discriminator得分
        _output_dict_discriminator1 = self.discriminator(**batch)
        logits_before = _output_dict_discriminator1['class_probabilities_simplabel']
        # 得到Generator之后的Discriminator得分
        out_dic = self.logits2embedding2dis(probabilities, batch['tokens']['bert'])
        logits_after = out_dic['class_probabilities_simplabel']
        logits_before = logits_before[:, 0].unsqueeze(1)
        logits_after = logits_after[:, 0].unsqueeze(1)
        # .to(self._cuda_devices[0])
        loss = self.contrast_loss(logits_before=logits_before.to(self._cuda_devices[0]), logits_after=logits_after.to(self._cuda_devices[0]))
        out_dic = {"loss": loss}
        return out_dic

    def batch_loss(self, batch_group: List[TensorDict], target_group: List[TensorDict], for_training: bool,
                   need_post_tag_batch=False, batch_num=0, epoch_num=0) -> Tuple[Union[int, Any], Union[int, Any], Any, Union[float, Any]]:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """

        assert len(batch_group) == 1
        # unsimple batch
        batch = batch_group[0]
        batch = nn_util.move_to_device(batch, self._cuda_devices[0])
        # simple batch (non-parallel with unsimple batch)
        batch_target = target_group[0]
        batch_target = nn_util.move_to_device(batch_target, self._cuda_devices[0])
        # print('batch', batch)
        # print(batch['tokens']['bert'])
        # print('batch_target', batch_target)
        # exit()
        output_dict_generator, post_tag_batch, tags, cls_embedding, orig_batch = self.generator(**batch, need_post_tag_batch=True, return_cls=True)
        # print('output_dict_generator', output_dict_generator)
        # print('post_tag_batch', post_tag_batch)
        # print('tags', tags)
        # print('cls_embedding', cls_embedding)
        # exit()

        # print('batch_target', batch_target)
        # print('post_tag_batch', post_tag_batch)
        # print('output_dict_generator', output_dict_generator['logits_labels'])
        # print(output_dict_generator['logits_labels'].shape)  # torch.Size([32, 40, 5002]) -> torch.Size([32, 40, 30522])
        # print('discriminator', self.discriminator)
        # print('nnembedding', self.discriminator.text_field_embedder.token_embedder_bert.bert_model.embeddings.word_embeddings.weight.shape)  # torch.Size([30523, 768])



        # --------------------Calculate generator loss--------------------
        # g_loss1: simp_loss
        # 输入：使用generator输出的logits_labels，simplabels=simple
        # 输出：discriminator的loss和accuracy
        if self.g_loss1_hp > 0:
            out_dic = self.logits2embedding2dis(output_dict_generator['class_probabilities_labels'], batch['tokens']['bert'],
                                                ground_truth='simple')
            g_loss1 = out_dic['loss']
            g_acc1 = out_dic['accuracy']
            print()
            print('g_loss1', self.g_loss1_hp * g_loss1, end='  ')
            print('g_acc1', g_acc1)
            # exit()

        # g_loss2: remain_loss
        # 输入：source_batch的token_ids 和 generator生成的sentence的token_ids的logits
        # 输出：CrossEntropyLoss
        self.g_loss2_hp = 0  # 已经不用啦
        if self.g_loss2_hp > 0:
            out_dic2 = self.remainloss(output_dict_generator['logits_labels'], batch['tokens']['bert'])
            g_loss2 = out_dic2['loss']
            print('g_loss2', self.g_loss2_hp * g_loss2)
            # exit()

        # g_loss3: semantic_loss
        # 输入：output_dict_generator['class_probabilities_labels']代表每一个token的[keep]的程度 & 原始的token
        # 输出：CosineEmbeddingLoss
        print()
        if self.g_loss3_hp > 0:
            out_dic3 = self.logits2embedding2similarity(output_dict_generator['class_probabilities_labels'], batch['tokens']['bert'])
            g_loss3 = out_dic3['loss']
            print('g_loss3', self.g_loss3_hp * g_loss3)
            # exit()

        # g_loss4: keep_loss
        # 输入：output_dict_generator['logits_labels']，希望更多的keep，以保留原句的语义信息
        # 输出：crossentropyloss（定义正确的label是[KEEP]）
        # self.g_loss4_hp = 0
        if self.g_loss4_hp > 0:
            out_dic4 = self.keep_loss_calculate(output_dict_generator['logits_labels'])
            g_loss4 = out_dic4['loss']
            print('g_loss4', self.g_loss4_hp * g_loss4)
            # exit()

        # g_loss5: confusion_loss
        # 输入：output_dict_generator['class_probabilities_labels'], batch['tokens']['bert']
        # 模型：Discriminator
        # 输出：Discriminator的困惑度
        # 困惑度的计算方式：转换为回归问题，期望discriminator输出的class_probabilities_labels的每一个数字都接近0.5
        # self.g_loss5_hp = 1
        if self.g_loss5_hp > 0:
            out_dic5 = self.confusion_loss_calculate(output_dict_generator['class_probabilities_labels'], batch['tokens']['bert'])
            g_loss5 = out_dic5['loss']
            print('g_loss5', self.g_loss5_hp * g_loss5)
            # exit()

        # g_loss6: llmcwi_loss
        # 输入：output_dict_generator['logits_labels']
        # 输出：crossentropy_loss
        # self.g_loss6_hp = 1
        if self.g_loss6_hp > 0:
            out_dic6 = self.llmcwi_loss_calculate(output_dict_generator['logits_labels'], batch['labels'], batch['tokens'])
            g_loss6 = out_dic6['loss']
            print('g_loss6', self.g_loss6_hp * g_loss6)
            # exit()

        # g_loss7: margin_confusion_loss
        # 输入：output_dict_generator['logits_labels']
        # 输出：marginLoss
        # self.g_loss7_hp = 1
        if self.g_loss7_hp > 0:
            out_dic7 = self.margin_comfusion_loss(output_dict_generator['class_probabilities_labels'], batch['tokens']['bert'])
            g_loss7 = out_dic7['loss']
            print('g_loss7', self.g_loss7_hp * g_loss7)
            # exit()

        # g_loss8: contrast_loss
        # 输入：
        # 输出：contrast_loss
        if self.g_loss8_hp > 0:
            out_dic8 = self.contrast_loss_calculate(batch, output_dict_generator['class_probabilities_labels'])
            g_loss8 = out_dic8['loss']
            print('g_loss8', self.g_loss8_hp * g_loss8)
            # exit()



        # --------------------Calculate discriminator loss--------------------
        # d_loss1: true_unsimple_loss
        # input: [batch, simplabels=unsimple]
        # model: discriminator
        # output: loss, accuracy
        self.d_loss1_hp = 1
        if self.d_loss1_hp > 0:
            _output_dict_discriminator1 = self.discriminator(**batch)
            d_loss1 = _output_dict_discriminator1['loss']
            d_acc1 = _output_dict_discriminator1["accuracy"]
            print('d_loss1', d_loss1, end='  ')
            print('d_acc1', d_acc1)

        # d_loss2: true_simple_loss
        # input: [batch_target, simplabels=simple]
        # model: discriminator
        # output: loss, accuracy
        # print('batch_target', batch_target)
        if self.d_loss2_hp > 0:
            _output_dict_discriminator2 = self.discriminator(**batch_target)
            d_loss2 = _output_dict_discriminator2['loss']
            d_acc2 = _output_dict_discriminator2['accuracy']
            print('d_loss2', d_loss2, end='  ')
            print('d_acc2', d_acc2)

        # d_loss3: generated_loss
        # 输入：使用generator输出的logits_labels，simplabels=unsimple
        # 输出：discriminator的loss和accuracy
        if self.d_loss3_hp > 0:
            _out_dic = self.logits2embedding2dis(output_dict_generator['class_probabilities_labels'], batch['tokens']['bert'],
                                                ground_truth='unsimple')
            d_loss3 = _out_dic['loss']
            d_acc3 = _out_dic['accuracy']
            print('d_loss3', d_loss3, end='  ')
            print('d_acc3', d_acc3)


        # loss_generator = self.g_loss1_hp * g_loss1 + self.g_loss2_hp * g_loss2 + self.g_loss3_hp * g_loss3
        # loss_discriminator = d_loss1 + d_loss2 + d_loss3
        # accuracy_generator = g_acc1
        # accuracy_discriminator = (d_acc1 + d_acc2 + d_acc3) / 3

        accuracy_generator = 0.0
        accuracy_discriminator = 0.0
        loss_generator = 0.0
        if self.g_loss1_hp > 0:
            loss_generator += self.g_loss1_hp * g_loss1
        if self.g_loss2_hp > 0:
            loss_generator += self.g_loss2_hp * g_loss2
        if self.g_loss3_hp > 0:
            loss_generator += self.g_loss3_hp * g_loss3
        if self.g_loss4_hp > 0:
            loss_generator += self.g_loss4_hp * g_loss4
        if self.g_loss5_hp > 0:
            loss_generator += self.g_loss5_hp * g_loss5
        if self.g_loss6_hp > 0:
            loss_generator += self.g_loss6_hp * g_loss6
        if self.g_loss7_hp > 0:
            loss_generator += self.g_loss7_hp * g_loss7
        if self.g_loss8_hp > 0:
            loss_generator += self.g_loss8_hp * g_loss8
        # loss_generator = self.g_loss1_hp * g_loss1 + self.g_loss2_hp * g_loss2 + self.g_loss3_hp * g_loss3 + self.g_loss4_hp * g_loss4 + \
        #                  self.g_loss5_hp * g_loss5 + self.g_loss6_hp * g_loss6 + self.g_loss7_hp * g_loss7
        # loss_discriminator = d_loss1
        loss_discriminator = 0.0
        if self.d_loss1_hp > 0:
            loss_discriminator += self.d_loss1_hp * d_loss1
        if self.d_loss2_hp > 0:
            loss_discriminator += self.d_loss2_hp * d_loss2
        if self.d_loss3_hp > 0:
            loss_discriminator += self.d_loss3_hp * d_loss3
        # accuracy_generator = g_acc1
        # accuracy_discriminator = (d_acc1 + d_acc2 + d_acc3) / 3

        # loss_generator = self.g_loss1_hp * g_loss1 + self.g_loss3_hp * g_loss3
        # loss_discriminator = torch.tensor(0)
        # accuracy_generator = g_acc1
        # accuracy_discriminator = 1

        # # 提供一些example供训练的时候参考
        # if batch_num % 1000 == 0:
        #     original_sentence_all = []
        #     post_tag_batch_all = []
        #     tags_all = []
        #     for iii in range(len(orig_batch)):
        #         orig = []
        #         for i in orig_batch[iii]:
        #             p = self.generator.vocab.get_token_from_index(int(i), namespace='labels')
        #             orig.append(p)
        #         original_sentence_all.append(orig)
        #         post_tag_batch_all.append(post_tag_batch[iii])
        #         tags_all.append(tags[iii])
        #
        #     dic = {
        #         "original_sentence_all": original_sentence_all,
        #         "post_tag_batch_all": post_tag_batch_all,
        #         "tags_all": tags_all
        #     }
        #     file_name = self.train_temp_path + '/train_' + str(epoch_num) + '_' + str(batch_num) + '.json'
        #     write_json_for(dic, file_name)

        if need_post_tag_batch:
            return loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator, post_tag_batch, tags
        else:
            return loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        print("Epoch ", epoch, "/", self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        print((f"Peak CPU memory usage MB: " + str(peak_cpu_usage)))
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")
            print((f"GPU memory usage MB: " + str(memory)))

        train_generator_loss = 0.0
        train_discriminator_loss = 0.0
        # Set the model to "train" mode.
        self.generator.train()
        self.discriminator.train()

        num_gpus = len(self._cuda_devices)  # 0

        # Get tqdm for the training batches
        raw_train_source_generator = self.iterator(self.train_source, num_epochs=1, shuffle=self.shuffle)
        raw_train_target_generator = self.iterator(self.train_target, num_epochs=1, shuffle=self.shuffle)
        # print('raw_train_source_generator', raw_train_source_generator)
        # print('raw_train_target_generator', raw_train_target_generator)

        train_source_generator = lazy_groups_of(raw_train_source_generator, num_gpus)
        train_target_generator = lazy_groups_of(raw_train_target_generator, num_gpus)
        num_training_batches_source = math.ceil(self.iterator.get_num_batches(self.train_source) / num_gpus)  # 1
        residue_source = num_training_batches_source % self.accumulated_batch_count  # 1
        num_training_batches_target = math.ceil(self.iterator.get_num_batches(self.train_target) / num_gpus)  # 1
        residue_target = num_training_batches_target % self.accumulated_batch_count  # 1
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        train_loss_generator_all = 0.0
        train_loss_discriminator_all = 0.0
        train_accuracy_generator_all = 0.0
        train_accuracy_discriminator_all = 0.0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        logger.info("Training")
        print('Training')
        train_generator_tqdm_source = Tqdm.tqdm(train_source_generator, total=num_training_batches_source)
        self.optimizer_generator.zero_grad()
        self.optimizer_discriminator.zero_grad()

        # for count whether update G or D or both
        batch_num = 0
        for batch_group, target_group in zip(train_generator_tqdm_source, train_target_generator):

            # print(batch_group)
            # print(target_group)
            # print(batch_group[0]['metadata'])
            # sentences = metadata2senteces(batch_group[0]['metadata'])
            # print('sentences', sentences)
            # temp = wordlistlist2sentence(sentences)
            # print(temp)
            # write_json_data(sentences, sentences, '0.json')
            # exit()
            # print(batch_group[0]['tokens']['bert'].shape)
            # print(target_group)
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            iter_len = self.accumulated_batch_count \
                if batches_this_epoch <= (num_training_batches_source - residue_source) else residue_source  # 1

            loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator = \
                self.batch_loss(batch_group, target_group, for_training=True, batch_num=batch_num, epoch_num=epoch)
            assert iter_len == 1
            batch_num += 1
            # exit()
            # loss_generator = loss_generator / iter_len
            # loss_discriminator = loss_discriminator / iter_len
            # print()
            # print('loss_generator', loss_generator)
            # print('loss_discriminator', loss_discriminator)
            # print('accuracy_generator_', accuracy_generator)
            # print('accuracy_discriminator_', accuracy_discriminator)
            train_loss_generator_all += loss_generator.detach().cpu().numpy()
            train_loss_discriminator_all += loss_discriminator.detach().cpu().numpy()
            train_accuracy_generator_all += accuracy_generator
            train_accuracy_discriminator_all += accuracy_discriminator

            # print('batch_num', batch_num)
            # print('self.generator_update_batch', self.generator_update_batch)
            # print('self.discriminator_update_batch', self.discriminator_update_batch)
            # print('batch_num % self.generator_update_batch', batch_num % self.generator_update_batch)
            # print('batch_num % self.discriminator_update_batch', batch_num % self.discriminator_update_batch)
            # for generator to update
            if batch_num % self.discriminator_update_batch != 0:
                # for generator to update
                print('Update generator.')
                loss_generator.backward()
                # torch.cuda.empty_cache()
                self.optimizer_generator.step()
                self.optimizer_generator.zero_grad()

            # for discriminator to update
            # if batch_num % self.generator_update_batch != 0 and batch_num % self.discriminator_update_batch == 0:
            if batch_num % self.discriminator_update_batch == 0:
                print('Update discriminator.')
                loss_discriminator.backward()
                # torch.cuda.empty_cache()
                self.optimizer_discriminator.step()
                self.optimizer_discriminator.zero_grad()

            del batch_group, target_group, loss_generator, loss_discriminator
            # self.optimizer_generator.zero_grad()
            # self.optimizer_discriminator.zero_grad()
            torch.cuda.empty_cache()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            # metrics_generator = training_util.get_metrics(self.generator, train_generator_loss, batches_this_epoch)
            # metrics_discriminator = training_util.get_metrics(self.discriminator, train_discriminator_loss, batches_this_epoch)
            metrics_all = {
                'generator_loss': train_loss_generator_all / batches_this_epoch,
                'generator_accuracy': train_accuracy_generator_all / batches_this_epoch,
                'discriminator_loss': train_loss_discriminator_all / batches_this_epoch,
                'discriminator_accuracy': train_accuracy_discriminator_all / batches_this_epoch
            }
            description = training_util.description_from_metrics(metrics_all)
            print(description)
            train_generator_tqdm_source.set_description(description, refresh=False)

        metrics_generator = {}
        metrics_discriminator = {}
        metrics_generator['loss'] = train_loss_generator_all / batches_this_epoch
        metrics_generator['accuracy'] = train_accuracy_generator_all / batches_this_epoch
        metrics_discriminator['loss'] = train_loss_discriminator_all / batches_this_epoch
        metrics_discriminator['accuracy'] = train_accuracy_discriminator_all / batches_this_epoch

        metrics_generator["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics_generator["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        metrics_discriminator["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics_discriminator["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        print('metrics_generator', metrics_generator)
        print('metrics_discriminator', metrics_discriminator)
        return metrics_generator, metrics_discriminator

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")
        print('Validating')

        self.generator.eval()
        self.discriminator.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)  # 1

        raw_val_generator_source = val_iterator(self.valid_source, num_epochs=1, shuffle=False)
        val_generator_source = lazy_groups_of(raw_val_generator_source, num_gpus)
        num_validation_batches_source = math.ceil(
            val_iterator.get_num_batches(self.valid_source) / num_gpus
        )
        val_generator_tqdm_source = Tqdm.tqdm(val_generator_source, total=num_validation_batches_source)

        raw_val_generator_target = val_iterator(self.valid_target, num_epochs=1, shuffle=False)
        val_generator_target = lazy_groups_of(raw_val_generator_target, num_gpus)
        num_validation_batches_source = math.ceil(
            val_iterator.get_num_batches(self.valid_source) / num_gpus
        )
        # val_generator_tqdm_target = Tqdm.tqdm(val_generator_target, total=num_validation_batches_source)

        batches_this_epoch = 0
        val_loss_generator_all = 0.0
        val_loss_discriminator_all = 0.0
        val_accuracy_generator_all = 0.0
        val_accuracy_discriminator_all = 0.0
        for source_group, target_group in zip(val_generator_tqdm_source, val_generator_target):
            iter_len, _ = source_group[0]['tokens']['bert'].shape
            loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator = \
                self.batch_loss(source_group, target_group, for_training=False)
            if loss_generator is not None and loss_discriminator is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss_generator_all += loss_generator.detach().cpu().numpy()
                val_loss_discriminator_all += loss_discriminator.detach().cpu().numpy()
                val_accuracy_generator_all += accuracy_generator
                val_accuracy_discriminator_all += accuracy_discriminator

            # Update the description with the latest metrics
            metrics_all = {
                'generator_loss': val_loss_generator_all / batches_this_epoch,
                'generator_accuracy': val_accuracy_generator_all / batches_this_epoch,
                'discriminator_loss': val_loss_discriminator_all / batches_this_epoch,
                'discriminator_accuracy': val_accuracy_discriminator_all / batches_this_epoch
            }
            description = training_util.description_from_metrics(metrics_all)
            print(description)
            val_generator_tqdm_source.set_description(description, refresh=False)


        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()
        metrics_generator = {}
        metrics_discriminator = {}
        metrics_generator['loss'] = val_loss_generator_all / batches_this_epoch
        metrics_generator['accuracy'] = val_accuracy_generator_all / batches_this_epoch
        metrics_discriminator['loss'] = val_loss_discriminator_all / batches_this_epoch
        metrics_discriminator['accuracy'] = val_accuracy_discriminator_all / batches_this_epoch
        print('metrics_generator', metrics_generator)
        print('metrics_discriminator', metrics_discriminator)
        return metrics_generator, metrics_discriminator

    def _test_predict(self) -> Tuple[float, int]:
        logger.info("Testing")
        print('Testing')

        self.generator.eval()
        self.discriminator.eval()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)  # 1

        raw_val_generator_source = val_iterator(self.test_source, num_epochs=1, shuffle=False)
        val_generator_source = lazy_groups_of(raw_val_generator_source, num_gpus)
        num_validation_batches_source = math.ceil(
            val_iterator.get_num_batches(self.test_source) / num_gpus
        )

        val_generator_tqdm_source = Tqdm.tqdm(val_generator_source, total=num_validation_batches_source)

        raw_val_generator_target = val_iterator(self.valid_target, num_epochs=1, shuffle=False)
        val_generator_target = lazy_groups_of(raw_val_generator_target, num_gpus)
        num_validation_batches_source = math.ceil(
            val_iterator.get_num_batches(self.test_source) / num_gpus
        )
        # val_generator_tqdm_target = Tqdm.tqdm(val_generator_target, total=num_validation_batches_source)

        batches_this_epoch = 0
        test_loss_generator_all = 0.0
        test_loss_discriminator_all = 0.0
        test_accuracy_generator_all = 0.0
        test_accuracy_discriminator_all = 0.0

        source_batch_all = []
        post_tag_batch_all = []
        tags_all = []


        for source_group, target_group in zip(val_generator_tqdm_source, val_generator_target):

            iter_len, _ = source_group[0]['tokens']['bert'].shape
            loss_generator, loss_discriminator, accuracy_generator, accuracy_discriminator, post_tag_batch, tags = \
                self.batch_loss(source_group, target_group, for_training=False, need_post_tag_batch=True)

            # add to all list
            for i in metadata2senteces(source_group[0]['metadata']):
                source_batch_all.append(i)
            for i in post_tag_batch:
                post_tag_batch_all.append(i)
            for i in tags:
                tags_all.append(i)

            print('source_bath_all', source_batch_all)
            # print(len(source_batch_all))
            print('post_tag_batch_all', post_tag_batch_all)
            # print(len(post_tag_batch_all))
            print('tags_all', tags_all)
            #
            # write_json_data(source_sentences=source_batch_all, predict_sentences=post_tag_batch_all, filename='0.json')
            # exit()

            if loss_generator is not None and loss_discriminator is not None:
                batches_this_epoch += 1
                test_loss_generator_all += loss_generator.detach().cpu().numpy()
                test_loss_discriminator_all += loss_discriminator.detach().cpu().numpy()
                test_accuracy_generator_all += accuracy_generator
                test_accuracy_discriminator_all += accuracy_discriminator

            # Update the description with the latest metrics
            metrics_all = {
                'generator_loss': test_loss_generator_all / batches_this_epoch,
                'generator_accuracy': test_accuracy_generator_all / batches_this_epoch,
                'discriminator_loss': test_loss_discriminator_all / batches_this_epoch,
                'discriminator_accuracy': test_accuracy_discriminator_all / batches_this_epoch
            }
            description = training_util.description_from_metrics(metrics_all)
            # print(description)
            val_generator_tqdm_source.set_description(description, refresh=False)


        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()
        metrics_generator = {}
        metrics_discriminator = {}
        metrics_generator['loss'] = test_loss_generator_all / batches_this_epoch
        metrics_generator['accuracy'] = test_accuracy_generator_all / batches_this_epoch
        metrics_discriminator['loss'] = test_loss_discriminator_all / batches_this_epoch
        metrics_discriminator['accuracy'] = test_accuracy_discriminator_all / batches_this_epoch
        print('metrics_generator', metrics_generator)
        print('metrics_discriminator', metrics_discriminator)
        return metrics_generator, metrics_discriminator, source_batch_all, post_tag_batch_all, tags_all

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()


        for epoch in range(self._num_epochs):
            epoch_start_time = time.time()
            metrics_generator, metrics_discriminator = self._train_epoch(epoch)

            # clear cache before validation
            torch.cuda.empty_cache()
            source_batch_all_all = []
            post_tag_batch_all_all = []
            tags_all_all = []
            if self.valid_target is not None and self.valid_source is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    metrics_generator_val, metrics_discriminator_val = self._validation_loss()
                    metrics_generator_test, metrics_discriminator_test, source_batch_all, post_tag_batch_all, tags_all = self._test_predict()
                    source_batch_all_all += source_batch_all
                    post_tag_batch_all_all += post_tag_batch_all
                    tags_all_all += tags_all
            train_metrics = {
                'generator': metrics_generator,
                'discriminator': metrics_discriminator
            }
            val_metrics = {
                'generator': metrics_generator_val,
                'discriminator': metrics_discriminator_val
            }
            test_metrics = {
                'generator': metrics_generator_test,
                'discriminator': metrics_discriminator_test
            }
            print('train_metrics', train_metrics)
            print('val_metrics', val_metrics)
            print('test_metrics', test_metrics)

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value
            for key, value in test_metrics.items():
                metrics["test_" + key] = value


            if self._serialization_dir:
                dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
                )

            print('Saving test with predict.')
            print('source_batch_all_all', source_batch_all_all)
            print('post_tag_batch_all_all', post_tag_batch_all_all)
            print('tags_all_all', tags_all_all)
            filename = os.path.join(self._serialization_dir, f"test_predict_{epoch}.json")
            write_json_data(source_sentences=source_batch_all_all, predict_sentences=post_tag_batch_all_all, tags=tags_all_all, filename=filename)
            # exit()

            # self._save_checkpoint(epoch)
            print('Saving model.')
            out_model = os.path.join(self._serialization_dir, f'generator_{epoch}.th', )
            with open(out_model, 'wb') as f:
                torch.save(self.generator.state_dict(), f)
            print("Generator is dumped")

            out_model = os.path.join(self._serialization_dir, f'discriminator_{epoch}.th')
            with open(out_model, 'wb') as f:
                torch.save(self.discriminator.state_dict(), f)
            print("Discriminator is dumped")

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))
            print("Epoch duration: ", datetime.timedelta(seconds=epoch_elapsed_time))

            epochs_trained += 1

        print('metrics', metrics)
        return metrics