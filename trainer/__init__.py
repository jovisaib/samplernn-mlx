import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import heapq


# Based on mlx.utils.trainer.Trainer code.
# Allows multiple inputs to the model, not all need to be Arrays.
class Trainer(object):

    def __init__(self, model, criterion, optimizer, dataset):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.iterations = 0
        self.epochs = 0
        self.stats = {}
        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }

    def register_plugin(self, plugin):
        plugin.register(self)

        intervals = plugin.trigger_interval
        if not isinstance(intervals, list):
            intervals = [intervals]
        for (duration, unit) in intervals:
            queue = self.plugin_queues[unit]
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        args = (time,) + args
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            heapq.heapify(q)

        for self.epochs in range(self.epochs + 1, self.epochs + epochs + 1):
            self.train()
            self.call_plugins('epoch', self.epochs)

    def train(self):
        for (self.iterations, data) in \
                enumerate(self.dataset, self.iterations + 1):
            batch_inputs = data[: -1]
            batch_target = data[-1]
            self.call_plugins(
                'batch', self.iterations, batch_inputs, batch_target
            )

            def wrap(input):
                if isinstance(input, mx.array):
                    return input
                return mx.array(input)
            batch_inputs = list(map(wrap, batch_inputs))

            batch_target = mx.array(batch_target)

            plugin_data = [None, None]

            def loss_fn(model, *inputs):
                batch_output = model(*inputs)
                loss = self.criterion(batch_output, batch_target)
                if plugin_data[0] is None:
                    plugin_data[0] = batch_output
                    plugin_data[1] = loss
                return loss

            loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
            loss, grads = loss_and_grad_fn(*batch_inputs)

            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters())

            self.call_plugins(
                'iteration', self.iterations, batch_inputs, batch_target,
                *plugin_data
            )
            self.call_plugins('update', self.iterations, self.model)