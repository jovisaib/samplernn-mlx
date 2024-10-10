import mlx.core as mx
import mlx.nn as nn

import math


class LearnedUpsampling1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()

        self.conv_t = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False
        )

        if bias:
            self.bias = mx.zeros((out_channels, kernel_size))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        # MLX doesn't have a direct equivalent to reset_parameters
        # We'll initialize the conv_t weights manually
        self.conv_t.weight = mx.random.uniform(
            low=-0.05, high=0.05, shape=self.conv_t.weight.shape
        )
        if self.bias is not None:
            self.bias = mx.zeros(self.bias.shape)

    def __call__(self, x):
        batch_size, _, length = x.shape
        kernel_size = self.conv_t.kernel_size[0]
        
        output = self.conv_t(x)
        
        if self.bias is not None:
            bias = mx.broadcast_to(
                self.bias.reshape(1, self.conv_t.out_channels, 1, kernel_size),
                (batch_size, self.conv_t.out_channels, length, kernel_size)
            )
            bias = bias.reshape(batch_size, self.conv_t.out_channels, length * kernel_size)
            output = output + bias
        
        return output


def lecun_uniform(tensor):
    fan_in = tensor.shape[0]  # Assuming the first dimension is fan_in
    limit = math.sqrt(3 / fan_in)
    return mx.random.uniform(low=-limit, high=limit, shape=tensor.shape)


def concat_init(tensor, inits):
    length, fan_out = tensor.shape
    fan_in = length // len(inits)

    chunks = []
    for init in inits:
        chunk = init((fan_in, fan_out))
        chunks.append(chunk)
    
    return mx.concatenate(chunks, axis=0)


def sequence_nll_loss_bits(input, target, *args, **kwargs):
    _, _, n_classes = input.shape
    input_flat = input.reshape(-1, n_classes)
    target_flat = target.reshape(-1)
    
    # MLX doesn't have a direct equivalent to nn.functional.nll_loss
    # We'll implement it manually
    log_probs = mx.log_softmax(input_flat, axis=-1)
    nll_loss = -log_probs[mx.arange(target_flat.size), target_flat].mean()
    
    return nll_loss * math.log(math.e, 2)