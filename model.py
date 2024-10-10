import mlx.core as mx
import mlx.nn as nn
import nn as custom_nn
import utils

import numpy as np


class SampleRNN(nn.Module):

    def __init__(self, frame_sizes, n_rnn, dim, learn_h0, q_levels,
                 weight_norm):
        super().__init__()

        self.dim = dim
        self.q_levels = q_levels

        ns_frame_samples = map(int, np.cumprod(frame_sizes))
        self.frame_level_rnns = [
            FrameLevelRNN(
                frame_size, n_frame_samples, n_rnn, dim, learn_h0, weight_norm
            )
            for (frame_size, n_frame_samples) in zip(
                frame_sizes, ns_frame_samples
            )
        ]

        self.sample_level_mlp = SampleLevelMLP(
            frame_sizes[0], dim, q_levels, weight_norm
        )

    @property
    def lookback(self):
        return self.frame_level_rnns[-1].n_frame_samples


class FrameLevelRNN(nn.Module):

    def __init__(self, frame_size, n_frame_samples, n_rnn, dim,
                 learn_h0, weight_norm):
        super().__init__()

        self.frame_size = frame_size
        self.n_frame_samples = n_frame_samples
        self.dim = dim

        h0 = mx.zeros((n_rnn, dim))
        if learn_h0:
            self.h0 = mx.array(h0)
        else:
            self.h0 = h0

        self.input_expand = nn.Conv1d(
            in_channels=n_frame_samples,
            out_channels=dim,
            kernel_size=1
        )
        self.input_expand.weight = mx.random.normal(0, 0.01, self.input_expand.weight.shape)
        self.input_expand.bias = mx.zeros(self.input_expand.bias.shape)

        self.rnn = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=n_rnn,
        )

        self.upsampling = custom_nn.LearnedUpsampling1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=frame_size
        )

    def __call__(self, prev_samples, upper_tier_conditioning, hidden):
        batch_size, _, _ = prev_samples.shape

        input = self.input_expand(mx.transpose(prev_samples, (0, 2, 1)))
        input = mx.transpose(input, (0, 2, 1))
        if upper_tier_conditioning is not None:
            input += upper_tier_conditioning

        reset = hidden is None

        if hidden is None:
            n_rnn, _ = self.h0.shape
            hidden = mx.broadcast_to(self.h0[None, :, :], (batch_size, n_rnn, self.dim))

        output, hidden = self.rnn(input, hidden)

        output = self.upsampling(mx.transpose(output, (0, 2, 1)))
        output = mx.transpose(output, (0, 2, 1))
        return output, hidden


class SampleLevelMLP(nn.Module):

    def __init__(self, frame_size, dim, q_levels, weight_norm):
        super().__init__()

        self.q_levels = q_levels

        self.embedding = nn.Embedding(
            self.q_levels,
            self.q_levels
        )

        self.input = nn.Conv1d(
            in_channels=q_levels,
            out_channels=dim,
            kernel_size=frame_size,
            bias=False
        )
        self.input.weight = mx.random.normal(0, 0.01, self.input.weight.shape)

        self.hidden = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1
        )
        self.hidden.weight = mx.random.normal(0, 0.01, self.hidden.weight.shape)
        self.hidden.bias = mx.zeros(self.hidden.bias.shape)

        self.output = nn.Conv1d(
            in_channels=dim,
            out_channels=q_levels,
            kernel_size=1
        )
        self.output.weight = custom_nn.lecun_uniform(self.output.weight.shape)
        self.output.bias = mx.zeros(self.output.bias.shape)

    def __call__(self, prev_samples, upper_tier_conditioning):
        batch_size, _, _ = upper_tier_conditioning.shape

        prev_samples = self.embedding(prev_samples.reshape(-1)).reshape(
            batch_size, -1, self.q_levels
        )

        prev_samples = mx.transpose(prev_samples, (0, 2, 1))
        upper_tier_conditioning = mx.transpose(upper_tier_conditioning, (0, 2, 1))

        x = mx.maximum(self.input(prev_samples) + upper_tier_conditioning, 0)
        x = mx.maximum(self.hidden(x), 0)
        x = self.output(x)
        x = mx.transpose(x, (0, 2, 1))

        return mx.log_softmax(x.reshape(-1, self.q_levels)).reshape(batch_size, -1, self.q_levels)


class Runner:

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.reset_hidden_states()

    def reset_hidden_states(self):
        self.hidden_states = {rnn: None for rnn in self.model.frame_level_rnns}

    def run_rnn(self, rnn, prev_samples, upper_tier_conditioning):
        output, new_hidden = rnn(
            prev_samples, upper_tier_conditioning, self.hidden_states[rnn]
        )
        self.hidden_states[rnn] = new_hidden
        return output


class Predictor(Runner):

    def __init__(self, model):
        super().__init__(model)

    def __call__(self, input_sequences, reset):
        if reset:
            self.reset_hidden_states()

        batch_size, _ = input_sequences.shape

        upper_tier_conditioning = None
        for rnn in reversed(self.model.frame_level_rnns):
            from_index = self.model.lookback - rnn.n_frame_samples
            to_index = -rnn.n_frame_samples + 1
            prev_samples = 2 * utils.linear_dequantize(
                input_sequences[:, from_index : to_index],
                self.model.q_levels
            )
            prev_samples = prev_samples.reshape(
                batch_size, -1, rnn.n_frame_samples
            )

            upper_tier_conditioning = self.run_rnn(
                rnn, prev_samples, upper_tier_conditioning
            )

        bottom_frame_size = self.model.frame_level_rnns[0].frame_size
        mlp_input_sequences = input_sequences \
            [:, self.model.lookback - bottom_frame_size :]

        return self.model.sample_level_mlp(
            mlp_input_sequences, upper_tier_conditioning
        )


class Generator(Runner):

    def __init__(self, model):
        super().__init__(model)

    def __call__(self, n_seqs, seq_len):
        self.reset_hidden_states()

        bottom_frame_size = self.model.frame_level_rnns[0].n_frame_samples
        sequences = mx.full((n_seqs, self.model.lookback + seq_len), utils.q_zero(self.model.q_levels))
        frame_level_outputs = [None for _ in self.model.frame_level_rnns]

        for i in range(self.model.lookback, self.model.lookback + seq_len):
            for (tier_index, rnn) in \
                    reversed(list(enumerate(self.model.frame_level_rnns))):
                if i % rnn.n_frame_samples != 0:
                    continue

                prev_samples = 2 * utils.linear_dequantize(
                    sequences[:, i - rnn.n_frame_samples : i],
                    self.model.q_levels
                )[:, None, :]

                if tier_index == len(self.model.frame_level_rnns) - 1:
                    upper_tier_conditioning = None
                else:
                    frame_index = (i // rnn.n_frame_samples) % \
                        self.model.frame_level_rnns[tier_index + 1].frame_size
                    upper_tier_conditioning = \
                        frame_level_outputs[tier_index + 1][:, frame_index, :][:, None, :]

                frame_level_outputs[tier_index] = self.run_rnn(
                    rnn, prev_samples, upper_tier_conditioning
                )

            prev_samples = sequences[:, i - bottom_frame_size : i]
            upper_tier_conditioning = \
                frame_level_outputs[0][:, i % bottom_frame_size, :][:, None, :]
            sample_dist = mx.exp(self.model.sample_level_mlp(
                prev_samples, upper_tier_conditioning
            ).squeeze(1))
            sequences[:, i] = mx.random.categorical(sample_dist)

        return sequences[:, self.model.lookback :]