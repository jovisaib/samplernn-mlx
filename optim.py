import mlx.core as mx


def gradient_clipping(optimizer, min=-1, max=1):

    class OptimizerWrapper(object):

        def step(self, loss_and_grad_fn):
            def wrapped_loss_and_grad_fn(*args, **kwargs):
                loss, grads = loss_and_grad_fn(*args, **kwargs)
                for param, grad in grads.items():
                    grads[param] = mx.clip(grad, min, max)
                return loss, grads
            
            return optimizer.step(wrapped_loss_and_grad_fn)

        def __getattr__(self, attr):
            return getattr(optimizer, attr)

    return OptimizerWrapper()