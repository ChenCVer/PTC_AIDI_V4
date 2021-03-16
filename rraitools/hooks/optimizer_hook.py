from torch.nn.utils import clip_grad

from .hook import Hook


class OptimizerHook(Hook):

    def __init__(self, grad_clip=None, gradient_cumulative_num=1):
        self.grad_clip = grad_clip  # 梯度裁剪
        self.gradient_cumulative_num = gradient_cumulative_num  # 累计梯度次数

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip)

    def after_train_iter(self, runner):
        runner.batch_iter_output['batch_loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.modeler.parameters())
        if (runner.inner_iter + 1) % self.gradient_cumulative_num == 0:
            runner.optimizer.step()
            runner.optimizer.zero_grad()
