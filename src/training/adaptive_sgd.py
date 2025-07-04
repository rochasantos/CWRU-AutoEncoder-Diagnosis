import torch
from torch.optim.optimizer import Optimizer, required

class AdaptiveSGD(Optimizer):
    def __init__(self, params, lr=0.01, a=1.1, b=0.9, beta=0.9):
        """
        Implements the ADCNN adaptive learning rate with momentum.

        Args:
            params: model.parameters()
            lr (float): initial learning rate
            a (float): growth factor when gradients align (suggested > 1)
            b (float): decay factor when gradients change direction (suggested < 1)
            beta (float): momentum factor (0 < beta < 1)
        """
        defaults = dict(lr=lr, a=a, b=b, beta=beta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            a = group["a"]
            b = group["b"]
            beta = group["beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]
                
                # Inicializa buffers se necessário
                if "prev_grad" not in state:
                    state["prev_grad"] = torch.zeros_like(p.data)
                    state["lr_tensor"] = torch.full_like(p.data, lr)

                prev_grad = state["prev_grad"]
                lr_tensor = state["lr_tensor"]

                # Determina se o gradiente mudou de direção (sinal)
                same_dir = grad * prev_grad > 0
                opp_dir = ~same_dir

                # Atualiza taxas de aprendizado elemento a elemento
                lr_tensor[same_dir] = lr_tensor[same_dir] * a
                lr_tensor[opp_dir] = lr_tensor[opp_dir] * b

                # Aplica atualização com momentum
                update = -lr_tensor * ((1 - beta) * grad + beta * prev_grad)
                p.data.add_(update)

                # Salva gradiente para próxima iteração
                state["prev_grad"] = grad.clone()
                state["lr_tensor"] = lr_tensor
