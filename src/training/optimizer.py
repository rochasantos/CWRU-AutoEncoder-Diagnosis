import torch

class AdaptiveMomentumOptimizer:
    def __init__(self, model, initial_lr=1e-3, alpha=1.05, beta=0.9, min_lr=1e-6, max_lr=1.0):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.device = next(model.parameters()).device

        self.rho = {}
        self.prev_grads = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.rho[name] = torch.full_like(param.data, initial_lr, device=self.device)
                self.prev_grads[name] = torch.zeros_like(param.data, device=self.device)

    def step(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue

                g = param.grad
                g_prev = self.prev_grads[name]

                # Atualização adaptativa
                same_sign = torch.sign(g) * torch.sign(g_prev) > 0
                rho_update = torch.where(same_sign, self.rho[name] * self.alpha, self.rho[name] / self.alpha)
                self.rho[name] = torch.clamp(rho_update, min=self.min_lr, max=self.max_lr)

                # Atualização com momento
                update = self.rho[name] * ((1 - self.beta) * g + self.beta * g_prev)
                param.data -= update

                self.prev_grads[name] = g.clone().detach()

    def zero_grad(self):
        self.model.zero_grad()

    def state_dict(self):
        return {
            'rho': {k: v.clone().detach().cpu() for k, v in self.rho.items()},
            'prev_grads': {k: v.clone().detach().cpu() for k, v in self.prev_grads.items()},
            'alpha': self.alpha,
            'beta': self.beta,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr
        }

    def load_state_dict(self, state_dict):
        self.alpha = state_dict['alpha']
        self.beta = state_dict['beta']
        self.min_lr = state_dict.get('min_lr', 1e-6)
        self.max_lr = state_dict.get('max_lr', 1.0)

        for k in state_dict['rho']:
            self.rho[k] = state_dict['rho'][k].to(self.device)
        for k in state_dict['prev_grads']:
            self.prev_grads[k] = state_dict['prev_grads'][k].to(self.device)
