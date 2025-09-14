import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalClassifier(nn.Module):
    """
    Combina dois classificadores:
      - binary_clf: logits para [N, F] (shape: [N, 2])
      - multi_clf : logits para [B, I, O] (shape: [N, 3])

    Forward retorna log-probabilidades para 4 classes na ordem [N, B, I, O],
    compatíveis com nn.NLLLoss (target: 0=N, 1=B, 2=I, 3=O).
    """
    def __init__(self, binary_clf: nn.Module, multi_clf: nn.Module, device="cpu"):
        super().__init__()
        self.binary_clf = binary_clf.to(device)
        self.multi_clf  = multi_clf.to(device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)

        # Logits dos dois classificadores
        bin_logits   = self.binary_clf(x)         # [N, 2] => [N, (N,F)]
        multi_logits = self.multi_clf(x)          # [N, 3] => [N, (B,I,O)]

        # Convertemos para log-probabilidades (log-softmax)
        bin_logp   = F.log_softmax(bin_logits,   dim=1)  # [N,2]
        multi_logp = F.log_softmax(multi_logits, dim=1)  # [N,3]

        # Combinação hierárquica em log-espaco
        logP_N = bin_logp[:, 0]                                  # P(N)
        logP_B = bin_logp[:, 1] + multi_logp[:, 0]               # P(F)*P(B|F)
        logP_I = bin_logp[:, 1] + multi_logp[:, 1]               # P(F)*P(I|F)
        logP_O = bin_logp[:, 1] + multi_logp[:, 2]               # P(F)*P(O|F)

        # Empilha em [N, B, I, O] => shape [N,4]
        logP_4 = torch.stack([logP_N, logP_B, logP_I, logP_O], dim=1)
        return logP_4  # use NLLLoss

    @torch.no_grad()
    def predict(self, x):
        logp = self.forward(x)
        preds = torch.argmax(logp, dim=1)  # 0=N,1=B,2=I,3=O
        idx2label = {0: "N", 1: "B", 2: "I", 3: "O"}
        return [idx2label[i.item()] for i in preds]
