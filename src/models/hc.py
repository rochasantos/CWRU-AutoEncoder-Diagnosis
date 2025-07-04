import torch
import torch.nn.functional as F

class HierarchicalClassifier(torch.nn.Module):
    def __init__(self, clf_nf, clf_bi, clf_bo, clf_io):
        """
        :param clf_nf: binary CNN that classifies normal vs faulty
        :param clf_bi: binary CNN that classifies ball (B) vs inner race (I)
        :param clf_bo: binary CNN that classifies ball (B) vs outer race (O)
        :param clf_io: binary CNN that classifies inner race (I) vs outer race (O)
        """
        super().__init__()
        self.clf_nf = clf_nf
        self.clf_bi = clf_bi
        self.clf_bo = clf_bo
        self.clf_io = clf_io

    def forward(self, x):
        # Step 1: Normal vs Faulty
        prob_nf = F.softmax(self.clf_nf(x), dim=1)
        normal_prob = prob_nf[:, 0]
        faulty_prob = prob_nf[:, 1]

        # Initialize output list with zeros: [B, I, O]
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, 3, device=x.device)

        # If faulty, apply remaining classifiers and sum scores
        # clf_io: I vs O -> [I, O]
        prob_io = F.softmax(self.clf_io(x), dim=1)
        scores[:, 1] += prob_io[:, 0]  # I
        scores[:, 2] += prob_io[:, 1]  # O

        # clf_bi: B vs I -> [B, I]
        prob_bi = F.softmax(self.clf_bi(x), dim=1)
        scores[:, 0] += prob_bi[:, 0]  # B
        scores[:, 1] += prob_bi[:, 1]  # I

        # clf_bo: B vs O -> [B, O]
        prob_bo = F.softmax(self.clf_bo(x), dim=1)
        scores[:, 0] += prob_bo[:, 0]  # B
        scores[:, 2] += prob_bo[:, 1]  # O

        # Final output: if normal_prob > faulty_prob, classify as N
        final_preds = []
        for i in range(batch_size):
            if normal_prob[i] > faulty_prob[i]:
                final_preds.append("N")
            else:
                max_idx = torch.argmax(scores[i]).item()
                final_preds.append(["B", "I", "O"][max_idx])

        return final_preds, scores
