import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1(nn.Module):
    def __init__(self, input_channels=1, input_length=4096, num_classes=2):        
        super(CNN1, self).__init__()
        self.input_channels = input_channels
        self.input_length = input_length

        # self.ln = nn.LayerNorm(input_length)

        self.conv1 = nn.Conv1d(input_channels, out_channels=16, kernel_size=8, padding=1)
        self.pool1 = nn.MaxPool1d(8)
        self.conv2 = nn.Conv1d(16, out_channels=32, kernel_size=32, padding=1)
        self.pool2 = nn.MaxPool1d(8)

        # Dynamically compute flatten size
        self.flatten_size = self._get_flatten_size()

        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _get_flatten_size(self):
        """
        Pass a dummy input through conv and pool layers to calculate the flatten size.
        """
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, self.input_length)
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            flatten_size = x.view(1, -1).shape[1]
        return flatten_size

    def forward(self, x):
        x = self.ln(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.sigmoid(x)


class HierarchicalClassifier(nn.Module):
    def __init__(self, clf_n_bio, clf_b_i, clf_b_o, clf_i_o, device='cuda'):
        super(HierarchicalClassifier, self).__init__()
        
        # Classificadores
        self.clf_n_bio = clf_n_bio.to(device)  # N/BIO
        self.clf_b_i = clf_b_i.to(device)      # B/I
        self.clf_b_o = clf_b_o.to(device)      # B/O
        self.clf_i_o = clf_i_o.to(device)      # I/O
        
        self.device = device

    def forward(self, x):
        """
        x: Tensor [batch_size, channels, input_length]
        Retorna: lista de rótulos ['N', 'B', 'I', 'O']
        """
        x = x.to(self.device)
        out_n_bio = self.clf_n_bio(x)
        pred_n_bio = torch.argmax(out_n_bio, dim=1)  # 0: N, 1: BIO

        predictions = []

        for idx, p in enumerate(pred_n_bio):
            sample = x[idx].unsqueeze(0)

            if p.item() == 0:
                predictions.append('N')
            else:
                out_b_i = self.clf_b_i(sample)
                pred_b_i = torch.argmax(out_b_i, dim=1)  # 0: B, 1: I

                if pred_b_i.item() == 0:
                    out_b_o = self.clf_b_o(sample)
                    pred_b_o = torch.argmax(out_b_o, dim=1)  # 0: B, 1: O

                    if pred_b_o.item() == 0:
                        predictions.append('B')
                    else:
                        out_i_o = self.clf_i_o(sample)
                        pred_i_o = torch.argmax(out_i_o, dim=1)  # 0: I, 1: O

                        if pred_i_o.item() == 0:
                            predictions.append('I')
                        else:
                            predictions.append('O')
                else:
                    out_b_o = self.clf_b_o(sample)
                    pred_b_o = torch.argmax(out_b_o, dim=1)  # 0: B, 1: O

                    if pred_b_o.item() == 0:
                        predictions.append('B')
                    else:
                        out_i_o = self.clf_i_o(sample)
                        pred_i_o = torch.argmax(out_i_o, dim=1)  # 0: I, 1: O

                        if pred_i_o.item() == 0:
                            predictions.append('I')
                        else:
                            predictions.append('O')

        return predictions