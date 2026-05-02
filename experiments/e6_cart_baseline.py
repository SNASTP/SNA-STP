"""CART distillation baseline for Breast Cancer BNN — 3 seeds."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn
import warnings; warnings.filterwarnings('ignore')

data = load_breast_cancer()
X_raw = data.data[:, :10]
y_raw = data.target
X_bin = np.zeros_like(X_raw, dtype=np.float32)
for j in range(10):
    t = np.median(X_raw[:, j])
    X_bin[:, j] = (X_raw[:, j] > t).astype(np.float32)

class BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 12); self.fc2 = nn.Linear(12, 8); self.fc3 = nn.Linear(8, 1)
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(m.weight, 0, 0.5); nn.init.zeros_(m.bias)
        self._binarize()
    def _binarize(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            m.weight.data = torch.sign(m.weight.data)
    def forward(self, x):
        x = (torch.sign(self.fc1(x)+self.fc1.bias)+1)/2
        x = (torch.sign(self.fc2(x)+self.fc2.bias)+1)/2
        x = torch.sigmoid(self.fc3(x)+self.fc3.bias)
        return x

all_inputs = torch.tensor([[(k>>j)&1 for j in range(9,-1,-1)] for k in range(1024)], dtype=torch.float32)

for seed in [42, 123, 456]:
    np.random.seed(seed); torch.manual_seed(seed)
    X_t = torch.tensor(X_bin); y_t = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)
    X_tr, X_te, y_tr, y_te = train_test_split(X_t, y_t, test_size=0.2, random_state=seed)
    model = BNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(500):
        opt.zero_grad()
        loss = nn.BCELoss()(model(X_tr), y_tr)
        loss.backward(); opt.step(); model._binarize()
    model.eval()
    with torch.no_grad():
        te_acc = ((model(X_te)>0.5).float()==y_te).float().mean().item()
        bnn_out = (model(all_inputs)>0.5).int().squeeze().numpy()
    for depth in [3, 5, 7, None]:
        cart = DecisionTreeClassifier(max_depth=depth, random_state=42)
        cart.fit(all_inputs.int().numpy(), bnn_out)
        fid = (cart.predict(all_inputs.int().numpy())==bnn_out).mean()
        n = cart.tree_.node_count
        d_label = str(depth) if depth else 'unlim'
        print(f"seed={seed} BNN_acc={te_acc*100:.1f}% CART_d={d_label:>5} fid={fid*100:.1f}% nodes={n}")
