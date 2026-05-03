"""
SNA-STP CIFAR-10 ResNet Bottleneck Experiment (Table C1)
=========================================================
Frozen ResNet-18 backbone + k-dim discrete bottleneck + classifier.
Multi-seed: k=12, k=16, 3 seeds. Reports mean ± std accuracy + extraction time.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from snap_lib.core.structure_matrix import StructureMatrixBuilder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ===== Config =====
SEEDS = [42, 123, 999]
K_VALUES = [12, 16]
BATCH_SIZE = 128
EPOCHS_BACKBONE = 80
EPOCHS_BOTTLENECK = 40
LR_BACKBONE = 0.1
LR_BOTTLENECK = 1e-3
QAT_LAMBDA_START = 0.1
QAT_LAMBDA_END = 0.5
NUM_CLASSES = 10
BACKBONE_DIM = 512

# ===== Data =====
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


def make_cifar_resnet18():
    """Standard ResNet-18 adapted for CIFAR-10 (3x3 conv, no maxpool)."""
    model = torchvision.models.resnet18(num_classes=NUM_CLASSES)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class BottleneckHead(nn.Module):
    """k-dim bottleneck + classifier on top of frozen backbone features."""
    def __init__(self, k=12, backbone_dim=BACKBONE_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.bottleneck = nn.Linear(backbone_dim, k)
        self.classifier = nn.Linear(k, num_classes)

    def forward(self, feat, hard=False):
        z = self.bottleneck(feat)
        z_gate = torch.sigmoid(z)
        if hard:
            z_hard = (z_gate > 0.5).float() + z_gate - z_gate.detach()
            out = self.classifier(z_hard)
        else:
            out = self.classifier(z_gate)
        return out, z_gate


class FullModel(nn.Module):
    """Backbone (frozen) + BottleneckHead."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, hard=False):
        with torch.no_grad():
            feat = self.backbone(x).flatten(1)
        return self.head(feat, hard=hard)


def train_backbone():
    print("Training CIFAR-10 ResNet-18 backbone from scratch (80 epochs)...")
    model = make_cifar_resnet18().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR_BACKBONE, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_BACKBONE)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for epoch in range(EPOCHS_BACKBONE):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, pred = outputs.max(1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    baseline_acc = 100.0 * correct / total
    print(f"Baseline accuracy: {baseline_acc:.2f}%")

    # Remove FC to get feature extractor
    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone_state = backbone.state_dict()
    torch.save(backbone_state, 'cifar_resnet18_backbone.pth')
    return backbone_state, baseline_acc


def train_bottleneck(backbone_state, k, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone = nn.Sequential(*list(make_cifar_resnet18().children())[:-1])
    backbone.load_state_dict(backbone_state)
    for p in backbone.parameters():
        p.requires_grad = False

    head = BottleneckHead(k=k).to(DEVICE)
    model = FullModel(backbone.to(DEVICE), head).to(DEVICE)

    optimizer = optim.Adam(head.parameters(), lr=LR_BOTTLENECK)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_BOTTLENECK)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for epoch in range(EPOCHS_BOTTLENECK):
        qat_lambda = QAT_LAMBDA_START + (QAT_LAMBDA_END - QAT_LAMBDA_START) * (epoch / EPOCHS_BOTTLENECK)
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, z_gate = model(inputs)
            ce_loss = criterion(outputs, targets)
            qat_loss = qat_lambda * torch.mean(z_gate * (1 - z_gate))
            loss = ce_loss + 0.1 * qat_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate with hard binarization
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, _ = model(inputs, hard=True)
            _, pred = outputs.max(1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    acc = 100.0 * correct / total
    return acc, head


def extract_sna_stp(head, k):
    builder = StructureMatrixBuilder()
    t0 = time.time()
    W_cls = head.classifier.weight.detach().cpu().numpy()
    b_cls = head.classifier.bias.detach().cpu().numpy()
    M_local = builder.build_layer_matrix(W_cls, b_cls, activation='sigmoid')
    elapsed = time.time() - t0
    return M_local, elapsed


# ===== Main =====
if __name__ == '__main__':
    # Phase 1: Train backbone
    print("=" * 70)
    print("Phase 1: Train CIFAR-10 ResNet-18 backbone")
    if os.path.exists('cifar_resnet18_backbone.pth'):
        print("Loading saved backbone...")
        backbone_state = torch.load('cifar_resnet18_backbone.pth', map_location='cpu')
        # Quick eval of baseline
        backbone = nn.Sequential(*list(make_cifar_resnet18().children())[:-1])
        backbone.load_state_dict(backbone_state)
        temp_model = nn.Sequential(backbone, nn.Flatten(1), nn.Linear(512, 10)).to(DEVICE)
        temp_model.eval()
        correct = 0
        total = 0
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = temp_model(inputs)
                _, pred = outputs.max(1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
        baseline_acc = 100.0 * correct / total
        print(f"Baseline accuracy: {baseline_acc:.2f}%")
    else:
        backbone_state, baseline_acc = train_backbone()

    # Phase 2: Train bottleneck variants
    print("\n" + "=" * 70)
    print("Phase 2: Train bottleneck models (k=12, 16, 3 seeds each)")
    results = {}
    for k in K_VALUES:
        accs = []
        times = []
        print(f"\n--- k = {k} ---")
        for seed in SEEDS:
            print(f"  Seed {seed}: training...", end=' ', flush=True)
            acc, head = train_bottleneck(backbone_state, k, seed)
            M_local, ext_time = extract_sna_stp(head, k)
            accs.append(acc)
            times.append(ext_time)
            print(f"Acc={acc:.2f}% Extract={ext_time:.4f}s M_shape={M_local.shape}")
        results[k] = {'accs': accs, 'times': times}

    # Phase 3: Report
    print("\n" + "=" * 70)
    print("RESULTS: mean ± std (3 seeds)")
    print("=" * 70)
    for k in K_VALUES:
        a_mean = np.mean(results[k]['accs'])
        a_std = np.std(results[k]['accs'], ddof=1)
        t_mean = np.mean(results[k]['times'])
        t_std = np.std(results[k]['times'], ddof=1)
        print(f"k={k}: Acc = {a_mean:.2f}% ± {a_std:.2f}  |  Extract = {t_mean:.4f}s ± {t_std:.4f}")

    print(f"\nBaseline (no bottleneck): {baseline_acc:.2f}%")
    print("\n=== For Table C1 (ResNet-18 rows) ===")
    for k in K_VALUES:
        a_mean = np.mean(results[k]['accs'])
        a_std = np.std(results[k]['accs'], ddof=1)
        t_mean = np.mean(results[k]['times'])
        print(f"SNA-STP Bottleneck & {k} & {2**k:,} & ResNet-18 & {a_mean:.2f}\\% ± {a_std:.2f} & \\textbf{{{t_mean:.3f}s}} \\\\")
    print("\nDone.")
