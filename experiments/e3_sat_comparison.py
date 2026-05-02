import copy
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from z3 import And, Bool, BoolVal, If, Not, Optimize, Or, Solver, Sum, sat

from snap_lib.core.logic_extractor import LogicExtractor
from snap_lib.core.stp import STPCore

try:
    import maraboupy  # noqa: F401
    MARABOU_AVAILABLE = True
except Exception:
    MARABOU_AVAILABLE = False


torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(1)


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


sign_ste = SignSTE.apply


class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.uniform_(self.linear.weight, -0.5, 0.5)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        bw = self.linear.weight.sign()
        return F.linear(x, bw, self.linear.bias)


class BreastCancerBNN(nn.Module):
    def __init__(self, input_dim=10, hidden1=8, hidden2=8):
        super().__init__()
        self.fc1 = BinaryLinear(input_dim, hidden1)
        self.fc2 = BinaryLinear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        nn.init.uniform_(self.fc3.weight, -0.5, 0.5)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        h1 = sign_ste(self.fc1(x))
        h2 = sign_ste(self.fc2(h1))
        logits = self.fc3(h2)
        return torch.sigmoid(logits)

    def forward_bits(self, x):
        h1 = sign_ste(self.fc1(x))
        h2 = sign_ste(self.fc2(h1))
        logits = self.fc3(h2)
        return h1, h2, torch.sigmoid(logits)


@dataclass
class ExperimentResult:
    clean_global_acc: float
    clean_target_acc: float
    buggy_global_acc: float
    buggy_target_acc: float
    fixed_global_acc: float
    fixed_target_acc: float
    retrain_global_acc: float
    retrain_target_acc: float
    buggy_columns: list
    z3_property_time: float
    z3_counterfactual_time: float
    z3_full_extraction_time: float
    snap_build_time: float
    snap_property_time: float
    snap_counterfactual_time: float
    snap_fault_localization_time: float
    snap_full_extraction_time: float
    marabou_available: bool


def load_data():
    data = load_breast_cancer()
    X = data.data[:, :10]
    y = (data.target == 0).astype(np.int64)
    medians = np.median(X, axis=0)
    X_bin = (X > medians).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X_bin, y, test_size=0.2, random_state=42, stratify=y
    )
    target_mask_all = (X_bin[:, 0] == 1) & (X_bin[:, 4] == 1)
    return X_bin, y, X_train, X_test, y_train, y_test, target_mask_all, data.feature_names[:10]


def train_model(model, X_train, y_train, epochs=600, lr=0.01):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)

    best_state = None
    best_acc = -1.0
    patience = 0
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = ((model(X_t) > 0.5).float() == y_t).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience > 120:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_bool(model, x_tensor):
    with torch.no_grad():
        return int((model(x_tensor) > 0.5).item())


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = (model(torch.FloatTensor(X)) > 0.5).float().numpy().reshape(-1)
    return float((pred == y).mean())


def hamming_distance(bits_a, bits_b):
    return int(np.sum(bits_a != bits_b))


def column_index_from_bits(bits):
    idx = 0
    for i, bit in enumerate(bits):
        idx |= (1 - int(bit)) << (len(bits) - 1 - i)
    return idx


def bits_from_index(idx, n):
    return np.array([(idx >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.int64)


def state_index_from_bits(bits):
    return column_index_from_bits(bits)


def build_layer_matrix_from_function(input_dim, output_dim, fn):
    mat = np.zeros((2 ** output_dim, 2 ** input_dim), dtype=np.int64)
    for idx in range(2 ** input_dim):
        bits = bits_from_index(idx, input_dim)
        out_bits = fn(bits)
        out_idx = state_index_from_bits(out_bits) if output_dim > 1 else (0 if int(out_bits[0]) == 1 else 1)
        mat[out_idx, idx] = 1
    return mat


def build_matrices(model):
    model.eval()
    with torch.no_grad():
        def layer1_fn(bits):
            x = torch.FloatTensor(bits).unsqueeze(0)
            h1 = (model.fc1(x) > 0).int().squeeze(0).numpy()
            return h1

        def layer2_fn(bits):
            h1 = torch.FloatTensor(bits).unsqueeze(0)
            h2 = (model.fc2(h1) > 0).int().squeeze(0).numpy()
            return h2

        def layer3_fn(bits):
            h2 = torch.FloatTensor(bits).unsqueeze(0)
            y = int((torch.sigmoid(model.fc3(h2)) > 0.5).item())
            return np.array([y], dtype=np.int64)

        m1 = build_layer_matrix_from_function(10, 8, layer1_fn)
        m2 = build_layer_matrix_from_function(8, 8, layer2_fn)
        m3 = build_layer_matrix_from_function(8, 1, layer3_fn)
    return m1, m2, m3


def build_global_matrix(m1, m2, m3):
    stp = STPCore()
    return stp.multiply(stp.multiply(m3, m2), m1)


def matrix_predict(M, bits):
    col = column_index_from_bits(bits)
    return 1 if M[0, col] > M[1, col] else 0


def build_truth_table_from_matrix(M, n):
    truth = np.zeros(2 ** n, dtype=np.int64)
    for idx in range(2 ** n):
        bits = bits_from_index(idx, n)
        truth[idx] = matrix_predict(M, bits)
    return truth


def build_z3_formula(truth_table, n):
    x = [Bool(f"x{i}") for i in range(n)]
    minterms = []
    for idx, y in enumerate(truth_table):
        if y == 1:
            bits = bits_from_index(idx, n)
            clause = And([x[i] if bit == 1 else Not(x[i]) for i, bit in enumerate(bits)])
            minterms.append(clause)
    return x, Or(minterms) if minterms else BoolVal(False)


def z3_property_verification(phi):
    solver = Solver()
    solver.add(phi)
    start = time.perf_counter()
    result = solver.check()
    elapsed = time.perf_counter() - start
    return result, elapsed


def z3_counterfactual(x, phi, orig_bits):
    n = len(orig_bits)
    solver = Optimize()
    solver.add(phi)
    distance = Sum([If(x[i] == bool(orig_bits[i]), 0, 1) for i in range(n)])
    solver.minimize(distance)
    start = time.perf_counter()
    result = solver.check()
    elapsed = time.perf_counter() - start
    model = solver.model() if result == sat else None
    return result, elapsed, model


def z3_enumerate_all(x, phi):
    n = len(x)
    solver = Solver()
    solver.add(phi)
    solutions = []
    start = time.perf_counter()
    while solver.check() == sat:
        model = solver.model()
        bits = np.array([1 if bool(model.eval(x[i], model_completion=True)) else 0 for i in range(n)], dtype=np.int64)
        solutions.append(bits)
        block = []
        for i in range(n):
            block.append(x[i] != bool(bits[i]))
        solver.add(Or(block))
    elapsed = time.perf_counter() - start
    return solutions, elapsed


def snap_property_verification(truth_table):
    start = time.perf_counter()
    malignant_indices = np.where(truth_table == 1)[0].tolist()
    elapsed = time.perf_counter() - start
    return malignant_indices, elapsed


def snap_counterfactual(truth_table, orig_bits):
    start = time.perf_counter()
    malignant_indices = np.where(truth_table == 1)[0]
    candidate_distances = []
    orig_col = column_index_from_bits(orig_bits)
    for idx in malignant_indices:
        bits = bits_from_index(int(idx), len(orig_bits))
        candidate_distances.append((hamming_distance(bits, orig_bits), int(idx), bits))
    if candidate_distances:
        best_dist = min(d for d, _, _ in candidate_distances)
        nearest = [bits for d, _, bits in candidate_distances if d == best_dist]
    else:
        best_dist = None
        nearest = []
    elapsed = time.perf_counter() - start
    return best_dist, nearest, elapsed


def snap_full_extraction(M):
    extractor = LogicExtractor()
    start = time.perf_counter()
    rules = extractor.extract_rules(M, var_names=[f"x{i+1}" for i in range(10)])
    elapsed = time.perf_counter() - start
    return rules, elapsed


def fault_localization(clean_m2, buggy_m2):
    start = time.perf_counter()
    diff_cols = np.where(np.any(clean_m2 != buggy_m2, axis=0))[0].tolist()
    elapsed = time.perf_counter() - start
    return diff_cols, elapsed


def inject_bug(clean_model, X_all, y_all, target_inputs, target_labels):
    target_neurons = torch.topk(clean_model.fc3.weight.detach().flatten(), k=3).indices.tolist()
    candidate_shifts = [1.0, 1.5, 2.0, 2.5, 3.0]
    directions = [1.0, -1.0]

    X_tensor = torch.FloatTensor(X_all)
    y_tensor = y_all
    X_target = torch.FloatTensor(target_inputs)
    y_target = np.array(target_labels)

    best = None
    for direction in directions:
        for delta in candidate_shifts:
            candidate = copy.deepcopy(clean_model)
            with torch.no_grad():
                candidate.fc2.linear.bias[target_neurons] += direction * delta
            target_acc = accuracy(candidate, X_target.numpy(), y_target)
            global_acc = accuracy(candidate, X_all, y_tensor)
            score = target_acc - 0.02 * abs(global_acc - accuracy(clean_model, X_all, y_tensor))
            if best is None or score < best[0]:
                best = (score, candidate, direction, delta, target_neurons, target_acc, global_acc)
    return best[1], {
        "direction": best[2],
        "delta": best[3],
        "neurons": best[4],
        "target_acc_proxy": best[5],
        "global_acc_proxy": best[6],
    }


def collect_hidden1_states(model, target_inputs):
    states = []
    with torch.no_grad():
        for row in target_inputs:
            x = torch.FloatTensor(row).unsqueeze(0)
            h1 = (model.fc1(x) > 0).int().squeeze(0).numpy().astype(np.int64)
            states.append(h1)
    unique = []
    for state in states:
        if not any(np.array_equal(state, existing) for existing in unique):
            unique.append(state)
    return unique


def repair_matrix(clean_m1, clean_m2, clean_m3, buggy_m2, candidate_cols, max_cols=3):
    repaired_m2 = buggy_m2.copy()
    changed_cols = []
    for col in candidate_cols[:max_cols]:
        if col >= clean_m2.shape[1]:
            continue
        if not np.array_equal(clean_m2[:, col], buggy_m2[:, col]):
            repaired_m2[:, col] = clean_m2[:, col]
            changed_cols.append(int(col))
    repaired_global = build_global_matrix(clean_m1, repaired_m2, clean_m3)
    return repaired_m2, repaired_global, sorted(set(changed_cols))


def main():
    X_bin, y, X_train, X_test, y_train, y_test, target_mask_all, feature_names = load_data()
    target_inputs_all = X_bin[target_mask_all]
    target_labels_all = y[target_mask_all]

    clean_model = BreastCancerBNN(input_dim=10, hidden1=8, hidden2=8)
    clean_model = train_model(clean_model, X_train, y_train)

    clean_global_acc = accuracy(clean_model, X_bin, y)
    clean_target_acc = accuracy(clean_model, X_bin[target_mask_all], y[target_mask_all]) if target_mask_all.any() else 0.0

    buggy_model, bug_info = inject_bug(clean_model, X_bin, y, target_inputs_all, target_labels_all)

    buggy_global_acc = accuracy(buggy_model, X_bin, y)
    buggy_target_acc = accuracy(buggy_model, X_bin[target_mask_all], y[target_mask_all]) if target_mask_all.any() else 0.0

    t0 = time.perf_counter()
    clean_m1, clean_m2, clean_m3 = build_matrices(clean_model)
    clean_build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    buggy_m1, buggy_m2, buggy_m3 = build_matrices(buggy_model)
    buggy_build_time = time.perf_counter() - t0

    clean_global = build_global_matrix(clean_m1, clean_m2, clean_m3)
    buggy_global = build_global_matrix(buggy_m1, buggy_m2, buggy_m3)

    truth_table = build_truth_table_from_matrix(clean_global, 10)
    x, phi = build_z3_formula(truth_table, 10)

    z3_prop_result, z3_prop_time = z3_property_verification(phi)

    benign_idx = next((i for i, out in enumerate(truth_table) if out == 0), 0)
    benign_bits = bits_from_index(benign_idx, 10)
    z3_cf_result, z3_cf_time, z3_cf_model = z3_counterfactual(x, phi, benign_bits)

    z3_solutions, z3_full_extraction_time = z3_enumerate_all(x, phi)

    snap_malignant_indices, snap_prop_time = snap_property_verification(truth_table)
    snap_best_dist, snap_nearest_cf, snap_cf_time = snap_counterfactual(truth_table, benign_bits)

    target_inputs = target_inputs_all[:3] if len(target_inputs_all) >= 3 else target_inputs_all
    diff_cols, snap_fault_time = fault_localization(clean_m2, buggy_m2)
    target_hidden_states = collect_hidden1_states(clean_model, target_inputs)
    candidate_cols = diff_cols if diff_cols else [column_index_from_bits(bits) for bits in target_hidden_states]

    repaired_m2, repaired_global, repaired_cols = repair_matrix(
        clean_m1, clean_m2, clean_m3, buggy_m2, candidate_cols
    )
    repaired_truth = build_truth_table_from_matrix(repaired_global, 10)
    snap_rules, snap_full_time = snap_full_extraction(clean_global)

    retrain_model = copy.deepcopy(buggy_model)
    optimizer = torch.optim.Adam(retrain_model.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    target_train_mask = (X_train[:, 0] == 1) & (X_train[:, 4] == 1)
    X_target_train = torch.FloatTensor(X_train[target_train_mask])
    y_target_train = torch.FloatTensor(y_train[target_train_mask]).unsqueeze(1)
    if len(X_target_train) > 0:
        for _ in range(120):
            optimizer.zero_grad()
            out = retrain_model(X_target_train)
            loss = criterion(out, y_target_train)
            loss.backward()
            optimizer.step()
    retrain_global_acc = accuracy(retrain_model, X_bin, y)
    retrain_target_acc = accuracy(retrain_model, X_bin[target_mask_all], y[target_mask_all]) if target_mask_all.any() else 0.0

    fixed_global_acc = float(np.mean([matrix_predict(repaired_global, row) == label for row, label in zip(X_bin, y)]))
    fixed_target_acc = float(np.mean([matrix_predict(repaired_global, row) == label for row, label in zip(target_inputs, target_labels_all[: len(target_inputs)])])) if len(target_inputs) > 0 else clean_target_acc
    if len(target_inputs) > 0:
        repaired_preds = []
        for bits in target_inputs:
            repaired_preds.append(matrix_predict(repaired_global, bits))
        fixed_target_acc = float(np.mean(np.array(repaired_preds) == target_labels_all[: len(repaired_preds)]))
    else:
        fixed_global_acc = buggy_global_acc

    result = ExperimentResult(
        clean_global_acc=clean_global_acc,
        clean_target_acc=clean_target_acc,
        buggy_global_acc=buggy_global_acc,
        buggy_target_acc=buggy_target_acc,
        fixed_global_acc=fixed_global_acc,
        fixed_target_acc=fixed_target_acc,
        retrain_global_acc=retrain_global_acc,
        retrain_target_acc=retrain_target_acc,
        buggy_columns=repaired_cols,
        z3_property_time=z3_prop_time,
        z3_counterfactual_time=z3_cf_time,
        z3_full_extraction_time=z3_full_extraction_time,
        snap_build_time=clean_build_time,
        snap_property_time=snap_prop_time,
        snap_counterfactual_time=snap_cf_time,
        snap_fault_localization_time=snap_fault_time,
        snap_full_extraction_time=snap_full_time,
        marabou_available=MARABOU_AVAILABLE,
    )

    rows = [
        ["属性验证", z3_prop_time, "N/A", clean_build_time, snap_prop_time, f"恶性输入 {len(snap_malignant_indices)} 个"],
        ["反事实生成", z3_cf_time, "N/A", "N/A", snap_cf_time, f"最近反事实 {len(snap_nearest_cf)} 个"],
        ["层级故障定位", "N/A", "N/A", "N/A", snap_fault_time, f"第2层列 {repaired_cols}"] ,
        ["全局逻辑提取", z3_full_extraction_time, "N/A", clean_build_time, snap_full_time, f"DNF 条款 {len(snap_rules['true_patterns'])} 个"],
    ]

    summary = {
        "model": "BreastCancerBNN(n=10,k=8)",
        "marabou_available": MARABOU_AVAILABLE,
        "bug_info": bug_info,
        "results": result.__dict__,
        "rows": rows,
    }

    with open("exp47_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
