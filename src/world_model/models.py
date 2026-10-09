"""Small low-dimensional world models (MLP, GRU), their ensembles, and the baselines they must beat.

Every model is used through a step function  delta_c, state = step(c, h, a, g, state)  on
batched numpy arrays, so rollouts, baselines and the planner share one code path.
"""
import numpy as np
import torch
from torch import nn

from src.world_model.data import transitions

torch.set_num_threads(4)


# ---------- baselines (gated by the gripper flag: the cube only turns while held) ----------

def persistence_step(c, h, a, g, state=None):
    return np.zeros_like(c), state


def follow_step(c, h, a, g, state=None):
    return a * g, state


def fit_gain(seqs):
    X, y = transitions(seqs)
    held = X[:, 3] > 0.5
    a = X[held, 2]
    return float(np.dot(a, y[held]) / max(np.dot(a, a), 1e-12))


def gain_step_fn(k):
    def step(c, h, a, g, state=None):
        return k * a * g, state
    return step


# ---------- learned models ----------

def make_mlp(hidden=64):
    return nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))


class GRUModel(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.cell = nn.GRUCell(4, hidden)
        self.head = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, x, state):
        state = self.cell(x, state)
        return self.head(state).squeeze(-1), state


def norm_stats(seqs):
    X, y = transitions(seqs)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    mu[3], sd[3] = 0.0, 1.0  # leave the gripper flag as 0 / 1
    return dict(mu=mu.astype(np.float32), sd=sd.astype(np.float32), y_sd=np.float32(y.std() + 1e-6))


def train_mlp(seqs, seed, epochs=1500, lr=3e-3):
    torch.manual_seed(seed)
    stats = norm_stats(seqs)
    X, y = transitions(seqs)
    X = torch.tensor((X - stats["mu"]) / stats["sd"], dtype=torch.float32)
    y = torch.tensor(y / stats["y_sd"], dtype=torch.float32)
    net = make_mlp()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(X).squeeze(-1) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return dict(kind="mlp", net=net, stats=stats)


def pad_sequences(seqs):
    T = max(len(s["t"]) for s in seqs)
    def pad(key, fill=0.0):
        return np.stack([np.r_[np.nan_to_num(s[key]), np.full(T - len(s[key]), fill)] for s in seqs]).astype(np.float32)
    obs = np.stack([np.r_[s["obs"], np.zeros(T - len(s["obs"]), bool)] for s in seqs])
    return pad("c"), pad("h"), pad("a"), pad("g"), obs


def train_gru(seqs, seed, epochs=600, lr=3e-3):
    """Trained on free-running rollouts from the window start (the cube rests there, c = 0),
    with the loss on observed steps only, so it learns to carry its own predictions."""
    torch.manual_seed(seed)
    stats = norm_stats(seqs)
    c, h, a, g, obs = (torch.tensor(x) for x in pad_sequences(seqs))
    mu, sd, y_sd = (torch.tensor(stats[k]) for k in ("mu", "sd", "y_sd"))
    net = GRUModel()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    B, T = c.shape
    for _ in range(epochs):
        opt.zero_grad()
        state = torch.zeros(B, net.hidden)
        pred = torch.where(obs[:, 0], c[:, 0], torch.zeros(B))
        loss, n = 0.0, 0
        for k in range(T - 1):
            x = (torch.stack([pred, h[:, k], a[:, k], g[:, k]], -1) - mu) / sd
            delta, state = net(x, state)
            pred = pred + delta * y_sd
            m = obs[:, k + 1]
            if m.any():
                loss = loss + ((pred[m] - c[m, k + 1]) ** 2).sum()
                n += int(m.sum())
        (loss / max(n, 1)).backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return dict(kind="gru", net=net, stats=stats)


def model_step_fn(model):
    net, stats = model["net"], model["stats"]
    mu, sd, y_sd = stats["mu"], stats["sd"], stats["y_sd"]

    def step(c, h, a, g, state=None):
        x = torch.tensor((np.stack([c, h, a, g], -1) - mu) / sd, dtype=torch.float32)
        with torch.no_grad():
            if model["kind"] == "mlp":
                return net(x).squeeze(-1).numpy() * y_sd, None
            if state is None:
                state = torch.zeros(len(c), net.hidden)
            delta, state = net(x, state)
            return delta.numpy() * y_sd, state
    return step


def train_ensemble(kind, seqs, n=5, seed=0):
    train = train_mlp if kind == "mlp" else train_gru
    return [train(seqs, seed + 100 * i) for i in range(n)]


# ---------- rollouts ----------

def rollout(step, seq, starts, h=None, a=None):
    """Predicted cube yaw for every step, one row per start index (B, T). Before its start, a row is
    fed the observed cube yaw where available; from the start on it runs on its own predictions.
    h / a override the hand trajectory (used by the planner)."""
    h = seq["h"] if h is None else h
    a = seq["a"] if a is None else a
    starts = np.asarray(starts)
    B, T = len(starts), len(h)
    c_obs = np.nan_to_num(seq["c"])
    pred = np.zeros((B, T))
    pred[:, 0] = c_obs[0] if seq["obs"][0] else 0.0
    state = None
    for k in range(T - 1):
        use_obs = (k < starts) & seq["obs"][k]
        c_in = np.where(use_obs, c_obs[k], pred[:, k])
        delta, state = step(c_in, np.full(B, h[k]), np.full(B, a[k]), np.full(B, seq["g"][k]), state)
        pred[:, k + 1] = c_in + delta
    return pred


def ensemble_rollout(steps, seq, starts, **kw):
    preds = np.stack([rollout(s, seq, starts, **kw) for s in steps])
    return preds.mean(0), preds.std(0)
