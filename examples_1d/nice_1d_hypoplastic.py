"""
NICE (Masi & Einav 2024) applied to a 1D elasto-hypoplastic material under lab-like conditions.

Reference material (identified from the TANN repository data, github.com/filippo-masi/Thermodynamics-Neural-Networks):
    u = 1/2 E (eps_e)^2,  sigma = E eps_e,  d(eps_e)/dt = eps_dot - |eps_dot| sigma / S,   E = 200 000, S = 141.42
Training data mimic standard laboratory tests:
    * strain-controlled, constant strain rate |eps_dot| = RATE
    * single monotonic loading (tension or compression) to eps_max, then unloading until sigma = 0  (NO cycles)
    * stress sampled at N_SAMPLES points per test; elastic/plastic strain, energy and dissipation NOT measured
    * optional Gaussian noise on stress: sigma* = sigma + mean|sigma| * N(0, delta)
Test:
    * held-out loading-unloading tests
    * random multi-cycle paths (protocol never seen in training)

NICE ingredients (1D version of NICE_reduced in nice/nice_module.py):
    state eps_e, evolution  d(eps_e)/dt = f_theta(eps_e, eps_dot)  (neural ODE, explicit midpoint, torchdiffeq)
    energy u_omega(eps_e), sigma = du/d(eps_e), dissipation d = sigma (eps_dot - d(eps_e)/dt) >= 0
    learnable initial elastic strain per test (residual sigma(eps_e(0)) - sigma_measured(0))
    loss = L_sigma + L_r0 + L_d + lambda * L2(weights)

Usage (from the repository root):
    python examples_1d/nice_1d_hypoplastic.py --noise 0
    python examples_1d/nice_1d_hypoplastic.py --noise 5 --epochs 3000
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torchdiffeq import odeint

DT = torch.float64
E_REF, S_REF = 2.0e5, 100 * np.sqrt(2)
RATE = 1e-3            # strain rate magnitude (1/s), quasi-static
H = 0.025              # integration step (s) -> strain increment 2.5e-5 per step
N_SAMPLES = 40         # stress measurements per loading-unloading test
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


# ----------------------------------------------------------------------------------------------- reference model
def reference_response(eps_dot_steps, h=H, sub=20):
    """Exact integration of sigma' = E(eps' - |eps'| sigma / S) for piecewise-constant strain rate (per step)."""
    sig = np.zeros(len(eps_dot_steps) + 1)
    s = 0.
    for k, r in enumerate(eps_dot_steps):
        de = r * h / sub
        for _ in range(sub):
            if de != 0:
                sg = np.sign(de)
                s = sg * S_REF + (s - sg * S_REF) * np.exp(-E_REF * abs(de) / S_REF)
        sig[k + 1] = s
    return sig


def _step(s, de, sub=20):
    """Exact stress update for a strain increment de (split into sub parts)."""
    for _ in range(sub):
        if de != 0:
            sg = np.sign(de)
            s = sg * S_REF + (s - sg * S_REF) * np.exp(-E_REF * abs(de / sub) / S_REF)
    return s


def loading_unloading_test(eps_max):
    """Load to eps_max at constant rate, then unload at the same rate until the stress reaches zero."""
    sg = np.sign(eps_max)
    n_load = int(round(abs(eps_max) / (RATE * H)))
    rates = [sg * RATE] * n_load
    sig = list(reference_response(rates))
    while True:
        s_new = _step(sig[-1], -sg * RATE * H)
        rates.append(-sg * RATE)
        sig.append(s_new)
        if np.sign(s_new) != sg:
            break
    return np.array(rates), np.array(sig)


def random_cyclic_path(rng, n_rev=6, eps_lim=0.006):
    """Random strain path with several reversals between -eps_lim and eps_lim, starting from 0."""
    e, rates = 0., []
    for _ in range(n_rev):
        target = rng.uniform(-eps_lim, eps_lim)
        n = int(round(abs(target - e) / (RATE * H)))
        rates += [np.sign(target - e) * RATE] * n
        e += np.sign(target - e) * RATE * H * n
    return np.array(rates), reference_response(rates)


def pad(list_rates, list_sig):
    """Stack paths of different duration: (K, n_paths) rates (0 after the end) and stresses (NaN after the end)."""
    K = max(len(r) for r in list_rates)
    R = np.zeros((K, len(list_rates)))
    S = np.full((K + 1, len(list_rates)), np.nan)
    for i, (r, s) in enumerate(zip(list_rates, list_sig)):
        R[:len(r), i] = r
        S[:len(s), i] = s
    lengths = np.array([len(r) for r in list_rates])
    return R, S, lengths


# ----------------------------------------------------------------------------------------------- NICE 1D
class NICE1D(torch.nn.Module):
    def __init__(self, n_paths, prm_e, prm_s, prm_r):
        super().__init__()
        self.prm_e, self.prm_s, self.prm_r = prm_e, prm_s, prm_r      # scales for eps_e, sigma, strain rate
        self.f = torch.nn.Sequential(torch.nn.Linear(2, 36, dtype=DT), torch.nn.GELU(),
                                     torch.nn.Linear(36, 36, dtype=DT), torch.nn.GELU(),
                                     torch.nn.Linear(36, 36, dtype=DT), torch.nn.GELU(),
                                     torch.nn.Linear(36, 1, dtype=DT))
        self.u = torch.nn.Sequential(torch.nn.Linear(1, 64, dtype=DT), torch.nn.Softplus(),
                                     torch.nn.Linear(64, 64, dtype=DT), torch.nn.Softplus(),
                                     torch.nn.Linear(64, 1, dtype=DT))
        self.e0 = torch.nn.Parameter(torch.zeros(n_paths, 1, dtype=DT))   # learnable initial elastic strain (normalised)
        self.rates = None

    def rate_of(self, eps_e, r):
        return self.f(torch.cat((eps_e / self.prm_e, r / self.prm_r), -1)) * self.prm_r

    def forward(self, t, y):                      # neural ODE right-hand side
        k = min(int(t.item() / H + 1e-6), self.rates.shape[0] - 1)
        return self.rate_of(y, self.rates[k][:, None])

    def energy(self, eps_e):
        return self.u(eps_e / self.prm_e) * self.prm_s * self.prm_e

    def stress(self, eps_e):
        if not eps_e.requires_grad:
            eps_e = eps_e.requires_grad_(True)
        u = self.energy(eps_e)
        return torch.autograd.grad(u, eps_e, torch.ones_like(u), create_graph=True)[0]

    def integrate(self, rates, y0):
        """rates: (K, n) tensor; y0: (n, 1). Returns eps_e (K+1, n, 1), sigma (K+1, n, 1), dissipation (K, n, 1)."""
        self.rates = rates
        K = rates.shape[0]
        t = torch.arange(K + 1, dtype=DT) * H
        y = odeint(self, y0, t, method='midpoint', options={'step_size': H})
        sig = self.stress(y)
        rate_e = self.rate_of(y[:-1], rates[..., None])
        diss = sig[1:] * (rates[..., None] - rate_e)
        return y, sig, diss

    def initial_elastic_strain(self, sigma0, iters=30):
        """Newton solve sigma(eps_e0) = sigma0 (inference)."""
        x = torch.zeros(len(sigma0), 1, dtype=DT)
        for _ in range(iters):
            x = x.detach().requires_grad_(True)
            s = self.stress(x)
            ds = torch.autograd.grad(s.sum(), x)[0]
            x = x - (s - sigma0[:, None]) / ds
        return x.detach()


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noise', type=float, default=0., help='noise amplitude delta in percent')
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tag = f'noise{args.noise:g}'
    os.makedirs(OUT, exist_ok=True)

    # ---- data: 24 loading-unloading tests (12 amplitudes, tension and compression) ----
    amps = np.linspace(0.0005, 0.006, 12)
    tests = [loading_unloading_test(s * a) for a in amps for s in (1, -1)]
    idx = rng.permutation(len(tests))
    n_tr, n_va = 14, 5
    i_tr, i_va, i_te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    R, S, L = pad([t[0] for t in tests], [t[1] for t in tests])

    # sparse sampling: N_SAMPLES stress values per test (+ initial state)
    samp = np.zeros((N_SAMPLES + 1, len(tests)), dtype=int)
    for i, n in enumerate(L):
        samp[:, i] = np.round(np.linspace(0, n, N_SAMPLES + 1)).astype(int)
    S_meas = np.take_along_axis(S, samp, 0)
    if args.noise > 0:
        noise = np.mean(np.abs(S_meas[1:])) * rng.normal(0, args.noise / 100, S_meas[1:].shape)
        S_meas[1:] += noise

    trv = np.concatenate((i_tr, i_va))
    prm_s = float(np.abs(S_meas[:, trv]).max())
    # scales: stress from data, elastic strain as a fraction of the imposed strain range (E is NOT used), rate
    prm_e, prm_r = float(np.abs(np.cumsum(R[:, trv] * H, 0)).max()) * 0.15, RATE
    model = NICE1D(len(trv), prm_e, prm_s, prm_r)

    Rt = torch.tensor(R[:, trv])
    Sm = torch.tensor(S_meas[:, trv])
    sp = torch.tensor(samp[:, trv])
    mask_tr = torch.zeros(len(trv), dtype=torch.bool); mask_tr[:n_tr] = True
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)
    mse = torch.nn.MSELoss()

    def losses(mask):
        y0 = model.e0 * prm_e
        _, sig, diss = model.integrate(Rt, y0)
        sig_s = torch.gather(sig[..., 0], 0, sp)                         # predicted stress at sample times
        l_sig = mse(sig_s[1:, mask] / prm_s, Sm[1:, mask] / prm_s)
        l_r0 = mse(model.stress(y0)[mask, 0] / prm_s, Sm[0, mask] / prm_s)
        valid = (torch.arange(Rt.shape[0])[:, None] < torch.tensor(L[trv])[None, :])
        dd = diss[..., 0][valid[:, :] & mask[None, :]]
        l_d = torch.mean((torch.relu(-dd) / (dd.abs().max().detach() + 1e-12)) ** 2)
        return l_sig, l_r0, l_d

    best, best_state, hist, t0 = np.inf, None, [], time.time()
    for ep in range(1, args.epochs + 1):
        opt.zero_grad()
        l_sig, l_r0, l_d = losses(mask_tr)
        l2 = sum((p ** 2).sum() for n, p in model.named_parameters() if 'weight' in n)
        loss = l_sig + l_r0 + l_d + 1e-5 * l2
        loss.backward(); opt.step()
        if sched.get_last_lr()[0] > 1e-4:
            sched.step()
        if ep % 25 == 0 or ep == 1:
            v_sig, v_r0, v_d = losses(~mask_tr)
            v = (v_sig + v_r0 + v_d).item()
            hist.append((ep, l_sig.item(), v))
            if v < best:
                best, best_state = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
            if ep % 250 == 0 or ep == 1:
                print(f'epoch {ep:5d}  L_sigma {l_sig.item():.3e}  L_d {l_d.item():.2e}  val {v:.3e}  ({time.time() - t0:.0f}s)', flush=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(OUT, f'nice1d_{tag}.pt'))

    # ---- evaluation (noise-free references) ----
    def predict(rates, sig0):
        y0 = model.initial_elastic_strain(torch.tensor(sig0))
        y, sig, diss = model.integrate(torch.tensor(rates), y0)
        return y[..., 0].detach().numpy(), sig[..., 0].detach().numpy(), diss[..., 0].detach().numpy()

    def rel_err(pred, ref, lengths):
        e = [np.sqrt(np.mean((pred[:n + 1, i] - ref[:n + 1, i]) ** 2)) for i, n in enumerate(lengths)]
        return float(np.mean(e) / S_REF)

    res = {'noise_percent': args.noise, 'n_train_tests': n_tr, 'n_val_tests': n_va, 'samples_per_test': N_SAMPLES,
           'data_used': 'stress at 40 times per loading-unloading test + imposed strain; no elastic/plastic strain, energy or dissipation',
           'epochs': args.epochs, 'train_time_s': round(time.time() - t0, 1)}
    _, s_te, _ = predict(R[:, i_te], S[0, i_te])
    res['test_loading_unloading_rel_rmse_sigma'] = rel_err(s_te, S[:, i_te], L[i_te])

    cyc = [random_cyclic_path(np.random.default_rng(100 + k)) for k in range(5)]
    Rc, Sc, Lc = pad([c[0] for c in cyc], [c[1] for c in cyc])
    ye_c, s_c, d_c = predict(Rc, Sc[0])
    res['test_cyclic_rel_rmse_sigma'] = rel_err(s_c, Sc, Lc)
    res['negative_dissipation_fraction_cyclic'] = float(np.mean(d_c[~np.isnan(Sc[1:])] < -1e-9))

    # discovered elastic strain vs truth (sigma/E), stiffness and limit stress
    ee_true = Sc / E_REF
    m = ~np.isnan(Sc)
    a, b = np.polyfit(ye_c[m], ee_true[m], 1)
    r2 = 1 - np.sum((ee_true[m] - (a * ye_c[m] + b)) ** 2) / np.sum((ee_true[m] - ee_true[m].mean()) ** 2)
    x = torch.tensor(ye_c[m].reshape(-1, 1)).requires_grad_(True)
    s = model.stress(x)
    E_nn = torch.autograd.grad(s.sum(), x)[0].detach().numpy().ravel()
    res['eps_e_vs_true'] = {'R2_linear': float(r2), 'slope': float(a), 'offset': float(b)}
    res['E_median'] = float(np.median(E_nn)); res['E_p5_p95'] = [float(np.percentile(E_nn, 5)), float(np.percentile(E_nn, 95))]
    # monotonic tension to 0.6 %
    n_m = int(round(0.006 / (RATE * H)))
    _, s_m, _ = predict(np.full((n_m, 1), RATE), np.zeros(1))
    eps_m = np.arange(n_m + 1) * RATE * H
    s_ref_m = reference_response(np.full(n_m, RATE))
    pl = eps_m - s_m[:, 0] / res['E_median']
    res['Rp0.2'] = float(np.interp(0.002, pl, s_m[:, 0]))
    res['Rp0.2_ref'] = float(np.interp(0.002, eps_m - s_ref_m / E_REF, s_ref_m))
    res['sigma_at_0.6pct'] = float(s_m[-1, 0]); res['sigma_at_0.6pct_ref'] = float(s_ref_m[-1])

    res['curves'] = {'train_example': {'eps': (np.cumsum(np.r_[0, R[:L[i_tr[0]], i_tr[0]]]) * H).tolist(),
                                       'sigma_ref': S[:L[i_tr[0]] + 1, i_tr[0]].tolist(),
                                       'samp_idx': samp[:, i_tr[0]].tolist(), 'sigma_meas': S_meas[:, i_tr[0]].tolist()},
                     'cyclic': [{'eps': (np.cumsum(np.r_[0, Rc[:Lc[i], i]]) * H).tolist(), 'sigma_ref': Sc[:Lc[i] + 1, i].tolist(),
                                 'sigma_pred': s_c[:Lc[i] + 1, i].tolist(), 'eps_e_pred': ye_c[:Lc[i] + 1, i].tolist(),
                                 'diss_pred': d_c[:Lc[i], i].tolist()} for i in range(2)],
                     'monotonic': {'eps': eps_m.tolist(), 'sigma_ref': s_ref_m.tolist(), 'sigma_pred': s_m[:, 0].tolist()},
                     'loss_hist': hist}
    with open(os.path.join(OUT, f'results_{tag}.json'), 'w') as f:
        json.dump(res, f)
    print(json.dumps({k: v for k, v in res.items() if k != 'curves'}, indent=1))


if __name__ == '__main__':
    main()
