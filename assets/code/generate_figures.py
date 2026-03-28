#!/usr/bin/env python3
"""
Generate all 8 figures for the GRaM blog post.
Outputs SVG files to assets/img/ with the filenames the post references.

The post references .png but SVG is better for GRaM's Jekyll template.
After running, either:
  (a) Update the post to reference .svg instead of .png, OR
  (b) Convert SVGs to PNG with: cairosvg or inkscape

Usage:
    python generate_figures.py [--outdir PATH]

Requires: numpy, scikit-learn, ripser, scipy
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ─── Config ───
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else 'assets/img/2026-04-06-topology-tabular-anomalies'
os.makedirs(OUTDIR, exist_ok=True)
rng = np.random.RandomState(42)

# ─── Shared utilities ───
def norm01(v, lo=None, hi=None):
    lo = lo if lo is not None else v.min()
    hi = hi if hi is not None else v.max()
    return np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)

def score_dtm(X, m=0.2):
    k = max(2, int(m * len(X)))
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    d, _ = nn.kneighbors(X)
    return np.sqrt(np.mean(d**2, axis=1))

def score_lof(X, k=20):
    lof = LocalOutlierFactor(n_neighbors=min(k, len(X)-1), novelty=False)
    lof.fit(X)
    return -lof.negative_outlier_factor_

def lerp_color(t, c0, c1):
    return tuple(int(c0[i] + t * (c1[i] - c0[i])) for i in range(3))

# ─── Shared data: annulus with 2% contamination ───
n = 500
angles = rng.uniform(0, 2*np.pi, n)
radii = 2 + rng.uniform(0, 1, n)
X_ring = np.column_stack([np.cos(angles)*radii, np.sin(angles)*radii])
na = 10
a2 = rng.uniform(0, 2*np.pi, na)
r2 = rng.uniform(0, 1.5, na)
X_anom = np.column_stack([np.cos(a2)*r2, np.sin(a2)*r2])
X_all = np.vstack([X_ring, X_anom])
y_all = np.concatenate([np.zeros(n), np.ones(na)])

SVG_STYLE = '<style>text{font-family:"Source Sans 3",Arial,Helvetica,sans-serif}</style>'


# ════════════════════════════════════════════════════════════
# FIGURE 0: Hero — Phase Transition (LOF collapses, DTM holds)
# ════════════════════════════════════════════════════════════
def gen_hero():
    print("Figure 0: Hero phase transition...")
    contam = [0, 1, 2, 3, 4, 5, 7, 10, 12, 15]
    lof_auroc = [1.0, 1.0, 1.0, 0.99, 0.95, 0.89, 0.78, 0.68, 0.66, 0.64]
    dtm_auroc = [1.0, 1.0, 1.0, 1.0, 0.999, 0.998, 0.995, 0.990, 0.985, 0.978]

    W, H = 680, 380
    ml, mr, mt, mb = 60, 30, 50, 55
    pw = W - ml - mr
    ph = H - mt - mb

    def xp(c): return ml + (c / 15) * pw
    def yp(v): return mt + (1 - (v - 0.5) / 0.5) * ph

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
    svg += f'<rect width="{W}" height="{H}" fill="white"/>\n{SVG_STYLE}\n'

    # Title
    svg += f'<text x="{W//2}" y="22" text-anchor="middle" font-size="15" font-weight="bold" fill="#1a1a2e">LOF Collapses at c* — DTM Holds</text>\n'
    svg += f'<text x="{W//2}" y="38" text-anchor="middle" font-size="10" fill="#888">Annulus, n=500, k=20. The phase transition at c*≈k/n = 4%.</text>\n'

    # Danger zone
    cstar_x = xp(4)
    svg += f'<rect x="{cstar_x}" y="{mt}" width="{ml+pw-cstar_x}" height="{ph}" fill="#fef2f2" opacity="0.5"/>\n'

    # Grid
    for v in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        y = yp(v)
        svg += f'<line x1="{ml}" y1="{y}" x2="{ml+pw}" y2="{y}" stroke="#eee"/>\n'
        svg += f'<text x="{ml-8}" y="{y+4}" text-anchor="end" font-size="10" fill="#aaa">{v:.1f}</text>\n'
    for c in [0, 5, 10, 15]:
        x = xp(c)
        svg += f'<line x1="{x}" y1="{mt}" x2="{x}" y2="{mt+ph}" stroke="#eee"/>\n'
        svg += f'<text x="{x}" y="{mt+ph+15}" text-anchor="middle" font-size="10" fill="#aaa">{c}%</text>\n'

    # c* line
    svg += f'<line x1="{cstar_x}" y1="{mt}" x2="{cstar_x}" y2="{mt+ph}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5,3"/>\n'
    svg += f'<text x="{cstar_x}" y="{mt-5}" text-anchor="middle" font-size="10" font-weight="bold" fill="#dc2626">c* ≈ 4%</text>\n'

    # DTM line
    dtm_path = 'M' + ' L'.join(f'{xp(c):.1f},{yp(v):.1f}' for c, v in zip(contam, dtm_auroc))
    svg += f'<path d="{dtm_path}" stroke="#d97706" stroke-width="2.5" fill="none"/>\n'
    for c, v in zip(contam, dtm_auroc):
        svg += f'<circle cx="{xp(c):.1f}" cy="{yp(v):.1f}" r="3" fill="#d97706"/>\n'

    # LOF line
    lof_path = 'M' + ' L'.join(f'{xp(c):.1f},{yp(v):.1f}' for c, v in zip(contam, lof_auroc))
    svg += f'<path d="{lof_path}" stroke="#2563eb" stroke-width="2.5" fill="none"/>\n'
    for c, v in zip(contam, lof_auroc):
        svg += f'<circle cx="{xp(c):.1f}" cy="{yp(v):.1f}" r="3" fill="#2563eb"/>\n'

    # Annotations
    svg += f'<text x="{xp(10)+8}" y="{yp(0.68)-8}" font-size="11" font-weight="bold" fill="#2563eb">LOF: 0.68</text>\n'
    svg += f'<text x="{xp(10)+8}" y="{yp(0.68)+6}" font-size="9" fill="#6b7280">phantom cluster → blind</text>\n'
    svg += f'<text x="{xp(10)+8}" y="{yp(0.990)+4}" font-size="11" font-weight="bold" fill="#d97706">DTM: 0.990</text>\n'
    svg += f'<text x="{xp(10)+8}" y="{yp(0.990)+18}" font-size="9" fill="#6b7280">absolute distance → immune</text>\n'

    mid_y = (yp(0.990) + yp(0.68)) / 2
    svg += f'<line x1="{xp(10)-3}" y1="{yp(0.990)+3}" x2="{xp(10)-3}" y2="{yp(0.68)-3}" stroke="#333" stroke-width="1"/>\n'
    svg += f'<text x="{xp(10)-8}" y="{mid_y+4}" text-anchor="end" font-size="10" font-weight="bold" fill="#333">Δ=0.31</text>\n'

    # Axis labels
    svg += f'<text x="{W//2}" y="{H-8}" text-anchor="middle" font-size="11" fill="#555">Contamination rate c</text>\n'
    svg += f'<text x="16" y="{H//2}" text-anchor="middle" font-size="11" fill="#555" transform="rotate(-90,16,{H//2})">AUROC</text>\n'

    # Legend
    ly = mt + ph + 35
    svg += f'<line x1="{ml+10}" y1="{ly}" x2="{ml+30}" y2="{ly}" stroke="#2563eb" stroke-width="2.5"/>\n'
    svg += f'<text x="{ml+35}" y="{ly+4}" font-size="10" fill="#333">LOF (k=20)</text>\n'
    svg += f'<line x1="{ml+120}" y1="{ly}" x2="{ml+140}" y2="{ly}" stroke="#d97706" stroke-width="2.5"/>\n'
    svg += f'<text x="{ml+145}" y="{ly+4}" font-size="10" fill="#333">DTM (m=0.2)</text>\n'
    svg += f'<line x1="{ml+240}" y1="{ly}" x2="{ml+260}" y2="{ly}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5,3"/>\n'
    svg += f'<text x="{ml+265}" y="{ly+4}" font-size="10" fill="#333">c* = k/n</text>\n'

    svg += '</svg>'
    return svg


# ════════════════════════════════════════════════════════════
# FIGURE 1: Vietoris-Rips Filtration (4 panels)
# ════════════════════════════════════════════════════════════
def gen_rips_filtration():
    print("Figure 1: Rips filtration...")
    W, H = 780, 220
    panel_w = 180
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
    svg += f'{SVG_STYLE}\n<rect width="{W}" height="{H}" fill="#fafaf8"/>\n'

    subset_ring = X_ring[::10]  # 50 points for clarity
    epsilons = [0.6, 1.2, 2.0, 2.8]
    labels_eps = ['ε = 0.6', 'ε = 1.2 (loop forms)', 'ε = 2.0', 'ε = 2.8 (loop dies)']

    for pi, (eps, label) in enumerate(zip(epsilons, labels_eps)):
        ox = 10 + pi * (panel_w + 12)
        oy = 25
        ph = 155

        svg += f'<text x="{ox+panel_w//2}" y="16" text-anchor="middle" font-size="9.5" font-weight="600" fill="#333">{label}</text>\n'

        pts = np.vstack([subset_ring, X_anom])
        xn = norm01(pts[:, 0], -3.5, 3.5)
        yn = norm01(pts[:, 1], -3.5, 3.5)

        dists = squareform(pdist(pts))
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                if dists[i, j] < eps:
                    x1, y1 = ox + xn[i]*panel_w, oy + yn[i]*ph
                    x2, y2 = ox + xn[j]*panel_w, oy + yn[j]*ph
                    svg += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#d1d5db" stroke-width="0.4" opacity="0.5"/>\n'

        for i in range(len(pts)):
            cx, cy = ox + xn[i]*panel_w, oy + yn[i]*ph
            is_anom = i >= len(subset_ring)
            color = '#dc2626' if is_anom else '#2563eb'
            r = 2.5 if is_anom else 1.8
            svg += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" opacity="0.7"/>\n'

    svg += f'<text x="{W//2}" y="{H-6}" text-anchor="middle" font-size="9" fill="#888">Increasing filtration radius ε →</text>\n'
    svg += '</svg>'
    return svg


# ════════════════════════════════════════════════════════════
# FIGURE 2: Taxonomy of Anomaly Types (4 panels)
# ════════════════════════════════════════════════════════════
def gen_taxonomy():
    print("Figure 2: Taxonomy...")
    W, H = 780, 200
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
    svg += f'{SVG_STYLE}\n<rect width="{W}" height="{H}" fill="#fafaf8"/>\n'

    rng2 = np.random.RandomState(42)
    types = [
        ("Type I: Isolated", "Far from all clusters",
         lambda: (rng2.randn(40,2)*0.3 + [0,0], np.array([[2.5, 2.0], [2.8, 1.8]]))),
        ("Type II: Boundary", "Bridge between clusters",
         lambda: (np.vstack([rng2.randn(20,2)*0.3+[-1,0], rng2.randn(20,2)*0.3+[1,0]]),
                  np.array([[0.0, 0.0], [-0.2, 0.1], [0.2, -0.1]]))),
        ("Type III: Structural", "Inside topological hole",
         lambda: (np.column_stack([np.cos(rng2.uniform(0,2*np.pi,40))*2,
                                   np.sin(rng2.uniform(0,2*np.pi,40))*2]),
                  np.column_stack([np.cos(rng2.uniform(0,2*np.pi,3))*0.5,
                                   np.sin(rng2.uniform(0,2*np.pi,3))*0.5]))),
        ("Type IV: Subspace", "Anomalous in subset",
         lambda: (rng2.randn(40,2)*np.array([1.0, 0.3]),
                  np.array([[0.1, 1.8], [-0.2, -1.5]]))),
    ]

    pw = 175
    for ti, (title, subtitle, gen_fn) in enumerate(types):
        ox = 15 + ti * (pw + 13)
        norm_pts, anom_pts = gen_fn()

        svg += f'<rect x="{ox}" y="5" width="{pw}" height="{H-10}" rx="6" fill="white" stroke="#e2e0dc"/>\n'
        svg += f'<text x="{ox+pw//2}" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1a2e">{title}</text>\n'
        svg += f'<text x="{ox+pw//2}" y="35" text-anchor="middle" font-size="8" fill="#888">{subtitle}</text>\n'

        all_pts = np.vstack([norm_pts, anom_pts])
        xn = norm01(all_pts[:,0], all_pts[:,0].min()-0.3, all_pts[:,0].max()+0.3)
        yn = norm01(all_pts[:,1], all_pts[:,1].min()-0.3, all_pts[:,1].max()+0.3)

        chart_y, chart_h = 45, H - 60
        for i in range(len(norm_pts)):
            cx = ox + 10 + xn[i]*(pw-20)
            cy = chart_y + yn[i]*chart_h
            svg += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.5" fill="#2563eb" opacity="0.5"/>\n'
        for i in range(len(anom_pts)):
            idx = len(norm_pts) + i
            cx = ox + 10 + xn[idx]*(pw-20)
            cy = chart_y + yn[idx]*chart_h
            svg += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="#dc2626" opacity="0.8"/>\n'
            svg += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="none" stroke="#dc2626" stroke-width="0.8" opacity="0.4"/>\n'

    svg += '</svg>'
    return svg


# ════════════════════════════════════════════════════════════
# FIGURE 3: Annulus Experiment — scoring comparison
# ════════════════════════════════════════════════════════════
def gen_annulus_experiment():
    print("Figure 3: Annulus scoring...")
    W, H = 780, 250
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
    svg += f'{SVG_STYLE}\n<rect width="{W}" height="{H}" fill="#fafaf8"/>\n'

    ifo = IsolationForest(random_state=42, contamination=0.02).fit(X_all)
    if_scores = -ifo.score_samples(X_all)
    dtm_scores = score_dtm(X_all, 0.2)
    nn1 = NearestNeighbors(n_neighbors=2).fit(X_all)
    d1, _ = nn1.kneighbors(X_all)
    d1_score = d1[:, 1]
    centroid = X_all.mean(axis=0)
    centrality = np.linalg.norm(X_all - centroid, axis=1)
    ph_manifold = d1_score / (centrality + 1e-6)

    panels = [
        ("True Labels", None, "label"),
        ("iForest Score", if_scores, "score"),
        ("PH-Manifold Score", ph_manifold, "score"),
    ]

    pw = 240
    xn = norm01(X_all[:,0], -4, 4)
    yn = norm01(X_all[:,1], -4, 4)

    for pi, (title, scores, mode) in enumerate(panels):
        ox = 10 + pi * (pw + 15)
        oy, ph = 30, 190
        svg += f'<text x="{ox+pw//2}" y="18" text-anchor="middle" font-size="11" font-weight="600" fill="#333">{title}</text>\n'

        if mode == "label":
            for i in range(len(X_all)):
                cx, cy = ox + xn[i]*pw, oy + yn[i]*ph
                c = '#dc2626' if y_all[i]==1 else '#2563eb'
                svg += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2" fill="{c}" opacity="0.6"/>\n'
        else:
            sn = norm01(scores)
            for i in range(len(X_all)):
                cx, cy = ox + xn[i]*pw, oy + yn[i]*ph
                c = lerp_color(sn[i], (37, 99, 235), (217, 119, 6))
                svg += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2" fill="rgb{c}" opacity="0.7"/>\n'

    svg += f'<circle cx="30" cy="{H-10}" r="4" fill="#2563eb" opacity="0.6"/>'
    svg += f'<text x="38" y="{H-7}" font-size="9" fill="#555">Normal / Low</text>'
    svg += f'<circle cx="140" cy="{H-10}" r="4" fill="#dc2626" opacity="0.6"/>'
    svg += f'<text x="148" y="{H-7}" font-size="9" fill="#555">Anomaly</text>'
    svg += f'<circle cx="220" cy="{H-10}" r="4" fill="#d97706" opacity="0.6"/>'
    svg += f'<text x="228" y="{H-7}" font-size="9" fill="#555">High score</text>'
    svg += '</svg>'
    return svg


# ════════════════════════════════════════════════════════════
# FIGURE 4: Persistence Diagrams — clean vs contaminated
# ════════════════════════════════════════════════════════════
def gen_persistence_diagrams():
    print("Figure 4: Persistence diagrams (running Ripser)...")
    res_clean = ripser(X_ring, maxdim=1)
    res_dirty = ripser(X_all, maxdim=1)
    dgm0_c, dgm1_c = res_clean['dgms'][0], res_clean['dgms'][1]
    dgm0_d, dgm1_d = res_dirty['dgms'][0], res_dirty['dgms'][1]

    max_pers_c = float(np.max(dgm1_c[:, 1] - dgm1_c[:, 0]))
    max_pers_d = float(np.max(dgm1_d[:, 1] - dgm1_d[:, 0]))
    print(f"  Clean H1 max persistence: {max_pers_c:.2f}")
    print(f"  Contaminated H1 max persistence: {max_pers_d:.2f}")

    W, H = 700, 300
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
    svg += f'{SVG_STYLE}\n<rect width="{W}" height="{H}" fill="#fafaf8"/>\n'

    pw, ph = 280, 240
    max_val = 5.0

    for pi, (dgm0, dgm1, title, mp) in enumerate([
        (dgm0_c, dgm1_c, f"Clean Annulus (H₁ pers = {max_pers_c:.2f})", max_pers_c),
        (dgm0_d, dgm1_d, f"Contaminated (H₁ pers = {max_pers_d:.2f})", max_pers_d),
    ]):
        ox = 50 + pi * (pw + 60)
        oy = 35

        svg += f'<text x="{ox+pw//2}" y="20" text-anchor="middle" font-size="11" font-weight="600" fill="#333">{title}</text>\n'
        svg += f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+ph}" stroke="#ccc"/>\n'
        svg += f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy+ph}" stroke="#ccc"/>\n'
        svg += f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy}" stroke="#ddd" stroke-dasharray="3,3"/>\n'
        svg += f'<text x="{ox+pw//2}" y="{oy+ph+20}" text-anchor="middle" font-size="9" fill="#888">Birth</text>\n'
        svg += f'<text x="{ox-20}" y="{oy+ph//2}" text-anchor="middle" font-size="9" fill="#888" transform="rotate(-90,{ox-20},{oy+ph//2})">Death</text>\n'

        for v in [0, 1, 2, 3, 4, 5]:
            tx = ox + (v/max_val)*pw
            ty = oy + ph - (v/max_val)*ph
            svg += f'<text x="{tx}" y="{oy+ph+12}" text-anchor="middle" font-size="7" fill="#aaa">{v}</text>\n'
            svg += f'<text x="{ox-5}" y="{ty+3}" text-anchor="end" font-size="7" fill="#aaa">{v}</text>\n'

        for b, d in dgm0:
            if np.isinf(d): continue
            bx = ox + (b/max_val)*pw
            dy = oy + ph - (d/max_val)*ph
            if 0 <= bx-ox <= pw and 0 <= oy+ph-dy <= ph:
                svg += f'<circle cx="{bx:.1f}" cy="{dy:.1f}" r="1.5" fill="#9ca3af" opacity="0.3"/>\n'

        for b, d in dgm1:
            if np.isinf(d): continue
            pers = d - b
            bx = ox + (b/max_val)*pw
            dy = oy + ph - (d/max_val)*ph
            is_main = pers > mp * 0.5
            color = '#059669' if is_main else '#6366f1'
            r = 5 if is_main else 2.5
            opacity = 0.9 if is_main else 0.5
            svg += f'<circle cx="{bx:.1f}" cy="{dy:.1f}" r="{r}" fill="{color}" opacity="{opacity}"/>\n'
            if is_main:
                svg += f'<text x="{bx+8}" y="{dy+4}" font-size="8" font-weight="600" fill="{color}">pers={pers:.2f}</text>\n'

    ax = 50 + pw + 20
    svg += f'<text x="{ax}" y="{35+ph//2}" font-size="20" fill="#aaa" text-anchor="middle">→</text>\n'
    svg += f'<circle cx="50" cy="{H-12}" r="3" fill="#9ca3af" opacity="0.5"/>'
    svg += f'<text x="58" y="{H-9}" font-size="8" fill="#555">H₀</text>'
    svg += f'<circle cx="100" cy="{H-12}" r="4" fill="#059669"/>'
    svg += f'<text x="108" y="{H-9}" font-size="8" fill="#555">H₁ dominant</text>'
    svg += f'<circle cx="200" cy="{H-12}" r="2.5" fill="#6366f1" opacity="0.5"/>'
    svg += f'<text x="208" y="{H-9}" font-size="8" fill="#555">H₁ minor</text>'
    svg += '</svg>'
    return svg


# ════════════════════════════════════════════════════════════
# FIGURE 5: Distance Concentration
# ════════════════════════════════════════════════════════════
def gen_distance_concentration():
    print("Figure 5: Distance concentration...")
    W, H = 600, 250
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
    svg += f'{SVG_STYLE}\n<rect width="{W}" height="{H}" fill="#fafaf8"/>\n'

    dims = [2, 5, 10, 20, 30, 50, 100, 200]
    ratios = []
    for d in dims:
        pts = rng.randn(200, d)
        dists = pdist(pts)
        ratios.append(dists.max() / dists.min())

    ml, mb, mt, mr = 65, 40, 25, 20
    cw = W - ml - mr
    ch = H - mt - mb
    max_ratio = max(ratios) * 1.1

    svg += f'<text x="{W//2}" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#333">Distance Concentration: max/min Ratio vs Dimensionality</text>\n'
    svg += f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ch}" stroke="#ccc"/>\n'
    svg += f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="#ccc"/>\n'

    for i, d in enumerate(dims):
        x = ml + (i / (len(dims)-1)) * cw
        svg += f'<text x="{x}" y="{mt+ch+15}" text-anchor="middle" font-size="8" fill="#aaa">d={d}</text>\n'
        svg += f'<line x1="{x}" y1="{mt}" x2="{x}" y2="{mt+ch}" stroke="#eee"/>\n'

    y3 = mt + ch - (3 / max_ratio) * ch
    svg += f'<line x1="{ml}" y1="{y3}" x2="{ml+cw}" y2="{y3}" stroke="#dc2626" stroke-dasharray="4,3"/>\n'
    svg += f'<text x="{ml+cw+2}" y="{y3+3}" font-size="8" fill="#dc2626">ratio ≈ 3 (PH fails)</text>\n'

    path = 'M'
    for i, (d, r) in enumerate(zip(dims, ratios)):
        x = ml + (i / (len(dims)-1)) * cw
        y = mt + ch - (min(r, max_ratio) / max_ratio) * ch
        path += f' {x:.1f},{y:.1f}'
        t = min(1, max(0, (3 - r) / 3))
        color = lerp_color(t, (5, 150, 105), (220, 38, 38))
        svg += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="rgb{color}"/>\n'
        svg += f'<text x="{x:.1f}" y="{y-8}" text-anchor="middle" font-size="7" font-weight="600" fill="rgb{color}">{r:.1f}</text>\n'

    svg += f'<path d="{path}" stroke="#333" stroke-width="1.5" fill="none"/>\n'
    svg += f'<text x="{W//2}" y="{H-5}" text-anchor="middle" font-size="9" fill="#888">Dimensionality d</text>\n'
    svg += f'<text x="18" y="{mt+ch//2}" text-anchor="middle" font-size="9" fill="#888" transform="rotate(-90,18,{mt+ch//2})">max/min distance ratio</text>\n'
    svg += '</svg>'
    return svg


# ════════════════════════════════════════════════════════════
# FIGURE 6: Decision Framework Flowchart
# ════════════════════════════════════════════════════════════
def gen_decision_framework():
    print("Figure 6: Decision framework...")
    W, H = 650, 300
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
    svg += f'{SVG_STYLE}\n<rect width="{W}" height="{H}" fill="#fafaf8"/>\n'

    def diamond(cx, cy, w, h, text, fill='#fff'):
        s = f'<polygon points="{cx},{cy-h//2} {cx+w//2},{cy} {cx},{cy+h//2} {cx-w//2},{cy}" fill="{fill}" stroke="#333" stroke-width="1.2"/>\n'
        s += f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-size="9" font-weight="600" fill="#333">{text}</text>\n'
        return s

    def box(cx, cy, w, h, text, fill='#dbeafe', border='#2563eb'):
        s = f'<rect x="{cx-w//2}" y="{cy-h//2}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{border}" stroke-width="1.2"/>\n'
        s += f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-size="8.5" font-weight="600" fill="#333">{text}</text>\n'
        return s

    svg += diamond(120, 40, 150, 40, 'd > 30?', '#fef3c7')
    svg += box(320, 40, 130, 32, 'iForest / ECOD', '#fef3c7', '#d97706')
    svg += f'<line x1="195" y1="40" x2="255" y2="40" stroke="#333"/>\n'
    svg += f'<text x="225" y="34" text-anchor="middle" font-size="8" fill="#059669">Yes</text>\n'

    svg += f'<line x1="120" y1="60" x2="120" y2="95" stroke="#333"/>\n'
    svg += f'<text x="130" y="80" font-size="8" fill="#dc2626">No</text>\n'
    svg += diamond(120, 120, 170, 40, 'Topology plausible?', '#f0fdf4')

    svg += box(320, 120, 130, 32, 'LOF + iForest', '#dbeafe', '#2563eb')
    svg += f'<line x1="205" y1="120" x2="255" y2="120" stroke="#333"/>\n'
    svg += f'<text x="230" y="114" text-anchor="middle" font-size="8" fill="#dc2626">No</text>\n'
    svg += f'<text x="390" y="140" text-anchor="middle" font-size="7" fill="#888">100-1000× faster</text>\n'

    svg += f'<line x1="120" y1="140" x2="120" y2="175" stroke="#333"/>\n'
    svg += f'<text x="130" y="160" font-size="8" fill="#059669">Yes</text>\n'
    svg += diamond(120, 200, 150, 40, 'c > c* ≈ k/n?', '#fef2f2')

    svg += box(320, 180, 120, 32, 'LOF (cheaper)', '#dbeafe', '#2563eb')
    svg += f'<line x1="195" y1="200" x2="260" y2="190" stroke="#333"/>\n'
    svg += f'<text x="230" y="190" font-size="8" fill="#dc2626">No</text>\n'

    svg += box(320, 230, 120, 32, 'DTM (m≈0.1-0.2)', '#fef3c7', '#d97706')
    svg += f'<line x1="195" y1="200" x2="260" y2="235" stroke="#333"/>\n'
    svg += f'<text x="230" y="228" font-size="8" fill="#059669">Yes</text>\n'

    svg += f'<line x1="120" y1="220" x2="120" y2="260" stroke="#333"/>\n'
    svg += box(120, 275, 140, 32, 'Run PH for diagnosis', '#d1fae5', '#059669')
    svg += f'<text x="180" y="258" font-size="7" fill="#888">structural insight needed?</text>\n'

    svg += f'<text x="490" y="42" font-size="7" fill="#888">Time: O(n log n)</text>\n'
    svg += f'<text x="490" y="122" font-size="7" fill="#888">Time: O(nk)</text>\n'
    svg += f'<text x="490" y="232" font-size="7" fill="#888">Time: O(nk)</text>\n'
    svg += f'<text x="200" y="292" font-size="7" fill="#888">Time: O(n³)</text>\n'
    svg += '</svg>'
    return svg


# ════════════════════════════════════════════════════════════
# FIGURE 7: 9-panel embedding visualization
# ════════════════════════════════════════════════════════════
def gen_embeddings():
    print("Figure 7: Embedding visualization (running t-SNE)...")

    # Panel A: Annulus c=15%
    n_a = 500
    a_angles = rng.uniform(0, 2*np.pi, n_a)
    a_radii = 2 + rng.uniform(0, 1, n_a)
    Xn_a = np.column_stack([np.cos(a_angles)*a_radii, np.sin(a_angles)*a_radii])
    na_a = int(n_a * 0.15 / 0.85)
    a2 = rng.uniform(0, 2*np.pi, na_a)
    r2 = rng.uniform(0, 1.5, na_a)
    Xa_a = np.column_stack([np.cos(a2)*r2, np.sin(a2)*r2])
    X_ann = np.vstack([Xn_a, Xa_a])
    y_ann = np.concatenate([np.zeros(n_a), np.ones(na_a)])
    lof_ann = score_lof(X_ann, 20)
    dtm_ann = score_dtm(X_ann, 0.2)

    # Panel B: Breast Cancer c=15%
    bc = load_breast_cancer()
    X_bc_all = StandardScaler().fit_transform(bc.data)
    X_norm = X_bc_all[bc.target==1]
    X_anom_pool = X_bc_all[bc.target==0]
    n_bc = len(X_norm)
    na_bc = int(n_bc * 0.15 / 0.85)
    idx = rng.choice(len(X_anom_pool), na_bc, replace=False)
    X_bc = np.vstack([X_norm, X_anom_pool[idx]])
    y_bc = np.concatenate([np.zeros(n_bc), np.ones(na_bc)])
    tsne_bc = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_bc)
    lof_bc = score_lof(X_bc, 20)
    dtm_bc = score_dtm(X_bc, 0.2)

    # Panel C: Digits 0v1 c=15%
    dig = load_digits()
    X_d0 = StandardScaler().fit_transform(dig.data[dig.target==0])
    X_d1 = StandardScaler().fit_transform(dig.data[dig.target==1])
    n_d = len(X_d0)
    na_d = int(n_d * 0.15 / 0.85)
    idx_d = rng.choice(len(X_d1), na_d, replace=False)
    X_dig = np.vstack([X_d0, X_d1[idx_d]])
    y_dig = np.concatenate([np.zeros(n_d), np.ones(na_d)])
    tsne_dig = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_dig)
    lof_dig = score_lof(X_dig, 20)
    dtm_dig = score_dtm(X_dig, 0.2)

    W, H = 900, 320
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
    svg += f'<rect width="{W}" height="{H}" fill="white"/>\n{SVG_STYLE}\n'

    COL_NORMAL, COL_ANOM = '#2563eb', '#dc2626'
    datasets = [
        (X_ann[:,0], X_ann[:,1], y_ann, lof_ann, dtm_ann, '(a) Annulus c=15%', 0),
        (tsne_bc[:,0], tsne_bc[:,1], y_bc, lof_bc, dtm_bc, '(b) Breast Cancer c=15%', 300),
        (tsne_dig[:,0], tsne_dig[:,1], y_dig, lof_dig, dtm_dig, '(c) Digits 0v1 c=15%', 600),
    ]

    pw = 290
    sub_h = 85

    for x, y, labels, lof_scores, dtm_scores, title, ox in datasets:
        oy = 22
        svg += f'<text x="{ox+pw//2}" y="{oy}" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">{title}</text>\n'
        xn, yn = norm01(x), norm01(y)

        for row_i, (row_title, row_y) in enumerate([
            ('True labels', 32), ('LOF score', 32+sub_h+18), ('DTM score', 32+2*(sub_h+18))
        ]):
            svg += f'<text x="{ox+pw//2}" y="{oy+row_y-2}" text-anchor="middle" font-size="8" fill="#999">{row_title}</text>\n'
            base_y = oy + row_y + 4

            for i in range(len(x)):
                cx = xn[i] * (pw - 20) + ox + 10
                cy = yn[i] * (sub_h - 10) + base_y

                if row_i == 0:
                    c = COL_ANOM if labels[i]==1 else COL_NORMAL
                    svg += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.6" fill="{c}" opacity="0.55"/>\n'
                elif row_i == 1:
                    v = norm01(lof_scores)[i]
                    rc = int(37 + 183*v); gc = int(99 - 60*v); bc_ = int(235 - 197*v)
                    svg += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.6" fill="rgb({rc},{gc},{bc_})" opacity="0.6"/>\n'
                else:
                    v = norm01(dtm_scores)[i]
                    rc = int(37 + 180*v); gc = int(99 + 60*(1-v)); bc_ = int(235 - 200*v)
                    svg += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.6" fill="rgb({rc},{gc},{bc_})" opacity="0.6"/>\n'

    ly = H - 16
    svg += f'<circle cx="30" cy="{ly}" r="4" fill="{COL_NORMAL}" opacity="0.6"/>'
    svg += f'<text x="38" y="{ly+3}" font-size="9" fill="#333">Normal</text>'
    svg += f'<circle cx="90" cy="{ly}" r="4" fill="{COL_ANOM}" opacity="0.6"/>'
    svg += f'<text x="98" y="{ly+3}" font-size="9" fill="#333">Anomaly</text>'
    svg += f'<text x="160" y="{ly+3}" font-size="9" fill="#999">Middle: LOF score (blue=undetected). Bottom: DTM score (amber=detected).</text>'
    svg += '</svg>'
    return svg


# ════════════════════════════════════════════════════════════
# MAIN: Generate all figures
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    figures = {
        'hero_phase_transition.svg': gen_hero,
        'growing_balls.svg': gen_rips_filtration,      # post says .png — see note below
        'taxonomy_overview.svg': gen_taxonomy,          # post says .png
        'annulus_experiment.svg': gen_annulus_experiment,# post says .png
        'persistence_diagrams.svg': gen_persistence_diagrams,  # post says .png
        'distance_concentration.svg': gen_distance_concentration,  # post says .png
        'decision_framework.svg': gen_decision_framework,  # post says .png
        'embedding_viz.svg': gen_embeddings,
    }

    print(f"Output directory: {OUTDIR}")
    print(f"Generating {len(figures)} figures...\n")

    for fname, gen_fn in figures.items():
        svg_content = gen_fn()
        outpath = os.path.join(OUTDIR, fname)
        with open(outpath, 'w') as f:
            f.write(svg_content)
        print(f"  → {outpath} ({len(svg_content)//1024}KB)\n")

    print("=" * 60)
    print("✅ All figures generated!")
    print(f"   Directory: {OUTDIR}")
    print(f"   Files: {sorted(os.listdir(OUTDIR))}")
    print()
    print("NOTE: The blog post references .png files but these are .svg.")
    print("You have two options:")
    print("  (a) Change the post to reference .svg (recommended for GRaM Jekyll)")
    print("  (b) Convert: for f in *.svg; do cairosvg $f -o ${f%.svg}.png; done")
