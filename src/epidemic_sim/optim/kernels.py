"""Noyaux vectorisés indépendants du framework (loi exacte, sans boucle O(N) par agent).

* ``proximity_hazard`` : risque de contamination par proximité (arbre spatial cKDTree).
* ``rewire_rejection`` : tirage successif pondéré des contacts par rejet à deux niveaux
  (cellule spatiale x classe d'âge), ~O(N k) ; exact en loi (idée GIRG, arXiv 1511.00576).
* ``rewire_gumbel``    : même loi par Gumbel top-k, O(N^2) vectorisé (référence exacte).

Règle (identique au code ECS) : pour chaque agent i vivant, k_i contacts tirés SANS remise,
de façon successive, parmi les vivants j != i avec probabilité proportionnelle à
w_ij = exp(-d_ij/alpha) * exp(-|age_i - age_j|/tau).
Les fonctions travaillent sur des tableaux numpy (positions de l'espace Mesa, âges, etc.).
"""
import numpy as np
from scipy.spatial import cKDTree


def _weights(x: np.ndarray, y: np.ndarray, age: np.ndarray, i: np.ndarray, j: np.ndarray,
             alpha: float, tau: float) -> np.ndarray:
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    return np.exp(-d / alpha - np.abs(age[i] - age[j]) / tau)


def rewire_gumbel(x: np.ndarray, y: np.ndarray, age: np.ndarray, alive: np.ndarray, k: np.ndarray,
                  alpha: float, tau: float, rng: np.random.Generator,
                  block: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gumbel top-k : argtop-k_i de (log w_ij + G_ij), G ~ Gumbel(0,1) i.i.d.
    donne exactement un tirage successif proportionnel à w (Kool et al. 2019)."""
    idx = np.flatnonzero(alive)
    xs, ys, ags = x[idx], y[idx], age[idx]
    ks = np.minimum(k[idx], len(idx) - 1)
    src, dst, ww = [], [], []
    for a in range(0, len(idx), block):
        b = min(a + block, len(idx))
        rows = np.arange(a, b)
        kb = ks[a:b]
        kmax = int(kb.max()) if len(kb) else 0
        if kmax == 0:
            continue
        logw = (-np.hypot(xs[a:b, None] - xs[None, :], ys[a:b, None] - ys[None, :]) / alpha
                - np.abs(ags[a:b, None] - ags[None, :]) / tau)
        score = logw + rng.gumbel(size=logw.shape)
        score[np.arange(b - a), rows] = -np.inf          # pas d'auto-contact
        top = np.argpartition(-score, kmax - 1, axis=1)[:, :kmax]
        ts = np.take_along_axis(score, top, 1)
        order = np.argsort(-ts, axis=1)
        top = np.take_along_axis(top, order, 1)
        keep = np.arange(kmax)[None, :] < kb[:, None]
        r, c = np.nonzero(keep)
        src.append(idx[rows[r]]); dst.append(idx[top[r, c]])
        ww.append(np.exp(np.take_along_axis(logw, top, 1)[r, c]))
    if not src:
        z = np.zeros(0, int); return z, z, np.zeros(0)
    return np.concatenate(src), np.concatenate(dst), np.concatenate(ww)


def rewire_rejection(x: np.ndarray, y: np.ndarray, age: np.ndarray, alive: np.ndarray, k: np.ndarray,
                     alpha: float, tau: float, rng: np.random.Generator,
                     cell: float | None = None, age_bin: float | None = None,
                     batch_factor: int = 3, stats: dict | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tirage i.i.d. ~ w par rejet à deux niveaux, puis "premières occurrences"
    jusqu'à k contacts distincts (= tirage successif exact).

    Niveau 1 : groupes g = (cellule, classe d'âge). Pour un agent du groupe g, on choisit
               un groupe g' avec proba ∝ n(g') * wmax(g,g'), wmax = borne sup de w entre groupes.
    Niveau 2 : membre uniforme de g', accepté avec proba w_ij / wmax(g,g').
    Les rejets (auto-contact, mort) restreignent simplement le support : loi exacte."""
    idx = np.flatnonzero(alive)
    n = len(idx)
    xs, ys, ags = x[idx], y[idx], age[idx]
    ks = np.minimum(k[idx], n - 1).astype(int)
    cell = cell or alpha / 2
    age_bin = age_bin or tau / 5
    cx = np.floor((xs - xs.min()) / cell).astype(int)
    cy = np.floor((ys - ys.min()) / cell).astype(int)
    ab = np.floor((ags - ags.min()) / age_bin).astype(int)
    ny, nb = cy.max() + 1, ab.max() + 1
    gid = (cx * ny + cy) * nb + ab
    uniq, g_of, counts = np.unique(gid, return_inverse=True, return_counts=True)
    G = len(uniq)
    gcx, rest = np.divmod(uniq, ny * nb)
    gcy, gab = np.divmod(rest, nb)
    gapx = np.maximum(np.abs(gcx[:, None] - gcx[None, :]) - 1, 0) * cell
    gapy = np.maximum(np.abs(gcy[:, None] - gcy[None, :]) - 1, 0) * cell
    gapa = np.maximum(np.abs(gab[:, None] - gab[None, :]) - 1, 0) * age_bin
    wmax = np.exp(-np.hypot(gapx, gapy) / alpha - gapa / tau)          # G x G
    prop = wmax * counts[None, :]
    cdf = np.cumsum(prop, axis=1)
    cdf /= cdf[:, -1:]
    order = np.argsort(g_of, kind="stable")                             # membres triés par groupe
    start = np.concatenate([[0], np.cumsum(counts)])
    # état : contacts déjà obtenus (clé agent*n+cand, rang d'apparition)
    got_key = np.zeros(0, np.int64); got_rank = np.zeros(0, np.int64)
    need = ks.copy()
    rank0 = 0
    proposals = accepted = 0
    active = np.flatnonzero(need > 0)
    while len(active):
        reps = np.minimum(need[active] * batch_factor + 2, 400)
        a = np.repeat(active, reps)                                     # agent (local)
        ga = g_of[a]
        u = rng.random(len(a))
        gp = np.empty(len(a), int)
        for g in np.unique(ga):                                         # G petit
            m = ga == g
            gp[m] = np.minimum(np.searchsorted(cdf[g], u[m]), G - 1)
        j = order[start[gp] + np.minimum((rng.random(len(a)) * counts[gp]).astype(int), counts[gp] - 1)]
        w = _weights(xs, ys, ags, a, j, alpha, tau)
        ok = (j != a) & (rng.random(len(a)) * wmax[ga, gp] < w)
        proposals += len(a); accepted += int(ok.sum())
        a, j = a[ok], j[ok]
        key = np.concatenate([got_key, a.astype(np.int64) * n + j])
        rank = np.concatenate([got_rank, rank0 + np.arange(len(a), dtype=np.int64)])
        rank0 += len(a)
        o = np.lexsort((rank, key))                                      # première occurrence par clé
        key, rank = key[o], rank[o]
        first = np.ones(len(key), bool); first[1:] = key[1:] != key[:-1]
        key, rank = key[first], rank[first]
        ag = key // n
        o = np.lexsort((rank, ag))                                       # par agent, ordre d'apparition
        key, rank, ag = key[o], rank[o], ag[o]
        pos = np.arange(len(ag)) - np.searchsorted(ag, ag)               # rang dans l'agent
        keep = pos < ks[ag]
        got_key, got_rank = key[keep], rank[keep]
        have = np.bincount(got_key // n, minlength=n)
        need = ks - have
        active = np.flatnonzero(need > 0)
    if stats is not None:
        stats.update(acceptance=accepted / max(proposals, 1), groups=G, proposals=proposals)
    s_loc, d_loc = got_key // n, got_key % n
    return idx[s_loc], idx[d_loc], _weights(xs, ys, ags, s_loc, d_loc, alpha, tau)


def proximity_hazard(s_pos: np.ndarray, s_immunity: np.ndarray, i_pos: np.ndarray,
                     i_load: np.ndarray, radius: float, beta: float) -> np.ndarray:
    """hazard[s] = somme sur les infectieux i à distance <= radius de
    (beta * charge_i / 1000) * (1 - immunité_s).  s_pos, i_pos : tableaux (n, 2)."""
    out = np.zeros(len(s_pos))
    if len(i_pos) == 0 or len(s_pos) == 0:
        return out
    tree = cKDTree(s_pos)
    lists = tree.query_ball_point(i_pos, r=radius)
    lengths = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
    if lengths.sum() == 0:
        return out
    targets = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists if len(l)])
    np.add.at(out, targets, beta * np.repeat(i_load, lengths) / 1000)
    return out * (1 - s_immunity)
