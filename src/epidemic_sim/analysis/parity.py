"""Epidemiological parity between two implementations of the model.

Structural parity (same parameters, same mechanisms, same outputs) is checked by unit
tests. *Epidemiological* parity asks whether the two implementations give the same
distribution of epidemic indicators over independent replicates, in particular the same
mean AND the same variance.

Estimands (one value per replicated run, ``n`` = number of agents):

* ``attack_rate``      = (n - S_final) / n : share of the population ever infected;
* ``peak_prevalence``  = max_t I(t) / n    : highest share infected at the same time;
* ``time_to_peak``     = argmax_t I(t)     : step at which the peak is reached.

For each estimand, ECS and Mesa are compared with:

* the difference of means (Welch t-test) and the two-sample KS test (whole distribution);
* the ratio of variances Var_mesa / Var_ecs with a bootstrap percentile confidence
  interval, and the Brown-Forsythe (median-centred Levene) test of equal variances.
  The F test is not used because the estimands are not normal.
Parity is accepted for an estimand when no test rejects at ``alpha`` and the
(1 - alpha) confidence interval of the variance ratio contains 1 (same error level as
the tests, otherwise a 95 % interval would raise false alarms on its own).
"""
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

ESTIMANDS: tuple[str, ...] = ("attack_rate", "peak_prevalence", "time_to_peak")


def summarise(series: Mapping[str, Sequence[float]], n_agents: int) -> dict[str, float]:
    """The three estimands of one run, from the time series of the model."""
    infected = list(series["infected"])
    return {"attack_rate": (n_agents - series["susceptible"][-1]) / n_agents,
            "peak_prevalence": max(infected) / n_agents,
            "time_to_peak": float(infected.index(max(infected)))}


def variance_ratio_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                      n_boot: int = 4000, level: float = 0.95) -> tuple[float, float, float]:
    """Var(b) / Var(a) and its bootstrap percentile confidence interval."""
    ratio = float(np.var(b, ddof=1) / np.var(a, ddof=1))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = (np.var(rng.choice(b, len(b)), ddof=1) /
                   max(np.var(rng.choice(a, len(a)), ddof=1), 1e-300))
    low, high = np.percentile(boot, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return ratio, float(low), float(high)


def compare_estimand(ecs: Sequence[float], mesa: Sequence[float], rng: np.random.Generator,
                     alpha: float = 0.01) -> dict[str, float]:
    """Compare one estimand between the two implementations (replicates are independent)."""
    a, b = np.asarray(ecs, dtype=float), np.asarray(mesa, dtype=float)
    ratio, low, high = variance_ratio_ci(a, b, rng, level=1 - alpha)
    result = {"mean_ecs": a.mean(), "mean_mesa": b.mean(),
              "sd_ecs": a.std(ddof=1), "sd_mesa": b.std(ddof=1),
              "p_mean_welch": stats.ttest_ind(a, b, equal_var=False).pvalue,
              "p_ks": stats.ks_2samp(a, b, method="asymp").pvalue,
              "p_variance_brown_forsythe": stats.levene(a, b, center="median").pvalue,
              "variance_ratio": ratio, "variance_ratio_low": low, "variance_ratio_high": high, "ci_level": 1 - alpha}
    result["parity"] = bool(min(result["p_mean_welch"], result["p_ks"],
                                result["p_variance_brown_forsythe"]) > alpha
                            and low <= 1.0 <= high)
    return result
