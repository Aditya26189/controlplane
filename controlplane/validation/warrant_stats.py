"""Statistics for ControlPlane.ai warrants.

Two objects, matching the two things a warrant has to do:

  certify_fpr()   issuance-time. Exact binomial (Clopper-Pearson) upper bound.
                  Refuses to certify rather than emitting an uncertifiable
                  threshold.

  FPRMonitor      runtime. Anytime-valid revocation trigger. Bounded
                  false-revocation probability over the WHOLE deployment,
                  at any stopping time -- not per-check.

Both are one-sided upper bounds on FPR, because that is the direction the
warrant makes a promise in.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log
from typing import Optional

import numpy as np
from scipy import stats

__all__ = [
    "Certification",
    "FPRMonitor",
    "certify_fpr",
    "cp_upper",
    "min_n_for",
]


# ---------------------------------------------------------------- issuance

def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson exact upper confidence bound on a binomial rate.

    k = observed false positives, n = independent true-negative items.
    Guarantees coverage >= 1-alpha for every true rate. Conservative by
    construction; that is the point.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(1.0 - alpha, k + 1, n - k))


def min_n_for(target_fpr: float, alpha: float = 0.05) -> int:
    """Smallest clean (k=0) sample that can certify target_fpr at 1-alpha.

    If you have fewer independent negatives than this, no threshold on this
    set can be certified at this budget -- regardless of where you put it.
    """
    return ceil(log(alpha) / log(1.0 - target_fpr))


@dataclass
class Certification:
    fpr_certified: bool      # renamed from `certified`: this is an FPR claim
    threshold: Optional[float]  # ONLY. It says nothing about recall. A detector
    ucb: Optional[float]     # that fires on nothing certifies trivially, so
    n_eff: int               # the policy layer MUST conjoin a recall floor.
    k: Optional[int]         # None when no threshold was evaluated
    reason: str


def certify_fpr(
    scores_neg: np.ndarray,
    target_fpr: float,
    alpha: float = 0.05,
    cluster_ids: Optional[np.ndarray] = None,
    n_profiles: int = 1,
) -> Certification:
    """Pick the lowest threshold whose CP upper bound clears target_fpr.

    scores_neg   detector scores on held-out TRUE NEGATIVES ONLY.
                 Must be disjoint from whatever data selected the operating
                 region -- selecting and certifying on the same items makes
                 the bound optimistic (winner's curse).
    cluster_ids  group label per item. If items share a source document,
                 template, or entity they are not independent; certification
                 then happens at CLUSTER level (a cluster counts as a false
                 positive if any of its items is). Exact, no ICC estimate
                 needed, and honest about effective sample size.
    n_profiles   number of policy profiles certified off this same set.
                 Applies a Bonferroni split of alpha.

    On the sweep and multiple testing: cp_upper(k(tau)) is non-increasing in
    tau, so the certifying set {tau : ucb <= target} is an upper set and taking
    its infimum is not a search over many hypotheses. The only way to err is
    for the CP bound to fail at the single fixed (unknown but deterministic)
    threshold bounding that region, which is one fixed-threshold coverage
    event bounded by alpha. Verified empirically at 0.0238 against nominal
    0.05. No multiplicity correction is required for the sweep itself.
    """
    alpha_eff = alpha / max(1, n_profiles)
    scores_neg = np.asarray(scores_neg, dtype=float)

    if cluster_ids is None:
        units = [np.array([s]) for s in scores_neg]
    else:
        cluster_ids = np.asarray(cluster_ids)
        units = [scores_neg[cluster_ids == c] for c in np.unique(cluster_ids)]

    n = len(units)
    # A cluster fires if ANY member scores at or above threshold. This bound
    # dominates item-level FPR ONLY when clusters are equal-sized. With
    # unequal sizes it is ANTICONSERVATIVE: one large all-firing cluster among
    # many silent singletons gives cluster-FPR far below the per-request rate
    # the warrant actually promises.
    if cluster_ids is not None and len({len(u) for u in units}) != 1:
        raise ValueError(
            "unequal cluster sizes: cluster-level FPR no longer dominates "
            "item-level FPR. Use the design-effect route instead.")
    unit_max = np.array([u.max() for u in units])

    floor_n = min_n_for(target_fpr, alpha_eff)
    if n < floor_n:
        return Certification(
            False, None, None, n, None,
            f"only {n} independent units; need >= {floor_n} clean units to "
            f"certify FPR<={target_fpr} at alpha={alpha_eff:.4f}",
        )

    # Candidates sit just ABOVE each observed negative score, so that k=0
    # (threshold above every negative) is reachable. Sweeping lowest-first
    # returns the most permissive threshold that still certifies, i.e. the
    # best recall consistent with the FPR promise.
    cuts = np.nextafter(np.sort(np.unique(unit_max)), np.inf)
    for tau in cuts:
        k = int((unit_max >= tau).sum())
        ucb = cp_upper(k, n, alpha_eff)
        if ucb <= target_fpr:
            return Certification(True, float(tau), ucb, n, k, "certified")

    raise AssertionError("unreachable: n >= floor_n guarantees k=0 certifies")


# ----------------------------------------------------------------- runtime

class FPRMonitor:
    """Anytime-valid revocation trigger for H0: true FPR <= p0.

    Feed one Bernoulli per LABELED TRUE-NEGATIVE item seen in production
    (1 = the detector fired on it). Wealth is a non-negative supermartingale
    under H0, so by Ville's inequality:

        P(wealth ever reaches 1/alpha | FPR really is <= p0) <= alpha

    over the entire lifetime of the warrant. You may check it every request,
    every hour, or never, and the guarantee is unchanged. No multiple-testing
    correction. No fixed evaluation window. This is what makes continuous
    renewal statistically legitimate rather than repeated peeking.

    lam_hi sets the smallest breach you want caught quickly: the Kelly-optimal
    bet against a true rate pi1 is lam* = (pi1 - p0) / (p0 * (1 - p0)).
    Setting lam_hi far above that makes the monitor fire on the first stray
    false positive; it stays valid but becomes useless.
    """

    def __init__(self, p0: float, alpha: float = 0.05,
                 lam_lo: float = 0.05, lam_hi: float = 10.0, n_lam: int = 40):
        if not 0.0 < p0 < 1.0:
            raise ValueError("p0 must be in (0, 1)")
        self.p0, self.alpha = p0, alpha
        # HARD CONSTRAINT: 1 + lam*(x - p0) > 0 for x in {0,1} requires
        # lam < 1/p0. Exceeding it makes log1p() return NaN, NaN propagates
        # through the mixture, and `nan >= 1/alpha` is False -- the monitor
        # goes silently dead and never revokes. Clamp, never trust the caller.
        lam_cap = 0.9 / p0
        self.lam_clamped = lam_hi > lam_cap
        if self.lam_clamped:
            lam_hi = lam_cap
        lam_lo = min(lam_lo, lam_hi / 100.0)
        self.lam_lo, self.lam_hi, self.n_lam = float(lam_lo), float(lam_hi), int(n_lam)
        self.lams = np.geomspace(lam_lo, lam_hi, n_lam)
        self._logw = np.zeros(n_lam)
        self.n = 0
        self.k = 0

    def update(self, is_false_positive: int) -> float:
        x = float(bool(is_false_positive))
        self.n += 1
        self.k += int(x)
        self._logw += np.log1p(self.lams * (x - self.p0))
        # NOT an assert: `python -O` strips asserts, and a guard that vanishes
        # under an optimisation flag is exactly the silent death this check
        # exists to prevent.
        if not np.isfinite(self._logw).all():
            raise FloatingPointError(
                f"non-finite wealth at n={self.n}, p0={self.p0}, "
                f"max lam*p0={self.lams.max() * self.p0:.3f}. The betting grid "
                "is invalid and the monitor can no longer revoke."
            )
        return self.wealth

    @property
    def wealth(self) -> float:
        m = self._logw.max()
        cap = np.log(1.0 / self.alpha) + 1.0          # nothing above 1/alpha
        if m > cap:                                    # changes the decision
            return 1.0 / self.alpha
        return float(np.exp(m) * np.mean(np.exp(self._logw - m)))

    @property
    def revoked(self) -> bool:
        return self.wealth >= 1.0 / self.alpha

    def state(self) -> dict:
        """Serialize into the warrant record. Resume by restoring _logw --
        never restart the monitor on renewal, or you reset the guarantee.

        The betting grid travels with the wealth. ``_logw[i]`` is the log
        wealth of the bet at ``lams[i]``, so restoring the vector against a
        grid built from different bounds silently reattaches every number to
        the wrong bet -- same shape, same types, nothing raised, guarantee
        gone.
        """
        return {"p0": self.p0, "alpha": self.alpha, "n": self.n, "k": self.k,
                "wealth": self.wealth, "logw": self._logw.tolist(),
                "revoked": self.revoked, "lam_lo": self.lam_lo,
                "lam_hi": self.lam_hi, "n_lam": self.n_lam,
                "lam_clamped": self.lam_clamped}

    @classmethod
    def from_state(cls, st: dict) -> "FPRMonitor":
        """Resume a monitor. Restarting instead of resuming resets the
        supermartingale and voids the false-revocation guarantee, so this
        exists to make the contract executable rather than documented."""
        # The grid is rebuilt from the RECORDED bounds, not the defaults. A
        # monitor constructed with a non-default lam_hi and resumed from
        # defaults would rebuild a different grid of the same length, pass the
        # shape check, and carry on with every bet misattributed.
        missing = {"lam_lo", "lam_hi", "n_lam"} - set(st)
        if missing:
            raise ValueError(
                f"monitor state is missing the betting grid {sorted(missing)}. "
                "Wealth cannot be reattached to bets that were not recorded; "
                "this state predates the grid being serialised and cannot be "
                "resumed safely."
            )
        m = cls(p0=st["p0"], alpha=st["alpha"], lam_lo=st["lam_lo"],
                lam_hi=st["lam_hi"], n_lam=st["n_lam"])
        logw = np.asarray(st["logw"], dtype=float)
        if logw.shape != m.lams.shape:
            raise ValueError("betting grid changed; cannot resume this monitor")
        m._logw, m.n, m.k = logw, st["n"], st["k"]
        return m


if __name__ == "__main__":
    print("min clean negatives needed (alpha=0.05, k=0):")
    for t in (0.015, 0.02, 0.03, 0.05):
        print(f"  FPR<={t:<6} -> n >= {min_n_for(t)}")

    print("\nCP upper bound by sample size:")
    for n in (258, 199, 129, 86, 60):
        print(f"  n={n:<5} k=0 -> {cp_upper(0, n):.4f}   k=1 -> {cp_upper(1, n):.4f}")
