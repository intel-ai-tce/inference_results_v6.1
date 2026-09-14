#!/usr/bin/env python3
"""Estimate P(a VBench run clears the MLPerf threshold) from REPLICATE runs.

This consumes several *duplicate* accuracy runs of the same config -- runs that
differ only because video generation is non-deterministic (GPU kernels, fp8 /
mxfp4 quantization, autotune) so the aggregate ``vbench_score`` wobbles run to
run. It estimates the probability that a single future run passes, three ways,
and cross-checks the key modelling assumption.

Estimators (all target: P a SINGLE new run scores >= threshold):

  A. empirical            fraction of the observed runs that pass. Nonparametric
                          and correlation-preserving, but coarse (only takes
                          values 0, 1/R, ..., 1) and useless once all runs land
                          on the same side of the threshold.

  B. whole-run predictive fit a Gaussian to the R aggregate scores and report
                          P(new run >= thr). Two flavours:
                            * normal    : Phi((mean - thr)/s)
                            * Student-t : predictive t_{R-1} with scale
                                          s*sqrt(1+1/R) -- the honest small-R
                                          version (heavier tails for R=5).
                          Preserves whatever run-to-run correlation exists
                          (it uses whole runs) and is smooth.

  C. per-prompt bootstrap the mix-and-match bootstrap: build many synthetic runs
                          by drawing, for EACH prompt independently, that
                          prompt's video from one of the R runs; score each
                          synthetic run; take the pass fraction. Smooth and
                          data-efficient, but assumes generation noise is
                          INDEPENDENT across prompts -- it erases any
                          whole-run common-mode drift.

  D. independence check   compares the run-to-run variance implied by C (prompts
                          independent) against the variance actually observed
                          across the R whole runs (B). If C's variance is much
                          smaller, a common-mode component exists and C is
                          overconfident -- prefer B. If they agree, independence
                          holds and C's precision is trustworthy.

Resampling unit is the PROMPT (a prompt contributes clips to several dimensions
-- e.g. scene, appearance_style, plus the consistency dims -- and the pass
criterion is the macro mean over dimensions). Picking one run per prompt keeps
each prompt's cross-dimension scores jointly, only breaking cross-prompt links.

Directory selection supports shell-style wildcards (expanded here too, so quote
them to let this tool do recursive ``**`` matching):

    python3 vbench_pass_probability.py 'runs/wan22/*__myconfig/SingleStream/accuracy'
    python3 vbench_pass_probability.py 'runs/wan22/**/Offline/accuracy'
    python3 vbench_pass_probability.py run1/.../accuracy run2/.../accuracy ...

Each match must resolve to an accuracy dir (…/<Scenario>/accuracy) or a vbench/
dir containing ``results_*_eval_results.json`` with per-video records.

Every pass-probability point estimate is reported with a confidence interval on
the probability itself (not just on the score): a Wilson interval for the
empirical proportion (A), a bootstrap over the run scores for the whole-run
predictive probability (B), and a leave-one-run-out jackknife for the per-prompt
bootstrap probability (C). With only a handful of replicate runs these are wide
on purpose -- that width IS the small-sample uncertainty.

Options:
    --threshold FLOAT   Pass threshold on the 0-100 scale (default: 99% of ref).
    --bootstrap N       Synthetic runs for estimator C (default 20000).
    --pass-ci-bootstrap N  Run-score resamples for the B pass-prob CI (default 10000).
    --ci PCT            Confidence level for reported intervals (default 95).
    --seed N            RNG seed (default 1234).
    --dimension DIM     Restrict to these dimensions (repeatable).
    --json PATH         Write the structured result to PATH.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import logging
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.vbench_viz import (  # noqa: E402
    load_dimension_series,
    resolve_vbench_run,
)
from wan_harness.vbench import (  # noqa: E402
    ACCURACY_THRESHOLD_99,
    REFERENCE_ACCURACY,
)

_log = logging.getLogger("vbench_pass_probability")

_INDEX_SUFFIX_RE = re.compile(r"-\d+$")
_WILDCARD_CHARS = set("*?[")


def _prompt_of(video_key: str) -> str:
    stem = video_key.rsplit(".", 1)[0]
    return _INDEX_SUFFIX_RE.sub("", stem)


# ----------------------------------------------------------------------
# Small stats helpers (no scipy dependency).
# ----------------------------------------------------------------------


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: float) -> float:
    """CDF of Student's t with df degrees of freedom."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    half = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - half if t > 0 else half


def _z_for_ci(ci_level: float) -> float:
    target = 1.0 - (1.0 - ci_level / 100.0) / 2.0
    lo, hi = 0.0, 8.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _normal_cdf(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo_i = int(pos)
    hi_i = min(lo_i + 1, len(sorted_vals) - 1)
    frac = pos - lo_i
    return sorted_vals[lo_i] * (1 - frac) + sorted_vals[hi_i] * frac


def _wilson_interval(k: int, n: int, ci_level: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n.

    Better-behaved than the normal (Wald) interval at small n and near 0/1,
    which is exactly the regime here (a handful of runs, often all passing).
    """
    if n <= 0:
        return float("nan"), float("nan")
    z = _z_for_ci(ci_level)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _pass_frac(scores: list[float], threshold: float) -> float:
    if not scores:
        return float("nan")
    return sum(1 for s in scores if s >= threshold) / len(scores)


def _predictive_pass_prob(mean: float, sd: float, threshold: float, R: int) -> float:
    """P(a single new run >= threshold) under the Student-t predictive."""
    if sd > 0:
        scale = sd * math.sqrt(1.0 + 1.0 / R)
        return _t_cdf((mean - threshold) / scale, df=R - 1)
    return 1.0 if mean >= threshold else 0.0


def _bootstrap_pred_pass_ci(
    run_scores: list[float],
    *,
    threshold: float,
    ci_level: float,
    n_ci: int,
    rng: random.Random,
) -> tuple[float, float]:
    """Percentile CI for the whole-run predictive pass probability (B).

    Resamples the R run scores with replacement, refits (mean, sd) and
    recomputes the Student-t predictive pass probability each time; the spread
    of those probabilities is the estimation uncertainty in B's point estimate.
    """
    R = len(run_scores)
    if R < 2:
        return float("nan"), float("nan")
    probs: list[float] = []
    for _ in range(n_ci):
        sample = [run_scores[rng.randrange(R)] for _ in range(R)]
        m = statistics.mean(sample)
        s = statistics.stdev(sample)
        probs.append(_predictive_pass_prob(m, s, threshold, R))
    probs.sort()
    alpha = (1.0 - ci_level / 100.0) / 2.0
    return _percentile(probs, alpha), _percentile(probs, 1.0 - alpha)


# ----------------------------------------------------------------------
# Run loading.
# ----------------------------------------------------------------------


@dataclass
class RunData:
    label: str
    run_dir: Path
    results_file: Path
    # prompt -> dim -> list of clip scores for that prompt in that dim
    prompt_dim: dict[str, dict[str, list[float]]]
    dims: list[str]

    def dim_mean(self, dim: str) -> float:
        clips = [c for pd in self.prompt_dim.values() for c in pd.get(dim, ())]
        return statistics.mean(clips) if clips else float("nan")

    def score(self, dims: list[str]) -> float:
        means = [self.dim_mean(d) for d in dims]
        means = [m for m in means if m == m]  # drop NaN
        return 100.0 * statistics.mean(means) if means else float("nan")


def _expand_patterns(patterns: list[str]) -> list[Path]:
    """Expand wildcard patterns into concrete directory paths (deduped, sorted)."""
    found: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        if _WILDCARD_CHARS & set(pat):
            matches = sorted(_glob.glob(pat, recursive=True))
            if not matches:
                _log.warning("pattern matched nothing: %s", pat)
            for m in matches:
                p = Path(m)
                if p.is_dir() and p.resolve() not in seen:
                    seen.add(p.resolve())
                    found.append(p)
        else:
            p = Path(pat)
            if p.resolve() not in seen:
                seen.add(p.resolve())
                found.append(p)
    return found


def _load_run(run_dir: Path, wanted: set[str] | None) -> RunData:
    resolved = resolve_vbench_run(run_dir)
    series = list(load_dimension_series(resolved.results_file))
    if wanted:
        series = [d for d in series if d.name in wanted]
    if not series:
        raise ValueError(f"{resolved.results_file}: no usable dimensions")
    prompt_dim: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in series:
        if not d.videos:
            raise ValueError(
                f"{resolved.results_file}: dimension {d.name!r} has no per-video "
                f"records; need a vbench_standard/MLPerf run (custom_input omits them)."
            )
        for v in d.videos:
            prompt_dim[_prompt_of(v.key)][d.name].append(v.score)
    plain = {p: {dm: list(cl) for dm, cl in dd.items()} for p, dd in prompt_dim.items()}
    return RunData(
        label=resolved.label,
        run_dir=resolved.run_dir,
        results_file=resolved.results_file,
        prompt_dim=plain,
        dims=[d.name for d in series],
    )


def _per_prompt_pass(
    runs: list[RunData],
    indices: list[int],
    dims: list[str],
    n_boot: int,
    rng: random.Random,
) -> list[float]:
    """Mix-and-match bootstrap over the runs in ``indices``.

    Builds ``n_boot`` synthetic runs by drawing, for each prompt independently,
    that prompt's clips from one of the runs (restricted to ``indices``) that
    contain it, then scores each synthetic run. Returns the sorted score list.
    """
    prompt_runs: dict[str, list[int]] = defaultdict(list)
    for i in indices:
        for p in runs[i].prompt_dim:
            prompt_runs[p].append(i)
    prompts = sorted(prompt_runs)
    dims_set = set(dims)
    out: list[float] = []
    for _ in range(n_boot):
        dim_clips: dict[str, list[float]] = defaultdict(list)
        for p in prompts:
            avail = prompt_runs[p]
            r_idx = avail[rng.randrange(len(avail))]
            pd = runs[r_idx].prompt_dim.get(p)
            if not pd:
                continue
            for d, clips in pd.items():
                if d in dims_set:
                    dim_clips[d].extend(clips)
        means = [statistics.mean(dim_clips[d]) for d in dims if dim_clips.get(d)]
        if means:
            out.append(100.0 * statistics.mean(means))
    out.sort()
    return out


# ----------------------------------------------------------------------
# Estimators.
# ----------------------------------------------------------------------


@dataclass
class Result:
    labels: list[str]
    run_dirs: list[Path]
    dimensions: list[str]
    threshold: float
    ci_level: float
    n_runs: int
    n_prompts: int
    # per-run
    run_scores: list[float]
    point_score: float
    dim_means: dict[str, float]
    dim_run_std: dict[str, float]
    # A empirical
    emp_pass: int
    emp_frac: float
    emp_ci_lo: float      # Wilson interval on the pass probability
    emp_ci_hi: float
    # B whole-run predictive
    mean_score: float
    sd_total: float
    p_normal: float
    p_t: float
    p_t_ci_lo: float      # bootstrap CI on the predictive pass probability
    p_t_ci_hi: float
    pred_lo: float        # predictive interval for a new run's SCORE
    pred_hi: float
    # C per-prompt bootstrap
    boot_mean: float
    sd_indep: float
    p_boot: float
    p_boot_ci_lo: float   # jackknife CI on the pass probability
    p_boot_ci_hi: float
    p_boot_se: float
    boot_lo: float        # percentile interval on the synthetic SCORE
    boot_hi: float
    n_boot: int
    universal_prompts: bool
    # D independence check
    var_ratio: float  # var_indep / var_total

    def to_json(self) -> dict:
        return {
            "labels": self.labels,
            "run_dirs": [str(p) for p in self.run_dirs],
            "dimensions": self.dimensions,
            "threshold": self.threshold,
            "ci_level": self.ci_level,
            "n_runs": self.n_runs,
            "n_prompts": self.n_prompts,
            "run_scores": self.run_scores,
            "point_score": self.point_score,
            "dimension_means": self.dim_means,
            "dimension_run_std": self.dim_run_std,
            "empirical": {
                "n_pass": self.emp_pass,
                "pass_fraction": self.emp_frac,
                "pass_prob_ci_lo": self.emp_ci_lo,
                "pass_prob_ci_hi": self.emp_ci_hi,
            },
            "whole_run_predictive": {
                "mean": self.mean_score,
                "sd_total": self.sd_total,
                "pass_prob_normal": self.p_normal,
                "pass_prob_t": self.p_t,
                "pass_prob_t_ci_lo": self.p_t_ci_lo,
                "pass_prob_t_ci_hi": self.p_t_ci_hi,
                "predictive_score_ci_lo": self.pred_lo,
                "predictive_score_ci_hi": self.pred_hi,
            },
            "per_prompt_bootstrap": {
                "mean": self.boot_mean,
                "sd_indep": self.sd_indep,
                "pass_prob": self.p_boot,
                "pass_prob_ci_lo": self.p_boot_ci_lo,
                "pass_prob_ci_hi": self.p_boot_ci_hi,
                "pass_prob_se": self.p_boot_se,
                "score_ci_lo": self.boot_lo,
                "score_ci_hi": self.boot_hi,
                "n_bootstrap": self.n_boot,
                "universal_prompts": self.universal_prompts,
            },
            "independence_check": {"variance_ratio_indep_over_total": self.var_ratio},
        }


def estimate(
    run_dirs: list[Path],
    *,
    threshold: float,
    ci_level: float,
    n_boot: int,
    n_pred_ci: int,
    seed: int,
    dimensions: list[str] | None,
) -> Result:
    wanted = set(dimensions) if dimensions else None
    runs: list[RunData] = []
    for rd in run_dirs:
        try:
            runs.append(_load_run(rd, wanted))
        except (FileNotFoundError, ValueError) as exc:
            _log.warning("skipping %s: %s", rd, exc)
    if len(runs) < 2:
        raise ValueError(
            f"need >= 2 replicate runs with VBench results to estimate run-to-run "
            f"pass probability (usable={len(runs)}); this variance cannot come "
            f"from a single run."
        )

    # Dimensions common to every run (the aggregate is a macro mean over these).
    common = set(runs[0].dims)
    for r in runs[1:]:
        common &= set(r.dims)
    if not common:
        raise ValueError("runs share no common dimensions")
    dims = [d for d in runs[0].dims if d in common]

    run_scores = [r.score(dims) for r in runs]
    R = len(runs)
    rng = random.Random(seed)

    # per-dimension across-run std (diagnostic) + averaged dim means.
    dim_means: dict[str, float] = {}
    dim_run_std: dict[str, float] = {}
    for d in dims:
        per_run = [r.dim_mean(d) for r in runs]
        dim_means[d] = statistics.mean(per_run)
        dim_run_std[d] = statistics.stdev(per_run) if R >= 2 else 0.0
    point_score = 100.0 * statistics.mean(dim_means[d] for d in dims)

    # --- A: empirical (+ Wilson CI on the pass probability) ---
    emp_pass = sum(1 for s in run_scores if s >= threshold)
    emp_frac = emp_pass / R
    emp_ci_lo, emp_ci_hi = _wilson_interval(emp_pass, R, ci_level)

    # --- B: whole-run predictive (+ bootstrap CI on the pass probability) ---
    mean_score = statistics.mean(run_scores)
    sd_total = statistics.stdev(run_scores) if R >= 2 else 0.0
    if sd_total > 0:
        p_normal = _normal_cdf((mean_score - threshold) / sd_total)
        pred_scale = sd_total * math.sqrt(1.0 + 1.0 / R)
        p_t = _t_cdf((mean_score - threshold) / pred_scale, df=R - 1)
        # predictive interval for a single new run's SCORE (t-based)
        t_mult = _t_ppf(1.0 - (1.0 - ci_level / 100.0) / 2.0, df=R - 1)
        pred_lo = mean_score - t_mult * pred_scale
        pred_hi = mean_score + t_mult * pred_scale
        p_t_ci_lo, p_t_ci_hi = _bootstrap_pred_pass_ci(
            run_scores, threshold=threshold, ci_level=ci_level, n_ci=n_pred_ci, rng=rng,
        )
    else:
        p_normal = p_t = 1.0 if mean_score >= threshold else 0.0
        pred_lo = pred_hi = mean_score
        p_t_ci_lo = p_t_ci_hi = p_t

    # --- C: per-prompt (independence) bootstrap ---
    # prompt -> list of run indices that contain it
    prompt_runs: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(runs):
        for p in r.prompt_dim:
            prompt_runs[p].append(i)
    prompts = sorted(prompt_runs)
    n_prompts = len(prompts)
    universal = all(len(v) == R for v in prompt_runs.values())
    if not universal:
        miss = sum(1 for v in prompt_runs.values() if len(v) != R)
        _log.warning(
            "%d/%d prompts are not present in all %d runs; bootstrap draws each "
            "prompt among the runs that have it.", miss, n_prompts, R,
        )

    boot_scores = _per_prompt_pass(runs, list(range(R)), dims, n_boot, rng)
    boot_mean = statistics.mean(boot_scores) if boot_scores else float("nan")
    sd_indep = statistics.stdev(boot_scores) if len(boot_scores) >= 2 else 0.0
    p_boot = _pass_frac(boot_scores, threshold)
    alpha = (1.0 - ci_level / 100.0) / 2.0
    boot_lo = _percentile(boot_scores, alpha)
    boot_hi = _percentile(boot_scores, 1.0 - alpha)

    # CI on C's pass probability via a leave-one-run-out (cluster) jackknife:
    # the runs are the resampling clusters, so dropping each in turn measures
    # how much the finite number of replicate runs moves the estimate. Cheap
    # (R extra bootstraps) and the honest small-R story -- the SE is large when
    # you only have a few runs.
    jk_inner = min(n_boot, 4000)
    p_boot_ci_lo = p_boot_ci_hi = p_boot_se = float("nan")
    if R >= 2 and boot_scores:
        pseudo: list[float] = []
        for i in range(R):
            sub = [j for j in range(R) if j != i]
            pf = _pass_frac(_per_prompt_pass(runs, sub, dims, jk_inner, rng), threshold)
            if pf == pf:  # not NaN
                pseudo.append(pf)
        if len(pseudo) >= 2:
            mean_j = statistics.mean(pseudo)
            var_j = (R - 1) / R * sum((x - mean_j) ** 2 for x in pseudo)
            p_boot_se = math.sqrt(var_j)
            zc = _z_for_ci(ci_level)
            p_boot_ci_lo = max(0.0, p_boot - zc * p_boot_se)
            p_boot_ci_hi = min(1.0, p_boot + zc * p_boot_se)

    # --- D: independence check ---
    var_total = sd_total ** 2
    var_indep = sd_indep ** 2
    var_ratio = (var_indep / var_total) if var_total > 0 else float("nan")

    return Result(
        labels=[r.label for r in runs],
        run_dirs=[r.run_dir for r in runs],
        dimensions=dims,
        threshold=threshold,
        ci_level=ci_level,
        n_runs=R,
        n_prompts=n_prompts,
        run_scores=run_scores,
        point_score=point_score,
        dim_means=dim_means,
        dim_run_std=dim_run_std,
        emp_pass=emp_pass,
        emp_frac=emp_frac,
        emp_ci_lo=emp_ci_lo,
        emp_ci_hi=emp_ci_hi,
        mean_score=mean_score,
        sd_total=sd_total,
        p_normal=p_normal,
        p_t=p_t,
        p_t_ci_lo=p_t_ci_lo,
        p_t_ci_hi=p_t_ci_hi,
        pred_lo=pred_lo,
        pred_hi=pred_hi,
        boot_mean=boot_mean,
        sd_indep=sd_indep,
        p_boot=p_boot,
        p_boot_ci_lo=p_boot_ci_lo,
        p_boot_ci_hi=p_boot_ci_hi,
        p_boot_se=p_boot_se,
        boot_lo=boot_lo,
        boot_hi=boot_hi,
        n_boot=len(boot_scores),
        universal_prompts=universal,
        var_ratio=var_ratio,
    )


def _t_ppf(q: float, df: float) -> float:
    """Inverse Student-t CDF via bisection (adequate for interval multipliers)."""
    if df <= 0:
        return float("nan")
    lo, hi = -100.0, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# Rendering.
# ----------------------------------------------------------------------


def render(r: Result) -> str:
    bar = "=" * 74
    thin = "-" * 74
    L: list[str] = [bar, "VBench pass-probability from replicate runs", bar]
    L.append(f"replicate runs : {r.n_runs}   prompts : {r.n_prompts}   "
             f"dimensions : {len(r.dimensions)}")
    for lbl, rd, sc in zip(r.labels, r.run_dirs, r.run_scores):
        L.append(f"  {sc:8.4f}  {lbl}")
    L.append(f"threshold : {r.threshold:.4f}   point score (mean over runs) : {r.point_score:.4f}")
    L.append("")
    L.append("per-dimension mean / across-run std:")
    for d in r.dimensions:
        L.append(f"  {d:26s} {r.dim_means[d]:.4f}  (run-std {r.dim_run_std[d]:.4f})")
    L.append(thin)

    ci = f"{r.ci_level:.0f}%"
    L.append("[A] empirical whole-run")
    L.append(f"    pass {r.emp_pass}/{r.n_runs} = {r.emp_frac:.3f}   "
             f"(coarse; step size {1.0 / r.n_runs:.2f})")
    L.append(f"    P(pass) {ci} CI (Wilson) [{r.emp_ci_lo:.3f}, {r.emp_ci_hi:.3f}]")
    L.append(thin)

    L.append("[B] whole-run predictive  (Gaussian fit to the run scores; keeps correlation)")
    L.append(f"    mean = {r.mean_score:.4f}   run-to-run sd = {r.sd_total:.4f}")
    L.append(f"    P(new run >= thr): normal = {r.p_normal:.3f}   "
             f"Student-t (df={r.n_runs - 1}) = {r.p_t:.3f}")
    L.append(f"    P(pass) {ci} CI (bootstrap over runs) "
             f"[{r.p_t_ci_lo:.3f}, {r.p_t_ci_hi:.3f}]")
    L.append(f"    {ci} predictive interval for a new run's score "
             f"[{r.pred_lo:.4f}, {r.pred_hi:.4f}]")
    L.append(thin)

    L.append("[C] per-prompt bootstrap  (mix-and-match; assumes prompts independent)")
    L.append(f"    mean = {r.boot_mean:.4f}   indep sd = {r.sd_indep:.4f}   "
             f"n_synth = {r.n_boot}")
    L.append(f"    P(new run >= thr) = {r.p_boot:.3f}")
    if r.p_boot_se == r.p_boot_se:  # not NaN
        L.append(f"    P(pass) {ci} CI (leave-one-run-out jackknife, se={r.p_boot_se:.3f}) "
                 f"[{r.p_boot_ci_lo:.3f}, {r.p_boot_ci_hi:.3f}]")
    L.append(f"    {ci} interval for the synthetic score [{r.boot_lo:.4f}, {r.boot_hi:.4f}]")
    if not r.universal_prompts:
        L.append("    (note: some prompts missing from some runs; see warning above)")
    L.append(thin)

    L.append("[D] independence check  (is the per-prompt model trustworthy?)")
    if r.var_ratio != r.var_ratio:  # NaN
        L.append("    run-to-run variance is ~0 (all runs identical); check N/A.")
    else:
        pct = 100.0 * r.var_ratio
        L.append(f"    indep sd = {r.sd_indep:.4f}  vs  whole-run sd = {r.sd_total:.4f}")
        L.append(f"    independent noise explains {pct:.0f}% of the run-to-run variance")
        if r.var_ratio < 0.5:
            L.append("    -> LARGE common-mode component: [C] is OVERCONFIDENT; prefer [B].")
        elif r.var_ratio < 0.8:
            L.append("    -> some common-mode drift: treat [C] as a lower bound on spread.")
        else:
            L.append("    -> independence roughly holds: [C]'s precision is trustworthy.")
    L.append(bar)
    L.append("Estimates the pass probability of ONE future run under generation noise.")
    L.append("Not the official (deterministic) MLPerf verdict for a fixed video set.")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vbench_pass_probability.py",
        description="Estimate P(vbench run passes) from replicate runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("patterns", nargs="+",
                   help="Run dirs or shell wildcard patterns (quote to enable "
                        "recursive ** matching), each resolving to a "
                        "…/<Scenario>/accuracy or vbench/ dir.")
    p.add_argument("--threshold", type=float, default=ACCURACY_THRESHOLD_99,
                   help=f"Pass threshold on the 0-100 scale (default {ACCURACY_THRESHOLD_99}).")
    p.add_argument("--bootstrap", type=int, default=20000,
                   help="Synthetic runs for the per-prompt bootstrap (default 20000).")
    p.add_argument("--pass-ci-bootstrap", type=int, default=10000, dest="pass_ci_bootstrap",
                   help="Run-score resamples for the [B] pass-probability CI (default 10000).")
    p.add_argument("--ci", type=float, default=95.0,
                   help="Confidence level for reported intervals (default 95).")
    p.add_argument("--seed", type=int, default=1234, help="RNG seed (default 1234).")
    p.add_argument("--dimension", action="append", dest="dimensions", default=None,
                   help="Restrict to this dimension (repeatable).")
    p.add_argument("--json", type=Path, default=None,
                   help="Write structured result to this path.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.bootstrap < 1:
        _log.error("--bootstrap must be >= 1")
        return 2
    if args.pass_ci_bootstrap < 1:
        _log.error("--pass-ci-bootstrap must be >= 1")
        return 2

    dirs = _expand_patterns(args.patterns)
    if not dirs:
        _log.error("no directories matched the given patterns")
        return 2
    _log.info("reference=%.2f  threshold=%.4f  runs=%d",
              REFERENCE_ACCURACY, args.threshold, len(dirs))
    for d in dirs:
        _log.info("  run: %s", d)

    try:
        res = estimate(
            dirs,
            threshold=args.threshold,
            ci_level=args.ci,
            n_boot=args.bootstrap,
            n_pred_ci=args.pass_ci_bootstrap,
            seed=args.seed,
            dimensions=args.dimensions,
        )
    except (FileNotFoundError, ValueError) as exc:
        _log.error("%s", exc)
        return 2

    sys.stdout.write(render(res))
    sys.stdout.flush()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res.to_json(), indent=2) + "\n", encoding="utf-8")
        _log.info("wrote %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
