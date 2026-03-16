from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Dict, List, Optional, Set, Tuple

from mawarith_benchmark.prediction.heirs import HEIRS
# ============================================================
# Evaluation tolerances
# ============================================================

# Tolerance for numeric share comparison (fractions in [0,1])
SHARE_EPS = 0.1

# Tolerance for final per-head percentage comparison
FINAL_EPS = 0.1


# ============================================================
# Arabic normalization utilities
# ============================================================

_TASHKEEL = re.compile(r"[\u064B-\u0652\u0670]")
_WS = re.compile(r"\s+")


def normalize_ar(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ى", "ي")
    s = s.replace("ـ", "")
    s = _TASHKEEL.sub("", s)
    s = _WS.sub(" ", s)
    return s




# ============================================================
# Heir name canonicalization (robust)
# ============================================================

def _heir_key(s: Any) -> str:
    """
    Internal normalized key used for matching heir names across gold/pred.
    Unifies variants such as:
      الأب / اب / أب  -> اب
      الأم / ام / أم  -> ام
      الأخ / اخ / أخ  -> اخ
      الأخت / اخت / أخت -> اخت
      الابن / ابن -> ابن
    Also strips leading 'ال' token-wise.
    """
    if not isinstance(s, str):
        return ""

    s = normalize_ar(s)  # IMPORTANT: now أب -> اب, أم -> ام, ...
    tokens = s.split()
    out = []

    for t in tokens:
        # strip leading definite article
        if t.startswith("ال") and len(t) > 2:
            t = t[2:]

        # lexical unification after normalize_ar
        # (normalize_ar already removed hamza on alif, so use hamza-less forms here)
        if t in {"اب"}:
            out.append("اب")
        elif t in {"ام"}:
            out.append("ام")
        elif t in {"اخ"}:
            out.append("اخ")
        elif t in {"اخت"}:
            out.append("اخت")
        elif t in {"ابن"}:
            out.append("ابن")
        else:
            out.append(t)

    return " ".join(out).strip()


# Build alias map once from official HEIRS list:
# key (normalized internal) -> official string from HEIRS
_HEIR_KEY_TO_OFFICIAL: Dict[str, str] = {}
for _h in HEIRS:
    k = _heir_key(_h)
    if k and k not in _HEIR_KEY_TO_OFFICIAL:
        _HEIR_KEY_TO_OFFICIAL[k] = _h


def canon_heir_name(h: Any) -> Optional[str]:
    """
    Return a canonical heir name.
    - Prefer the official HEIRS form when we can map to it.
    - Otherwise return the internal normalized key.
    """
    if not isinstance(h, str):
        return None

    key = _heir_key(h)
    if not key:
        return None

    return _HEIR_KEY_TO_OFFICIAL.get(key, key)






def round4(x: Optional[float]) -> Optional[float]:
    """Round numeric values to 4 decimal places for logging."""
    return round(x, 4) if isinstance(x, (int, float)) else None


# ============================================================
# Fraction parsing and share normalization
# ============================================================

def _parse_fraction_str(s: str) -> Optional[Fraction]:
    """
    Parse a fractional string of the form 'a/b' into a Fraction object.
    """
    m = re.search(r"(\d+)\s*/\s*(\d+)", s)
    if not m:
        return None
    num, den = int(m.group(1)), int(m.group(2))
    return Fraction(num, den) if den != 0 else None

def _extract_fraction_str(x: Any) -> Optional[str]:
    """
    Robust extraction of fraction-like values from heterogeneous formats.
    Supported inputs: str, int, float, list, dict.
    """
    if isinstance(x, str):
        return x.strip() or None
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, list):
        for v in x:
            s = _extract_fraction_str(v)
            if s:
                return s
    if isinstance(x, dict):
        for k in ("fraction", "value", "raw"):
            if k in x:
                s = _extract_fraction_str(x[k])
                if s:
                    return s
    return None

def _normalize_share(raw: Any) -> Tuple[Optional[float], str]:
    """
    Normalize a share expression into a numeric value or symbolic type.

    Returns:
        (value, kind) where kind ∈ {'value', 'remainder', 'all', 'bad'}
    """
    if not isinstance(raw, str):
        return None, "bad"

    s = normalize_ar(raw)

    if s == normalize_ar("كل التركة"):
        return 1.0, "all"

    if s == normalize_ar("باقي التركة"):
        return None, "remainder"

    frac = _parse_fraction_str(s)
    if frac is not None:
        return float(frac), "value"

    try:
        v = float(s)
        if 0.0 <= v <= 1.0:
            return v, "value"
    except Exception:
        pass

    return None, "bad"


# ============================================================
# Heir extraction helpers
# ============================================================

def _extract_heirs_with_counts(lst: Any) -> Dict[str, int]:
    """
    Extract heirs and their multiplicities from a list representation.
    """
    out: Dict[str, int] = {}
    if not isinstance(lst, list):
        return out
    for it in lst:
        if not isinstance(it, dict):
            continue
        h = canon_heir_name(it.get("heir"))
        c = it.get("count")
        if h and isinstance(c, int):
            out[h] = c
    return out

def _extract_heir_names(lst: Any) -> Set[str]:
    """
    Extract a set of heir names from a list (used for blocking relations).
    """
    if not isinstance(lst, list):
        return set()
    return {
        canon_heir_name(it.get("heir"))
        for it in lst
        if isinstance(it, dict) and canon_heir_name(it.get("heir"))
    }


# ============================================================
# Heirs identification and blocking evaluation
# ============================================================

def score_heirs_blocked(
    gold: Dict[str, Any],
    pred: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Score the identification of heirs and blocking relations.
    Combines name-level F1, count accuracy, and blocking penalties.
    """
    gold_counts = _extract_heirs_with_counts(gold.get("heirs"))
    pred_counts = _extract_heirs_with_counts(pred.get("heirs"))

    gold_blocked = _extract_heir_names(gold.get("blocked"))
    pred_blocked = _extract_heir_names(pred.get("blocked"))

    gold_effective = set(gold_counts)
    pred_effective = set(pred_counts) - pred_blocked
    inter = gold_effective & pred_effective

    if not gold_effective and not pred_effective:
        f1_name = 1.0
    elif not gold_effective or not pred_effective:
        f1_name = 0.0
    else:
        f1_name = 2 * len(inter) / (len(gold_effective) + len(pred_effective))

    correct_count = sum(1 for h in inter if gold_counts[h] == pred_counts[h])
    acc_count = (correct_count / len(inter)) if inter else 1.0

    step1 = 0.7 * f1_name + 0.3 * acc_count

    added = sorted(pred_effective & gold_blocked)
    wrongly_blocked = sorted(pred_blocked & gold_effective)

    penalty = len(added) + len(wrongly_blocked)
    denom = len(gold_effective) or 1
    step2 = max(0.0, 1.0 - penalty / denom)

    score = 0.6 * step1 + 0.4 * step2

    return score, {
        "gold_names": sorted(gold_effective),
        "pred_names": sorted(pred_effective),
        "gold_blocked": sorted(gold_blocked),
        "pred_blocked": sorted(pred_blocked),
        "pred_effective_after_blocking": sorted(pred_effective),
        "missing_names": sorted(gold_effective - pred_effective),
        "spurious_names": sorted(pred_effective - gold_effective),
        "gold_counts": gold_counts,
        "pred_counts": pred_counts,
        "added_eligible_should_be_blocked": added,
        "wrongly_blocked_should_be_eligible": wrongly_blocked,
        "penalty_count": penalty,
        "score": round4(score),
    }






def score_shares(
    gold: Dict[str, Any],
    pred: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Accept a prediction as correct if it matches ANY gold representation:
      - gold.fraction OR gold.heir_fraction (when provided)

    Matching rules:
      - symbolic match for remainder / full estate
      - numeric comparison for fractional shares with tolerance
      - EXTRA equivalences:
          * all <-> value==1.0
          * remainder <-> computed remainder value (from gold fixed shares)
    """

    # -------- GOLD: heir -> list of acceptable raw strings --------
    gold_raw_cands: Dict[str, List[str]] = {}

    for it in gold.get("shares", []) or []:
        if not isinstance(it, dict):
            continue
        h = canon_heir_name(it.get("heir"))
        if not h:
            continue

        cands: List[str] = []
        for key in ("fraction", "heir_fraction"):
            s = _extract_fraction_str(it.get(key))
            if s:
                cands.append(normalize_ar(s))

        if cands:
            gold_raw_cands.setdefault(h, []).extend(cands)

    # -------- PRED: heir -> list of predicted raw strings --------
    pred_raw: Dict[str, List[str]] = {}

    for it in pred.get("shares", []) or []:
        if not isinstance(it, dict):
            continue
        h = canon_heir_name(it.get("heir"))
        if not h:
            continue

        cands: List[str] = []
        for key in ("fraction", "heir_fraction"):
            s = _extract_fraction_str(it.get(key))
            if s:
                cands.append(normalize_ar(s))

        if cands:
            pred_raw.setdefault(h, []).extend(cands)

    # -------- helper: compute remainder from gold fixed shares --------
    def _compute_gold_remainder_for(h_target: str) -> Optional[float]:
        """
        Compute remainder = 1 - sum(other fixed shares), using gold fractions.
        Only uses other heirs with at least one parseable numeric 'value' share.
        Ignores heirs that are 'remainder' or 'all' in their candidates.
        Returns None if not enough info.
        """
        total = 0.0
        used_any = False

        for h2, cands2 in gold_raw_cands.items():
            if h2 == h_target:
                continue

            # pick first numeric "value" candidate from this heir (if any)
            best_val: Optional[float] = None
            for raw2 in cands2:
                v2, k2 = _normalize_share(raw2)
                if k2 == "value" and v2 is not None:
                    best_val = v2
                    break
                # if heir itself is remainder/all in gold, skip it entirely
                if k2 in {"remainder", "all"}:
                    best_val = None
                    break

            if best_val is not None:
                total += best_val
                used_any = True

        if not used_any:
            return None

        rem = 1.0 - total
        # keep within [0,1] with tiny tolerance
        if rem < -1e-6 or rem > 1.0 + 1e-6:
            return None
        return max(0.0, min(1.0, rem))

    # -------- SCORING per heir --------
    per_heir: Dict[str, Any] = {}
    scores: List[float] = []

    for h, gold_cands in gold_raw_cands.items():
        sc = 0.0
        matched_gold_raw = None
        matched_pred_raw = None
        matched_pred_val = None

        gold_remainder_val = _compute_gold_remainder_for(h)

        # Try all (gold_candidate, pred_candidate) pairs
        for graw in gold_cands:
            gv, gkind = _normalize_share(graw)

            for praw in (pred_raw.get(h) or []):
                pv, pkind = _normalize_share(praw)

                # (1) exact symbolic matches
                if gkind == pkind == "remainder":
                    sc = 1.0
                    matched_gold_raw = graw
                    matched_pred_raw = praw
                    break

                if gkind == pkind == "all":
                    sc = 1.0
                    matched_gold_raw = graw
                    matched_pred_raw = praw
                    matched_pred_val = 1.0
                    break

                # (2) numeric match
                if gkind == pkind == "value" and gv is not None and pv is not None:
                    if abs(pv - gv) <= SHARE_EPS:
                        sc = 1.0
                        matched_gold_raw = graw
                        matched_pred_raw = praw
                        matched_pred_val = pv
                        break

                # --------------------------------------------------
                # EXTRA equivalence A: all <-> value==1.0
                # --------------------------------------------------
                if gv is not None:
                    # gold value 1.0 vs pred all
                    if gkind == "value" and abs(gv - 1.0) <= SHARE_EPS and pkind == "all":
                        sc = 1.0
                        matched_gold_raw = graw
                        matched_pred_raw = praw
                        matched_pred_val = 1.0
                        break

                if pv is not None:
                    # gold all vs pred value 1.0
                    if gkind == "all" and pkind == "value" and abs(pv - 1.0) <= SHARE_EPS:
                        sc = 1.0
                        matched_gold_raw = graw
                        matched_pred_raw = praw
                        matched_pred_val = pv
                        break

                # --------------------------------------------------
                # EXTRA equivalence B: remainder <-> computed remainder value
                # pred says "باقي التركة", gold gives numeric remainder (e.g., 0.375)
                # --------------------------------------------------
                if pkind == "remainder" and gkind == "value" and gv is not None and gold_remainder_val is not None:
                    if abs(gv - gold_remainder_val) <= SHARE_EPS:
                        sc = 1.0
                        matched_gold_raw = graw
                        matched_pred_raw = praw
                        matched_pred_val = gold_remainder_val
                        break

                # (optional symmetric) gold says remainder but pred provides numeric remainder
                if gkind == "remainder" and pkind == "value" and pv is not None and gold_remainder_val is not None:
                    if abs(pv - gold_remainder_val) <= SHARE_EPS:
                        sc = 1.0
                        matched_gold_raw = graw
                        matched_pred_raw = praw
                        matched_pred_val = pv
                        break

            if sc == 1.0:
                break

        entry = {
            "gold_raw": gold_cands[0] if gold_cands else None,
            "gold_raw_candidates": list(dict.fromkeys(gold_cands)),
            "pred_raw_candidates": pred_raw.get(h),
            "matched_gold_raw": matched_gold_raw,
            "matched_pred_raw": matched_pred_raw,
        }

        if matched_pred_val is not None:
            entry["matched_pred_val"] = round4(matched_pred_val)

        entry["score"] = sc  # score ALWAYS last

        per_heir[h] = entry
        scores.append(sc)

    score = sum(scores) / len(scores) if scores else 1.0

    return score, {
        "per_heir": per_heir,
        "score": round4(score),
    }

# ============================================================
# Shares evaluation
# ============================================================

def score_shares2(
    gold: Dict[str, Any],
    pred: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Accept a prediction as correct if it matches ANY gold representation:
      - gold.fraction OR gold.heir_fraction (when provided)

    Matching rules:
      - symbolic match for remainder / full estate
      - numeric comparison for fractional shares with tolerance
    """

    # -------- GOLD: heir -> list of acceptable raw strings --------
    gold_raw_cands: Dict[str, List[str]] = {}

    for it in gold.get("shares", []) or []:
        if not isinstance(it, dict):
            continue
        h = canon_heir_name(it.get("heir"))
        if not h:
            continue

        cands: List[str] = []
        for key in ("fraction", "heir_fraction"):
            s = _extract_fraction_str(it.get(key))
            if s:
                cands.append(normalize_ar(s))

        if cands:
            gold_raw_cands.setdefault(h, []).extend(cands)

    # -------- PRED: heir -> list of predicted raw strings --------
    pred_raw: Dict[str, List[str]] = {}

    for it in pred.get("shares", []) or []:
        if not isinstance(it, dict):
            continue
        h = canon_heir_name(it.get("heir"))
        if not h:
            continue

        cands: List[str] = []
        for key in ("fraction", "heir_fraction"):
            s = _extract_fraction_str(it.get(key))
            if s:
                cands.append(normalize_ar(s))

        if cands:
            pred_raw.setdefault(h, []).extend(cands)

    # -------- SCORING per heir --------
    per_heir: Dict[str, Any] = {}
    scores: List[float] = []

    for h, gold_cands in gold_raw_cands.items():
        sc = 0.0
        matched_gold_raw = None
        matched_pred_raw = None
        matched_pred_val = None

        # Try all (gold_candidate, pred_candidate) pairs
        for graw in gold_cands:
            gv, gkind = _normalize_share(graw)

            for praw in (pred_raw.get(h) or []):
                pv, pkind = _normalize_share(praw)

                # remainder
                if gkind == pkind == "remainder":
                    sc = 1.0
                    matched_gold_raw = graw
                    matched_pred_raw = praw
                    break

                # all estate
                if gkind == pkind == "all":
                    sc = 1.0
                    matched_gold_raw = graw
                    matched_pred_raw = praw
                    matched_pred_val = 1.0
                    break

                # numeric value
                if gkind == pkind == "value" and gv is not None and pv is not None:
                    if abs(pv - gv) <= SHARE_EPS:
                        sc = 1.0
                        matched_gold_raw = graw
                        matched_pred_raw = praw
                        matched_pred_val = pv
                        break

            if sc == 1.0:
                break

        # ---- logging per heir ----
        entry = {
            "gold_raw": gold_cands[0] if gold_cands else None,
            "gold_raw_candidates": list(dict.fromkeys(gold_cands)),
            "pred_raw_candidates": pred_raw.get(h),
            "matched_gold_raw": matched_gold_raw,
            "matched_pred_raw": matched_pred_raw,
        }

        if matched_pred_val is not None:
            entry["matched_pred_val"] = round4(matched_pred_val)

        # score ALWAYS last
        entry["score"] = sc

        per_heir[h] = entry
        scores.append(sc)

    # -------- GLOBAL SCORE --------
    score = sum(scores) / len(scores) if scores else 1.0

    return score, {
        "per_heir": per_heir,
        "score": round4(score),
    }



def score_awl(
    gold: Dict[str, Any],
    pred: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Score the identification of awl / radd cases.

    Rule (final):
    - This function is called ONLY if shares_score == 1.0
    - If gold.awl_or_radd == pred.awl_or_radd
      => awl_score = 1.0
    - total_shares is IGNORED here (checked in final step)
    """

    g = normalize_ar(gold.get("awl_or_radd"))
    p = normalize_ar(pred.get("awl_or_radd"))

    def is_no(x: Any) -> bool:
        return x in {"", "لا", "no", None}

    g_norm = "no" if is_no(g) else g
    p_norm = "no" if is_no(p) else p

    if g_norm == p_norm:
        return 1.0, {
            "gold_awl": g_norm,
            "pred_awl": p_norm,
            "score": 1.0,
        }

    return 0.0, {
        "gold_awl": g_norm,
        "pred_awl": p_norm,
        "score": 0.0,
    }


# ============================================================
# Final distribution evaluation
# ============================================================

def extract_final_percent_map(obj: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract final per-head percentage distribution after normalization.
    """
    dist = obj.get("post_tasil", {}).get("distribution")
    if not isinstance(dist, list):
        return {}

    out: Dict[str, float] = {}
    for it in dist:
        if not isinstance(it, dict):
            continue
        h = canon_heir_name(it.get("heir"))
        pc = it.get("per_head_percent")
        if h and isinstance(pc, (int, float)):
            out[h] = float(pc)
    return out

def score_final_distribution(
    gold: Dict[str, Any],
    pred: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Score final per-head percentage distribution for each heir.
    """
    gold_map = extract_final_percent_map(gold)
    pred_map = extract_final_percent_map(pred)

    per_heir: Dict[str, Any] = {}
    scores: List[float] = []

    for h, gv in gold_map.items():
        pv = pred_map.get(h)
        sc = 1.0 if (pv is not None and abs(pv - gv) <= FINAL_EPS) else 0.0

        per_heir[h] = {
            "gold_percent": round4(gv),
            "pred_percent": round4(pv) if pv is not None else None,
            "score": sc,
        }
        scores.append(sc)

    score = (sum(scores) / len(scores)) if scores else 0.0

    return score, {
        "gold_final": {k: round4(v) for k, v in gold_map.items()},
        "pred_final": {k: round4(v) for k, v in pred_map.items()},
        "per_heir": per_heir,
        "score": round4(score),
    }


# ============================================================
# MIR-E aggregation
# ============================================================

WEIGHTS = {
    "heirs_blocked": 0.30,
    "shares": 0.30,
    "awl": 0.10,
    "final": 0.30,
}

def compute_mire(gold: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    MIR-E: Multi-step Inheritance Reasoning Evaluation.
    Aggregates step-level scores using fixed weights.
    """
    s_hb, d_hb = score_heirs_blocked(gold, pred)
    s_sh, d_sh = score_shares(gold, pred)

    # Awl is only evaluated if shares are fully correct
    if s_sh < 1.0:
        s_aw, d_aw = 0.0, {"skipped_due_to": "previous_step_error"}
    else:
        s_aw, d_aw = score_awl(gold, pred)

    s_fi, d_fi = score_final_distribution(gold, pred)

    mire = (
        WEIGHTS["heirs_blocked"] * s_hb +
        WEIGHTS["shares"] * s_sh +
        WEIGHTS["awl"] * s_aw +
        WEIGHTS["final"] * s_fi
    )

    return {
        "MIR-E": round4(mire),
        "subscores": {
            "heirs_blocked": round4(s_hb),
            "shares": round4(s_sh),
            "awl": round4(s_aw),
            "final": round4(s_fi),
        },
        "details": {
            "heirs_blocked": d_hb,
            "shares": d_sh,
            "awl": d_aw,
            "final": d_fi,
        },
    }
