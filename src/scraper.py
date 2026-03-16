"""
Scraper for almwareeth.com — fills the inheritance form and parses results.

Flow:
  1. Navigate to /new_masaala
  2. Set gender + heir dropdowns
  3. Submit → /tagheez
  4. Click "تقسيم تفصيلى"
  5. Parse /masaala/[hash] results page

Usage:
    from src.scraper import solve_with_scraper
    result = solve_with_scraper(heirs_dict)
"""

import re
import time
from dataclasses import dataclass, field
from playwright.sync_api import sync_playwright, Page, Browser

BASE_URL = "https://almwareeth.com"

# ── Heir name → form field mapping ────────────────────────────
# Maps competition heir names to (form_field_id, type)
# type: "bool" = yes/no dropdown, "count" = 1-49 dropdown (needs hover)

HEIR_FIELD_MAP: dict[str, tuple[str, str]] = {
    # Boolean fields
    "زوج":              ("zawg1",    "bool"),
    "أم":               ("om1",      "bool"),
    "أب":               ("ab1",      "bool"),
    "أم الأم":           ("omom1",    "bool"),
    "أب الأب":           ("abab1",    "bool"),
    "أب أب الأب":        ("ababab1",  "bool"),
    "أم أم الأم":        ("omomom1",  "bool"),
    "أم الأب":           ("omab1",    "bool"),
    "أم أب الأب":        ("omabab1",  "bool"),
    "أم أم الأب":        ("omomab1",  "bool"),
    # Count fields
    "زوجة":             ("zawgat",   "count"),
    "ابن":              ("s1",       "count"),
    "بنت":              ("s2",       "count"),
    "ابن ابن":           ("s3",       "count"),
    "بنت ابن":           ("s4",       "count"),
    "ابن ابن ابن":       ("s5",       "count"),
    "بنت ابن ابن":       ("s6",       "count"),
    "أخ شقيق":          ("s7",       "count"),
    "أخت شقيقة":        ("s8",       "count"),
    "أخ لأب":           ("s9",       "count"),
    "أخت لأب":          ("s10",      "count"),
    "أخ لأم":           ("s11",      "count"),
    "أخت لأم":          ("s12",      "count"),
    "ابن أخ شقيق":      ("s13",      "count"),
    "ابن أخ لأب":       ("s14",      "count"),
    "ابن ابن أخ شقيق":   ("s15",      "count"),
    "ابن ابن أخ لأب":    ("s16",      "count"),
    "عم شقيق":          ("s17",      "count"),
    "عم لأب":           ("s18",      "count"),
    "ابن عم شقيق":      ("s19",      "count"),
    "ابن عم لأب":       ("s20",      "count"),
    "ابن ابن عم شقيق":   ("s21",      "count"),
    "ابن ابن عم لأب":    ("s22",      "count"),
    "عم الأب شقيق الجد":  ("s23",      "count"),
    "عم الأب لأب":       ("s24",      "count"),
    "بن عم الأب الشقيق":  ("s25",      "count"),
    "بن عم الأب لأب":    ("s26",      "count"),
}

# Alternative heir name spellings that map to the same fields
HEIR_ALIASES: dict[str, str] = {
    "أب الأب":                "أب الأب",
    "جد":                     "أب الأب",
    "أب أب الأب":              "أب أب الأب",
    "عم الأب الشقيق":     "عم الأب شقيق الجد",
    "عم الأب (أخ الجد للأب)":   "عم الأب لأب",
    "ابن عم الأب الشقيق":     "بن عم الأب الشقيق",
    "بن عم الأب (من الأب)":     "بن عم الأب لأب",
    "عم الأب شقيق":            "عم الأب شقيق الجد",
    "عم الأب أخ الجد للأب":     "عم الأب لأب",
    "ابن عم الأب": "بن عم الأب الشقيق",
    "ابن ابن عم الأب": "ابن ابن عم لأب",
    "ابن عم الأب شقيق الجد": "بن عم الأب الشقيق",
    "عم الأب": "عم الأب شقيق الجد",
    # Deeper heirs not on almwareeth — map to closest available
    "ابن ابن ابن أخ شقيق": "ابن ابن أخ شقيق",
    "ابن ابن ابن أخ لأب": "ابن ابن أخ لأب",
    "ابن ابن ابن عم شقيق": "ابن ابن عم شقيق",
    "ابن ابن ابن عم لأب": "ابن ابن عم لأب",
}


@dataclass
class ScraperResult:
    """Parsed result from almwareeth.com"""
    heirs: list[dict] = field(default_factory=list)
    blocked: list[dict] = field(default_factory=list)
    shares: list[dict] = field(default_factory=list)
    awl_or_radd: str = "لا"
    total_shares: int = 0
    distribution: list[dict] = field(default_factory=list)
    tawdeeh: dict[str, str] = field(default_factory=dict)  # heir_name → explanation
    raw_html: str = ""


def _resolve_heir_name(name: str) -> str:
    """Resolve heir name aliases to canonical form."""
    name = name.strip()
    return HEIR_ALIASES.get(name, name)


def _select_count(page: Page, selector: str, value: str):
    """Hover to populate lazy-loaded options, then select."""
    page.hover(selector)
    time.sleep(0.3)
    page.select_option(selector, value)


def _parse_fraction(td_html: str) -> str:
    """
    Parse fraction from table cell HTML.

    Formats:
      - "1<div class="k">ــــــــ</div>6" → "1/6"
      - "باقى التركة" or "باقي التركة" → "باقي التركة"
      - "—" → blocked (return "—")
    """
    text = re.sub(r"<[^>]+>", " ", td_html).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"ــ+", "/", text)
    text = _strip_kashida(text)

        # Check for blocked
    if text.strip() == "—" or text.strip() == "":
        return "—"

    # Try to extract fraction pattern: "num / denom" — PRIORITY over باقي
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if match:
        return f"{match.group(1)}/{match.group(2)}"

    # Check for "باقى التركة" or "باقي التركة" (only if no fraction found)
    if "باق" in text and "التركة" in text:
        return "باقي التركة"
    return text.strip() or "—"

def _strip_kashida(text: str) -> str:
    """Remove Arabic kashida (tatweel) character ـ from text."""
    return text.replace("\u0640", "")


def _parse_heir_name_and_count(text: str) -> list[tuple[str, int]]:
    """
    Parse heir name(s) and count from cell text.
    Returns a list of (name, count) tuples to handle combined rows.

    Examples:
      "أم الأب" → [("أم الأب", 1)]
      "أخت شقيقة (2)" → [("أخت شقيقة", 2)]
      "ابن عم شقيق(3)" → [("ابن عم شقيق", 3)]
      "ابن (2) و بنت (4)" → [("ابن", 2), ("بنت", 4)]
      "أم أم الأم و أم أم الأب و أم أب الأب" → [("أم أم الأم", 1), ("أم أم الأب", 1), ("أم أب الأب", 1)]
    """
    text = re.sub(r"<[^>]+>", "", text).strip()
    text = _strip_kashida(text)

    # Check for combined names — split on " و " and validate against known heirs
    if " و " in text:
        known_names = set(HEIR_FIELD_MAP.keys()) | set(HEIR_ALIASES.keys())
        segments = text.split(" و ")
        matched = []
        for seg in segments:
            seg = seg.strip()
            # Try matching with count suffix
            m = re.search(r"^(.*?)\s*[\(（](\d+)[\)）]\s*$", seg)
            if m:
                name = m.group(1).strip()
                count = int(m.group(2))
                if name in known_names or _resolve_heir_name(name) in HEIR_FIELD_MAP:
                    matched.append((name, count))
            elif seg in known_names or _resolve_heir_name(seg) in HEIR_FIELD_MAP:
                matched.append((seg, 1))
        # Only use this split if we matched all segments
        if len(matched) == len(segments):
            return matched

    # Single heir with count
    match = re.search(r"^(.*?)\s*[\(（](\d+)[\)）]\s*$", text)
    if match:
        return [(match.group(1).strip(), int(match.group(2)))]

    # Single heir, no count
    return [(text.strip(), 1)]


def _parse_percent(text: str) -> float:
    """Parse percentage like '%16.67' or '16.67%' → 16.67"""
    text = re.sub(r"<[^>]+>", "", text).strip()
    text = text.replace("%", "").strip()
    try:
        return round(float(text), 2)
    except ValueError:
        return 0.0


def fill_form_and_submit(
    page: Page,
    gender: str,
    heirs: dict[str, int],
) -> str:
    """
    Fill the almwareeth.com form and navigate to results page.

    Args:
        page:   Playwright page object
        gender: "ذكر" or "أنثى"
        heirs:  Dict of {heir_name: count}

    Returns:
        URL of the results page (/masaala/[hash])
    """
    # Block ads to prevent networkidle hangs
    page.route("**/*", lambda route: route.abort()
        if any(d in route.request.url for d in [
            "googlesyndication", "googletagmanager", "doubleclick",
            "google-analytics", "adsbygoogle", "pagead",
        ])
        else route.continue_()
    )

    # Navigate to form
    page.goto(f"{BASE_URL}/new_masaala")
    page.wait_for_load_state("load")

    # Set gender
    if gender == "أنثى":
        page.click("input#lll")
        time.sleep(0.3)  # let JS show/hide spouse fields
    else:
        page.click("input#kj")

    # Fill heirs
    for heir_name, count in heirs.items():
        resolved = _resolve_heir_name(heir_name)

        if resolved not in HEIR_FIELD_MAP:
            print(f"    [WARN] Unknown heir: '{heir_name}' (resolved: '{resolved}'), skipping")
            continue

        field_id, field_type = HEIR_FIELD_MAP[resolved]
        selector = f"select#{field_id}"

        if field_type == "bool":
            value = "yes" if count > 0 else "no"
            # Special case: zawg1 uses "1" instead of "yes"
            if field_id == "zawg1":
                value = "1" if count > 0 else "no"
            page.select_option(selector, value)
        else:
            value = str(count) if count > 0 else "no"
            _select_count(page, selector, value)

    # Madhab = الجمهور (default, but be explicit)
    page.click("input#gm")

    # Submit
    page.click("input#insert2")
    page.wait_for_load_state("load")
    time.sleep(2)

    # We're on /tagheez — find and click "تفصيلى"
    detail_link = page.query_selector("a:has-text('تفصيل')")
    if not detail_link:
        detail_link = page.query_selector("a:has-text('تفصيلى')")
    if not detail_link:
        # Try clicking any link that goes to /masaala/
        detail_link = page.query_selector("a[href*='/masaala/']")

    if detail_link:
        detail_link.click()
    else:
        # Maybe the page already redirected or uses JS
        # Try finding a button/input
        detail_btn = page.query_selector("input[value*='تفصيل']")
        if detail_btn:
            detail_btn.click()
        else:
            raise RuntimeError(f"Could not find detailed calculation link on {page.url}")

    page.wait_for_load_state("load")
    time.sleep(2)

    return page.url


def parse_results_page(page: Page) -> ScraperResult:
    """
    Parse the /masaala/[hash] results page.

    Returns:
        ScraperResult with all parsed inheritance data
    """
    result = ScraperResult()
    result.raw_html = page.content()

    # ── Detect awl/radd ───────────────────────────────────
    if "المسألة فيها عول" in result.raw_html:
        result.awl_or_radd = "عول"
    elif "المسألة فيها رد" in result.raw_html:
        result.awl_or_radd = "رد"
    else:
        result.awl_or_radd = "لا"

    # ── Parse main table (shares + blocked) ───────────────
    main_table = page.query_selector("div#gadwalnaseeb table")
    if main_table:
        rows = main_table.query_selector_all("tr")
        for row in rows[1:]:  # skip header
            cells = row.query_selector_all("td")
            if len(cells) < 3:
                continue

            heir_cell = cells[2]  # الوارث column (rightmost in RTL)
            share_cell = cells[1]  # نصيبه column

            heir_html = heir_cell.inner_html()
            share_html = share_cell.inner_html()

            # Extract tawdeeh (explanation) text
            tawdeeh_cell = cells[0]  # التوضيح column
            tawdeeh_el = tawdeeh_cell.query_selector("span.tawdeeh")
            tawdeeh_text = tawdeeh_el.inner_text().strip() if tawdeeh_el else ""
            
            heir_entries = _parse_heir_name_and_count(heir_html)
            fraction = _parse_fraction(share_html)

            is_blocked = "mahgob" in heir_html or fraction == "—"

            # For combined residual rows, get correct counts from distribution table
            # (parsed later), so we defer count fixing after distribution is parsed
            for heir_name, count in heir_entries:
                if not heir_name:
                    continue

                # Store tawdeeh for this heir
                if tawdeeh_text:
                    result.tawdeeh[heir_name] = tawdeeh_text

                if is_blocked:
                    result.blocked.append({"heir": heir_name, "count": count})
                else:
                    result.heirs.append({"heir": heir_name, "count": count})
                    result.shares.append({
                        "heir": heir_name,
                        "count": count,
                        "fraction": fraction,
                    })

    # ── Parse final per-person table ──────────────────────
    all_tables = page.query_selector_all("table")
    final_table = None
    for table in all_tables:
        header = table.inner_text()
        if "نصيب" in header and "مئوي" in header:
            final_table = table
            break

    if final_table:
        rows = final_table.query_selector_all("tr")
        for row in rows[1:]:  # skip header
            cells = row.query_selector_all("td")
            if len(cells) < 4:
                continue

            percent_html = cells[0].inner_html()
            share_html = cells[1].inner_html()
            count_text = cells[2].inner_text().strip()
            heir_html = cells[3].inner_html()

            heir_name = re.sub(r"<[^>]+>", "", heir_html).strip()
            heir_name = _strip_kashida(heir_name)
            per_head_share = _parse_fraction(share_html)
            per_head_percent = _parse_percent(percent_html)

            try:
                count = int(count_text)
            except ValueError:
                count = 1

            result.distribution.append({
                "heir": heir_name,
                "count": count,
                "per_head_shares": per_head_share,
                "per_head_percent": per_head_percent,
            })

    # ── Fix counts for heirs/shares that came from combined rows ──
    # The main table gives count=1 for split heirs, but distribution has correct counts
    dist_counts = {d["heir"]: d["count"] for d in result.distribution}
    for h in result.heirs:
        if h["heir"] in dist_counts:
            h["count"] = dist_counts[h["heir"]]
    for s in result.shares:
        if s["heir"] in dist_counts:
            s["count"] = dist_counts[s["heir"]]

    # ── Extract total_shares ──
    if result.distribution:
        first_share = result.distribution[0]["per_head_shares"]
        match = re.search(r"\d+/(\d+)", first_share)
        if match:
            result.total_shares = int(match.group(1))

    return result


def build_reasoning(result: ScraperResult) -> str:
    """
    Build Arabic reasoning text from scraper results + tawdeeh.
    Credits almwareeth.com as the solver.
    """
    lines = []
    lines.append("تم حل هذه المسألة باستخدام حاسبة المواريث (almwareeth.com) وفق مذهب الجمهور.")
    lines.append("")

    # Inheriting heirs
    if result.shares:
        lines.append("الورثة وأنصبتهم:")
        for share in result.shares:
            heir = share["heir"]
            count = share["count"]
            fraction = share["fraction"]
            heir_label = f"{heir} ({count})" if count > 1 else heir

            tawdeeh = result.tawdeeh.get(heir, "")
            if tawdeeh:
                lines.append(f"- {heir_label}: {fraction} — {tawdeeh}")
            else:
                lines.append(f"- {heir_label}: {fraction}")
        lines.append("")

    # Blocked heirs
    if result.blocked:
        lines.append("المحجوبون:")
        for b in result.blocked:
            heir = b["heir"]
            count = b["count"]
            heir_label = f"{heir} ({count})" if count > 1 else heir

            tawdeeh = result.tawdeeh.get(heir, "")
            if tawdeeh:
                lines.append(f"- {heir_label}: محجوب — {tawdeeh}")
            else:
                lines.append(f"- {heir_label}: محجوب")
        lines.append("")

    # Awl/Radd
    if result.awl_or_radd == "عول":
        lines.append(f"المسألة فيها عول، أصل المسألة بعد العول: {result.total_shares}")
    elif result.awl_or_radd == "رد":
        lines.append(f"المسألة فيها رد، أصل المسألة بعد الرد: {result.total_shares}")
    else:
        lines.append(f"المسألة عادلة (لا عول ولا رد)، أصل المسألة: {result.total_shares}")

    return "\n".join(lines)


def result_to_competition_json(result: ScraperResult) -> dict:
    """Convert ScraperResult to competition answer_structured format."""
    return {
        "reasoning": build_reasoning(result),
        "heirs": result.heirs,
        "blocked": result.blocked,
        "shares": result.shares,
        "awl_or_radd": result.awl_or_radd,
        "post_tasil": {
            "total_shares": result.total_shares,
            "distribution": result.distribution,
        },
    }


def solve_with_scraper(
    gender: str,
    heirs: dict[str, int],
    browser: Browser | None = None,
    headless: bool = True,
) -> dict:
    """
    Full pipeline: fill form → scrape → parse → return competition JSON.

    Args:
        gender:   "ذكر" or "أنثى"
        heirs:    Dict of {heir_name: count}
        browser:  Reuse existing browser instance (optional)
        headless: Run browser headlessly

    Returns:
        Competition-format answer_structured dict
    """
    own_browser = False

    if browser is None:
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=headless)
        own_browser = True

    try:
        page = browser.new_page()
        url = fill_form_and_submit(page, gender, heirs)
        result = parse_results_page(page)
        page.close()
        return result_to_competition_json(result)
    finally:
        if own_browser:
            browser.close()