"""
endes_codes.py
==============
INEI proyecto codes for **ENDES** (Encuesta Demografica y de Salud Familiar — Peru's
DHS), as a parallel track to ENAHO (`enaho_codes.py`) and the ENAHO Panel
(`panel_codes.py`). Same INEI download scheme:

    https://proyectos.inei.gob.pe/iinei/srienaho/descarga/STATA/{CODE}-Modulo{NN}.zip

CODE is the global proyecto id (NOT the year). The year->code map below was harvested
from the INEI catalogue `codigosencuesta.rds` (the same table that gave the ENAHO and
ENAHO-Panel maps; 2026-06-19). UNLIKE ENAHO, the ENDES module-number set is NOT yet
verified here — ENDES ships DHS "recode" modules (household, women 15-49, children,
domestic violence, anthropometry...) under INEI module numbers we must DISCOVER against
the server (`discover()`), then content-verify each before trusting it. Do that before
any analysis, exactly as we did for ENAHO.
"""
from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# year -> INEI proyecto code (harvested from codigosencuesta.rds, NOT content-verified)
ENDES_CODE = {
    1996: 32, 2000: 35, 2004: 120, 2005: 150, 2006: 183, 2007: 194,
    2008: 209, 2009: 238, 2010: 260, 2011: 290, 2012: 323, 2013: 407,
    2014: 441, 2015: 504, 2016: 548, 2017: 605, 2018: 638, 2019: 691,
    2020: 739, 2021: 760, 2022: 786, 2023: 910, 2024: 968,
}

# IMPORTANT: ENDES is served in SPSS for ALL years (STATA only exists 2020+). See
# docs/NOTES_endes.md. Default format = SPSS.
BASE = "https://proyectos.inei.gob.pe/iinei/srienaho/descarga"
DEFAULT_FORMAT = "SPSS"
UA = "Mozilla/5.0 (ENAHO_ANALYSIS scraper; research use)"


def url(year: int, module: str | int, fmt: str = DEFAULT_FORMAT) -> str:
    """Download-zip URL for an ENDES year and module number (SPSS by default)."""
    return f"{BASE}/{fmt}/{ENDES_CODE[year]}-Modulo{module}.zip"


def years() -> list[int]:
    return sorted(ENDES_CODE)


def _head_ok(u: str, timeout: int = 20) -> bool:
    try:
        with urlopen(Request(u, headers={"User-Agent": UA}, method="HEAD"), timeout=timeout) as r:
            return r.status == 200
    except (HTTPError, URLError):
        return False


def discover(year: int, candidates: list[int], fmt: str = DEFAULT_FORMAT) -> list[int]:
    """HEAD-probe `candidates` module numbers for `year`; return those that exist.

    ENDES module numbers are unknown a priori — probe a range, keep the hits.
    Prints each hit so we can map module-number -> content on the next pass.
    """
    code = ENDES_CODE[year]
    hits = []
    for m in candidates:
        if _head_ok(f"{BASE}/{fmt}/{code}-Modulo{m}.zip"):
            hits.append(m)
            print(f"  HIT  {year} (code {code})  Modulo{m}")
    return hits


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--lo", type=int, default=1, help="low module number")
    ap.add_argument("--hi", type=int, default=99, help="high module number (inclusive)")
    a = ap.parse_args()
    print(f"probing ENDES {a.year} (code {ENDES_CODE[a.year]}) modules {a.lo}..{a.hi}")
    hits = discover(a.year, list(range(a.lo, a.hi + 1)))
    print(f"\n{len(hits)} modules found: {hits}")
