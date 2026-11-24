"""
endes_units.py
==============
Fix for the ENDES 2004-2008 CUMULATIVE-file trap.

Verified 2026-06-21 from the interview-date CMC (v008): the early ENDES "annual" files
are CUMULATIVE, not single-year:
  code 120 (2004): interviews 2003-2004      code 194 (2007): interviews 2003-2007
  code 150 (2005): interviews 2003-2005      code 209 (2008): interviews 2005-2008
  code 183 (2006): interviews 2003-2006      code 238 (2009)+: single calendar year (clean)
Plotting these as separate years creates spurious "jumps" (each point is a running pooled
average over largely the SAME interviews). FIX: assign every record to its TRUE calendar
year via v008/hv008 and take each year from ONE source file (no duplication):
  2004-2007 -> code 194 (folder 2007_194), filtered to v008's year
  2008      -> code 209 (folder 2008_209), filtered to 2008
  2009-2024 -> the annual file (already single-year)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from endes_codes import ENDES_CODE

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "endes"

# calendar year -> (folder name) source. 2004-2007 from cumulative 194, 2008 from 209.
_SRC = {2004: "2007_194", 2005: "2007_194", 2006: "2007_194", 2007: "2007_194", 2008: "2008_209"}
for _y in range(2009, 2025):
    _SRC[_y] = f"{_y}_{ENDES_CODE[_y]}"


def years(y0=2004, y1=2024):
    return [y for y in range(y0, y1 + 1) if y in _SRC]


def dir_for(y: int) -> Path:
    return RAW / _SRC[y]


def cmc_year(s) -> pd.Series:
    """CMC (century-month code) -> calendar year, vectorized."""
    v = pd.to_numeric(s, errors="coerce")
    return 1900 + ((v - 1) // 12)


def true_year_mask(df: pd.DataFrame, y: int, col: str) -> pd.Series:
    """Boolean mask: records whose v008/hv008 (col) falls in calendar year y.
    For clean annual files (>=2009) the file is already single-year, so if the
    column is absent we keep everything."""
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    return cmc_year(df[col]).eq(y)
