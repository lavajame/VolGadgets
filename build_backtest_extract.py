from __future__ import annotations

import argparse
import ast
import configparser
import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

MONTH_DIR_RE = re.compile(r"\d{4}-\d{2}$")
DAILY_FILE_RE = re.compile(r"(?P<quote_date>\d{4}-\d{2}-\d{2})(?P<kind>options|stocks)\.csv$")
DATE_FORMATS = (
    "%Y-%m-%d",
    "%d %b %Y",
    "%d %B %Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%d/%m/%Y",
)

OUTPUT_FIELDS = [
    "quote_date",
    "selection_group",
    "underlying_close",
    "contract",
    "expiration",
    "option_type",
    "strike",
    "style",
    "bid",
    "bid_size",
    "ask",
    "ask_size",
    "mid",
    "volume",
    "open_interest",
    "delta",
    "gamma",
    "theta",
    "vega",
    "implied_volatility",
    "quote_missing",
]


@dataclass(frozen=True)
class ExtractConfig:
    data_root: Path
    output_dir: Path
    as_of_date: date
    ticker: str
    expiry_targets_days: tuple[int, int]
    centre_pct: float
    range_offsets: tuple[float, float]
    strike_count: int


@dataclass(frozen=True)
class StrikePair:
    strike: float
    call_row: dict[str, str]
    put_row: dict[str, str]
    cumulative_open_interest: int


@dataclass(frozen=True)
class SelectedLeg:
    selection_group: str
    selection_rank: int
    contract: str
    expiration: date
    option_type: str
    strike: float
    style: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a consolidated backtest CSV from daily CBOE option and stock files."
        )
    )
    parser.add_argument(
        "--config",
        default="extract_config.ini",
        help="Path to the config file. Defaults to extract_config.ini in the working directory.",
    )
    return parser.parse_args()


def parse_config_date(raw_value: str) -> date:
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", raw_value.strip(), flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", "")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"Could not parse as_of_date '{raw_value}'. Use ISO format or a date like 6th Feb 2013."
    )


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    option_files, stock_files = discover_daily_files(config.data_root)

    if config.as_of_date not in option_files:
        raise ValueError(f"No option file found for as_of_date={config.as_of_date}.")
    if config.as_of_date not in stock_files:
        raise ValueError(f"No stock file found for as_of_date={config.as_of_date}.")

    as_of_stock_row = load_stock_row(stock_files[config.as_of_date], config.ticker)
    as_of_spot = float(as_of_stock_row["close"])
    as_of_option_rows = load_underlying_option_rows(
        option_files[config.as_of_date],
        config.ticker,
    )

    near_expiry, far_expiry = select_expiries(
        as_of_option_rows,
        config.as_of_date,
        config.expiry_targets_days,
    )
    near_pairs = build_strike_pairs(as_of_option_rows, near_expiry)
    far_pairs = build_strike_pairs(as_of_option_rows, far_expiry)

    target_centre_cash = as_of_spot * config.centre_pct
    target_pair = select_target_pair(far_pairs, target_centre_cash)
    selected_near_pairs = select_near_pairs(
        near_pairs,
        target_pair.strike,
        config.range_offsets,
        config.strike_count,
    )
    selected_legs = build_selected_legs(target_pair, selected_near_pairs)

    quote_dates = collect_quote_dates(
        option_files,
        stock_files,
        config.as_of_date,
        near_expiry,
    )

    output_path = config.output_dir / f"{config.as_of_date:%y%m%d}_{config.ticker}" / "mkt_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = write_output(
        output_path=output_path,
        config=config,
        quote_dates=quote_dates,
        stock_files=stock_files,
        option_files=option_files,
        selected_legs=selected_legs,
        as_of_spot=as_of_spot,
        near_expiry=near_expiry,
        far_expiry=far_expiry,
        target_centre_cash=target_centre_cash,
        target_strike=target_pair.strike,
    )

    near_strikes = ", ".join(str(pair.strike) for pair in selected_near_pairs)
    print(f"As-of date: {config.as_of_date}")
    print(f"Ticker: {config.ticker}")
    print(f"Spot close on as-of date: {as_of_spot}")
    print(
        "Selected expiries: "
        f"{near_expiry} (near), {far_expiry} (far) for requested days {config.expiry_targets_days}"
    )
    print(f"Target centre cash: {target_centre_cash}")
    print(f"Target strike on far expiry: {target_pair.strike}")
    print(f"Near-expiry strikes: {near_strikes}")
    print(f"Wrote {row_count} rows to {output_path}")


def load_config(config_path: Path) -> ExtractConfig:
    parser = configparser.ConfigParser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    parser.read(config_path)
    if "extract" not in parser:
        raise ValueError("Config file must contain an [extract] section.")

    section = parser["extract"]
    base_dir = config_path.parent.resolve()

    as_of_date = parse_config_date(section["as_of_date"])
    ticker = section["ticker"].strip().strip("\"'").upper()
    expiry_targets_days = parse_expiry_targets(section["expiry_targets_days"])
    centre_pct, range_offsets, strike_count = parse_strike_selection(
        section["strike_selection"]
    )

    data_root = resolve_path(section.get("data_root", "."), base_dir)
    output_dir = resolve_path(section.get("output_dir", "backtests"), base_dir)

    return ExtractConfig(
        data_root=data_root,
        output_dir=output_dir,
        as_of_date=as_of_date,
        ticker=ticker,
        expiry_targets_days=expiry_targets_days,
        centre_pct=centre_pct,
        range_offsets=range_offsets,
        strike_count=strike_count,
    )


def parse_expiry_targets(raw_value: str) -> tuple[int, int]:
    parsed = parse_literal(raw_value, "expiry_targets_days")
    if len(parsed) != 2:
        raise ValueError("expiry_targets_days must contain exactly two integers.")

    first = int(parsed[0])
    second = int(parsed[1])
    if first <= 0 or second <= first:
        raise ValueError(
            "expiry_targets_days must be two ascending positive integers, for example (30, 60)."
        )
    return first, second


def parse_strike_selection(raw_value: str) -> tuple[float, tuple[float, float], int]:
    parsed = parse_literal(raw_value, "strike_selection")
    if len(parsed) != 3:
        raise ValueError(
            "strike_selection must have the shape (centre_pct, (low_offset, high_offset), count)."
        )

    centre_pct = float(parsed[0])
    offsets = parsed[1]
    strike_count = int(parsed[2])

    if not isinstance(offsets, (tuple, list)) or len(offsets) != 2:
        raise ValueError("strike_selection offsets must be a two-item tuple or list.")

    low_offset = float(offsets[0])
    high_offset = float(offsets[1])

    if centre_pct <= 0:
        raise ValueError("centre_pct must be positive.")
    if low_offset >= high_offset:
        raise ValueError("strike_selection offsets must satisfy low_offset < high_offset.")
    if strike_count <= 0:
        raise ValueError("strike_selection count must be a positive integer.")

    return centre_pct, (low_offset, high_offset), strike_count


def parse_literal(raw_value: str, field_name: str) -> tuple:
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse {field_name}: {raw_value}") from exc

    if not isinstance(parsed, (tuple, list)):
        raise ValueError(f"{field_name} must be written as a tuple or list literal.")
    return tuple(parsed)


def resolve_path(raw_value: str, base_dir: Path) -> Path:
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def discover_daily_files(data_root: Path) -> tuple[dict[date, Path], dict[date, Path]]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    option_files: dict[date, Path] = {}
    stock_files: dict[date, Path] = {}

    for month_dir in sorted(data_root.iterdir()):
        if not month_dir.is_dir() or not MONTH_DIR_RE.fullmatch(month_dir.name):
            continue

        for file_path in sorted(month_dir.iterdir()):
            match = DAILY_FILE_RE.fullmatch(file_path.name)
            if not match:
                continue

            quote_date = date.fromisoformat(match.group("quote_date"))
            kind = match.group("kind")
            if kind == "options":
                option_files[quote_date] = file_path
            else:
                stock_files[quote_date] = file_path

    return option_files, stock_files


def load_stock_row(file_path: Path, ticker: str) -> dict[str, str]:
    with file_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["symbol"].strip().upper() == ticker:
                return row

    raise ValueError(f"Ticker {ticker} not found in stock file {file_path}")


def load_underlying_option_rows(file_path: Path, ticker: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with file_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["underlying"].strip().upper() == ticker:
                rows.append(row)

    if not rows:
        raise ValueError(f"Ticker {ticker} not found in option file {file_path}")
    return rows


def select_expiries(
    option_rows: list[dict[str, str]],
    as_of_date: date,
    expiry_targets_days: tuple[int, int],
) -> tuple[date, date]:
    expiries = sorted({date.fromisoformat(row["expiration"]) for row in option_rows})
    if len(expiries) < 2:
        raise ValueError("At least two expiries are required on the as-of date.")

    selected = [
        pick_nearest_expiry(expiries, as_of_date + timedelta(days=target_days))
        for target_days in expiry_targets_days
    ]

    if selected[0] == selected[1]:
        raise ValueError(
            "The requested expiry targets resolved to the same listed expiry. Use a wider day tuple."
        )

    return selected[0], selected[1]


def pick_nearest_expiry(expiries: list[date], target_date: date) -> date:
    return min(expiries, key=lambda expiry: (abs((expiry - target_date).days), expiry))


def build_strike_pairs(
    option_rows: list[dict[str, str]],
    expiry: date,
) -> list[StrikePair]:
    grouped: dict[float, dict[str, dict[str, str] | None]] = {}

    for row in option_rows:
        if date.fromisoformat(row["expiration"]) != expiry:
            continue

        option_type = row["type"].strip().lower()
        if option_type not in {"call", "put"}:
            continue

        strike = float(row["strike"])
        slot = grouped.setdefault(strike, {"call": None, "put": None})
        slot[option_type] = choose_better_row(slot[option_type], row)

    pairs: list[StrikePair] = []
    for strike, slot in grouped.items():
        call_row = slot["call"]
        put_row = slot["put"]
        if call_row is None or put_row is None:
            continue

        pairs.append(
            StrikePair(
                strike=strike,
                call_row=call_row,
                put_row=put_row,
                cumulative_open_interest=(
                    safe_int(call_row["open_interest"]) + safe_int(put_row["open_interest"])
                ),
            )
        )

    if not pairs:
        raise ValueError(f"No call/put pairs were found for expiry {expiry}.")
    return sorted(pairs, key=lambda pair: pair.strike)


def choose_better_row(
    current_row: dict[str, str] | None,
    candidate_row: dict[str, str],
) -> dict[str, str]:
    if current_row is None:
        return candidate_row

    current_key = (
        safe_int(current_row["open_interest"]),
        safe_int(current_row["volume"]),
        current_row["contract"],
    )
    candidate_key = (
        safe_int(candidate_row["open_interest"]),
        safe_int(candidate_row["volume"]),
        candidate_row["contract"],
    )
    return candidate_row if candidate_key > current_key else current_row


def select_target_pair(strike_pairs: list[StrikePair], centre_cash: float) -> StrikePair:
    return min(
        strike_pairs,
        key=lambda pair: (
            abs(pair.strike - centre_cash),
            -pair.cumulative_open_interest,
            pair.strike,
        ),
    )


def select_near_pairs(
    strike_pairs: list[StrikePair],
    target_strike: float,
    range_offsets: tuple[float, float],
    strike_count: int,
) -> list[StrikePair]:
    lower_bound = target_strike * (1 + range_offsets[0])
    upper_bound = target_strike * (1 + range_offsets[1])
    candidates = [
        pair for pair in strike_pairs if lower_bound <= pair.strike <= upper_bound
    ]

    if len(candidates) < strike_count:
        raise ValueError(
            f"Only {len(candidates)} strike pairs are available in range "
            f"[{lower_bound}, {upper_bound}] for requested count={strike_count}."
        )

    if strike_count == 1:
        target_grid = [(lower_bound + upper_bound) / 2]
    else:
        span = upper_bound - lower_bound
        target_grid = [
            lower_bound + (span * index / (strike_count - 1))
            for index in range(strike_count)
        ]

    chosen: list[StrikePair] = []
    used_strikes: set[float] = set()

    for grid_strike in target_grid:
        remaining = [pair for pair in candidates if pair.strike not in used_strikes]
        best_pair = min(
            remaining,
            key=lambda pair: (
                abs(pair.strike - grid_strike),
                -pair.cumulative_open_interest,
                abs(pair.strike - target_strike),
                pair.strike,
            ),
        )
        chosen.append(best_pair)
        used_strikes.add(best_pair.strike)

    return sorted(chosen, key=lambda pair: pair.strike)


def build_selected_legs(
    target_pair: StrikePair,
    near_pairs: list[StrikePair],
) -> list[SelectedLeg]:
    legs: list[SelectedLeg] = []

    for row in (target_pair.call_row, target_pair.put_row):
        legs.append(build_leg_from_row("far_target", 0, row))

    for selection_rank, pair in enumerate(near_pairs, start=1):
        for row in (pair.call_row, pair.put_row):
            legs.append(build_leg_from_row("near_grid", selection_rank, row))

    return legs


def build_leg_from_row(
    selection_group: str,
    selection_rank: int,
    row: dict[str, str],
) -> SelectedLeg:
    return SelectedLeg(
        selection_group=selection_group,
        selection_rank=selection_rank,
        contract=row["contract"],
        expiration=date.fromisoformat(row["expiration"]),
        option_type=row["type"].strip().lower(),
        strike=float(row["strike"]),
        style=row["style"],
    )


def collect_quote_dates(
    option_files: dict[date, Path],
    stock_files: dict[date, Path],
    start_date: date,
    end_date: date,
) -> list[date]:
    quote_dates = sorted(
        quote_date
        for quote_date in option_files.keys() & stock_files.keys()
        if start_date <= quote_date <= end_date
    )
    if not quote_dates:
        raise ValueError(f"No shared stock/option dates found between {start_date} and {end_date}.")
    return quote_dates


def write_output(
    output_path: Path,
    config: ExtractConfig,
    quote_dates: list[date],
    stock_files: dict[date, Path],
    option_files: dict[date, Path],
    selected_legs: list[SelectedLeg],
    as_of_spot: float,
    near_expiry: date,
    far_expiry: date,
    target_centre_cash: float,
    target_strike: float,
) -> int:
    selected_contracts = {leg.contract for leg in selected_legs}
    row_count = 0

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for quote_date in quote_dates:
            stock_row = load_stock_row(stock_files[quote_date], config.ticker)
            option_rows = load_selected_option_rows(
                option_files[quote_date],
                selected_contracts,
            )
            base_row = build_base_row(
                config=config,
                quote_date=quote_date,
                stock_row=stock_row,
                as_of_spot=as_of_spot,
                near_expiry=near_expiry,
                far_expiry=far_expiry,
                target_centre_cash=target_centre_cash,
                target_strike=target_strike,
            )

            writer.writerow(build_underlying_row(base_row))
            row_count += 1

            for leg in selected_legs:
                writer.writerow(build_option_row(base_row, leg, option_rows.get(leg.contract)))
                row_count += 1

    return row_count


def load_selected_option_rows(
    file_path: Path,
    selected_contracts: set[str],
) -> dict[str, dict[str, str]]:
    remaining = set(selected_contracts)
    found_rows: dict[str, dict[str, str]] = {}

    with file_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            contract = row["contract"]
            if contract not in remaining:
                continue

            found_rows[contract] = row
            remaining.remove(contract)
            if not remaining:
                break

    return found_rows


def build_base_row(
    config: ExtractConfig,
    quote_date: date,
    stock_row: dict[str, str],
    as_of_spot: float,
    near_expiry: date,
    far_expiry: date,
    target_centre_cash: float,
    target_strike: float,
) -> dict[str, object]:
    return {
        "as_of_date": config.as_of_date.isoformat(),
        "quote_date": quote_date.isoformat(),
        "requested_near_days": config.expiry_targets_days[0],
        "requested_far_days": config.expiry_targets_days[1],
        "selected_near_expiry": near_expiry.isoformat(),
        "selected_far_expiry": far_expiry.isoformat(),
        "spot_on_as_of": as_of_spot,
        "target_centre_pct": config.centre_pct,
        "target_centre_cash": target_centre_cash,
        "target_strike": target_strike,
        "range_low_offset": config.range_offsets[0],
        "range_high_offset": config.range_offsets[1],
        "underlying_symbol": config.ticker,
        "underlying_open": stock_row["open"],
        "underlying_high": stock_row["high"],
        "underlying_low": stock_row["low"],
        "underlying_close": stock_row["close"],
        "underlying_volume": stock_row["volume"],
        "underlying_adjust_close": stock_row.get("adjust_close", ""),
    }


def build_underlying_row(base_row: dict[str, object]) -> dict[str, object]:
    row = blank_output_row()
    row["quote_date"] = base_row["quote_date"]
    row["selection_group"] = "underlying"
    row["underlying_close"] = base_row["underlying_close"]
    row["quote_missing"] = 0
    return row


def build_option_row(
    base_row: dict[str, object],
    leg: SelectedLeg,
    option_row: dict[str, str] | None,
) -> dict[str, object]:
    row = blank_output_row()

    if option_row is None:
        row["quote_date"] = base_row["quote_date"]
        row["selection_group"] = leg.selection_group
        row["underlying_close"] = base_row["underlying_close"]
        row["contract"] = leg.contract
        row["expiration"] = leg.expiration.isoformat()
        row["option_type"] = leg.option_type
        row["strike"] = leg.strike
        row["style"] = leg.style
        row["quote_missing"] = 1
        return row

    bid = parse_optional_float(option_row["bid"])
    ask = parse_optional_float(option_row["ask"])

    # Only emit the reduced set of fields requested: quote_date, underlying_close, and option columns.
    row_fields = {
        "quote_date": base_row["quote_date"],
        "selection_group": leg.selection_group,
        "underlying_close": base_row["underlying_close"],
        "contract": option_row["contract"],
        "expiration": option_row["expiration"],
        "option_type": option_row["type"],
        "strike": option_row["strike"],
        "style": option_row["style"],
        "bid": option_row["bid"],
        "bid_size": option_row.get("bid_size", ""),
        "ask": option_row["ask"],
        "ask_size": option_row.get("ask_size", ""),
        "mid": format_number(compute_mid(bid, ask)),
        "volume": option_row.get("volume", ""),
        "open_interest": option_row.get("open_interest", ""),
        "delta": option_row.get("delta", ""),
        "gamma": option_row.get("gamma", ""),
        "theta": option_row.get("theta", ""),
        "vega": option_row.get("vega", ""),
        "implied_volatility": option_row.get("implied_volatility", ""),
        "quote_missing": 0,
    }
    row.update(row_fields)
    return row


def blank_output_row() -> dict[str, object]:
    return {field: "" for field in OUTPUT_FIELDS}


def safe_int(raw_value: str) -> int:
    return int(raw_value) if raw_value.strip() else 0


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    return format(value, ".10g")


def parse_optional_float(raw_value: str) -> float | None:
    value = raw_value.strip()
    if not value:
        return None
    return float(value)


def compute_mid(bid: float | None, ask: float | None) -> float | None:
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    if bid is not None:
        return bid
    if ask is not None:
        return ask
    return None


if __name__ == "__main__":
    main()