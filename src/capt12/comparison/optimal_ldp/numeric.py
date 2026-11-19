"""Outward-rounded arithmetic for statistical and optimization bounds."""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal
from fractions import Fraction

ZERO = Decimal(0)
ONE = Decimal(1)
DOWN = Context(prec=40, rounding=ROUND_FLOOR)
UP = Context(prec=40, rounding=ROUND_CEILING)


def decimal_ratio(value: Fraction, ctx: Context) -> Decimal:
    return ctx.divide(Decimal(value.numerator), Decimal(value.denominator))


def log_bound(value: Decimal, ctx: Context) -> Decimal:
    if value == ONE:
        return ZERO
    # Decimal's ln and exp are correctly rounded to nearest, even in a
    # directed context. Stepping one representable number encloses the result.
    result = ctx.ln(value)
    return ctx.next_minus(result) if ctx is DOWN else ctx.next_plus(result)


def exp_bound(value: Decimal, ctx: Context) -> Decimal:
    if value == ZERO:
        return ONE
    result = ctx.exp(value)
    return ctx.next_minus(result) if ctx is DOWN else ctx.next_plus(result)


def sqrt_upper(value: Decimal) -> Decimal:
    return UP.next_plus(UP.sqrt(value)) if value else ZERO


def float_bound(value: Decimal, ctx: Context) -> float:
    result = float(value)
    represented = Decimal.from_float(result)
    if ctx is DOWN and represented > value:
        return math.nextafter(result, -math.inf)
    if ctx is UP and represented < value:
        return math.nextafter(result, math.inf)
    return result


def entropy_mass(a: Decimal, b: Decimal, ctx: Context) -> Decimal:
    """Enclose (a+b) h(a/(a+b)), increasing in each nonnegative mass."""
    if not a or not b:
        return ZERO
    return ctx.add(
        ctx.multiply(a, log_bound(ctx.add(ONE, ctx.divide(b, a)), ctx)),
        ctx.multiply(b, log_bound(ctx.add(ONE, ctx.divide(a, b)), ctx)),
    )


def multinomial_tv_radius(n: int, alphabet: int, alpha: float) -> float:
    """One-sided Hoeffding over all subsets, at fixed sample size."""
    if n <= 0 or alphabet < 1 or not 0 < alpha < 1:
        raise ValueError("TV coverage requires samples, an alphabet, and alpha in (0,1)")
    numerator = UP.add(
        UP.multiply(Decimal(alphabet), log_bound(Decimal(2), UP)),
        log_bound(Decimal.from_float(alpha), DOWN).copy_negate(),
    )
    return min(1.0, float_bound(sqrt_upper(UP.divide(numerator, Decimal(2 * n))), UP))
