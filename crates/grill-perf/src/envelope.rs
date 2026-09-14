//! Exact observed-envelope comparisons shared by serving and domain policies.
//! Floating bounds are display-only; every decision uses checked integer products.

use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub struct Rational {
    pub numerator: u64,
    pub denominator: u64,
}

impl From<(u64, u64)> for Rational {
    fn from((numerator, denominator): (u64, u64)) -> Self {
        Self { numerator, denominator }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    LowerBetter,
    HigherBetter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvelopeDecision {
    Pass,
    Regression,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Assessment {
    pub decision: EnvelopeDecision,
    pub adverse_bounds: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnvelopeReason {
    InvalidRational,
    InvalidBounds,
    ArithmeticOverflow { adverse_bounds: Option<[f64; 2]> },
    NonpositiveReference,
    ReferenceSpreadExceeded,
    EnvelopeStraddlesTolerance { adverse_bounds: [f64; 2] },
}

// Two u64 factors always fit in u128. Callers validate denominators first.
fn order(a: Rational, b: Rational) -> std::cmp::Ordering {
    (u128::from(a.numerator) * u128::from(b.denominator))
        .cmp(&(u128::from(b.numerator) * u128::from(a.denominator)))
}

fn scaled(a: Rational, scale: u32, b: Rational) -> Option<u128> {
    u128::from(a.numerator)
        .checked_mul(u128::from(scale))
        .and_then(|n| n.checked_mul(u128::from(b.denominator)))
}

pub fn range(values: &[Rational]) -> Result<Option<[Rational; 2]>, EnvelopeReason> {
    if values.iter().any(|value| value.denominator == 0) {
        return Err(EnvelopeReason::InvalidRational);
    }
    let Some(&first) = values.first() else { return Ok(None); };
    Ok(Some(values.iter().copied().fold([first, first], |[low, high], value| {
        [
            if order(value, low).is_lt() { value } else { low },
            if order(value, high).is_gt() { value } else { high },
        ]
    })))
}

pub fn assess(
    reference: [Rational; 2],
    candidate: [Rational; 2],
    adverse_bps: u32,
    spread_bps: u32,
    direction: Direction,
) -> Result<Assessment, EnvelopeReason> {
    let [low, high] = reference;
    let [c_low, c_high] = candidate;
    if [low, high, c_low, c_high].iter().any(|v| v.denominator == 0) {
        return Err(EnvelopeReason::InvalidRational);
    }
    if adverse_bps > 9999 || spread_bps > 1_000_000
        || order(low, high).is_gt() || order(c_low, c_high).is_gt()
    {
        return Err(EnvelopeReason::InvalidBounds);
    }
    if low.numerator == 0 {
        return Err(EnvelopeReason::NonpositiveReference);
    }
    let reference_overflow = EnvelopeReason::ArithmeticOverflow { adverse_bounds: None };
    if scaled(high, 10000, low).ok_or(reference_overflow)?
        > scaled(low, 10000 + spread_bps, high).ok_or(reference_overflow)?
    {
        return Err(EnvelopeReason::ReferenceSpreadExceeded);
    }
    let ratio = |a: Rational, b: Rational| {
        (a.numerator as f64 / a.denominator as f64) / (b.numerator as f64 / b.denominator as f64)
    };
    let adverse_bounds = match direction {
        Direction::LowerBetter => [ratio(c_low, high) - 1.0, ratio(c_high, low) - 1.0],
        Direction::HigherBetter => [1.0 - ratio(c_high, low), 1.0 - ratio(c_low, high)],
    };
    let overflow = EnvelopeReason::ArithmeticOverflow { adverse_bounds: Some(adverse_bounds) };
    // Keep both comparisons checked, even when the first establishes PASS.
    // Legacy policy1 fails closed on an overflowing second comparison too.
    let (pass, regression) = match direction {
        Direction::LowerBetter => (
            scaled(c_high, 10000, low).ok_or(overflow)?
                <= scaled(low, 10000 + adverse_bps, c_high).ok_or(overflow)?,
            scaled(c_low, 10000, high).ok_or(overflow)?
                > scaled(high, 10000 + adverse_bps, c_low).ok_or(overflow)?,
        ),
        Direction::HigherBetter => (
            scaled(c_low, 10000, high).ok_or(overflow)?
                >= scaled(high, 10000 - adverse_bps, c_low).ok_or(overflow)?,
            scaled(c_high, 10000, low).ok_or(overflow)?
                < scaled(low, 10000 - adverse_bps, c_high).ok_or(overflow)?,
        ),
    };
    let decision = if pass {
        EnvelopeDecision::Pass
    } else if regression {
        EnvelopeDecision::Regression
    } else {
        return Err(EnvelopeReason::EnvelopeStraddlesTolerance { adverse_bounds });
    };
    Ok(Assessment { decision, adverse_bounds })
}

#[cfg(test)]
#[path = "../tests/support/envelope.rs"]
mod tests;
