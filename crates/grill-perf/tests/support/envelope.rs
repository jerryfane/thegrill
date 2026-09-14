use super::*;

fn r(n: u64, d: u64) -> Rational {
    (n, d).into()
}
fn point(n: u64, d: u64) -> [Rational; 2] {
    [r(n, d); 2]
}

#[test]
fn rational_validation_covers_every_endpoint_and_range_input() {
    assert_eq!(range(&[]), Ok(None));
    assert_eq!(range(&[r(1, 2), r(2, 4)]), Ok(Some(point(1, 2))));
    assert_eq!(
        range(&[r(1, 1), r(0, 0)]),
        Err(EnvelopeReason::InvalidRational)
    );
    for index in 0..4 {
        let mut endpoints = [r(1, 1); 4];
        endpoints[index] = r(0, 0);
        assert_eq!(
            assess(
                [endpoints[0], endpoints[1]],
                [endpoints[2], endpoints[3]],
                0,
                0,
                Direction::LowerBetter
            ),
            Err(EnvelopeReason::InvalidRational)
        );
    }
    assert_eq!(
        assess(
            [r(2, 1), r(1, 1)],
            point(1, 1),
            0,
            0,
            Direction::LowerBetter
        ),
        Err(EnvelopeReason::InvalidBounds)
    );
    assert_eq!(
        assess(
            point(1, 1),
            [r(2, 1), r(1, 1)],
            0,
            0,
            Direction::LowerBetter
        ),
        Err(EnvelopeReason::InvalidBounds)
    );
    assert_eq!(
        assess(point(0, 1), point(0, 1), 0, 0, Direction::LowerBetter),
        Err(EnvelopeReason::NonpositiveReference)
    );
    assert_eq!(
        assess(point(1, 1), point(1, 1), 10000, 0, Direction::LowerBetter),
        Err(EnvelopeReason::InvalidBounds)
    );
    assert_eq!(
        assess(
            point(1, 1),
            point(1, 1),
            0,
            1_000_001,
            Direction::LowerBetter
        ),
        Err(EnvelopeReason::InvalidBounds)
    );
}

#[test]
fn exact_tolerance_and_spread_boundaries_have_no_epsilon() {
    assert_eq!(
        assess(
            point(10000, 1),
            point(19999, 1),
            9999,
            0,
            Direction::LowerBetter
        )
        .unwrap()
        .decision,
        EnvelopeDecision::Pass
    );
    assert_eq!(
        assess(
            point(10000, 1),
            point(20000, 1),
            9999,
            0,
            Direction::LowerBetter
        )
        .unwrap()
        .decision,
        EnvelopeDecision::Regression
    );
    assert_eq!(
        assess(
            point(10000, 1),
            point(1, 1),
            9999,
            0,
            Direction::HigherBetter
        )
        .unwrap()
        .decision,
        EnvelopeDecision::Pass
    );
    assert_eq!(
        assess(
            point(10000, 1),
            point(0, 1),
            9999,
            0,
            Direction::HigherBetter
        )
        .unwrap()
        .decision,
        EnvelopeDecision::Regression
    );
    assert!(matches!(
        assess(
            [r(100, 1), r(200, 1)],
            point(150, 1),
            0,
            10000,
            Direction::LowerBetter
        ),
        Err(EnvelopeReason::EnvelopeStraddlesTolerance { .. })
    ));
    assert_eq!(
        assess(
            [r(100, 1), r(201, 1)],
            point(150, 1),
            0,
            10000,
            Direction::LowerBetter
        ),
        Err(EnvelopeReason::ReferenceSpreadExceeded)
    );
}

#[test]
fn exact_integer_decision_survives_display_rounding() {
    let result = assess(
        point(u64::MAX - 1, 1),
        point(u64::MAX, 1),
        0,
        0,
        Direction::LowerBetter,
    )
    .unwrap();
    assert_eq!(result.decision, EnvelopeDecision::Regression);
    assert_eq!(result.adverse_bounds, [0.0, 0.0]);
}

#[test]
fn overflow_retains_only_diagnostics_available_at_the_failure_stage() {
    assert_eq!(
        assess(
            point(u64::MAX, u64::MAX),
            point(1, 1),
            0,
            0,
            Direction::LowerBetter
        ),
        Err(EnvelopeReason::ArithmeticOverflow {
            adverse_bounds: None
        })
    );
    assert!(matches!(
        assess(
            point(u64::MAX, 1),
            point(1, u64::MAX),
            0,
            0,
            Direction::LowerBetter
        ),
        Err(EnvelopeReason::ArithmeticOverflow {
            adverse_bounds: Some(_)
        })
    ));
    assert_eq!(
        range(&[r(u64::MAX, 1), r(1, u64::MAX)]),
        Ok(Some([r(1, u64::MAX), r(u64::MAX, 1)]))
    );
}
