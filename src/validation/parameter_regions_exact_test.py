"""Exact rational checks for the reduced two-player actuator game."""

from fractions import Fraction as F


PROFILES = ((0, 0), (0, 1), (1, 0), (1, 1))
PLAYER_WEIGHTS = (
    (F(1, 5), F(3, 10), F(1, 2)),
    (F(2, 5), F(1, 10), F(1, 2)),
)
CHANNEL_ENERGIES = ((F(1), F(0), F(1)), (F(1), F(1), F(1)))
RHO = F(2, 3)


def dot(first, second):
    return sum((x * y for x, y in zip(first, second)), F(0))


CHANNEL_VALUES = tuple(
    tuple(dot(weights, channel) for channel in CHANNEL_ENERGIES)
    for weights in PLAYER_WEIGHTS
)
FULL_VALUES = tuple(sum(values, F(0)) for values in CHANNEL_VALUES)


def normalized_benefit(profile, player):
    return sum(
        (F(profile[channel]) * CHANNEL_VALUES[player][channel]
         for channel in range(2)),
        F(0),
    ) / FULL_VALUES[player]


BENEFITS = {
    profile: tuple(normalized_benefit(profile, player) for player in range(2))
    for profile in PROFILES
}
A = BENEFITS[(1, 0)][0]
B = BENEFITS[(0, 1)][1]


def coefficients_from_ratios(g, r):
    alpha = F(1) / (F(1) + g + F(3, 2) * r)
    return alpha, g * alpha, F(3, 2) * r * alpha


def build_payoff(alpha, gamma, delta):
    payoff = {}
    for s1, s2 in PROFILES:
        redundancy = RHO * s1 * s2
        payoff[(s1, s2)] = (
            alpha * BENEFITS[(s1, s2)][0] - gamma * s1
            - delta * redundancy,
            alpha * BENEFITS[(s1, s2)][1] - gamma * s2
            - delta * redundancy,
        )
    return payoff


def pure_nash(payoff):
    result = set()
    for s1, s2 in PROFILES:
        if (payoff[(s1, s2)][0] >= payoff[(1 - s1, s2)][0]
                and payoff[(s1, s2)][1] >= payoff[(s1, 1 - s2)][1]):
            result.add((s1, s2))
    return result


def predicted_pure(g, r):
    result = set()
    if g >= B:
        result.add((0, 0))
    if g <= B and g + r >= A:
        result.add((0, 1))
    if g <= A and g + r >= B:
        result.add((1, 0))
    if g + r <= A:
        result.add((1, 1))
    return result


def activation_gains(payoff, p, q):
    gain_one = ((F(1) - q) * (payoff[(1, 0)][0] - payoff[(0, 0)][0])
                + q * (payoff[(1, 1)][0] - payoff[(0, 1)][0]))
    gain_two = ((F(1) - p) * (payoff[(0, 1)][1] - payoff[(0, 0)][1])
                + p * (payoff[(1, 1)][1] - payoff[(1, 0)][1]))
    return gain_one, gain_two


def action_is_optimal(probability, gain):
    if probability == 0:
        return gain <= 0
    if probability == 1:
        return gain >= 0
    return gain == 0


def is_mixed_nash(payoff, p, q):
    gain_one, gain_two = activation_gains(payoff, p, q)
    return action_is_optimal(p, gain_one) and action_is_optimal(q, gain_two)


def payoff_from_ratios(g, r):
    return build_payoff(*coefficients_from_ratios(g, r))


def welfare(payoff):
    return {
        profile: payoff[profile][0] + payoff[profile][1]
        for profile in PROFILES
    }


def product_distribution(p, q):
    return {
        (0, 0): (F(1) - p) * (F(1) - q),
        (0, 1): (F(1) - p) * q,
        (1, 0): p * (F(1) - q),
        (1, 1): p * q,
    }


def expected_welfare(values, p, q):
    probabilities = product_distribution(p, q)
    return sum(
        (probabilities[profile] * values[profile] for profile in PROFILES),
        F(0),
    )


def externality_compensation(payoff):
    """Return the other player's payoff change from the actor's baseline 0."""
    return {
        (s1, s2): (
            payoff[(s1, s2)][1] - payoff[(0, s2)][1],
            payoff[(s1, s2)][0] - payoff[(s1, 0)][0],
        )
        for s1, s2 in PROFILES
    }


def add_budget_balanced_transfer(payoff, tau):
    return {
        (s1, s2): (
            payoff[(s1, s2)][0] + tau * (s1 - s2),
            payoff[(s1, s2)][1] - tau * (s1 - s2),
        )
        for s1, s2 in PROFILES
    }


def main():
    assert CHANNEL_VALUES == ((F(7, 10), F(1)), (F(9, 10), F(1)))
    assert FULL_VALUES == (F(17, 10), F(19, 10))
    assert BENEFITS == {
        (0, 0): (F(0), F(0)),
        (0, 1): (F(10, 17), F(10, 19)),
        (1, 0): (F(7, 17), F(9, 19)),
        (1, 1): (F(1), F(1)),
    }
    assert A == F(7, 17)
    assert B == F(10, 19)
    assert A < B

    cases = (
        (F(1, 10), F(1, 10)),
        (F(1, 10), A - F(1, 10)),
        (F(1, 5), F(1, 4)),
        (F(1, 5), B - F(1, 5)),
        (F(1, 3), F(2, 9)),
        (A, F(1, 5)),
        (B, F(1, 10)),
        (F(7, 10), F(1, 10)),
        (A, F(0)),
        (B, F(0)),
    )
    for g, r in cases:
        assert pure_nash(payoff_from_ratios(g, r)) == predicted_pure(g, r)

    g = F(1, 3)
    r = F(2, 9)
    payoff = payoff_from_ratios(g, r)
    p_star = (B - g) / r
    q_star = (A - g) / r
    assert p_star == F(33, 38)
    assert q_star == F(6, 17)
    assert 0 < p_star < 1 and 0 < q_star < 1
    assert activation_gains(payoff, p_star, q_star) == (F(0), F(0))
    assert is_mixed_nash(payoff, p_star, q_star)

    values = welfare(payoff)
    assert values == {
        (0, 0): F(0),
        (0, 1): F(757, 1615),
        (1, 0): F(107, 323),
        (1, 1): F(8, 15),
    }
    assert {profile for profile, value in values.items()
            if value == max(values.values())} == {(1, 1)}
    probabilities = product_distribution(p_star, q_star)
    assert probabilities == {
        (0, 0): F(55, 646),
        (0, 1): F(30, 646),
        (1, 0): F(363, 646),
        (1, 1): F(198, 646),
    }
    mixed_welfare = expected_welfare(values, p_star, q_star)
    assert mixed_welfare == F(387459, 1043290)
    assert pure_nash(payoff) == {(0, 1), (1, 0)}
    worst_pure = min(values[profile] for profile in pure_nash(payoff))
    assert worst_pure == F(107, 323)
    assert mixed_welfare > worst_pure
    pure_poa = values[(1, 1)] / worst_pure
    all_poa = values[(1, 1)] / min(worst_pure, mixed_welfare)
    assert pure_poa == F(2584, 1605)
    assert all_poa == pure_poa

    # Exact externalities and the induced welfare differences.
    compensation = externality_compensation(payoff)
    assert compensation == {
        (0, 0): (F(0), F(0)),
        (0, 1): (F(0), F(6, 17)),
        (1, 0): (F(27, 95), F(0)),
        (1, 1): (F(43, 285), F(56, 255)),
    }
    expected_changes = {
        (1, 0): (F(4, 85), F(27, 95), F(107, 323)),
        (1, 1): (F(-22, 255), F(43, 285), F(313, 4845)),
        (2, 0): (F(11, 95), F(6, 17), F(757, 1615)),
        (2, 1): (F(-1, 57), F(56, 255), F(979, 4845)),
    }
    for opponent in (0, 1):
        own = payoff[(1, opponent)][0] - payoff[(0, opponent)][0]
        other = payoff[(1, opponent)][1] - payoff[(0, opponent)][1]
        total = values[(1, opponent)] - values[(0, opponent)]
        assert (own, other, total) == expected_changes[(1, opponent)]
        modified = (
            payoff[(1, opponent)][0] + compensation[(1, opponent)][0]
            - payoff[(0, opponent)][0] - compensation[(0, opponent)][0]
        )
        assert modified == total
    for opponent in (0, 1):
        own = payoff[(opponent, 1)][1] - payoff[(opponent, 0)][1]
        other = payoff[(opponent, 1)][0] - payoff[(opponent, 0)][0]
        total = values[(opponent, 1)] - values[(opponent, 0)]
        assert (own, other, total) == expected_changes[(2, opponent)]
        modified = (
            payoff[(opponent, 1)][1] + compensation[(opponent, 1)][1]
            - payoff[(opponent, 0)][1] - compensation[(opponent, 0)][1]
        )
        assert modified == total
    assert sum(compensation[(1, 1)], F(0)) == F(359, 969)

    # Exact thresholds and equilibrium changes under the budget-balanced rule.
    tau_two = payoff[(1, 0)][1] - payoff[(1, 1)][1]
    tau_one = payoff[(0, 1)][0] - payoff[(1, 1)][0]
    assert tau_two == F(1, 57)
    assert tau_one == F(22, 255)
    assert tau_two < tau_one < F(1, 10)

    transfer_cases = (
        (F(0), {(0, 1), (1, 0)}),
        (tau_two, {(0, 1), (1, 0)}),
        ((tau_two + tau_one) / 2, {(0, 1)}),
        (tau_one, {(0, 1), (1, 1)}),
        (F(1, 10), {(1, 1)}),
    )
    for tau, expected_equilibria in transfer_cases:
        modified_payoff = add_budget_balanced_transfer(payoff, tau)
        assert pure_nash(modified_payoff) == expected_equilibria
        assert welfare(modified_payoff) == values
        for profile in PROFILES:
            transfer_one = modified_payoff[profile][0] - payoff[profile][0]
            transfer_two = modified_payoff[profile][1] - payoff[profile][1]
            assert transfer_one + transfer_two == 0

    at_tau_one = add_budget_balanced_transfer(payoff, tau_one)
    assert all(is_mixed_nash(at_tau_one, p, F(1))
               for p in (F(0), F(1, 3), F(1)))
    assert values[(1, 1)] / values[(0, 1)] == F(2584, 2271)

    at_point_one = add_budget_balanced_transfer(payoff, F(1, 10))
    assert all(value > 0 for value in (
        at_point_one[(1, 0)][0] - at_point_one[(0, 0)][0],
        at_point_one[(1, 1)][0] - at_point_one[(0, 1)][0],
        at_point_one[(0, 1)][1] - at_point_one[(0, 0)][1],
        at_point_one[(1, 1)][1] - at_point_one[(1, 0)][1],
    ))

    # The n-player equal-sharing identity is exact, but its converse fails.
    sample_payoffs = (F(2, 7), F(-1, 5), F(11, 13))
    sample_welfare = sum(sample_payoffs, F(0))
    equal_share = sample_welfare / 3
    transfers = tuple(equal_share - value for value in sample_payoffs)
    assert sum(transfers, F(0)) == 0
    assert all(value + transfer == equal_share
               for value, transfer in zip(sample_payoffs, transfers))
    counterexample_welfare = {
        (0, 0): F(1), (0, 1): F(0),
        (1, 0): F(0), (1, 1): F(2),
    }
    counterexample_payoff = {
        profile: (value / 2, value / 2)
        for profile, value in counterexample_welfare.items()
    }
    assert pure_nash(counterexample_payoff) == {(0, 0), (1, 1)}
    assert counterexample_welfare[(0, 0)] < counterexample_welfare[(1, 1)]

    mixed_constant = F(10, 17) * A + F(9, 19) * B
    mixed_g_coefficient = F(10, 17) + F(9, 19)
    assert mixed_constant == F(51280, 104329)
    assert mixed_g_coefficient == F(343, 323)
    for parametric_g, parametric_r in (
            (F(1, 3), F(2, 9)),
            (F(1, 10), F(1)),
            (F(1, 5), F(1, 2))):
        assert parametric_g < A and parametric_g + parametric_r > B
        alpha, _, _ = coefficients_from_ratios(parametric_g, parametric_r)
        parametric_p = (B - parametric_g) / parametric_r
        parametric_q = (A - parametric_g) / parametric_r
        direct = expected_welfare(
            welfare(payoff_from_ratios(parametric_g, parametric_r)),
            parametric_p,
            parametric_q,
        )
        closed_form = alpha / parametric_r * (
            mixed_constant - mixed_g_coefficient * parametric_g
        )
        assert direct == closed_form

    # g + r = a: player 1 may use any mixture while player 2 activates.
    g = F(1, 10)
    payoff = payoff_from_ratios(g, A - g)
    assert all(is_mixed_nash(payoff, p, F(1)) for p in (F(0), F(1, 2), F(1)))

    # g = b: player 1 is inactive while player 2 may use any mixture.
    payoff = payoff_from_ratios(B, F(1, 10))
    assert all(is_mixed_nash(payoff, F(0), q) for q in (F(0), F(1, 2), F(1)))

    # g + r = b with g < a: the second player's boundary interval is exact.
    g = F(1, 5)
    r = B - g
    payoff = payoff_from_ratios(g, r)
    q_max = (A - g) / r
    assert is_mixed_nash(payoff, F(0), F(1))
    assert all(is_mixed_nash(payoff, F(1), q)
               for q in (F(0), q_max / 2, q_max))
    assert not is_mixed_nash(payoff, F(1), (q_max + 1) / 2)

    # g = a with r > b - a: the first player's boundary interval is exact.
    r = F(1, 5)
    payoff = payoff_from_ratios(A, r)
    p_min = (B - A) / r
    assert is_mixed_nash(payoff, F(0), F(1))
    assert all(is_mixed_nash(payoff, p, F(0))
               for p in (p_min, (p_min + 1) / 2, F(1)))
    assert not is_mixed_nash(payoff, p_min / 2, F(0))

    # Zero-redundancy and alpha = 0 degeneracies.
    payoff = payoff_from_ratios(A, F(0))
    assert all(is_mixed_nash(payoff, p, F(1)) for p in (F(0), F(1, 2), F(1)))
    payoff = payoff_from_ratios(B, F(0))
    assert all(is_mixed_nash(payoff, F(0), q) for q in (F(0), F(1, 2), F(1)))
    assert pure_nash(build_payoff(F(0), F(1), F(0))) == {(0, 0)}
    redundancy_only = build_payoff(F(0), F(0), F(1))
    assert pure_nash(redundancy_only) == {(0, 0), (0, 1), (1, 0)}
    assert is_mixed_nash(redundancy_only, F(0), F(2, 3))
    assert is_mixed_nash(redundancy_only, F(2, 3), F(0))
    assert not is_mixed_nash(redundancy_only, F(1, 2), F(1, 2))

    print("PASS: exact reduced-game equilibrium, welfare, PoA, and incentives")


if __name__ == "__main__":
    main()
