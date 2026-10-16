#include "bbm_gramian.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using Payoff = std::array<std::array<double, 2>, 4>;
using ProfileFlags = std::array<bool, 4>;

struct Parameters {
    double alpha;
    double gamma;
    double delta;
};

struct FixedModel {
    std::array<std::array<double, 2>, 4> normalized_benefit{};
    double rho{};
};

struct Case {
    std::string name;
    Parameters parameters;
    ProfileFlags expected_pure;
};

std::size_t profile_index(int s1, int s2) {
    return static_cast<std::size_t>(2 * s1 + s2);
}

bbm::Spectrum bbm_scaled(const bbm::Spectrum& fourier, int K) {
    bbm::Spectrum scaled = fourier;
    for (int k = -K; k <= K; ++k) {
        const double kd = static_cast<double>(k);
        scaled[k + K] /= 1.0 + kd * kd;
    }
    return scaled;
}

double modal_similarity(
    const bbm::Spectrum& first,
    const bbm::Spectrum& second
) {
    bbm::Complex inner{0.0, 0.0};
    double first_norm_squared = 0.0;
    double second_norm_squared = 0.0;
    for (std::size_t k = 0; k < first.size(); ++k) {
        inner += std::conj(first[k]) * second[k];
        first_norm_squared += std::norm(first[k]);
        second_norm_squared += std::norm(second[k]);
    }
    return std::norm(inner) /
        (first_norm_squared * second_norm_squared);
}

double weighted_diagonal(
    const bbm::Gramian& gramian,
    const std::array<double, 3>& weights
) {
    double result = 0.0;
    for (std::size_t k = 0; k < weights.size(); ++k) {
        result += weights[k] * gramian[k][k].real();
    }
    return result;
}

FixedModel reconstruct_fixed_model() {
    constexpr int K = 1;
    constexpr double horizon = 1.5;
    const bbm::Spectrum first{
        {2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}
    };
    const bbm::Spectrum second{
        {2.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}
    };
    const std::array<std::array<double, 3>, 2> weights{
        std::array<double, 3>{0.2, 0.3, 0.5},
        std::array<double, 3>{0.4, 0.1, 0.5}
    };

    const bbm::Gramian full = bbm::linearized_gramian(
        {first, second}, K, horizon
    );
    const std::array<double, 2> denominator{
        weighted_diagonal(full, weights[0]),
        weighted_diagonal(full, weights[1])
    };

    FixedModel model;
    model.rho = modal_similarity(
        bbm_scaled(first, K), bbm_scaled(second, K)
    );
    for (int s1 = 0; s1 <= 1; ++s1) {
        for (int s2 = 0; s2 <= 1; ++s2) {
            std::vector<bbm::Spectrum> selected;
            if (s1 == 1) {
                selected.push_back(first);
            }
            if (s2 == 1) {
                selected.push_back(second);
            }
            const bbm::Gramian gramian = bbm::linearized_gramian(
                selected, K, horizon
            );
            const std::size_t profile = profile_index(s1, s2);
            for (std::size_t player = 0; player < weights.size(); ++player) {
                model.normalized_benefit[profile][player] =
                    weighted_diagonal(gramian, weights[player]) /
                    denominator[player];
            }
        }
    }
    return model;
}

Parameters from_ratios(double g, double r) {
    const double alpha = 1.0 / (1.0 + g + 1.5 * r);
    return {alpha, g * alpha, 1.5 * r * alpha};
}

Payoff build_payoff(
    const FixedModel& model,
    const Parameters& parameters
) {
    Payoff payoff{};
    for (int s1 = 0; s1 <= 1; ++s1) {
        for (int s2 = 0; s2 <= 1; ++s2) {
            const std::size_t profile = profile_index(s1, s2);
            const double redundancy =
                model.rho * static_cast<double>(s1 * s2);
            payoff[profile][0] =
                parameters.alpha * model.normalized_benefit[profile][0] -
                parameters.gamma * s1 - parameters.delta * redundancy;
            payoff[profile][1] =
                parameters.alpha * model.normalized_benefit[profile][1] -
                parameters.gamma * s2 - parameters.delta * redundancy;
        }
    }
    return payoff;
}

ProfileFlags enumerate_pure_nash(const Payoff& payoff, double tolerance) {
    ProfileFlags result{};
    for (int s1 = 0; s1 <= 1; ++s1) {
        for (int s2 = 0; s2 <= 1; ++s2) {
            const std::size_t profile = profile_index(s1, s2);
            const std::size_t player_one_deviation =
                profile_index(1 - s1, s2);
            const std::size_t player_two_deviation =
                profile_index(s1, 1 - s2);
            result[profile] =
                payoff[profile][0] + tolerance >=
                    payoff[player_one_deviation][0] &&
                payoff[profile][1] + tolerance >=
                    payoff[player_two_deviation][1];
        }
    }
    return result;
}

ProfileFlags symbolic_pure_nash(
    const Parameters& parameters,
    double tolerance
) {
    const double lower = 7.0 * parameters.alpha / 17.0;
    const double upper = 10.0 * parameters.alpha / 19.0;
    const double effective_penalty =
        parameters.gamma + 2.0 * parameters.delta / 3.0;
    return {
        parameters.gamma + tolerance >= upper,
        parameters.gamma <= upper + tolerance &&
            effective_penalty + tolerance >= lower,
        parameters.gamma <= lower + tolerance &&
            effective_penalty + tolerance >= upper,
        effective_penalty <= lower + tolerance
    };
}

std::array<double, 4> welfare(const Payoff& payoff) {
    std::array<double, 4> result{};
    for (std::size_t profile = 0; profile < result.size(); ++profile) {
        result[profile] = payoff[profile][0] + payoff[profile][1];
    }
    return result;
}

ProfileFlags enumerate_welfare_maximizers(
    const std::array<double, 4>& values,
    double tolerance
) {
    const double maximum = *std::max_element(values.begin(), values.end());
    ProfileFlags result{};
    for (std::size_t profile = 0; profile < values.size(); ++profile) {
        result[profile] = values[profile] + tolerance >= maximum;
    }
    return result;
}

ProfileFlags symbolic_welfare_maximizers(
    const Parameters& parameters,
    double tolerance
) {
    if (parameters.alpha <= tolerance) {
        if (parameters.gamma <= tolerance) {
            return {true, true, true, false};
        }
        return {true, false, false, false};
    }

    const double inactive_boundary =
        360.0 * parameters.alpha / 323.0;
    const double joint_boundary =
        286.0 * parameters.alpha / 323.0;
    const double joint_expression =
        parameters.gamma + 4.0 * parameters.delta / 3.0;
    return {
        parameters.gamma + tolerance >= inactive_boundary,
        parameters.gamma <= inactive_boundary + tolerance &&
            joint_expression + tolerance >= joint_boundary,
        false,
        joint_expression <= joint_boundary + tolerance
    };
}

bool action_is_optimal(double probability, double gain, double tolerance) {
    if (probability <= tolerance) {
        return gain <= tolerance;
    }
    if (probability >= 1.0 - tolerance) {
        return gain >= -tolerance;
    }
    return std::abs(gain) <= tolerance;
}

bool is_mixed_nash(
    const Payoff& payoff,
    double p,
    double q,
    double tolerance
) {
    const double player_one_gain =
        (1.0 - q) * (payoff[2][0] - payoff[0][0]) +
        q * (payoff[3][0] - payoff[1][0]);
    const double player_two_gain =
        (1.0 - p) * (payoff[1][1] - payoff[0][1]) +
        p * (payoff[3][1] - payoff[2][1]);
    return action_is_optimal(p, player_one_gain, tolerance) &&
        action_is_optimal(q, player_two_gain, tolerance);
}

double expected_welfare(
    const std::array<double, 4>& values,
    double p,
    double q
) {
    return (1.0 - p) * (1.0 - q) * values[0] +
        (1.0 - p) * q * values[1] +
        p * (1.0 - q) * values[2] + p * q * values[3];
}

Payoff add_budget_balanced_transfer(const Payoff& payoff, double tau) {
    Payoff result = payoff;
    for (int s1 = 0; s1 <= 1; ++s1) {
        for (int s2 = 0; s2 <= 1; ++s2) {
            const std::size_t profile = profile_index(s1, s2);
            const double transfer = tau * static_cast<double>(s1 - s2);
            result[profile][0] += transfer;
            result[profile][1] -= transfer;
        }
    }
    return result;
}

} // namespace

int main() {
    const double tolerance =
        32768.0 * std::numeric_limits<double>::epsilon();
    const double a = 7.0 / 17.0;
    const double b = 10.0 / 19.0;
    const double mixed_boundary_g = 0.20;
    const double mixed_boundary_r =
        (51280.0 / 104329.0 - (343.0 / 323.0) * mixed_boundary_g) /
        (286.0 / 323.0 - mixed_boundary_g);
    const FixedModel model = reconstruct_fixed_model();
    const std::array<std::array<double, 2>, 4> expected_benefit{
        std::array<double, 2>{0.0, 0.0},
        std::array<double, 2>{10.0 / 17.0, 10.0 / 19.0},
        std::array<double, 2>{7.0 / 17.0, 9.0 / 19.0},
        std::array<double, 2>{1.0, 1.0}
    };

    bool passed = std::abs(model.rho - 2.0 / 3.0) <= tolerance;
    for (std::size_t profile = 0; profile < expected_benefit.size(); ++profile) {
        for (std::size_t player = 0; player < 2; ++player) {
            passed = passed && std::abs(
                model.normalized_benefit[profile][player] -
                expected_benefit[profile][player]
            ) <= tolerance;
        }
    }

    const std::vector<Case> cases{
        {"unique-11", from_ratios(0.10, 0.10),
            {false, false, false, true}},
        {"11-01-boundary", from_ratios(0.10, a - 0.10),
            {false, true, false, true}},
        {"unique-01", from_ratios(0.20, 0.25),
            {false, true, false, false}},
        {"10-01-boundary", from_ratios(0.20, b - 0.20),
            {false, true, true, false}},
        {"strict-anti-coordination", from_ratios(1.0 / 3.0, 2.0 / 9.0),
            {false, true, true, false}},
        {"mixed-pure-welfare-boundary",
            from_ratios(mixed_boundary_g, mixed_boundary_r),
            {false, true, true, false}},
        {"player-one-indifference", from_ratios(a, 0.20),
            {false, true, true, false}},
        {"00-01-boundary", from_ratios(b, 0.10),
            {true, true, false, false}},
        {"unique-00-positive-optimum", from_ratios(0.70, 0.10),
            {true, false, false, false}},
        {"unique-00-zero-optimum", from_ratios(1.20, 0.10),
            {true, false, false, false}},
        {"high-redundancy-mixed-worst", from_ratios(0.10, 5.0),
            {false, true, true, false}},
        {"zero-redundancy-player-one-tie", from_ratios(a, 0.0),
            {false, true, false, true}},
        {"zero-redundancy-player-two-tie", from_ratios(b, 0.0),
            {true, true, false, false}},
        {"zero-benefit-cost-only", {0.0, 1.0, 0.0},
            {true, false, false, false}},
        {"zero-benefit-redundancy-only", {0.0, 0.0, 1.0},
            {true, true, true, false}}
    };

    for (const Case& test_case : cases) {
        const Parameters& parameters = test_case.parameters;
        passed = passed && parameters.alpha >= 0.0 &&
            parameters.gamma >= 0.0 && parameters.delta >= 0.0 &&
            std::abs(
                parameters.alpha + parameters.gamma + parameters.delta - 1.0
            ) <= tolerance;

        const Payoff payoff = build_payoff(model, parameters);
        const ProfileFlags enumerated = enumerate_pure_nash(payoff, tolerance);
        const ProfileFlags symbolic =
            symbolic_pure_nash(parameters, tolerance);
        passed = passed && enumerated == symbolic &&
            enumerated == test_case.expected_pure;

        const double d = 2.0 * parameters.delta / 3.0;
        const double first_gain = 7.0 * parameters.alpha / 17.0 -
            parameters.gamma;
        const double second_gain = 10.0 * parameters.alpha / 19.0 -
            parameters.gamma;
        passed = passed &&
            std::abs((payoff[2][0] - payoff[0][0]) - first_gain)
                <= tolerance &&
            std::abs((payoff[3][0] - payoff[1][0]) - (first_gain - d))
                <= tolerance &&
            std::abs((payoff[1][1] - payoff[0][1]) - second_gain)
                <= tolerance &&
            std::abs((payoff[3][1] - payoff[2][1]) - (second_gain - d))
                <= tolerance;

        const std::array<double, 4> values = welfare(payoff);
        const std::array<double, 4> expected_values{
            0.0,
            360.0 * parameters.alpha / 323.0 - parameters.gamma,
            286.0 * parameters.alpha / 323.0 - parameters.gamma,
            2.0 * parameters.alpha - 2.0 * parameters.gamma -
                4.0 * parameters.delta / 3.0
        };
        for (std::size_t profile = 0; profile < values.size(); ++profile) {
            passed = passed &&
                std::abs(values[profile] - expected_values[profile])
                    <= tolerance;
        }
        passed = passed &&
            enumerate_welfare_maximizers(values, tolerance) ==
                symbolic_welfare_maximizers(parameters, tolerance);

        const bool symbolic_completely_mixed =
            d > tolerance && first_gain > tolerance &&
            second_gain > tolerance && first_gain < d - tolerance &&
            second_gain < d - tolerance;
        const bool direct_completely_mixed =
            payoff[2][0] - payoff[0][0] > tolerance &&
            payoff[3][0] - payoff[1][0] < -tolerance &&
            payoff[1][1] - payoff[0][1] > tolerance &&
            payoff[3][1] - payoff[2][1] < -tolerance;
        passed = passed &&
            symbolic_completely_mixed == direct_completely_mixed;

        if (symbolic_completely_mixed) {
            const double p = second_gain / d;
            const double q = first_gain / d;
            const double mixed = expected_welfare(values, p, q);
            const double indifference_value =
                q * 10.0 * parameters.alpha / 17.0 +
                p * 9.0 * parameters.alpha / 19.0;
            const double g = parameters.gamma / parameters.alpha;
            const double r = d / parameters.alpha;
            const double expected_mixed_normalized =
                ((10.0 / 17.0) * (a - g) +
                 (9.0 / 19.0) * (b - g)) / r;
            const double closed_form_mixed_normalized =
                (51280.0 / 104329.0 - (343.0 / 323.0) * g) / r;
            const double normalized_first_pure = 360.0 / 323.0 - g;
            const double normalized_second_pure = 286.0 / 323.0 - g;
            const double critical_r =
                (51280.0 / 104329.0 - (343.0 / 323.0) * g) /
                normalized_second_pure;
            passed = passed && is_mixed_nash(payoff, p, q, tolerance) &&
                std::abs(mixed - indifference_value) <= tolerance &&
                std::abs(
                    mixed / parameters.alpha - expected_mixed_normalized
                ) <= tolerance &&
                std::abs(
                    expected_mixed_normalized - closed_form_mixed_normalized
                ) <= tolerance &&
                expected_mixed_normalized > 0.0 &&
                expected_mixed_normalized < normalized_first_pure &&
                critical_r > b - g;
            if (r < critical_r - tolerance) {
                passed = passed &&
                    expected_mixed_normalized > normalized_second_pure;
            } else if (r > critical_r + tolerance) {
                passed = passed &&
                    expected_mixed_normalized < normalized_second_pure;
            } else {
                passed = passed && std::abs(
                    expected_mixed_normalized - normalized_second_pure
                ) <= tolerance;
            }

            double worst_pure = std::numeric_limits<double>::infinity();
            for (std::size_t profile = 0; profile < values.size(); ++profile) {
                if (enumerated[profile]) {
                    worst_pure = std::min(worst_pure, values[profile]);
                }
            }
            const double optimum =
                *std::max_element(values.begin(), values.end());
            const double pure_price = optimum / worst_pure;
            const double all_price = optimum / std::min(worst_pure, mixed);
            const double normalized_optimum = std::max({
                0.0,
                normalized_first_pure,
                2.0 * (1.0 - g - r)
            });
            const double expected_pure_price =
                normalized_optimum / normalized_second_pure;
            const double expected_all_price =
                normalized_optimum /
                std::min(normalized_second_pure, expected_mixed_normalized);
            passed = passed &&
                std::abs(pure_price - expected_pure_price) <= tolerance &&
                std::abs(all_price - expected_all_price) <= tolerance &&
                all_price + tolerance >= pure_price;
            if (test_case.name == "high-redundancy-mixed-worst") {
                passed = passed && mixed < worst_pure &&
                    all_price > pure_price;
            }
        }

        for (const double tau : {0.0, 0.03, 0.10, 0.25}) {
            const Payoff transferred =
                add_budget_balanced_transfer(payoff, tau);
            Parameters effective = parameters;
            effective.gamma -= tau;
            passed = passed &&
                enumerate_pure_nash(transferred, tolerance) ==
                    symbolic_pure_nash(effective, tolerance);
            const std::array<double, 4> transferred_welfare =
                welfare(transferred);
            for (std::size_t profile = 0; profile < values.size(); ++profile) {
                passed = passed && std::abs(
                    transferred_welfare[profile] - values[profile]
                ) <= tolerance;
            }
        }
    }

    const Parameters lower_boundary = from_ratios(0.10, a - 0.10);
    const Payoff lower_payoff = build_payoff(model, lower_boundary);
    passed = passed && is_mixed_nash(lower_payoff, 0.37, 1.0, tolerance);

    const double boundary_g = 0.20;
    const double boundary_r = b - boundary_g;
    const Parameters upper_boundary = from_ratios(boundary_g, boundary_r);
    const Payoff upper_payoff = build_payoff(model, upper_boundary);
    const double upper_q = (a - boundary_g) / boundary_r;
    passed = passed &&
        is_mixed_nash(upper_payoff, 1.0, 0.5 * upper_q, tolerance);

    const double indifference_r = 0.20;
    const Parameters player_one_tie = from_ratios(a, indifference_r);
    const Payoff player_one_tie_payoff = build_payoff(model, player_one_tie);
    const double lower_p = (b - a) / indifference_r;
    passed = passed && is_mixed_nash(
        player_one_tie_payoff, 0.5 * (1.0 + lower_p), 0.0, tolerance
    );

    const Parameters player_two_tie = from_ratios(b, 0.10);
    passed = passed && is_mixed_nash(
        build_payoff(model, player_two_tie), 0.0, 0.41, tolerance
    );

    const Payoff redundancy_only = build_payoff(
        model, {0.0, 0.0, 1.0}
    );
    passed = passed &&
        is_mixed_nash(redundancy_only, 0.0, 0.73, tolerance) &&
        is_mixed_nash(redundancy_only, 0.42, 0.0, tolerance) &&
        !is_mixed_nash(redundancy_only, 0.42, 0.73, tolerance);

    std::cout << std::setprecision(16)
              << "Validated parameter cases: " << cases.size() << '\n'
              << "Pure boundaries: a=" << a << ", b=" << b << '\n'
              << "Welfare boundaries: 286/323=" << 286.0 / 323.0
              << ", 360/323=" << 360.0 / 323.0 << '\n'
              << (passed ? "PASS\n" : "FAIL\n");
    return passed ? 0 : 1;
}
