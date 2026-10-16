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

struct Outcome {
    int s1;
    int s2;
    std::array<double, 2> benefit;
    std::array<double, 2> payoff;
};

std::size_t profile_index(int s1, int s2) {
    return static_cast<std::size_t>(2 * s1 + s2);
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

bool contains(const std::vector<int>& actions, int action) {
    return std::find(actions.begin(), actions.end(), action) !=
        actions.end();
}

std::string action_set(const std::vector<int>& actions) {
    std::string result = "{";
    for (std::size_t i = 0; i < actions.size(); ++i) {
        if (i != 0) {
            result += ',';
        }
        result += std::to_string(actions[i]);
    }
    result += '}';
    return result;
}

} // namespace

int main() {
    constexpr int K = 1;
    constexpr double horizon = 1.5;
    constexpr double control_weight = 0.7;
    constexpr double cost_weight = 0.3;
    constexpr double selection_cost = 0.6;
    constexpr double intervention_cost = 0.4;
    constexpr double budget = 1.0;
    const double tolerance =
        2048.0 * std::numeric_limits<double>::epsilon();

    // Reduced game: beta_i = eta_i = delta_i = 0 and
    // alpha_i = 0.7, gamma_i = 0.3. Each selected channel has
    // c_i = c_i^sel + c_i^int = 0.6 + 0.4 = M_i = 1, so
    // K_i(s_i) equals the binary activation decision.
    const double normalized_activation_cost =
        (selection_cost + intervention_cost) / budget;

    // Fixed library: player 1 owns the cosine channel and player 2
    // owns the independent mean-mode channel.
    const bbm::Spectrum actuator_one{
        bbm::Complex{0.5, 0.0},
        bbm::Complex{0.0, 0.0},
        bbm::Complex{0.5, 0.0}
    };
    const bbm::Spectrum actuator_two{
        bbm::Complex{0.0, 0.0},
        bbm::Complex{1.0, 0.0},
        bbm::Complex{0.0, 0.0}
    };
    const std::array<std::array<double, 3>, 2> weights{
        std::array<double, 3>{0.2, 0.3, 0.5},
        std::array<double, 3>{0.4, 0.1, 0.5}
    };

    const bbm::Gramian full_gramian = bbm::linearized_gramian(
        {actuator_one, actuator_two},
        K,
        horizon
    );
    const std::array<double, 2> full_benefit{
        weighted_diagonal(full_gramian, weights[0]),
        weighted_diagonal(full_gramian, weights[1])
    };

    std::array<Outcome, 4> outcomes{};
    for (int s1 = 0; s1 <= 1; ++s1) {
        for (int s2 = 0; s2 <= 1; ++s2) {
            std::vector<bbm::Spectrum> selected;
            if (s1 == 1) {
                selected.push_back(actuator_one);
            }
            if (s2 == 1) {
                selected.push_back(actuator_two);
            }

            const bbm::Gramian gramian =
                bbm::linearized_gramian(selected, K, horizon);
            Outcome& outcome = outcomes[profile_index(s1, s2)];
            outcome.s1 = s1;
            outcome.s2 = s2;

            for (std::size_t i = 0; i < weights.size(); ++i) {
                outcome.benefit[i] =
                    weighted_diagonal(gramian, weights[i]);
                const double normalized_control =
                    outcome.benefit[i] / full_benefit[i];
                const int own_action = i == 0 ? s1 : s2;
                outcome.payoff[i] =
                    control_weight * normalized_control -
                    cost_weight * normalized_activation_cost * own_action;
            }
        }
    }

    const std::array<std::array<double, 2>, 4> expected_benefits{
        std::array<double, 2>{0.0, 0.0},
        std::array<double, 2>{0.45, 0.15},
        std::array<double, 2>{0.065625, 0.084375},
        std::array<double, 2>{0.515625, 0.234375}
    };
    const std::array<std::array<double, 2>, 4> expected_payoffs{
        std::array<double, 2>{0.0, 0.0},
        std::array<double, 2>{0.6109090909090909, 0.148},
        std::array<double, 2>{-0.2109090909090909, 0.252},
        std::array<double, 2>{0.4, 0.4}
    };

    bool passed =
        full_benefit[0] > tolerance &&
        full_benefit[1] > tolerance &&
        std::abs(control_weight + cost_weight - 1.0) <= tolerance;

    for (std::size_t p = 0; p < outcomes.size(); ++p) {
        for (std::size_t i = 0; i < weights.size(); ++i) {
            passed = passed &&
                std::abs(
                    outcomes[p].benefit[i] - expected_benefits[p][i]
                ) <= tolerance &&
                std::abs(
                    outcomes[p].payoff[i] - expected_payoffs[p][i]
                ) <= tolerance;
        }
    }

    std::array<std::vector<int>, 2> best_response_one;
    std::array<std::vector<int>, 2> best_response_two;

    for (int s2 = 0; s2 <= 1; ++s2) {
        const double maximum = std::max(
            outcomes[profile_index(0, s2)].payoff[0],
            outcomes[profile_index(1, s2)].payoff[0]
        );
        for (int s1 = 0; s1 <= 1; ++s1) {
            if (std::abs(
                outcomes[profile_index(s1, s2)].payoff[0] - maximum
            ) <= tolerance) {
                best_response_one[s2].push_back(s1);
            }
        }
    }

    for (int s1 = 0; s1 <= 1; ++s1) {
        const double maximum = std::max(
            outcomes[profile_index(s1, 0)].payoff[1],
            outcomes[profile_index(s1, 1)].payoff[1]
        );
        for (int s2 = 0; s2 <= 1; ++s2) {
            if (std::abs(
                outcomes[profile_index(s1, s2)].payoff[1] - maximum
            ) <= tolerance) {
                best_response_two[s1].push_back(s2);
            }
        }
    }

    passed = passed &&
        best_response_one[0] == std::vector<int>{0} &&
        best_response_one[1] == std::vector<int>{0} &&
        best_response_two[0] == std::vector<int>{1} &&
        best_response_two[1] == std::vector<int>{1};

    std::array<bool, 4> is_nash{};
    for (const auto& outcome : outcomes) {
        const std::size_t p = profile_index(outcome.s1, outcome.s2);
        is_nash[p] =
            contains(best_response_one[outcome.s2], outcome.s1) &&
            contains(best_response_two[outcome.s1], outcome.s2);
    }
    passed = passed &&
        !is_nash[0] && is_nash[1] && !is_nash[2] && !is_nash[3];

    std::cout << std::setprecision(16)
              << "profile H1 H2 J1 J2 BR1(s2) BR2(s1) Nash\n";
    for (std::size_t p = 0; p < outcomes.size(); ++p) {
        const auto& outcome = outcomes[p];
        std::cout << '(' << outcome.s1 << ',' << outcome.s2 << ") "
                  << outcome.benefit[0] << ' '
                  << outcome.benefit[1] << ' '
                  << outcome.payoff[0] << ' '
                  << outcome.payoff[1] << ' '
                  << action_set(best_response_one[outcome.s2]) << ' '
                  << action_set(best_response_two[outcome.s1]) << ' '
                  << (is_nash[p] ? "yes" : "no") << '\n';
    }
    std::cout << (passed ? "PASS\n" : "FAIL\n");
    return passed ? 0 : 1;
}
