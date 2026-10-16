#include "bbm_gramian.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using PayoffTable = std::array<std::array<double, 2>, 4>;

struct Analysis {
    std::array<std::vector<int>, 2> best_one;
    std::array<std::vector<int>, 2> best_two;
    std::array<bool, 4> nash{};
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

bool contains(const std::vector<int>& actions, int action) {
    return std::find(actions.begin(), actions.end(), action) !=
        actions.end();
}

Analysis analyze(const PayoffTable& payoff, double tolerance) {
    Analysis result;
    for (int s2 = 0; s2 <= 1; ++s2) {
        const double maximum = std::max(
            payoff[profile_index(0, s2)][0],
            payoff[profile_index(1, s2)][0]
        );
        for (int s1 = 0; s1 <= 1; ++s1) {
            if (std::abs(payoff[profile_index(s1, s2)][0] - maximum)
                <= tolerance) {
                result.best_one[s2].push_back(s1);
            }
        }
    }
    for (int s1 = 0; s1 <= 1; ++s1) {
        const double maximum = std::max(
            payoff[profile_index(s1, 0)][1],
            payoff[profile_index(s1, 1)][1]
        );
        for (int s2 = 0; s2 <= 1; ++s2) {
            if (std::abs(payoff[profile_index(s1, s2)][1] - maximum)
                <= tolerance) {
                result.best_two[s1].push_back(s2);
            }
        }
    }
    for (int s1 = 0; s1 <= 1; ++s1) {
        for (int s2 = 0; s2 <= 1; ++s2) {
            result.nash[profile_index(s1, s2)] =
                contains(result.best_one[s2], s1) &&
                contains(result.best_two[s1], s2);
        }
    }
    return result;
}

std::string action_set(const std::vector<int>& actions) {
    std::string result = "{";
    for (std::size_t i = 0; i < actions.size(); ++i) {
        result += (i == 0 ? "" : ",") + std::to_string(actions[i]);
    }
    return result + '}';
}

} // namespace

int main() {
    constexpr int K = 1;
    constexpr double horizon = 1.5;
    constexpr double alpha = 0.6;
    constexpr double gamma = 0.2;
    constexpr double delta = 0.2;
    constexpr double alpha_without_redundancy = 0.75;
    constexpr double gamma_without_redundancy = 0.25;
    const double tolerance =
        4096.0 * std::numeric_limits<double>::epsilon();

    // The Gramian implementation divides Fourier coefficients by
    // (1+k^2). These original profiles therefore produce exactly
    // c1=(1,0,1) and c2=(1,1,1) for modes (-1,0,1).
    const bbm::Spectrum fourier_one{{2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}};
    const bbm::Spectrum fourier_two{{2.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}};
    const bbm::Spectrum c1 = bbm_scaled(fourier_one, K);
    const bbm::Spectrum c2 = bbm_scaled(fourier_two, K);
    const bbm::Spectrum expected_c1{{1.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    const bbm::Spectrum expected_c2{{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};

    bool passed = true;
    for (std::size_t k = 0; k < c1.size(); ++k) {
        passed = passed &&
            std::abs(c1[k] - expected_c1[k]) <= tolerance &&
            std::abs(c2[k] - expected_c2[k]) <= tolerance;
    }

    // theta=0 because no physical node locations are available.
    const double rho = modal_similarity(c1, c2);
    passed = passed && std::abs(rho - 2.0 / 3.0) <= tolerance;

    const std::array<std::array<double, 3>, 2> weights{
        std::array<double, 3>{0.2, 0.3, 0.5},
        std::array<double, 3>{0.4, 0.1, 0.5}
    };
    const bbm::Gramian full_gramian = bbm::linearized_gramian(
        {fourier_one, fourier_two}, K, horizon
    );
    const std::array<double, 2> full_benefit{
        weighted_diagonal(full_gramian, weights[0]),
        weighted_diagonal(full_gramian, weights[1])
    };

    std::array<std::array<double, 2>, 4> benefits{};
    std::array<std::array<double, 2>, 4> normalized_benefits{};
    std::array<double, 4> redundancy{};
    PayoffTable payoff_with_redundancy{};
    PayoffTable payoff_without_redundancy{};

    for (int s1 = 0; s1 <= 1; ++s1) {
        for (int s2 = 0; s2 <= 1; ++s2) {
            const std::size_t p = profile_index(s1, s2);
            std::vector<bbm::Spectrum> selected;
            if (s1 == 1) {
                selected.push_back(fourier_one);
            }
            if (s2 == 1) {
                selected.push_back(fourier_two);
            }
            const bbm::Gramian gramian =
                bbm::linearized_gramian(selected, K, horizon);
            redundancy[p] = static_cast<double>(s1 * s2) * rho;

            for (std::size_t i = 0; i < weights.size(); ++i) {
                benefits[p][i] = weighted_diagonal(gramian, weights[i]);
                normalized_benefits[p][i] =
                    benefits[p][i] / full_benefit[i];
                const int own_action = i == 0 ? s1 : s2;
                payoff_with_redundancy[p][i] =
                    alpha * normalized_benefits[p][i] - gamma * own_action -
                    delta * redundancy[p];
                payoff_without_redundancy[p][i] =
                    alpha_without_redundancy * normalized_benefits[p][i] -
                    gamma_without_redundancy * own_action;
            }
        }
    }

    const std::array<std::array<double, 2>, 4> expected_benefits{
        std::array<double, 2>{0.0, 0.0},
        std::array<double, 2>{1.5, 1.5},
        std::array<double, 2>{1.05, 1.35},
        std::array<double, 2>{2.55, 2.85}
    };
    const std::array<std::array<double, 2>, 4> expected_normalized{
        std::array<double, 2>{0.0, 0.0},
        std::array<double, 2>{10.0 / 17.0, 10.0 / 19.0},
        std::array<double, 2>{7.0 / 17.0, 9.0 / 19.0},
        std::array<double, 2>{1.0, 1.0}
    };
    const std::array<double, 4> expected_redundancy{
        0.0, 0.0, 0.0, 2.0 / 3.0
    };
    const PayoffTable expected_with{
        std::array<double, 2>{0.0, 0.0},
        std::array<double, 2>{6.0 / 17.0, 11.0 / 95.0},
        std::array<double, 2>{4.0 / 85.0, 27.0 / 95.0},
        std::array<double, 2>{4.0 / 15.0, 4.0 / 15.0}
    };
    const PayoffTable expected_without{
        std::array<double, 2>{0.0, 0.0},
        std::array<double, 2>{15.0 / 34.0, 11.0 / 76.0},
        std::array<double, 2>{1.0 / 17.0, 27.0 / 76.0},
        std::array<double, 2>{0.5, 0.5}
    };

    passed = passed &&
        std::abs(alpha + gamma + delta - 1.0) <= tolerance &&
        std::abs(
            alpha_without_redundancy + gamma_without_redundancy - 1.0
        ) <= tolerance &&
        std::abs(
            alpha / gamma -
            alpha_without_redundancy / gamma_without_redundancy
        ) <= tolerance;

    for (std::size_t p = 0; p < benefits.size(); ++p) {
        passed = passed &&
            std::abs(redundancy[p] - expected_redundancy[p]) <= tolerance;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            passed = passed &&
                std::abs(benefits[p][i] - expected_benefits[p][i])
                    <= tolerance &&
                std::abs(
                    normalized_benefits[p][i] - expected_normalized[p][i]
                ) <= tolerance &&
                std::abs(payoff_with_redundancy[p][i] - expected_with[p][i])
                    <= tolerance &&
                std::abs(payoff_without_redundancy[p][i] - expected_without[p][i])
                    <= tolerance;
        }
    }

    const Analysis with = analyze(payoff_with_redundancy, tolerance);
    const Analysis without = analyze(payoff_without_redundancy, tolerance);
    passed = passed &&
        with.best_one[0] == std::vector<int>{1} &&
        with.best_one[1] == std::vector<int>{0} &&
        with.best_two[0] == std::vector<int>{1} &&
        with.best_two[1] == std::vector<int>{0} &&
        !with.nash[0] && with.nash[1] && with.nash[2] && !with.nash[3] &&
        without.best_one[0] == std::vector<int>{1} &&
        without.best_one[1] == std::vector<int>{1} &&
        without.best_two[0] == std::vector<int>{1} &&
        without.best_two[1] == std::vector<int>{1} &&
        !without.nash[0] && !without.nash[1] &&
        !without.nash[2] && without.nash[3];

    std::cout << std::setprecision(16)
              << "rho=" << rho << '\n'
              << "profile H1 H2 R J1 J2 BR1 BR2 Nash\n";
    for (int s1 = 0; s1 <= 1; ++s1) {
        for (int s2 = 0; s2 <= 1; ++s2) {
            const std::size_t p = profile_index(s1, s2);
            std::cout << '(' << s1 << ',' << s2 << ") "
                      << benefits[p][0] << ' ' << benefits[p][1] << ' '
                      << redundancy[p] << ' '
                      << payoff_with_redundancy[p][0] << ' '
                      << payoff_with_redundancy[p][1] << ' '
                      << action_set(with.best_one[s2]) << ' '
                      << action_set(with.best_two[s1]) << ' '
                      << (with.nash[p] ? "yes" : "no") << '\n';
        }
    }
    std::cout << "Nash without redundancy:";
    for (std::size_t p = 0; p < without.nash.size(); ++p) {
        if (without.nash[p]) {
            std::cout << " (" << p / 2 << ',' << p % 2 << ')';
        }
    }
    std::cout << '\n' << (passed ? "PASS\n" : "FAIL\n");
    return passed ? 0 : 1;
}
