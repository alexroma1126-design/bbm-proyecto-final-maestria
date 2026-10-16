#pragma once

#include "bbm_spectral.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace bbm {

using Gramian = std::vector<std::vector<Complex>>;

// Finite-horizon Gramian for BBM linearized at u = 0.
// Modes are ordered -K, ..., K. Profiles contain Fourier
// coefficients of the physical actuator shapes b_j(x).
inline Gramian linearized_gramian(
    const std::vector<Spectrum>& profiles,
    int K,
    double T
) {
    if (K < 0 || K > 64 || !std::isfinite(T) || T <= 0.0) {
        throw std::invalid_argument("Invalid truncation or horizon");
    }

    const std::size_t N = static_cast<std::size_t>(2 * K + 1);

    for (const auto& profile : profiles) {
        if (profile.size() != N) {
            throw std::invalid_argument("Invalid actuator profile size");
        }
        for (const auto& coefficient : profile) {
            if (!std::isfinite(coefficient.real()) ||
                !std::isfinite(coefficient.imag())) {
                throw std::invalid_argument(
                    "Nonfinite actuator coefficient"
                );
            }
        }
    }

    Gramian G(N, std::vector<Complex>(N, {0.0, 0.0}));

    for (int k = -K; k <= K; ++k) {
        const double kd = static_cast<double>(k);
        const double dk = 1.0 + kd * kd;
        const double omega_k = -kd / dk;

        for (int ell = -K; ell <= K; ++ell) {
            const double ld = static_cast<double>(ell);
            const double dl = 1.0 + ld * ld;
            const double omega_ell = -ld / dl;

            Complex coupling{0.0, 0.0};

            for (const auto& profile : profiles) {
                const Complex Bk = profile[K + k] / dk;
                const Complex Bell = profile[K + ell] / dl;
                coupling += Bk * std::conj(Bell);
            }

            const double half_phase =
                (omega_k - omega_ell) * T / 2.0;

            const double sinc =
                std::abs(half_phase) < 1e-8
                    ? 1.0 - half_phase * half_phase / 6.0
                    : std::sin(half_phase) / half_phase;

            const Complex integral =
                T * sinc * std::exp(Complex{0.0, half_phase});

            G[K + k][K + ell] = coupling * integral;
        }
    }

    return G;
}

} // namespace bbm
