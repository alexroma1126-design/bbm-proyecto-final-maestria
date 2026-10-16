#pragma once

#include <complex>
#include <stdexcept>
#include <vector>

namespace bbm {

using Complex = std::complex<double>;
using Spectrum = std::vector<Complex>;

// Fourier-Galerkin right-hand side of the uncontrolled,
// periodic nonlinear BBM equation. Modes are indexed -K,...,K.
inline Spectrum rhs(const Spectrum& a, int K) {
    if (K < 0 || a.size() != static_cast<std::size_t>(2 * K + 1)) {
        throw std::invalid_argument("Invalid Fourier spectrum size");
    }

    const Complex imaginary{0.0, 1.0};
    Spectrum derivative(a.size(), Complex{0.0, 0.0});

    for (int k = -K; k <= K; ++k) {
        Complex convolution{0.0, 0.0};

        for (int p = -K; p <= K; ++p) {
            const int q = k - p;

            if (q >= -K && q <= K) {
                convolution += a[p + K] * a[q + K];
            }
        }

        const double kd = static_cast<double>(k);
        const double frequency = kd / (1.0 + kd * kd);

        derivative[k + K] =
            -imaginary * frequency *
            (a[k + K] + 0.5 * convolution);
    }

    return derivative;
}

} // namespace bbm
