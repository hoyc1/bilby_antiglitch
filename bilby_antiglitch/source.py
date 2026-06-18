# Licensed under an MIT style license -- see LICENSE.md

import numpy as np

__author__ = ["Charlie Hoy <charlie.hoy@port.ac.uk>"]


def antiglitch(frequency_array, psd, tstart, A, f, phi, tc, gamma=None, **kwargs):
    """Antiglitch frequency domain source model.
    """
    log_gamma = kwargs.get("log_gamma", None)
    if log_gamma is not None:
        gamma = 10**(log_gamma)
    elif gamma is None:
        raise ValueError("Either gamma or log_gamma must be provided")

    htilde = np.exp(-gamma / 2 * (np.log(frequency_array) - np.log(f)) ** 2)
    N = np.sqrt(np.sum(np.abs(htilde) ** 2 / psd))
    dt = tc - tstart
    hf = A * np.exp(2j * phi - 2 * np.pi * 1j * frequency_array * dt) * htilde / N
    return hf
