import numpy as np
from scipy.signal import butter, sosfiltfilt
import math


def tddr(data: np.array, sample_rate: int) -> np.array:
    """
    Apply Temporal Derivative Distribution Repair algorithm.

    Fishburn F.A., Ludlum R.S., Vaidya C.J., & Medvedev A.V. (2019).
    Temporal Derivative Distribution Repair (TDDR): A motion correction
    method for fNIRS. NeuroImage, 184, 171-179.
    https://doi.org/10.1016/j.neuroimage.2018.09.025
    """
    signal = np.array(data)
    if len(signal.shape) != 1:
        for ch in range(signal.shape[1]):
            signal[:, ch] = tddr(signal[:, ch], sample_rate)
        return signal

    # Preprocess: Separate high and low frequencies
    filter_cutoff = .5
    filter_order = 3
    signal_mean = math.fsum(signal) / len(signal)
    signal -= signal_mean
    sos = butter(N=filter_order, Wn=filter_cutoff, output='sos', fs=sample_rate)
    signal_low = sosfiltfilt(sos, signal)
    signal_high = signal - signal_low

    # Initialize
    tune = 4.685
    D = np.sqrt(np.finfo(signal.dtype).eps)
    mu = np.inf
    iter = 0

    # Step 1. Compute temporal derivative of the signal
    deriv = np.diff(signal_low)

    # Step 2. Initialize observation weights
    w = np.ones(deriv.shape)

    # Step 3. Iterative estimation of robust weights
    while iter < 50:

        iter = iter + 1
        mu0 = mu

        # Step 3a. Estimate weighted mean
        mu = np.sum(w * deriv) / np.sum(w)

        # Step 3b. Calculate absolute residuals of estimate
        dev = np.abs(deriv - mu)

        # Step 3c. Robust estimate of standard deviation of the residuals
        sigma = 1.4826 * np.median(dev)

        # Step 3d. Scale deviations by standard deviation and tuning parameter
        r = dev / (sigma * tune)

        # Step 3e. Calculate new weights according to Tukey's biweight function
        w = ((1 - r**2) * (r < 1)) ** 2

        # Step 3f. Terminate if new estimate is within machine-precision of old estimate
        if abs(mu - mu0) < D * max(abs(mu), abs(mu0)):
            break

    # Step 4. Apply robust weights to centered derivative
    new_deriv = w * (deriv - mu)

    # Step 5. Integrate corrected derivative
    signal_low_corrected = np.cumsum(np.insert(new_deriv, 0, 0.0))

    # Postprocess: Center the corrected signal
    signal_low_corrected = signal_low_corrected - np.mean(signal_low_corrected)

    # Postprocess: Merge back with uncorrected high frequency component
    signal_corrected = signal_low_corrected + signal_high + signal_mean

    return signal_corrected
