import json
import numpy as np
from scipy import fft as scipy_fft
from scipy.interpolate import CubicSpline, BSpline, make_interp_spline
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from tools.utils import parse_json_array, parse_json_matrix


def register_tools(mcp):

    @mcp.tool()
    def fft(data: str, sample_rate: float = 1.0) -> str:
        """Compute Fast Fourier Transform of a signal. data: JSON array of real values. sample_rate: samples per second. Returns frequencies and magnitudes."""
        try:
            signal = np.array(parse_json_array(data), dtype=float)
            n = len(signal)
            yf = scipy_fft.fft(signal)
            xf = scipy_fft.fftfreq(n, 1.0 / sample_rate)

            magnitudes = np.abs(yf[:n//2]).tolist()
            frequencies = xf[:n//2].tolist()
            phases = np.angle(yf[:n//2]).tolist()

            dominant_idx = np.argmax(magnitudes[1:]) + 1
            return json.dumps({
                "frequencies": frequencies,
                "magnitudes": magnitudes,
                "phases": phases,
                "dominant_frequency": frequencies[dominant_idx],
                "dominant_magnitude": magnitudes[dominant_idx],
                "num_samples": n,
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def ifft(real_parts: str, imag_parts: str = "") -> str:
        """Compute Inverse FFT. real_parts: JSON array. imag_parts: optional JSON array. Returns reconstructed signal."""
        try:
            real = np.array(parse_json_array(real_parts), dtype=float)
            if imag_parts:
                imag = np.array(parse_json_array(imag_parts), dtype=float)
                spectrum = real + 1j * imag
            else:
                spectrum = real
            signal = scipy_fft.ifft(spectrum)
            return json.dumps({"signal": np.real(signal).tolist()})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def spline_interpolate(x_data: str, y_data: str, x_new: str, kind: str = "cubic", bc_type: str = "not-a-knot") -> str:
        """Advanced spline interpolation. kind: 'cubic', 'bspline'. bc_type for cubic: 'not-a-knot', 'clamped', 'natural'. Returns interpolated y values and spline coefficients."""
        try:
            x = np.array(parse_json_array(x_data), dtype=float)
            y = np.array(parse_json_array(y_data), dtype=float)
            x_eval = np.array(parse_json_array(x_new), dtype=float)

            if kind == "cubic":
                cs = CubicSpline(x, y, bc_type=bc_type)
                y_eval = cs(x_eval)
                derivatives = cs(x_eval, 1)
                return json.dumps({
                    "y_interpolated": y_eval.tolist(),
                    "derivatives": derivatives.tolist(),
                    "coefficients_shape": list(cs.c.shape),
                })
            elif kind == "bspline":
                spline = make_interp_spline(x, y)
                y_eval = spline(x_eval)
                return json.dumps({"y_interpolated": y_eval.tolist()})
            else:
                return f"Error: Unknown kind '{kind}'. Use: cubic, bspline"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def sparse_eigenvalues(matrix: str, num_eigenvalues: int = 6, which: str = "LM") -> str:
        """Compute eigenvalues of a large/sparse matrix using iterative methods (ARPACK). matrix: JSON 2D array. which: 'LM' (largest magnitude), 'SM' (smallest), 'LR' (largest real), 'SR' (smallest real). Efficient for large matrices."""
        try:
            data = parse_json_matrix(matrix)
            m = np.array(data, dtype=float)
            n = m.shape[0]
            if num_eigenvalues >= n:
                num_eigenvalues = n - 2

            sparse_m = csr_matrix(m)
            eigenvalues, eigenvectors = eigs(sparse_m, k=num_eigenvalues, which=which)

            eig_sorted = sorted(zip(eigenvalues.tolist(), eigenvectors.T.tolist()), key=lambda x: abs(x[0]), reverse=True)

            return json.dumps({
                "eigenvalues": [{"real": complex(e[0]).real, "imag": complex(e[0]).imag} for e in eig_sorted],
                "num_computed": len(eig_sorted),
                "method": "ARPACK iterative",
                "which": which,
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def power_spectral_density(data: str, sample_rate: float = 1.0, window: str = "hann") -> str:
        """Compute Power Spectral Density (PSD) using Welch's method. data: JSON array. window: 'hann', 'hamming', 'blackman', 'boxcar'. Returns frequencies and PSD values."""
        try:
            from scipy.signal import welch
            signal = np.array(parse_json_array(data), dtype=float)
            freqs, psd = welch(signal, fs=sample_rate, window=window, nperseg=min(256, len(signal)))
            peak_idx = np.argmax(psd)
            return json.dumps({
                "frequencies": freqs.tolist(),
                "psd": psd.tolist(),
                "peak_frequency": float(freqs[peak_idx]),
                "peak_power": float(psd[peak_idx]),
            })
        except Exception as e:
            return f"Error: {e}"
