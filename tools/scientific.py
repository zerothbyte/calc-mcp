import json
import numpy as np
from scipy import optimize, integrate, interpolate as scipy_interp
from tools.utils import parse_json_array, expr_to_callable


def register_tools(mcp):

    @mcp.tool()
    def find_minimum(expression: str, variable: str = "x", bounds_lower: float = -100, bounds_upper: float = 100) -> str:
        """Find the minimum of a function within bounds. Expression uses the variable name (default 'x'). Example: find_minimum('(x-3)**2 + 1')."""
        try:
            f = expr_to_callable(expression, variable)
            result = optimize.minimize_scalar(f, bounds=(bounds_lower, bounds_upper), method="bounded")
            return json.dumps({"x_min": result.x, "f_min": result.fun, "success": result.success})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def find_root(expression: str, variable: str = "x", x0: float = 0.0, x1: float = None) -> str:
        """Find a numerical root of f(x) = 0 near x0. Example: find_root('x**3 - 2*x - 5', 'x', 2)."""
        try:
            f = expr_to_callable(expression, variable)
            if x1 is not None:
                root = optimize.brentq(f, x0, x1)
            else:
                root = optimize.newton(f, x0)
            return json.dumps({"root": root, "function_value": float(f(root))})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def numerical_integrate(expression: str, variable: str = "x", lower: float = 0, upper: float = 1) -> str:
        """Numerically integrate a function over [lower, upper]. Example: numerical_integrate('exp(-x**2)', 'x', 0, 1)."""
        try:
            f = expr_to_callable(expression, variable)
            result, error = integrate.quad(f, lower, upper)
            return json.dumps({"result": result, "error_estimate": error})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def curve_fit_data(x_data: str, y_data: str, model: str = "polynomial", degree: int = 2) -> str:
        """Fit a model to data. Models: 'polynomial', 'exponential', 'logarithmic', 'power'. x_data and y_data are JSON arrays."""
        try:
            x = np.array(parse_json_array(x_data), dtype=float)
            y = np.array(parse_json_array(y_data), dtype=float)
            if len(x) != len(y):
                return "Error: x_data and y_data must have same length"

            if model == "polynomial":
                coeffs = np.polyfit(x, y, degree)
                y_pred = np.polyval(coeffs, x)
                terms = [f"{c:.6g}*x^{degree-i}" if degree-i > 0 else f"{c:.6g}" for i, c in enumerate(coeffs)]
                equation = " + ".join(terms)
            elif model == "exponential":
                def exp_func(x, a, b):
                    return a * np.exp(b * x)
                popt, _ = optimize.curve_fit(exp_func, x, y, p0=[1, 0.1], maxfev=5000)
                y_pred = exp_func(x, *popt)
                coeffs = popt.tolist()
                equation = f"{popt[0]:.6g} * exp({popt[1]:.6g} * x)"
            elif model == "logarithmic":
                def log_func(x, a, b):
                    return a * np.log(x) + b
                popt, _ = optimize.curve_fit(log_func, x, y, maxfev=5000)
                y_pred = log_func(x, *popt)
                coeffs = popt.tolist()
                equation = f"{popt[0]:.6g} * ln(x) + {popt[1]:.6g}"
            elif model == "power":
                def power_func(x, a, b):
                    return a * np.power(x, b)
                popt, _ = optimize.curve_fit(power_func, x, y, p0=[1, 1], maxfev=5000)
                y_pred = power_func(x, *popt)
                coeffs = popt.tolist()
                equation = f"{popt[0]:.6g} * x^{popt[1]:.6g}"
            else:
                return f"Error: Unknown model '{model}'. Use: polynomial, exponential, logarithmic, power"

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return json.dumps({
                "coefficients": coeffs if isinstance(coeffs, list) else coeffs.tolist(),
                "equation": equation,
                "r_squared": r_squared,
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def interpolate_data(x_data: str, y_data: str, x_new: str, method: str = "cubic") -> str:
        """Interpolate to find y values at new x points. Methods: 'linear', 'cubic', 'nearest'. All inputs are JSON arrays."""
        try:
            x = np.array(parse_json_array(x_data), dtype=float)
            y = np.array(parse_json_array(y_data), dtype=float)
            x_eval = np.array(parse_json_array(x_new), dtype=float)
            if len(x) != len(y):
                return "Error: x_data and y_data must have same length"
            f = scipy_interp.interp1d(x, y, kind=method, fill_value="extrapolate")
            y_eval = f(x_eval)
            return json.dumps(y_eval.tolist())
        except Exception as e:
            return f"Error: {e}"
