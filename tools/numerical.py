import json
import numpy as np
from scipy import optimize
from tools.utils import parse_json_array, parse_json_matrix, expr_to_callable


def register_tools(mcp):

    @mcp.tool()
    def monte_carlo_integrate(expression: str, variable: str = "x", lower: float = 0, upper: float = 1, samples: int = 100000) -> str:
        """Monte Carlo integration of f(x) over [lower, upper]. Good for high-dimensional or complex integrals. Returns JSON with result and std_error."""
        try:
            f = expr_to_callable(expression, variable)
            rng = np.random.default_rng()
            x_samples = rng.uniform(lower, upper, samples)
            y_values = f(x_samples)
            width = upper - lower
            mean_val = np.mean(y_values)
            result = width * mean_val
            std_error = width * np.std(y_values) / np.sqrt(samples)
            return json.dumps({"result": result, "std_error": std_error, "samples": samples})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def monte_carlo_integrate_nd(expression: str, variables: str = "x,y", bounds: str = "[[0,1],[0,1]]", samples: int = 100000) -> str:
        """N-dimensional Monte Carlo integration. variables: comma-separated. bounds: JSON array of [lower, upper] for each variable. Expression uses variable names."""
        try:
            import sympy
            from tools.utils import safe_parse_expr

            vars_list = [v.strip() for v in variables.split(",")]
            bounds_arr = json.loads(bounds)
            if len(vars_list) != len(bounds_arr):
                return "Error: Number of variables must match number of bounds"

            syms = [sympy.Symbol(v) for v in vars_list]
            local_vars = {v: s for v, s in zip(vars_list, syms)}
            expr = safe_parse_expr(expression, local_vars)
            f = sympy.lambdify(syms, expr, modules=["numpy"])

            rng = np.random.default_rng()
            volume = 1.0
            sample_points = []
            for lo, hi in bounds_arr:
                sample_points.append(rng.uniform(lo, hi, samples))
                volume *= (hi - lo)

            y_values = f(*sample_points)
            result = volume * np.mean(y_values)
            std_error = volume * np.std(y_values) / np.sqrt(samples)
            return json.dumps({"result": float(result), "std_error": float(std_error), "dimensions": len(vars_list), "samples": samples})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def finite_difference(expression: str, variable: str = "x", point: float = 0, order: int = 1, h: float = 1e-5) -> str:
        """Compute numerical derivative using finite differences. order: 1 (first derivative), 2 (second), etc. Returns approximate derivative at point."""
        try:
            f = expr_to_callable(expression, variable)
            if order == 1:
                result = (f(point + h) - f(point - h)) / (2 * h)
            elif order == 2:
                result = (f(point + h) - 2 * f(point) + f(point - h)) / (h ** 2)
            elif order == 3:
                result = (f(point + 2*h) - 2*f(point + h) + 2*f(point - h) - f(point - 2*h)) / (2 * h**3)
            elif order == 4:
                result = (f(point + 2*h) - 4*f(point + h) + 6*f(point) - 4*f(point - h) + f(point - 2*h)) / (h**4)
            else:
                return "Error: Only orders 1-4 supported"
            return json.dumps({"derivative_order": order, "point": point, "value": float(result), "step_size": h})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def convex_optimize(expression: str, variables: str = "x,y", x0: str = "[0, 0]", method: str = "BFGS", bounds: str = "") -> str:
        """Minimize a multivariate function. variables: comma-separated. x0: JSON initial guess. methods: BFGS, Nelder-Mead, L-BFGS-B, Powell. bounds: optional JSON [[lo,hi],...] for L-BFGS-B."""
        try:
            import sympy
            from tools.utils import safe_parse_expr

            vars_list = [v.strip() for v in variables.split(",")]
            syms = [sympy.Symbol(v) for v in vars_list]
            local_vars = {v: s for v, s in zip(vars_list, syms)}
            expr = safe_parse_expr(expression, local_vars)
            f = sympy.lambdify(syms, expr, modules=["numpy"])

            initial = json.loads(x0)
            def objective(x):
                return float(f(*x))

            kwargs = {"method": method}
            if bounds:
                kwargs["bounds"] = json.loads(bounds)

            result = optimize.minimize(objective, initial, **kwargs)
            return json.dumps({
                "x_min": result.x.tolist(),
                "f_min": float(result.fun),
                "success": result.success,
                "iterations": result.nit,
                "message": result.message,
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def solve_bvp(equations: str, boundary_conditions: str, t_span: str = "[0, 1]", num_points: int = 50) -> str:
        """Solve boundary value problem. equations: semicolon-separated RHS of y' = f(t, y0, y1, ...). boundary_conditions: JSON object with 'left' and 'right' arrays. Example: equations='y1; -y0', boundary_conditions='{\"left\": [0], \"right\": [1]}' for y'' + y = 0, y(0)=0, y(1)=1."""
        try:
            from scipy.integrate import solve_bvp as scipy_solve_bvp
            import sympy
            from tools.utils import safe_parse_expr

            bc = json.loads(boundary_conditions)
            t_range = json.loads(t_span)
            exprs = [e.strip() for e in equations.split(";")]
            n = len(exprs)

            t_sym = sympy.Symbol("t")
            y_syms = [sympy.Symbol(f"y{i}") for i in range(n)]
            local_vars = {"t": t_sym}
            for i, s in enumerate(y_syms):
                local_vars[f"y{i}"] = s

            compiled = []
            for expr_str in exprs:
                parsed = safe_parse_expr(expr_str, local_vars)
                compiled.append(sympy.lambdify([t_sym] + y_syms, parsed, modules=["numpy"]))

            def fun(t, y):
                return np.array([f(t, *y) for f in compiled])

            def bc_func(ya, yb):
                residuals = []
                left = bc.get("left", [])
                right = bc.get("right", [])
                for i, val in enumerate(left):
                    if val is not None:
                        residuals.append(ya[i] - val)
                for i, val in enumerate(right):
                    if val is not None:
                        residuals.append(yb[i] - val)
                return np.array(residuals)

            t_mesh = np.linspace(t_range[0], t_range[1], num_points)
            y_init = np.ones((n, num_points))

            sol = scipy_solve_bvp(fun, bc_func, t_mesh, y_init)
            if not sol.success:
                return f"Error: {sol.message}"
            return json.dumps({"t": sol.x.tolist(), "y": sol.y.tolist(), "success": True})
        except Exception as e:
            return f"Error: {e}"
