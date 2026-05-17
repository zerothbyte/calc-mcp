import json
import numpy as np
import sympy
from scipy.integrate import solve_ivp
from tools.utils import safe_parse_expr


def register_tools(mcp):

    @mcp.tool()
    def sym_dsolve(equation: str, function: str = "y", variable: str = "x") -> str:
        """Solve an ODE symbolically. Shorthand: \"y'' + y\", \"y' - 2*y\", \"y'' + 3*y' + 2*y\". Expression is set equal to 0. function: dependent variable name (default 'y'). variable: independent variable (default 'x')."""
        try:
            x = sympy.Symbol(variable)
            f = sympy.Function(function)
            fx = f(x)

            expr_str = equation
            fx_str = f"{function}({variable})"

            expr_str = expr_str.replace(f"{function}'''", f"Derivative({fx_str}, {variable}, 3)")
            expr_str = expr_str.replace(f"{function}''", f"Derivative({fx_str}, {variable}, 2)")
            expr_str = expr_str.replace(f"{function}'", f"Derivative({fx_str}, {variable})")

            import re
            expr_str = re.sub(
                rf'\b{re.escape(function)}\b(?!\(|\')',
                fx_str,
                expr_str
            )

            try:
                expr = sympy.sympify(expr_str, locals={variable: x, function: f, fx_str.split("(")[0]: f})
            except Exception as parse_err:
                return (
                    f"Error parsing equation: {parse_err}\n"
                    f"Supported formats:\n"
                    f"  - Shorthand: \"y'' + y\", \"y' - 2*y\", \"y'' + 3*y' + 2*y\"\n"
                    f"  - Explicit: \"Derivative(y(x), x, 2) + y(x)\"\n"
                    f"  - With coefficients: \"y'' + 3*y' + 2*y\"\n"
                    f"  - Non-homogeneous: \"y'' + y - sin(x)\"\n"
                    f"Parsed as: {expr_str}"
                )

            try:
                sol = sympy.dsolve(expr, fx)
            except Exception as solve_err:
                return (
                    f"Error solving ODE: {solve_err}\n"
                    f"Equation parsed as: {expr} = 0\n"
                    f"Tip: Ensure the equation is a valid ODE in {function}({variable})."
                )

            return str(sol)
        except Exception as e:
            return (
                f"Error: {e}\n"
                f"Usage: equation like \"y'' + y\", \"y' - 2*y\", \"y'' + 3*y' + 2*y\"\n"
                f"function='{function}', variable='{variable}'"
            )

    @mcp.tool()
    def solve_ode_numeric(expression: str, t_span: str = "[0, 10]", y0: str = "[1]", t_eval_points: int = 100) -> str:
        """Solve ODE numerically using scipy. expression: RHS of dy/dt = f(t, y). Use 't' for time and 'y' for the state variable. Example: '-y', '-0.5*y + sin(t)'. t_span: JSON [t_start, t_end]. y0: JSON array of initial conditions."""
        try:
            t_range = json.loads(t_span)
            initial = json.loads(y0)
            t_eval = np.linspace(t_range[0], t_range[1], t_eval_points)

            if len(initial) == 1:
                t_sym = sympy.Symbol("t")
                y_sym = sympy.Symbol("y")
                parsed = safe_parse_expr(expression, {"t": t_sym, "y": y_sym})
                f = sympy.lambdify([t_sym, y_sym], parsed, modules=["numpy"])
                def func(t, y):
                    return [f(t, y[0])]
            else:
                var_names = [f"y{i}" for i in range(len(initial))]
                t_sym = sympy.Symbol("t")
                y_syms = [sympy.Symbol(name) for name in var_names]
                local_vars = {"t": t_sym}
                for name, sym in zip(var_names, y_syms):
                    local_vars[name] = sym
                exprs = [e.strip() for e in expression.split(";")]
                funcs = []
                for expr_str in exprs:
                    parsed = safe_parse_expr(expr_str, local_vars)
                    funcs.append(sympy.lambdify([t_sym] + y_syms, parsed, modules=["numpy"]))

                def func(t, y):
                    return [f(t, *y) for f in funcs]

            sol = solve_ivp(func, t_range, initial, t_eval=t_eval, method="RK45")
            if not sol.success:
                return f"Error: {sol.message}"
            result = {"t": sol.t.tolist(), "y": sol.y.tolist()}
            return json.dumps(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def solve_ode_system(equations: str, t_span: str = "[0, 10]", y0: str = "[1, 0]", t_eval_points: int = 100) -> str:
        """Solve a system of ODEs. equations: semicolon-separated RHS expressions using y0,y1,... and t. Example: '-0.1*y0 + 0.01*y0*y1; 0.05*y0*y1 - 0.5*y1' (Lotka-Volterra). Returns JSON with t and y arrays."""
        try:
            t_range = json.loads(t_span)
            initial = json.loads(y0)
            t_eval = np.linspace(t_range[0], t_range[1], t_eval_points)

            exprs = [e.strip() for e in equations.split(";")]
            n = len(initial)
            t_sym = sympy.Symbol("t")
            y_syms = [sympy.Symbol(f"y{i}") for i in range(n)]
            local_vars = {"t": t_sym}
            for i, sym in enumerate(y_syms):
                local_vars[f"y{i}"] = sym

            compiled = []
            for expr_str in exprs:
                parsed = safe_parse_expr(expr_str, local_vars)
                compiled.append(sympy.lambdify([t_sym] + y_syms, parsed, modules=["numpy"]))

            def func(t, y):
                return [f(t, *y) for f in compiled]

            sol = solve_ivp(func, t_range, initial, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9, max_step=np.inf)
            if not sol.success:
                sol = solve_ivp(func, t_range, initial, t_eval=t_eval, method="Radau", rtol=1e-6, atol=1e-9)
                if not sol.success:
                    return f"Error: {sol.message}"
            result = {"t": sol.t.tolist(), "y": sol.y.tolist(), "labels": [f"y{i}" for i in range(n)]}
            return json.dumps(result)
        except Exception as e:
            return f"Error: {e}"
