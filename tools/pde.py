import json
import numpy as np
import sympy
from tools.utils import safe_parse_expr


def register_tools(mcp):

    @mcp.tool()
    def sym_pde_solve(equation: str, function: str = "u", variables: str = "x,t") -> str:
        """Solve a PDE symbolically using SymPy's pdsolve. equation: PDE expression set equal to 0. function: name of unknown function. variables: comma-separated independent variables. Example: 'Derivative(u(x,t),t) - Derivative(u(x,t),x,2)' for heat equation."""
        try:
            vars_list = [v.strip() for v in variables.split(",")]
            syms = [sympy.Symbol(v) for v in vars_list]
            f = sympy.Function(function)
            f_call = f(*syms)

            local_vars = {function: f_call}
            for v, s in zip(vars_list, syms):
                local_vars[v] = s

            expr = sympy.sympify(equation, locals=local_vars)
            sol = sympy.pdsolve(expr, f_call)
            return str(sol)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def heat_equation_1d(length: float = 1.0, time_end: float = 0.5, nx: int = 50, nt: int = 500, alpha: float = 0.01, initial_condition: str = "sin(pi*x)", boundary_left: float = 0.0, boundary_right: float = 0.0) -> str:
        """Solve 1D heat equation u_t = alpha * u_xx numerically using finite differences. initial_condition: expression in x over [0, length]. Returns JSON with x, t, and u (2D solution array)."""
        try:
            dx = length / (nx - 1)
            dt = time_end / nt
            r = alpha * dt / dx**2

            if r > 0.5:
                nt = int(time_end / (0.4 * dx**2 / alpha)) + 1
                dt = time_end / nt
                r = alpha * dt / dx**2

            from tools.utils import expr_to_callable
            f_init = expr_to_callable(initial_condition, "x")

            x = np.linspace(0, length, nx)
            u = f_init(x)
            u[0] = boundary_left
            u[-1] = boundary_right

            t_samples = np.linspace(0, time_end, min(50, nt))
            sample_indices = set((np.linspace(0, nt, min(50, nt))).astype(int))
            results = [u.copy().tolist()]
            t_recorded = [0.0]

            for n in range(1, nt + 1):
                u_new = u.copy()
                u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
                u_new[0] = boundary_left
                u_new[-1] = boundary_right
                u = u_new
                if n in sample_indices:
                    results.append(u.tolist())
                    t_recorded.append(n * dt)

            return json.dumps({
                "x": x.tolist(),
                "t": t_recorded,
                "u": results,
                "parameters": {"alpha": alpha, "nx": nx, "nt": nt, "dx": dx, "dt": dt, "r": r},
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def wave_equation_1d(length: float = 1.0, time_end: float = 1.0, nx: int = 100, nt: int = 500, c: float = 1.0, initial_displacement: str = "sin(pi*x)", initial_velocity: str = "0") -> str:
        """Solve 1D wave equation u_tt = c^2 * u_xx numerically. initial_displacement and initial_velocity: expressions in x. Boundary: u(0,t)=u(L,t)=0. Returns JSON with x, t, u."""
        try:
            from tools.utils import expr_to_callable

            dx = length / (nx - 1)
            dt = time_end / nt
            r = (c * dt / dx) ** 2

            if r > 1.0:
                nt = int(time_end / (0.9 * dx / c)) + 1
                dt = time_end / nt
                r = (c * dt / dx) ** 2

            f_disp = expr_to_callable(initial_displacement, "x")
            f_vel = expr_to_callable(initial_velocity, "x")

            x = np.linspace(0, length, nx)
            u_prev = f_disp(x)
            u_prev[0] = 0.0
            u_prev[-1] = 0.0

            vel = f_vel(x)
            u_curr = np.copy(u_prev)
            u_curr[1:-1] = u_prev[1:-1] + 0.5 * r * (u_prev[2:] - 2*u_prev[1:-1] + u_prev[:-2]) + dt * vel[1:-1]
            u_curr[0] = 0.0
            u_curr[-1] = 0.0

            sample_indices = set((np.linspace(0, nt, min(50, nt))).astype(int))
            results = [u_prev.tolist()]
            t_recorded = [0.0]

            for n in range(1, nt + 1):
                u_next = np.zeros(nx)
                u_next[1:-1] = 2*u_curr[1:-1] - u_prev[1:-1] + r*(u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2])
                u_prev = u_curr
                u_curr = u_next
                if n in sample_indices:
                    results.append(u_curr.tolist())
                    t_recorded.append(n * dt)

            return json.dumps({"x": x.tolist(), "t": t_recorded, "u": results, "parameters": {"c": c, "nx": nx, "nt": nt, "courant": r}})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def laplace_equation_2d(nx: int = 50, ny: int = 50, x_range: str = "[0, 1]", y_range: str = "[0, 1]", boundary_top: float = 100.0, boundary_bottom: float = 0.0, boundary_left: float = 0.0, boundary_right: float = 0.0, max_iterations: int = 5000, tolerance: float = 1e-5) -> str:
        """Solve 2D Laplace equation (u_xx + u_yy = 0) using iterative relaxation (Gauss-Seidel). Dirichlet boundary conditions. Returns JSON with x, y, u (2D grid)."""
        try:
            xr = json.loads(x_range)
            yr = json.loads(y_range)
            x = np.linspace(xr[0], xr[1], nx)
            y = np.linspace(yr[0], yr[1], ny)
            u = np.zeros((ny, nx))

            u[0, :] = boundary_bottom
            u[-1, :] = boundary_top
            u[:, 0] = boundary_left
            u[:, -1] = boundary_right

            for iteration in range(max_iterations):
                u_old = u.copy()
                u[1:-1, 1:-1] = 0.25 * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2])
                diff = np.max(np.abs(u - u_old))
                if diff < tolerance:
                    break

            return json.dumps({
                "x": x.tolist(),
                "y": y.tolist(),
                "u": u.tolist(),
                "converged": bool(diff < tolerance),
                "iterations": iteration + 1,
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def poisson_equation_2d(nx: int = 50, ny: int = 50, x_range: str = "[0, 1]", y_range: str = "[0, 1]", source: str = "-2*pi**2*sin(pi*x)*sin(pi*y)", boundary: float = 0.0, max_iterations: int = 5000, tolerance: float = 1e-5) -> str:
        """Solve 2D Poisson equation (u_xx + u_yy = f(x,y)) using iterative relaxation. source: expression in x,y. Returns JSON with x, y, u."""
        try:
            import sympy as sp
            from tools.utils import safe_parse_expr as spe

            xr = json.loads(x_range)
            yr = json.loads(y_range)
            x_arr = np.linspace(xr[0], xr[1], nx)
            y_arr = np.linspace(yr[0], yr[1], ny)
            dx = (xr[1] - xr[0]) / (nx - 1)
            dy = (yr[1] - yr[0]) / (ny - 1)

            x_sym = sp.Symbol("x")
            y_sym = sp.Symbol("y")
            source_expr = spe(source, {"x": x_sym, "y": y_sym})
            f_source = sp.lambdify([x_sym, y_sym], source_expr, modules=["numpy"])

            X, Y = np.meshgrid(x_arr, y_arr)
            F = f_source(X, Y)

            u = np.full((ny, nx), boundary)
            u[0, :] = boundary
            u[-1, :] = boundary
            u[:, 0] = boundary
            u[:, -1] = boundary

            for iteration in range(max_iterations):
                u_old = u.copy()
                u[1:-1, 1:-1] = 0.25 * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - dx*dy*F[1:-1, 1:-1])
                diff = np.max(np.abs(u - u_old))
                if diff < tolerance:
                    break

            return json.dumps({
                "x": x_arr.tolist(),
                "y": y_arr.tolist(),
                "u": u.tolist(),
                "converged": bool(diff < tolerance),
                "iterations": iteration + 1,
            })
        except Exception as e:
            return f"Error: {e}"
