import json
import numpy as np
import sympy
from scipy.integrate import solve_ivp
from tools.utils import safe_parse_expr, parse_json_matrix


def _parse_metric(metric: str, coordinates: str):
    """Parse metric string and validate format. Returns (g, syms, local_vars, n) or raises ValueError."""
    coords = [v.strip() for v in coordinates.split(",")]
    syms = [sympy.Symbol(c) for c in coords]
    local_vars = {c: s for c, s in zip(coords, syms)}
    n = len(coords)

    rows = [row.strip() for row in metric.split(";")]
    if len(rows) != n:
        raise ValueError(
            f"Invalid metric format: got {len(rows)} rows but {n} coordinates ({coordinates}). "
            f"Ensure metric is a string with rows separated by ';' and elements by ','. "
            f"Example for 2D: '1,0;0,r**2' with coordinates 'r,theta'"
        )
    for i, row in enumerate(rows):
        elems = row.split(",")
        if len(elems) != n:
            raise ValueError(
                f"Invalid metric format: row {i+1} has {len(elems)} elements but expected {n}. "
                f"Each row must have exactly {n} comma-separated elements to match coordinates ({coordinates})."
            )

    g = sympy.Matrix(n, n, lambda i, j: safe_parse_expr(rows[i].split(",")[j].strip(), local_vars))

    # Validate metric symmetry: g_ij must equal g_ji
    diff = sympy.simplify(g - g.T)
    if diff != sympy.zeros(n, n):
        asymmetric = [(i, j) for i in range(n) for j in range(i+1, n) if diff[i, j] != 0]
        pairs = ", ".join(f"g_{coords[i]}{coords[j]} ≠ g_{coords[j]}{coords[i]}" for i, j in asymmetric)
        raise ValueError(
            f"Metric tensor must be symmetric (g_ij = g_ji). "
            f"Asymmetric entries: {pairs}."
        )

    return g, syms, local_vars, n


def register_tools(mcp):

    @mcp.tool()
    def einstein_tensor(metric: str, coordinates: str = "t,r,theta,phi") -> str:
        """Compute Einstein tensor G_mu_nu = R_mu_nu - (1/2)*g_mu_nu*R from a metric. metric: rows separated by ';', elements by ','. Example: '-(1-2*M/r),0;0,1/(1-2*M/r)' with coordinates 't,r'."""
        try:
            g, syms, local_vars, n = _parse_metric(metric, coordinates)
            g_inv = g.inv()

            def gamma(sigma, mu, nu):
                return sympy.Rational(1, 2) * sum(
                    g_inv[sigma, lam] * (
                        sympy.diff(g[lam, nu], syms[mu]) +
                        sympy.diff(g[lam, mu], syms[nu]) -
                        sympy.diff(g[mu, nu], syms[lam])
                    ) for lam in range(n)
                )

            ricci = sympy.zeros(n, n)
            for mu in range(n):
                for nu in range(n):
                    val = sum(
                        sympy.diff(gamma(lam, mu, nu), syms[lam])
                        - sympy.diff(gamma(lam, mu, lam), syms[nu])
                        + sum(gamma(lam, lam, sig) * gamma(sig, mu, nu) for sig in range(n))
                        - sum(gamma(lam, nu, sig) * gamma(sig, mu, lam) for sig in range(n))
                        for lam in range(n)
                    )
                    ricci[mu, nu] = sympy.simplify(val)

            R_scalar = sympy.simplify(sum(g_inv[mu, nu] * ricci[mu, nu] for mu in range(n) for nu in range(n)))

            einstein = {}
            for mu in range(n):
                for nu in range(n):
                    G_mn = sympy.simplify(ricci[mu, nu] - sympy.Rational(1, 2) * g[mu, nu] * R_scalar)
                    if G_mn != 0:
                        einstein[f"G_{coords[mu]}{coords[nu]}"] = str(G_mn)

            return json.dumps({
                "einstein_tensor": einstein if einstein else "zero (vacuum solution)",
                "ricci_scalar": str(R_scalar),
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def solve_geodesic_numeric(metric_components: str, coordinates: str = "r,theta", initial_position: str = "[10, 1.5708]", initial_velocity: str = "[0, 0.1]", tau_span: str = "[0, 100]", num_points: int = 500) -> str:
        """Solve geodesic equations numerically. metric_components: rows separated by ';', elements by ','. initial_position/velocity: JSON arrays. tau_span: proper time range. Returns trajectory."""
        try:
            g, syms, local_vars, n = _parse_metric(metric_components, coordinates)
            g_inv = g.inv()

            gamma_sym = {}
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        val = sympy.Rational(1, 2) * sum(
                            g_inv[sigma, lam] * (
                                sympy.diff(g[lam, nu], syms[mu]) +
                                sympy.diff(g[lam, mu], syms[nu]) -
                                sympy.diff(g[mu, nu], syms[lam])
                            ) for lam in range(n)
                        )
                        val = sympy.simplify(val)
                        if val != 0:
                            gamma_sym[(sigma, mu, nu)] = val

            gamma_funcs = {}
            for key, expr in gamma_sym.items():
                gamma_funcs[key] = sympy.lambdify(syms, expr, modules=["numpy"])

            def geodesic_rhs(tau, state):
                pos = state[:n]
                vel = state[n:]
                accel = np.zeros(n)
                for (sigma, mu, nu), func in gamma_funcs.items():
                    try:
                        gamma_val = func(*pos)
                        accel[sigma] -= gamma_val * vel[mu] * vel[nu]
                    except Exception:
                        pass
                return np.concatenate([vel, accel])

            pos0 = json.loads(initial_position)
            vel0 = json.loads(initial_velocity)
            y0 = pos0 + vel0
            t_range = json.loads(tau_span)
            t_eval = np.linspace(t_range[0], t_range[1], num_points)

            sol = solve_ivp(geodesic_rhs, t_range, y0, t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10)
            if not sol.success:
                sol = solve_ivp(geodesic_rhs, t_range, y0, t_eval=t_eval, method="Radau", rtol=1e-8, atol=1e-10)

            trajectory = {coords[i]: sol.y[i].tolist() for i in range(n)}
            velocities = {f"d{coords[i]}/dtau": sol.y[n+i].tolist() for i in range(n)}

            return json.dumps({
                "tau": sol.t.tolist(),
                "trajectory": trajectory,
                "velocities": velocities,
                "success": bool(sol.success),
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def kretschner_scalar(metric: str, coordinates: str = "r,theta") -> str:
        """Compute Kretschner scalar K = R_abcd * R^abcd (measures curvature singularities). metric: rows separated by ';', elements by ','."""
        try:
            g, syms, local_vars, n = _parse_metric(metric, coordinates)
            g_inv = g.inv()

            def gamma(sigma, mu, nu):
                return sympy.Rational(1, 2) * sum(
                    g_inv[sigma, lam] * (
                        sympy.diff(g[lam, nu], syms[mu]) +
                        sympy.diff(g[lam, mu], syms[nu]) -
                        sympy.diff(g[mu, nu], syms[lam])
                    ) for lam in range(n)
                )

            R = {}
            for rho in range(n):
                for sigma in range(n):
                    for mu in range(n):
                        for nu in range(n):
                            val = (sympy.diff(gamma(rho, sigma, nu), syms[mu])
                                   - sympy.diff(gamma(rho, sigma, mu), syms[nu])
                                   + sum(gamma(rho, mu, lam) * gamma(lam, sigma, nu) for lam in range(n))
                                   - sum(gamma(rho, nu, lam) * gamma(lam, sigma, mu) for lam in range(n)))
                            val = sympy.simplify(val)
                            if val != 0:
                                R[(rho, sigma, mu, nu)] = val

            K = sympy.Integer(0)
            for (rho, sigma, mu, nu), R_up in R.items():
                R_down = sum(
                    g[rho, a] * g[sigma, b] * R.get((a, b, mu, nu), 0)
                    for a in range(n) for b in range(n)
                ) if (rho, sigma, mu, nu) in R else 0
                K += R_up * R_down

            K = sympy.simplify(K)
            return json.dumps({"kretschner_scalar": str(K)})
        except Exception as e:
            return f"Error: {e}"
