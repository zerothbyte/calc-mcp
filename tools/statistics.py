import json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from tools.utils import parse_json_array, parse_json_matrix


def register_tools(mcp):

    @mcp.tool()
    def linear_regression(x_data: str, y_data: str) -> str:
        """Perform OLS linear regression. x_data: JSON 1D or 2D array (multiple predictors), y_data: JSON 1D array. Returns coefficients, intercept, R-squared, p-values."""
        try:
            import statsmodels.api as sm

            y = np.array(parse_json_array(y_data), dtype=float)
            x_raw = json.loads(x_data)

            if isinstance(x_raw[0], list):
                X = np.array(x_raw, dtype=float)
            else:
                X = np.array(x_raw, dtype=float).reshape(-1, 1)

            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()

            result = {
                "intercept": model.params[0],
                "coefficients": model.params[1:].tolist(),
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "p_values": model.pvalues.tolist(),
                "std_errors": model.bse.tolist(),
                "f_statistic": model.fvalue,
                "f_pvalue": model.f_pvalue,
                "summary": str(model.summary()),
            }
            return json.dumps(result, default=str)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def hypothesis_test(data: str, test_type: str = "ttest_1samp", alternative: str = "two-sided", value: float = 0, data2: str = "") -> str:
        """Run hypothesis test. Types: ttest_1samp, ttest_ind, ttest_paired, chi2, shapiro, mannwhitney, anova. data/data2: JSON arrays."""
        try:
            if not data or not data.strip():
                return "Error: data is empty. Provide a JSON array, e.g. [1.2, 2.3, 3.4]."
            arr = np.array(parse_json_array(data), dtype=float)
            if len(arr) == 0:
                return "Error: data array is empty. Provide at least one value, e.g. [1.2, 2.3, 3.4]."

            if test_type == "ttest_1samp":
                stat, p = scipy_stats.ttest_1samp(arr, value, alternative=alternative)
                test_name = f"One-sample t-test (H0: mean = {value})"
            elif test_type == "ttest_ind":
                arr2 = np.array(parse_json_array(data2), dtype=float)
                stat, p = scipy_stats.ttest_ind(arr, arr2, alternative=alternative)
                test_name = "Independent two-sample t-test"
            elif test_type == "ttest_paired":
                arr2 = np.array(parse_json_array(data2), dtype=float)
                stat, p = scipy_stats.ttest_rel(arr, arr2, alternative=alternative)
                test_name = "Paired t-test"
            elif test_type == "chi2":
                stat, p = scipy_stats.chisquare(arr)
                test_name = "Chi-squared goodness of fit"
            elif test_type == "shapiro":
                stat, p = scipy_stats.shapiro(arr)
                test_name = "Shapiro-Wilk normality test"
            elif test_type == "mannwhitney":
                arr2 = np.array(parse_json_array(data2), dtype=float)
                stat, p = scipy_stats.mannwhitneyu(arr, arr2, alternative=alternative)
                test_name = "Mann-Whitney U test"
            elif test_type == "anova":
                groups = [np.array(g, dtype=float) for g in json.loads(data)]
                stat, p = scipy_stats.f_oneway(*groups)
                test_name = "One-way ANOVA"
            else:
                return f"Error: Unknown test type '{test_type}'"

            significant = bool(p < 0.05)
            result = {
                "test": test_name,
                "statistic": float(stat),
                "p_value": float(p),
                "significant_at_0.05": significant,
                "interpretation": f"{'Reject' if significant else 'Fail to reject'} H0 at alpha=0.05 (p={p:.6f})",
            }
            return json.dumps(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def correlation_matrix(data: str, method: str = "pearson") -> str:
        """Compute correlation matrix. data: JSON 2D array (columns are variables). Methods: pearson, spearman, kendall."""
        try:
            matrix = parse_json_matrix(data)
            df = pd.DataFrame(matrix)
            corr = df.corr(method=method)
            result = {
                "correlation_matrix": corr.values.tolist(),
                "method": method,
            }
            return json.dumps(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def time_series_decompose(data: str, period: int = 12, model: str = "additive") -> str:
        """Decompose time series into trend, seasonal, residual. data: JSON 1D array. Models: additive, multiplicative."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            arr = parse_json_array(data)
            series = pd.Series(arr, dtype=float)
            decomposition = seasonal_decompose(series, model=model, period=period)
            result = {
                "trend": [None if np.isnan(v) else v for v in decomposition.trend.tolist()],
                "seasonal": decomposition.seasonal.tolist(),
                "residual": [None if np.isnan(v) else v for v in decomposition.resid.tolist()],
                "model": model,
                "period": period,
            }
            return json.dumps(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def probability_distribution(distribution: str = "normal", operation: str = "pdf", x: str = "0", params: str = "{}") -> str:
        """Work with probability distributions. Distributions: normal, t, chi2, f, binomial, poisson, exponential. Operations: pdf, cdf, ppf, random. params: JSON dict of distribution parameters."""
        try:
            p = json.loads(params)

            dist_map = {
                "normal": lambda: scipy_stats.norm(loc=p.get("mean", 0), scale=p.get("std", 1)),
                "t": lambda: scipy_stats.t(df=p.get("df", 10)),
                "chi2": lambda: scipy_stats.chi2(df=p.get("df", 1)),
                "f": lambda: scipy_stats.f(dfn=p.get("dfn", 1), dfd=p.get("dfd", 1)),
                "binomial": lambda: scipy_stats.binom(n=p.get("n", 10), p=p.get("p", 0.5)),
                "poisson": lambda: scipy_stats.poisson(mu=p.get("mu", 1)),
                "exponential": lambda: scipy_stats.expon(scale=p.get("scale", 1)),
            }

            if distribution not in dist_map:
                return f"Error: Unknown distribution '{distribution}'. Available: {list(dist_map.keys())}"

            dist = dist_map[distribution]()

            if operation == "random":
                n_samples = int(float(x)) if x and x.strip() not in ("0", "") else 10
                if n_samples <= 0:
                    n_samples = 10
                samples = dist.rvs(size=n_samples).tolist()
                return json.dumps({"samples": samples, "count": n_samples})
            else:
                x_vals = json.loads(x) if x.startswith("[") else [float(x)]
                if operation == "pdf":
                    result = dist.pdf(x_vals) if hasattr(dist, "pdf") else dist.pmf(x_vals)
                elif operation == "cdf":
                    result = dist.cdf(x_vals)
                elif operation == "ppf":
                    result = dist.ppf(x_vals)
                else:
                    return f"Error: Unknown operation '{operation}'. Use: pdf, cdf, ppf, random"

                result = np.atleast_1d(result).tolist()
                return json.dumps({"x": x_vals, "result": result, "operation": operation, "distribution": distribution})
        except Exception as e:
            return f"Error: {e}"
