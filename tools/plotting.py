import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.utils import parse_json_array, expr_to_callable


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def register_tools(mcp):

    @mcp.tool()
    def plot_function(expression: str, variable: str = "x", x_min: float = -10, x_max: float = 10, title: str = "", y_label: str = "y", num_points: int = 500) -> str:
        """Plot mathematical function(s). Separate multiple expressions with semicolons. Example: 'sin(x);cos(x)'. Returns base64-encoded PNG."""
        try:
            x = np.linspace(x_min, x_max, num_points)
            fig, ax = plt.subplots(figsize=(10, 6))

            expressions = [e.strip() for e in expression.split(";")]
            for expr_str in expressions:
                f = expr_to_callable(expr_str, variable)
                y = f(x)
                ax.plot(x, y, label=expr_str)

            ax.set_xlabel(variable)
            ax.set_ylabel(y_label)
            ax.set_title(title or f"Plot of {expression}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            return _fig_to_base64(fig)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def plot_scatter(x_data: str, y_data: str, title: str = "", x_label: str = "x", y_label: str = "y", trend_line: bool = False) -> str:
        """Create scatter plot from data. x_data and y_data are JSON arrays. Optional linear trend line. Returns base64-encoded PNG."""
        try:
            x = np.array(parse_json_array(x_data), dtype=float)
            y = np.array(parse_json_array(y_data), dtype=float)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x, y, alpha=0.7)

            if trend_line:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, label=f"y = {z[0]:.4f}x + {z[1]:.4f}")
                ax.legend()

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title or "Scatter Plot")
            ax.grid(True, alpha=0.3)
            return _fig_to_base64(fig)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def plot_histogram(data: str, bins: int = 30, title: str = "", x_label: str = "Value", y_label: str = "Frequency", density: bool = False) -> str:
        """Create histogram from data. Input: JSON array. Optional density normalization. Returns base64-encoded PNG."""
        try:
            arr = np.array(parse_json_array(data), dtype=float)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(arr, bins=bins, density=density, alpha=0.7, edgecolor="black")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Density" if density else y_label)
            ax.set_title(title or "Histogram")
            ax.grid(True, alpha=0.3)
            return _fig_to_base64(fig)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def plot_bar(categories: str, values: str, title: str = "", x_label: str = "", y_label: str = "Value", horizontal: bool = False) -> str:
        """Create bar chart. categories: JSON array of strings, values: JSON array of numbers. Returns base64-encoded PNG."""
        try:
            cats = parse_json_array(categories)
            vals = np.array(parse_json_array(values), dtype=float)

            fig, ax = plt.subplots(figsize=(10, 6))
            if horizontal:
                ax.barh(cats, vals, alpha=0.7)
                ax.set_xlabel(y_label)
                ax.set_ylabel(x_label)
            else:
                ax.bar(cats, vals, alpha=0.7)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
            ax.set_title(title or "Bar Chart")
            ax.grid(True, alpha=0.3, axis="x" if horizontal else "y")
            return _fig_to_base64(fig)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def plot_multi_series(x_data: str, y_series: str, labels: str = "", title: str = "", x_label: str = "x", y_label: str = "y") -> str:
        """Plot multiple data series on one chart. x_data: JSON array, y_series: JSON array of arrays, labels: JSON array of strings. Returns base64-encoded PNG."""
        try:
            x = np.array(parse_json_array(x_data), dtype=float)
            series = parse_json_array(y_series)
            label_list = parse_json_array(labels) if labels else [f"Series {i+1}" for i in range(len(series))]

            fig, ax = plt.subplots(figsize=(10, 6))
            for i, s in enumerate(series):
                y = np.array(s, dtype=float)
                lbl = label_list[i] if i < len(label_list) else f"Series {i+1}"
                ax.plot(x, y, label=lbl)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title or "Multi-Series Plot")
            ax.legend()
            ax.grid(True, alpha=0.3)
            return _fig_to_base64(fig)
        except Exception as e:
            return f"Error: {e}"
