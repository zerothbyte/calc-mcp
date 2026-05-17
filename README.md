# 🧮 calc-mcp

**Calculator MCP Server** — A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that provides advanced mathematical computation capabilities for AI clients such as Claude, Cursor, and other MCP-compatible tools.

---

## ✨ Features

This server exposes dozens of **math tools** that can be called directly by AI:

| Module | Description |
|---|---|
| `server.py` | Basic operations: addition, subtraction, multiplication, division, square root, logarithm, trigonometry, factorial |
| `symbolic` | Symbolic computation (derivatives, integrals, limits) via SymPy |
| `symbolic_algebra` | Symbolic algebra, expression simplification |
| `linalg` | Linear algebra: determinants, matrix inverse, eigenvalues |
| `advanced_linalg` | Matrix decomposition (SVD, QR, LU, Cholesky) |
| `vector_calculus` | Vector calculus: gradient, divergence, curl |
| `complex_analysis` | Complex number analysis |
| `analysis` | General mathematical analysis |
| `numerical` | Numerical methods (numerical integration, root finding) |
| `ode` | Ordinary differential equation (ODE) solver |
| `pde` | Partial differential equation (PDE) solver |
| `differential_geometry` | Differential geometry (curvature, geodesics) |
| `general_relativity` | Riemann, Ricci tensors, and general relativity metrics |
| `transforms` | Fourier & Laplace transforms |
| `signal_processing` | Digital signal processing (FFT, filters) |
| `statistics` | Descriptive statistics, regression, hypothesis testing |
| `discrete_math` | Discrete mathematics (combinatorics, number theory) |
| `graph_theory` | Graph theory (BFS, DFS, shortest path) |
| `logic` | Propositional and predicate logic |
| `scientific` | Scientific constants & unit conversions |
| `plotting` | Graph and chart generation via Matplotlib |

---

## 🚀 Installation

### Prerequisites
- Python **3.10+**
- `uv` or `pip`

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd calc-mcp

# 2. (Optional) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -e .
# or using uv:
# uv sync
```

---

## ▶️ Running the Server

### Stdio Mode (Default — for Claude Desktop, etc.)
```bash
python server.py
```

### SSE Mode (Server-Sent Events)
```bash
python server.py --transport sse --port 8000
```

### Streamable HTTP Mode
```bash
python server.py --transport streamable-http --port 8000
```

### All Mode (SSE + Streamable HTTP simultaneously)
```bash
python server.py --transport all --port 8000
```

When running in `all` mode, the available endpoints are:
| Endpoint | URL |
|---|---|
| SSE | `http://localhost:8000/calc/sse` |
| Streamable HTTP | `http://localhost:8000/calc/mcp` |

---

## 🔧 Configuration with Claude Desktop

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "calculator": {
      "command": "python",
      "args": ["C:/path/to/calc-mcp/server.py"]
    }
  }
}
```

---

## 🏗️ Project Structure

```
calc-mcp/
├── server.py           # Main entry point & basic operation tools
├── pyproject.toml      # Project configuration & dependencies
├── tools/              # Math tool modules
│   ├── __init__.py     # Registers all tools into MCP
│   ├── symbolic.py
│   ├── linalg.py
│   ├── statistics.py
│   └── ...             # (20 other modules)
└── README.md
```

---

## 📦 Main Dependencies

| Package | Purpose |
|---|---|
| `mcp[cli]` | Model Context Protocol framework |
| `sympy` | Symbolic computation |
| `numpy` | Array & numerical computation |
| `scipy` | Scientific & numerical algorithms |
| `matplotlib` | Plotting & visualization |
| `statsmodels` | Statistics & econometrics |
| `pandas` | Data manipulation |

---

## 📄 License

MIT License — free to use and modify.
