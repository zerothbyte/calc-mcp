from tools.symbolic import register_tools as register_symbolic
from tools.linalg import register_tools as register_linalg
from tools.scientific import register_tools as register_scientific
from tools.plotting import register_tools as register_plotting
from tools.statistics import register_tools as register_statistics
from tools.ode import register_tools as register_ode
from tools.vector_calculus import register_tools as register_vector_calculus
from tools.complex_analysis import register_tools as register_complex_analysis
from tools.discrete_math import register_tools as register_discrete_math
from tools.advanced_linalg import register_tools as register_advanced_linalg
from tools.numerical import register_tools as register_numerical
from tools.pde import register_tools as register_pde
from tools.differential_geometry import register_tools as register_differential_geometry
from tools.transforms import register_tools as register_transforms
from tools.analysis import register_tools as register_analysis
from tools.symbolic_algebra import register_tools as register_symbolic_algebra
from tools.graph_theory import register_tools as register_graph_theory
from tools.general_relativity import register_tools as register_general_relativity
from tools.signal_processing import register_tools as register_signal_processing
from tools.logic import register_tools as register_logic


def register_all(mcp):
    register_symbolic(mcp)
    register_linalg(mcp)
    register_scientific(mcp)
    register_plotting(mcp)
    register_statistics(mcp)
    register_ode(mcp)
    register_vector_calculus(mcp)
    register_complex_analysis(mcp)
    register_discrete_math(mcp)
    register_advanced_linalg(mcp)
    register_numerical(mcp)
    register_pde(mcp)
    register_differential_geometry(mcp)
    register_transforms(mcp)
    register_analysis(mcp)
    register_symbolic_algebra(mcp)
    register_graph_theory(mcp)
    register_general_relativity(mcp)
    register_signal_processing(mcp)
    register_logic(mcp)
