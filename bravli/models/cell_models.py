"""Point neuron model parameters for Drosophila cell types.

Two neuron types:
  - LIFParams: leaky integrate-and-fire (spiking neurons)
  - GradedParams: subthreshold LIF (non-spiking, graded potential)

Both are frozen dataclasses with confidence annotations.
"""

from dataclasses import dataclass
from typing import Optional


HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


@dataclass(frozen=True)
class LIFParams:
    """Leaky integrate-and-fire neuron parameters.

    Parameters
    ----------
    name : str
        Model name (e.g., "kenyon_cell").
    v_rest : float
        Resting membrane potential (mV).
    v_thresh : float
        Spike threshold (mV).
    v_reset : float
        Post-spike reset potential (mV).
    tau_m : float
        Membrane time constant (ms).
    t_ref : float
        Refractory period (ms).
    c_m : float
        Membrane capacitance (pF).
    r_input : float
        Input resistance (MOhm).
    confidence : str
        Evidence strength.
    source : str
        Key reference(s).
    notes : str
        Caveats or context.
    """
    name: str
    v_rest: float = -55.0
    v_thresh: float = -45.0
    v_reset: float = -55.0
    tau_m: float = 20.0
    t_ref: float = 2.0
    c_m: float = 8.0
    r_input: float = 500.0
    confidence: str = LOW
    source: str = ""
    notes: str = ""

    @property
    def g_leak(self):
        """Leak conductance (nS) = 1000 / R_input (MOhm)."""
        return 1000.0 / self.r_input if self.r_input > 0 else 0.0

    @property
    def mode(self):
        return "spiking"

    def to_dict(self):
        return {
            "name": self.name,
            "mode": self.mode,
            "v_rest_mV": self.v_rest,
            "v_thresh_mV": self.v_thresh,
            "v_reset_mV": self.v_reset,
            "tau_m_ms": self.tau_m,
            "t_ref_ms": self.t_ref,
            "c_m_pF": self.c_m,
            "r_input_MOhm": self.r_input,
            "g_leak_nS": self.g_leak,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(frozen=True)
class GradedParams:
    """Graded-potential (non-spiking) neuron parameters.

    Same equations as LIF but with threshold set unreachably high.
    The membrane potential is the output signal — it drives graded
    synaptic transmission to postsynaptic partners.

    Parameters
    ----------
    name : str
        Model name (e.g., "optic_lobe_interneuron").
    v_rest : float
        Resting membrane potential (mV).
    v_thresh : float
        Unreachable threshold (mV). Set to +100 mV by default.
    tau_m : float
        Membrane time constant (ms).
    c_m : float
        Membrane capacitance (pF).
    r_input : float
        Input resistance (MOhm).
    v_range : float
        Operating range of membrane potential (mV). Graded neurons
        typically operate over ~15 mV around rest.
    confidence : str
        Evidence strength.
    source : str
        Key reference(s).
    notes : str
        Caveats or context.
    """
    name: str
    v_rest: float = -55.0
    v_thresh: float = 100.0    # unreachable → never spikes
    tau_m: float = 10.0
    c_m: float = 5.0
    r_input: float = 500.0
    v_range: float = 15.0
    confidence: str = LOW
    source: str = ""
    notes: str = ""

    @property
    def g_leak(self):
        return 1000.0 / self.r_input if self.r_input > 0 else 0.0

    @property
    def mode(self):
        return "graded"

    @property
    def v_reset(self):
        """Graded neurons don't reset, but kept for API compatibility."""
        return self.v_rest

    def to_dict(self):
        return {
            "name": self.name,
            "mode": self.mode,
            "v_rest_mV": self.v_rest,
            "v_thresh_mV": self.v_thresh,
            "tau_m_ms": self.tau_m,
            "c_m_pF": self.c_m,
            "r_input_MOhm": self.r_input,
            "g_leak_nS": self.g_leak,
            "v_range_mV": self.v_range,
            "confidence": self.confidence,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Cell model database
# ---------------------------------------------------------------------------

class CellModelDB:
    """Registry of cell model parameter sets.

    Supports lookup by model name, cell class, or super class.
    """

    def __init__(self):
        self._models = {}
        self._class_map = {}     # cell_class → model name
        self._super_map = {}     # super_class → model name

    def register(self, model, cell_classes=None, super_classes=None):
        """Register a model with optional class mappings."""
        self._models[model.name] = model
        for cc in (cell_classes or []):
            self._class_map[cc] = model.name
        for sc in (super_classes or []):
            self._super_map[sc] = model.name

    def get(self, name):
        """Get a model by name."""
        if name not in self._models:
            raise KeyError(f"Unknown cell model '{name}'. "
                           f"Available: {list(self._models.keys())}")
        return self._models[name]

    def resolve(self, cell_class=None, super_class=None):
        """Resolve the best model for a given cell/super class.

        Priority: cell_class > super_class > default.
        """
        if cell_class and cell_class in self._class_map:
            return self._models[self._class_map[cell_class]]
        if super_class and super_class in self._super_map:
            return self._models[self._super_map[super_class]]
        return self._models.get("default_spiking")

    def list_models(self):
        """List all registered models."""
        return [m.to_dict() for m in self._models.values()]

    def __len__(self):
        return len(self._models)

    def __contains__(self, name):
        return name in self._models


# ---------------------------------------------------------------------------
# Build the database
# ---------------------------------------------------------------------------

CELL_MODEL_DB = CellModelDB()

# --- Spiking models ---

CELL_MODEL_DB.register(
    LIFParams(
        name="default_spiking",
        v_rest=-55.0, v_thresh=-45.0, v_reset=-55.0,
        tau_m=20.0, t_ref=2.0, c_m=8.0, r_input=500.0,
        confidence=LOW,
        source="Shiu et al. 2024 (adapted: V_rest corrected to -55 mV)",
        notes="Default for any neuron without class-specific data.",
    ),
)

CELL_MODEL_DB.register(
    LIFParams(
        name="shiu_uniform",
        v_rest=-52.0, v_thresh=-45.0, v_reset=-52.0,
        tau_m=20.0, t_ref=2.2, c_m=8.0, r_input=500.0,
        confidence=MEDIUM,
        source="Shiu et al. 2024, Nature 634:210-219",
        notes="Exact Shiu parameters. V_rest=V_reset=-52 mV. "
              "All neurons identical. Validated for activation patterns.",
    ),
)

CELL_MODEL_DB.register(
    LIFParams(
        name="projection_neuron",
        v_rest=-58.0, v_thresh=-45.0, v_reset=-55.0,
        tau_m=20.0, t_ref=2.0, c_m=8.0, r_input=600.0,
        confidence=HIGH,
        source="Gouwens & Wilson 2009, J Neurosci 29:6239",
        notes="Antennal lobe PNs. V_rest from cell-attached recording. "
              "R_input=598+/-69 MOhm. C_m fitted: 0.8-2.6 uF/cm^2.",
    ),
    cell_classes=["PN"],
)

CELL_MODEL_DB.register(
    LIFParams(
        name="kenyon_cell",
        v_rest=-55.0, v_thresh=-45.0, v_reset=-55.0,
        tau_m=5.0, t_ref=2.0, c_m=3.6, r_input=1360.0,
        confidence=HIGH,
        source="Su & O'Dowd 2003",
        notes="Mushroom body Kenyon cells. Very high R_input (1.36 GOhm), "
              "tiny C_m (3.6 pF). tau_m estimated from R*C.",
    ),
    cell_classes=["KC"],
)

CELL_MODEL_DB.register(
    LIFParams(
        name="motoneuron_fast",
        v_rest=-68.0, v_thresh=-45.0, v_reset=-60.0,
        tau_m=10.0, t_ref=2.0, c_m=15.0, r_input=150.0,
        confidence=HIGH,
        source="Azevedo et al. 2020",
        notes="Fast leg motoneurons. Low R_input, high C_m.",
    ),
    cell_classes=["MN_fast"],
)

CELL_MODEL_DB.register(
    LIFParams(
        name="motoneuron_slow",
        v_rest=-48.0, v_thresh=-40.0, v_reset=-48.0,
        tau_m=20.0, t_ref=2.0, c_m=10.0, r_input=700.0,
        confidence=HIGH,
        source="Azevedo et al. 2020",
        notes="Slow leg motoneurons. High R_input, depolarized V_rest.",
    ),
    cell_classes=["MN_slow"],
)

CELL_MODEL_DB.register(
    LIFParams(
        name="clock_neuron",
        v_rest=-50.0, v_thresh=-45.0, v_reset=-55.0,
        tau_m=4.0, t_ref=2.0, c_m=12.0, r_input=305.0,
        confidence=MEDIUM,
        source="Sheeba et al. 2008",
        notes="Large ventral lateral neurons (l-LNv). "
              "tau_m estimated from R*C ~ 305 MOhm * 12 pF.",
    ),
    cell_classes=["LNv"],
)

# --- Register super_class defaults ---

CELL_MODEL_DB.register(
    LIFParams(
        name="central_default",
        v_rest=-55.0, v_thresh=-45.0, v_reset=-55.0,
        tau_m=20.0, t_ref=2.0, c_m=8.0, r_input=500.0,
        confidence=LOW,
        source="Composite estimate",
        notes="Default for central brain neurons without specific data.",
    ),
    super_classes=["central"],
)

CELL_MODEL_DB.register(
    LIFParams(
        name="motor_default",
        v_rest=-60.0, v_thresh=-45.0, v_reset=-55.0,
        tau_m=15.0, t_ref=2.0, c_m=12.0, r_input=300.0,
        confidence=LOW,
        source="Average of fast/slow motoneuron data",
        notes="Default for motor, ascending, descending neurons.",
    ),
    super_classes=["motor", "ascending", "descending"],
)

CELL_MODEL_DB.register(
    LIFParams(
        name="sensory_default",
        v_rest=-60.0, v_thresh=-45.0, v_reset=-55.0,
        tau_m=10.0, t_ref=2.0, c_m=10.0, r_input=400.0,
        confidence=LOW,
        source="Estimated from olfactory receptor neuron literature",
        notes="Default for sensory neurons.",
    ),
    super_classes=["sensory"],
)

# --- Graded models ---

CELL_MODEL_DB.register(
    GradedParams(
        name="optic_graded",
        v_rest=-55.0, tau_m=10.0, c_m=5.0, r_input=500.0,
        v_range=15.0,
        confidence=LOW,
        source="Estimated; Stolz et al. 2021 approach",
        notes="Default for optic lobe non-spiking interneurons. "
              "Lamina monopolar cells, medulla interneurons. "
              "Operating range ~15 mV around rest.",
    ),
    super_classes=["optic"],
)

CELL_MODEL_DB.register(
    GradedParams(
        name="photoreceptor",
        v_rest=-60.0, tau_m=8.0, c_m=50.0, r_input=500.0,
        v_range=20.0,
        confidence=MEDIUM,
        source="Niven et al. 2003; Juusola et al. 2017",
        notes="R1-R6 photoreceptors. Large C_m (45-64 pF). "
              "Graded response to light, no spikes.",
    ),
    cell_classes=["photoreceptor"],
)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_cell_params(name):
    """Get cell model parameters by name."""
    return CELL_MODEL_DB.get(name)


def list_cell_models():
    """List all registered cell models."""
    return CELL_MODEL_DB.list_models()
