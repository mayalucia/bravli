"""Synapse model parameter database for Drosophila neurotransmitter types.

Each NT type maps to a SynapseModel with kinetic, conductance, and
short-term plasticity parameters. Confidence levels (HIGH, MEDIUM, LOW)
indicate the strength of the supporting evidence.

References:
    Su & O'Dowd 2003 — nAChR kinetics in Kenyon cells
    Lee & O'Dowd 2000 — Fast cholinergic transmission
    Lee et al. 2003 — Rdl GABA_A fast inhibition
    Han et al. 2024 — NMJ GluR gating with Neto
    Shiu et al. 2024 — Whole-brain LIF parameters
    Hallermann et al. 2010 — NMJ short-term plasticity
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------

HIGH = "HIGH"      # Direct Drosophila electrophysiology data
MEDIUM = "MEDIUM"  # Insect literature or close analogy
LOW = "LOW"        # Theoretical default, no direct measurement


# ---------------------------------------------------------------------------
# Short-term plasticity parameters (Tsodyks-Markram model)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class STPParams:
    """Tsodyks-Markram short-term plasticity parameters.

    U: initial release probability (utilization)
    tau_rec: recovery from depression (ms)
    tau_fac: facilitation decay (ms); 0 = no facilitation
    confidence: evidence strength for these values
    """
    U: float = 0.4
    tau_rec: float = 400.0
    tau_fac: float = 0.0
    confidence: str = LOW


# ---------------------------------------------------------------------------
# Synapse model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynapseModel:
    """Biophysical parameters for a synapse type.

    Parameters
    ----------
    name : str
        Human-readable name (e.g., "cholinergic_excitatory").
    nt_type : str
        Neurotransmitter name matching FlyWire labels.
    sign : int
        +1 (excitatory), -1 (inhibitory), 0 (modulatory).
    tau_rise : float
        Conductance rise time constant (ms). None for modulatory.
    tau_decay : float
        Conductance decay time constant (ms). None for modulatory.
    e_rev : float or None
        Reversal potential (mV). None for modulatory (no fast PSP).
    g_peak : float or None
        Peak unitary conductance (nS). None for modulatory.
    modulation_tau : float or None
        Effective time constant for modulatory NTs (ms). None for fast.
    modulation_gain : float
        Multiplicative gain factor for modulatory NTs (1.0 = no effect).
    stp : STPParams or None
        Short-term plasticity parameters.
    confidence : str
        Overall confidence in this parameter set.
    source : str
        Key reference(s) for these parameters.
    notes : str
        Additional context or caveats.
    """
    name: str
    nt_type: str
    sign: int
    tau_rise: Optional[float] = None
    tau_decay: Optional[float] = None
    e_rev: Optional[float] = None
    g_peak: Optional[float] = None
    modulation_tau: Optional[float] = None
    modulation_gain: float = 1.0
    stp: Optional[STPParams] = None
    confidence: str = LOW
    source: str = ""
    notes: str = ""

    @property
    def is_fast(self):
        """True if this is a fast ionotropic synapse (has tau_rise/decay)."""
        return self.tau_rise is not None and self.tau_decay is not None

    @property
    def is_modulatory(self):
        """True if this is a slow modulatory synapse."""
        return self.sign == 0

    @property
    def sign_label(self):
        """Human-readable sign label."""
        return {1: "excitatory", -1: "inhibitory", 0: "modulatory"}[self.sign]

    def to_dict(self):
        """Serialize to dict for export."""
        d = {
            "name": self.name,
            "nt_type": self.nt_type,
            "sign": self.sign,
            "sign_label": self.sign_label,
            "tau_rise_ms": self.tau_rise,
            "tau_decay_ms": self.tau_decay,
            "e_rev_mV": self.e_rev,
            "g_peak_nS": self.g_peak,
            "modulation_tau_ms": self.modulation_tau,
            "modulation_gain": self.modulation_gain,
            "confidence": self.confidence,
            "source": self.source,
            "notes": self.notes,
        }
        if self.stp is not None:
            d["stp_U"] = self.stp.U
            d["stp_tau_rec_ms"] = self.stp.tau_rec
            d["stp_tau_fac_ms"] = self.stp.tau_fac
            d["stp_confidence"] = self.stp.confidence
        return d


# ---------------------------------------------------------------------------
# The parameter database
# ---------------------------------------------------------------------------

SYNAPSE_DB = {

    "acetylcholine": SynapseModel(
        name="cholinergic_excitatory",
        nt_type="acetylcholine",
        sign=1,
        tau_rise=0.5,
        tau_decay=1.8,
        e_rev=0.0,
        g_peak=0.4,
        stp=STPParams(U=0.4, tau_rec=400.0, tau_fac=75.0, confidence=LOW),
        confidence=HIGH,
        source="Su & O'Dowd 2003; Lee & O'Dowd 2000",
        notes="nAChR-mediated. tau_rise 0.43-0.61 ms, tau_decay 1.4-2.1 ms "
              "from Kenyon cell recordings. g_peak from mEPSC ~25 pA / 75 mV.",
    ),

    "GABA": SynapseModel(
        name="gabaergic_inhibitory",
        nt_type="GABA",
        sign=-1,
        tau_rise=0.8,
        tau_decay=4.5,
        e_rev=-45.0,
        g_peak=0.3,
        stp=STPParams(U=0.4, tau_rec=400.0, tau_fac=0.0, confidence=LOW),
        confidence=HIGH,
        source="Lee et al. 2003; Su & O'Dowd 2003",
        notes="Rdl (GABA_A-like) Cl- channel. tau_rise 0.80-0.88 ms, "
              "tau_decay 3.7-5.4 ms. E_rev -37 to -45 mV depending on [Cl-]_i. "
              "Using -45 mV (physiological estimate for adult CNS).",
    ),

    "glutamate": SynapseModel(
        name="glutamatergic_inhibitory",
        nt_type="glutamate",
        sign=-1,
        tau_rise=0.8,
        tau_decay=5.0,
        e_rev=-45.0,
        g_peak=0.3,
        stp=STPParams(U=0.4, tau_rec=400.0, tau_fac=0.0, confidence=LOW),
        confidence=LOW,
        source="GABA_A analogy; Li et al. 2021 (GluCl); Shiu et al. 2024",
        notes="GluCl-mediated inhibition predominates in central brain. "
              "No direct kinetic recordings from Drosophila CNS GluCl. "
              "Parameters modeled on GABA_A (same Cl- channel family). "
              "A minority of central glutamatergic synapses may be excitatory "
              "(via iGluR), but GluCl is the default for connectome-scale models.",
    ),

    "dopamine": SynapseModel(
        name="dopaminergic_modulatory",
        nt_type="dopamine",
        sign=0,
        modulation_tau=5000.0,
        modulation_gain=1.5,
        confidence=MEDIUM,
        source="Yamamoto & Seto 2014; Shiu et al. 2024",
        notes="GPCR-mediated (Dop1R1/2 D1-like, Dop2R D2-like). "
              "No fast PSP; modulates excitability and synaptic gain. "
              "In MB: gates Hebbian learning (reward/punishment). "
              "modulation_tau ~seconds; modulation_gain is a placeholder. "
              "Shiu et al. treated as excitatory (sign=+1) for simplicity.",
    ),

    "serotonin": SynapseModel(
        name="serotonergic_modulatory",
        nt_type="serotonin",
        sign=0,
        modulation_tau=5000.0,
        modulation_gain=1.3,
        confidence=MEDIUM,
        source="insect 5-HT literature",
        notes="GPCR-mediated (5-HT receptors coupled to cAMP/IP3). "
              "Modulates sensory gain and feeding circuits. "
              "No fast conductance; operates on second timescale.",
    ),

    "octopamine": SynapseModel(
        name="octopaminergic_modulatory",
        nt_type="octopamine",
        sign=0,
        modulation_tau=5000.0,
        modulation_gain=1.3,
        confidence=MEDIUM,
        source="Roeder 2005",
        notes="Functionally analogous to vertebrate norepinephrine. "
              "Multiple receptor subtypes (alpha, beta). "
              "Modulates arousal, sensory sensitivity, locomotor vigor.",
    ),
}

# Alias: the Shiu model sign convention
_SHIU_SIGNS = {
    "acetylcholine": 1,
    "GABA": -1,
    "glutamate": -1,
    "dopamine": 1,    # Shiu treats as excitatory
    "serotonin": 1,   # Shiu treats as excitatory
    "octopamine": 1,  # Shiu treats as excitatory
}


def get_synapse_model(nt_type):
    """Look up a SynapseModel by neurotransmitter name.

    Parameters
    ----------
    nt_type : str
        Neurotransmitter name (e.g., "acetylcholine", "GABA").

    Returns
    -------
    SynapseModel

    Raises
    ------
    KeyError
        If nt_type is not in the database.
    """
    if nt_type not in SYNAPSE_DB:
        raise KeyError(
            f"Unknown NT type '{nt_type}'. "
            f"Available: {list(SYNAPSE_DB.keys())}"
        )
    return SYNAPSE_DB[nt_type]


def list_models():
    """List all synapse models in the database.

    Returns
    -------
    list of dict
        Summary of each model: name, nt_type, sign, confidence.
    """
    return [
        {
            "nt_type": m.nt_type,
            "name": m.name,
            "sign": m.sign,
            "sign_label": m.sign_label,
            "is_fast": m.is_fast,
            "confidence": m.confidence,
        }
        for m in SYNAPSE_DB.values()
    ]


def simple_sign(nt_type, mode="biophysical"):
    """Return the sign (+1/-1/0) for a neurotransmitter.

    Parameters
    ----------
    nt_type : str
        Neurotransmitter name.
    mode : str
        "biophysical" — modulatory NTs get 0.
        "shiu" — modulatory NTs get +1 (following Shiu et al. 2024).

    Returns
    -------
    int
    """
    if mode == "shiu":
        return _SHIU_SIGNS.get(nt_type, 0)
    return SYNAPSE_DB[nt_type].sign if nt_type in SYNAPSE_DB else 0
