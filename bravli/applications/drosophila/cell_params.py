"""Drosophila cell model parameters.

Canonical source for fly-specific LIF and graded neuron parameters.
The same registrations also exist in bravli.models.cell_models for
backward compatibility; this module is the authoritative location.
"""

from bravli.models.cell_models import (
    CellModelDB, LIFParams, GradedParams, HIGH, MEDIUM, LOW,
)


def build_drosophila_cell_db():
    """Build a CellModelDB with all Drosophila-specific models."""
    db = CellModelDB()

    # --- Spiking models ---

    db.register(
        LIFParams(
            name="default_spiking",
            v_rest=-55.0, v_thresh=-45.0, v_reset=-55.0,
            tau_m=20.0, t_ref=2.0, c_m=8.0, r_input=500.0,
            confidence=LOW,
            source="Shiu et al. 2024 (adapted: V_rest corrected to -55 mV)",
            notes="Default for any neuron without class-specific data.",
        ),
    )

    db.register(
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

    db.register(
        LIFParams(
            name="projection_neuron",
            v_rest=-58.0, v_thresh=-45.0, v_reset=-55.0,
            tau_m=20.0, t_ref=2.0, c_m=8.0, r_input=600.0,
            confidence=HIGH,
            source="Gouwens & Wilson 2009, J Neurosci 29:6239",
            notes="Antennal lobe PNs. V_rest from cell-attached recording. "
                  "R_input=598+/-69 MOhm. C_m fitted: 0.8-2.6 uF/cm^2.",
        ),
        cell_classes=["PN", "ALPN"],
    )

    db.register(
        LIFParams(
            name="mbon",
            v_rest=-55.0, v_thresh=-45.0, v_reset=-55.0,
            tau_m=15.0, t_ref=2.0, c_m=10.0, r_input=400.0,
            confidence=MEDIUM,
            source="Hige et al. 2015, Neuron 88:985; Aso et al. 2014, eLife 3:e04577",
            notes="Mushroom body output neurons. Moderate time constant, "
                  "receive sparse KC input.",
        ),
        cell_classes=["MBON"],
    )

    db.register(
        LIFParams(
            name="dan",
            v_rest=-55.0, v_thresh=-45.0, v_reset=-55.0,
            tau_m=20.0, t_ref=2.0, c_m=8.0, r_input=500.0,
            confidence=LOW,
            source="Estimated from dopaminergic neuron literature",
            notes="Dopaminergic neurons. Carry reinforcement signals to MB.",
        ),
        cell_classes=["DAN"],
    )

    db.register(
        LIFParams(
            name="kenyon_cell",
            v_rest=-55.0, v_thresh=-45.0, v_reset=-55.0,
            tau_m=5.0, t_ref=2.0, c_m=3.6, r_input=1360.0,
            confidence=HIGH,
            source="Su & O'Dowd 2003",
            notes="Mushroom body Kenyon cells. Very high R_input (1.36 GOhm), "
                  "tiny C_m (3.6 pF).",
        ),
        cell_classes=["KC", "Kenyon_Cell"],
    )

    db.register(
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

    db.register(
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

    db.register(
        LIFParams(
            name="clock_neuron",
            v_rest=-50.0, v_thresh=-45.0, v_reset=-55.0,
            tau_m=4.0, t_ref=2.0, c_m=12.0, r_input=305.0,
            confidence=MEDIUM,
            source="Sheeba et al. 2008",
            notes="Large ventral lateral neurons (l-LNv).",
        ),
        cell_classes=["LNv"],
    )

    # --- Super-class defaults ---

    db.register(
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

    db.register(
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

    db.register(
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

    db.register(
        GradedParams(
            name="optic_graded",
            v_rest=-55.0, tau_m=10.0, c_m=5.0, r_input=500.0,
            v_range=15.0,
            confidence=LOW,
            source="Estimated; Stolz et al. 2021 approach",
            notes="Default for optic lobe non-spiking interneurons.",
        ),
        super_classes=["optic"],
    )

    db.register(
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

    return db


DROSOPHILA_CELL_MODEL_DB = build_drosophila_cell_db()
