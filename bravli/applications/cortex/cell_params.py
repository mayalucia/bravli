"""Cortical cell model parameters.

LIF parameters for mammalian cortical neuron types, derived from
BBP recipe e-types and Allen Institute cell-type data.

These are effective single-compartment parameters — the BBP recipe
specifies detailed morphologies, but for our boundary-condition approach
we reduce each m-type to an effective point neuron that reproduces
the key firing statistics (rate, CV_ISI, adaptation).
"""

from bravli.models.cell_models import CellModelDB, LIFParams, HIGH, MEDIUM, LOW


def build_cortical_cell_db():
    """Build a CellModelDB with rodent cortical neuron types.

    Parameters based on:
    - Markram et al. 2015 (Cell): BBP reconstruction of rat SSCx
    - Allen Institute Cell Types Database: mouse V1 electrophysiology
    - Ramaswamy & Markram 2015: effective parameters for e-types
    """
    db = CellModelDB()

    # --- Default ---
    db.register(
        LIFParams(
            name="default_spiking",
            v_rest=-70.0, v_thresh=-50.0, v_reset=-65.0,
            tau_m=20.0, t_ref=2.0, c_m=150.0, r_input=133.0,
            confidence=LOW,
            source="Composite cortical estimate",
            notes="Default for unclassified cortical neurons.",
        ),
    )

    # --- Excitatory: pyramidal cells ---
    db.register(
        LIFParams(
            name="L4_spiny_stellate",
            v_rest=-70.0, v_thresh=-50.0, v_reset=-65.0,
            tau_m=15.0, t_ref=2.0, c_m=120.0, r_input=125.0,
            confidence=MEDIUM,
            source="Feldmeyer et al. 1999; Markram et al. 2015",
            notes="L4 SSC — primary thalamic recipients. Short tau_m, "
                  "compact dendritic tree.",
        ),
        cell_classes=["L4_SSC"],
    )

    db.register(
        LIFParams(
            name="L23_pyramidal",
            v_rest=-72.0, v_thresh=-50.0, v_reset=-65.0,
            tau_m=25.0, t_ref=2.0, c_m=200.0, r_input=125.0,
            confidence=MEDIUM,
            source="Feldmeyer et al. 2006; Mason & Larkman 1990",
            notes="L2/3 TPC — prediction error neurons. Longer tau_m, "
                  "larger C_m from extended apical dendrite.",
        ),
        cell_classes=["L2_TPC", "L3_TPC", "L23_PC"],
    )

    db.register(
        LIFParams(
            name="L5_thick_tufted",
            v_rest=-68.0, v_thresh=-48.0, v_reset=-60.0,
            tau_m=20.0, t_ref=2.0, c_m=300.0, r_input=67.0,
            confidence=MEDIUM,
            source="Ramaswamy & Markram 2015; Hay et al. 2011",
            notes="L5 TPC — prediction/output neurons. Large soma, "
                  "low R_input, large C_m. Prominent back-propagating APs.",
        ),
        cell_classes=["L5_TPC", "L5_TTPC"],
    )

    # --- Inhibitory: fast-spiking (PV+) ---
    db.register(
        LIFParams(
            name="fast_spiking_basket",
            v_rest=-70.0, v_thresh=-45.0, v_reset=-65.0,
            tau_m=10.0, t_ref=1.0, c_m=100.0, r_input=100.0,
            confidence=HIGH,
            source="Markram et al. 2004; Hu et al. 2014",
            notes="PV+ basket cells (LBC, NBC). Short tau_m, short t_ref, "
                  "can sustain high-frequency firing. In predictive coding: "
                  "divisive precision control.",
        ),
        cell_classes=["LBC", "NBC", "FS"],
        super_classes=["PV"],
    )

    # --- Inhibitory: SST+ Martinotti ---
    db.register(
        LIFParams(
            name="martinotti",
            v_rest=-65.0, v_thresh=-50.0, v_reset=-60.0,
            tau_m=20.0, t_ref=2.0, c_m=120.0, r_input=167.0,
            confidence=MEDIUM,
            source="Silberberg & Markram 2007; Wang et al. 2004",
            notes="SST+ Martinotti cells (MC). Dendritic-targeting inhibition. "
                  "In predictive coding: subtractive prediction signal.",
        ),
        cell_classes=["MC"],
        super_classes=["SST"],
    )

    # --- Inhibitory: VIP+ ---
    db.register(
        LIFParams(
            name="vip_interneuron",
            v_rest=-65.0, v_thresh=-48.0, v_reset=-60.0,
            tau_m=15.0, t_ref=2.0, c_m=80.0, r_input=188.0,
            confidence=LOW,
            source="Lee et al. 2013; Prönneke et al. 2015",
            notes="VIP+ interneurons (BP, SBC). Inhibit other interneurons "
                  "(disinhibitory circuit). Not in BBP recipe explicitly, "
                  "but critical for predictive coding gating.",
        ),
        cell_classes=["BP", "SBC"],
        super_classes=["VIP"],
    )

    return db


CORTICAL_CELL_MODEL_DB = build_cortical_cell_db()
