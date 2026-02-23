"""Drosophila synapse model parameters.

Canonical source for fly-specific neurotransmitter kinetics and
short-term plasticity parameters. The same data also exists in
bravli.physiology.synapse_models for backward compatibility;
this module is the authoritative location.
"""

from bravli.physiology.synapse_models import (
    SynapseModel, STPParams, HIGH, MEDIUM, LOW,
)


def build_drosophila_synapse_db():
    """Build a synapse database with all Drosophila NT types."""
    return {
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
                  "from Kenyon cell recordings.",
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
            notes="Rdl (GABA_A-like) Cl- channel.",
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
            notes="GluCl-mediated inhibition predominates in central brain.",
        ),

        "dopamine": SynapseModel(
            name="dopaminergic_modulatory",
            nt_type="dopamine",
            sign=0,
            modulation_tau=5000.0,
            modulation_gain=1.5,
            confidence=MEDIUM,
            source="Yamamoto & Seto 2014; Shiu et al. 2024",
            notes="GPCR-mediated. No fast PSP; modulates excitability.",
        ),

        "serotonin": SynapseModel(
            name="serotonergic_modulatory",
            nt_type="serotonin",
            sign=0,
            modulation_tau=5000.0,
            modulation_gain=1.3,
            confidence=MEDIUM,
            source="insect 5-HT literature",
            notes="GPCR-mediated. Modulates sensory gain and feeding circuits.",
        ),

        "octopamine": SynapseModel(
            name="octopaminergic_modulatory",
            nt_type="octopamine",
            sign=0,
            modulation_tau=5000.0,
            modulation_gain=1.3,
            confidence=MEDIUM,
            source="Roeder 2005",
            notes="Functionally analogous to vertebrate norepinephrine.",
        ),
    }


DROSOPHILA_SYNAPSE_DB = build_drosophila_synapse_db()

# Shiu model sign convention (treats modulatory NTs as excitatory)
DROSOPHILA_SHIU_SIGNS = {
    "acetylcholine": 1,
    "GABA": -1,
    "glutamate": -1,
    "dopamine": 1,
    "serotonin": 1,
    "octopamine": 1,
}
