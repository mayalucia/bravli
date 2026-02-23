"""Neurotransmitter constants for connectivity analysis.

These constants define the neurotransmitter types, their column names
in connectome datasets, and their sign classification. They are
organism-agnostic in principle — the same NT types appear across
species — though the column naming convention originates from FlyWire.
"""

# NT probability columns (as named in FlyWire feather files)
NT_COLUMNS = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]

# Human-readable names for NT columns
NT_NAMES = {
    "gaba_avg": "GABA",
    "ach_avg": "acetylcholine",
    "glut_avg": "glutamate",
    "oct_avg": "octopamine",
    "ser_avg": "serotonin",
    "da_avg": "dopamine",
}

# NT sign classification
NT_SIGN = {
    "acetylcholine": "excitatory",
    "GABA": "inhibitory",
    "glutamate": "mixed",  # GluCl → inhibitory; AMPA/NMDA-like → excitatory
    "octopamine": "modulatory",
    "serotonin": "modulatory",
    "dopamine": "modulatory",
}
