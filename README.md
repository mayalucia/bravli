# bravli

Brain Reconstruction Analysis and Validation Library -- for the *Drosophila* whole-brain connectome.

## The real story lives in `codev/`

This is a literate codebase. The Python files in `bravli/` are **tangled output** from
the org-mode lessons in `codev/`. To understand the code, read the lessons. To modify the
code, edit the lessons and re-tangle.

```
codev/00-foundations.org      # Dataset abstraction, logging
codev/01-parcellation.org    # Neuropil region tree, FlyWire loader
codev/02-composition.org     # Cell type counts, neurotransmitter profiles
codev/03-factology.org       # Structured measurement system
codev/04-visualization.org   # 3D rendering via navis
codev/05-explore-mushroom-body.org  # Integration walkthrough
codev/06-atlas.org           # Brain atlas and neuropil geometry
codev/07-plan-fly-brain-2026.org  # Research roadmap
codev/08-connectivity.org    # Synaptic connectivity analysis
codev/09-synaptic-physiology.org  # Synapse models and neurotransmitters
codev/10-cell-models.org     # LIF and graded cell models
codev/11-simulation.org      # LIF simulation engine
codev/12-portal.org          # Interactive digital twin portal
codev/13-mushroom-body-circuit.org  # MB microcircuit extraction + sparseness
codev/14-isn-and-learning.org # ISN regime + three-factor STDP conditioning
codev/15-brunel-phase-diagram.org  # Brunel phase diagram + FlyWire regime
codev/16-neuromodulatory-switching.org  # Neuromodulatory state switching (Marder's principle)
codev/17-stochastic-synapses.org  # Stochastic synapses, noise, and resonance
```

## Quick start

```bash
pip install -e ".[all]"
python -c "import bravli; print('ready')"
```

## Tangle (requires Emacs)

```bash
make tangle
```

## Test

```bash
make test
```

## Heritage

This project inherits ideas from the Blue Brain Project's cell atlas pipeline,
the `bravlibpy` circuit analysis library, and the `circuit-factology` measurement
framework -- adapted for the publicly available FlyWire connectome (139K neurons,
50M synapses, 8,453 cell types). It is a domain application within the MayaLucIA
personal scientific computing environment.
