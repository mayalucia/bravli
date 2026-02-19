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
