"""Atlas: neuropil meshes, neuron skeletons, and whole-brain rendering."""

from bravli.atlas.neuropils import (
    load_neuropil,
    load_neuropils,
    list_neuropils,
    NEUROPIL_GROUPS,
)
from bravli.atlas.skeletons import (
    load_skeletons,
    sample_neuron_ids,
)
from bravli.atlas.render import (
    render_atlas,
    render_neuropil_detail,
    render_morphologies,
)
