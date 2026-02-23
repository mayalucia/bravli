"""Parse BBP CircuitBuildRecipe into Python data structures.

The BBP recipe files encode decades of curated experimental knowledge:
  - cell_composition.yaml: m-type densities, layer assignments, e-type distributions
  - mtype_taxonomy.tsv: morphological class (PYR/INT) and synaptic class (EXC/INH)
  - builderRecipeAllPathways.xml: synapse parameters per pathway
  - mini_frequencies.tsv: spontaneous miniature frequencies per layer
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MType:
    """A morphological neuron type from the BBP taxonomy."""
    name: str
    layer: int
    morph_class: str = ""      # PYR or INT
    synapse_class: str = ""    # EXC or INH
    etype_distribution: Dict[str, float] = field(default_factory=dict)

    @property
    def is_excitatory(self) -> bool:
        return self.synapse_class == "EXC"


@dataclass
class SynapseClass:
    """Tsodyks-Markram synapse parameters from the BBP recipe."""
    id: str
    gsyn: float        # peak conductance (nS)
    gsyn_sd: float     # standard deviation
    u: float           # initial release probability (USE)
    u_sd: float
    d: float           # depression time constant (ms)
    d_sd: float
    f: float           # facilitation time constant (ms)
    f_sd: float
    dtc: float         # decay time constant (ms)
    dtc_sd: float
    nrrp: float        # number of readily releasable vesicles
    nmda_ratio: float = 0.0   # NMDA/AMPA ratio (gsynSRSF)
    u_hill: float = 2.79      # Hill coefficient for calcium dependence


@dataclass
class PathwayRule:
    """Maps pre -> post m-type pattern to a synapse class."""
    from_sclass: str        # EXC or INH
    to_sclass: str          # EXC or INH
    from_mtype: str = ""    # specific m-type pattern (empty = any)
    to_mtype: str = ""
    synapse_class_id: str = ""


@dataclass
class BBPRecipe:
    """Parsed BBP CircuitBuildRecipe."""
    mtypes: List[MType]
    taxonomy: Dict[str, Tuple[str, str]]
    synapse_classes: Dict[str, SynapseClass]
    pathway_rules: List[PathwayRule]

    def excitatory_mtypes(self) -> List[MType]:
        return [m for m in self.mtypes if m.is_excitatory]

    def inhibitory_mtypes(self) -> List[MType]:
        return [m for m in self.mtypes if not m.is_excitatory]

    def mtypes_in_layer(self, layer: int) -> List[MType]:
        return [m for m in self.mtypes if m.layer == layer]

    def get_synapse_class(self, class_id: str) -> Optional[SynapseClass]:
        return self.synapse_classes.get(class_id)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_cell_composition(path: Path) -> List[MType]:
    """Parse cell_composition.yaml into a list of MType objects.

    Requires pyyaml (a bravli dependency).
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    mtypes = []
    for entry in data.get("neurons", []):
        traits = entry.get("traits", {})
        mtype = MType(
            name=traits.get("mtype", ""),
            layer=int(traits.get("layer", 0)),
            etype_distribution=traits.get("etype", {}),
        )
        mtypes.append(mtype)
    return mtypes


def parse_taxonomy(path: Path) -> Dict[str, Tuple[str, str]]:
    """Parse mtype_taxonomy.tsv -> {mtype: (morph_class, synapse_class)}.

    Returns dict mapping m-type name to (PYR|INT, EXC|INH).
    """
    taxonomy = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                taxonomy[parts[0]] = (parts[1], parts[2])
    return taxonomy


def _parse_xml_with_entities(path: Path) -> ET.Element:
    """Parse XML that may use external entity references.

    The BBP recipe XML uses DTD entity references like
    &connectivityRecipe; pointing to sibling files. We resolve
    these by reading the referenced files inline.
    """
    text = path.read_text(encoding="latin-1")

    # Resolve SYSTEM entities: <!ENTITY name SYSTEM "file.xml">
    import re
    entity_pattern = re.compile(
        r'<!ENTITY\s+(\w+)\s+SYSTEM\s+"([^"]+)"\s*>'
    )
    entities = entity_pattern.findall(text)

    for ent_name, ent_file in entities:
        ent_path = path.parent / ent_file
        if ent_path.exists():
            ent_content = ent_path.read_text(encoding="latin-1")
            text = text.replace(f"&{ent_name};", ent_content)
        else:
            # Referenced file missing — remove the entity reference
            text = text.replace(f"&{ent_name};", "")

    # Strip the DOCTYPE declaration (entities already resolved)
    text = re.sub(r'<!DOCTYPE[^>]*\[.*?\]>', '', text, flags=re.DOTALL)

    return ET.fromstring(text)


def parse_synapse_classes(path: Path) -> Dict[str, SynapseClass]:
    """Parse <SynapsesClassification> from the BBP recipe XML."""
    root = _parse_xml_with_entities(path)
    classes = {}
    for cls in root.iter("class"):
        sc = SynapseClass(
            id=cls.get("id"),
            gsyn=float(cls.get("gsyn", 0)),
            gsyn_sd=float(cls.get("gsynSD", 0)),
            u=float(cls.get("u", 0)),
            u_sd=float(cls.get("uSD", 0)),
            d=float(cls.get("d", 0)),
            d_sd=float(cls.get("dSD", 0)),
            f=float(cls.get("f", 0)),
            f_sd=float(cls.get("fSD", 0)),
            dtc=float(cls.get("dtc", 0)),
            dtc_sd=float(cls.get("dtcSD", 0)),
            nrrp=float(cls.get("nrrp", 0)),
            nmda_ratio=float(cls.get("gsynSRSF", 0)),
            u_hill=float(cls.get("uHillCoefficient", 2.79)),
        )
        classes[sc.id] = sc
    return classes


def parse_pathway_rules(path: Path) -> List[PathwayRule]:
    """Parse <SynapsesReposition> pathway rules from the BBP recipe XML."""
    root = _parse_xml_with_entities(path)
    rules = []
    for rule in root.iter("rule"):
        rules.append(PathwayRule(
            from_sclass=rule.get("fromSClass", ""),
            to_sclass=rule.get("toSClass", ""),
            from_mtype=rule.get("fromMType", ""),
            to_mtype=rule.get("toMType", ""),
            synapse_class_id=rule.get("class", ""),
        ))
    return rules


def load_recipe(recipe_dir: Path) -> BBPRecipe:
    """Load a complete BBP CircuitBuildRecipe from a directory.

    Expected layout:
        recipe_dir/
            inputs/1_cell_placement/cell_composition.yaml
            inputs/1_cell_placement/mtype_taxonomy.tsv
            inputs/4_synapse_generation/ALL/builderRecipeAllPathways.xml

    Parameters
    ----------
    recipe_dir : Path
        Root of the CircuitBuildRecipe directory.

    Returns
    -------
    BBPRecipe
    """
    recipe_dir = Path(recipe_dir)

    cell_comp_path = recipe_dir / "inputs" / "1_cell_placement" / "cell_composition.yaml"
    taxonomy_path = recipe_dir / "inputs" / "1_cell_placement" / "mtype_taxonomy.tsv"
    pathways_path = recipe_dir / "inputs" / "4_synapse_generation" / "ALL" / "builderRecipeAllPathways.xml"

    mtypes = parse_cell_composition(cell_comp_path)
    taxonomy = parse_taxonomy(taxonomy_path)

    # Annotate m-types with taxonomy info
    for mt in mtypes:
        if mt.name in taxonomy:
            mt.morph_class, mt.synapse_class = taxonomy[mt.name]

    synapse_classes = parse_synapse_classes(pathways_path)
    pathway_rules = parse_pathway_rules(pathways_path)

    return BBPRecipe(
        mtypes=mtypes,
        taxonomy=taxonomy,
        synapse_classes=synapse_classes,
        pathway_rules=pathway_rules,
    )
