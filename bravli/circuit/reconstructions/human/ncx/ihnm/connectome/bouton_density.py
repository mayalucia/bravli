from pathlib import Path
from bravli.circuit.model import BlueBrainCircuitModel
from bravli.circuit.adapter.neocortical import BlueBrainModelAdapter
from bravli.circuit.reconstructions.human.ncx.ihnm import CONFIG


def get_model(path_circuit):
    return BlueBrainCircuitModel(path_circuit_data=path_circuit)


def get_adapter(config=CONFIG):
    return BlueBrainModelAdapter(**config.field_dict)


def report(path_circuit, config=CONFIG, *args, **kwargs):
    """
    Generate a little report for bouton density analysis.
    """
    from bravli.circuit.analysis.connectome import bouton_density
    article = bouton_density.ART.get(*args, **kwargs)
    adapter = get_adapter(config)
    model = get_model(path_circuit)
    analysis =  article.generate_analysis(adapter, model, *args, **kwargs)
    return analysis

