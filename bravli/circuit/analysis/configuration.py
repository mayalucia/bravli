"""
Configuration of analysis, common stuff...
"""

from pathlib import Path
import seaborn as sn
from dmt.tk.field import NA, Field, LambdaField, lazyfield, WithFields

class CircuitAnalysisConfiguration(WithFields):
    """
    Common methods and attributes to configure a circuit analysis.
    """
    subregions = Field(
        """
        Named subregions contained in the larger modeled brain region.
        """,
        __default_value__=["S1HL", "S1FL", "S1Tr", "S1Sh"])
    path_resources = Field(
        """
        Path to the directory that contains resources required by this analysis.
        """,
        __default_value__=Path.cwd().joinpath("resources"))
    path_reference_data = Field(
        """
        Path where the reference data sits.
        """,
        __default_value__=Path(__file__).parent.parent.parent / "reference_data")

    use_roi_sampling = Field(
        """
        Boolean that indicates if spatial measurements of cell density,
        inhibitory fractions, etc. should be made in randomly sampled
        regions of interest (ROI).
        """,
        __default_value__=True)
    size_roi = Field(
        """
        Size of region-of-interest that will be sampled.
        """,
        __default_value__=50.)
    sample_size = Field(
        """
        Number of individuals to sample for stochastic measurements.
        """,
        __default_value__=20)
    connection_sample_size = Field(
        """
        Number of connections to sample for a single pathway.
        """,
        __default_value__=10000)
    apposition_sample_size = LambdaField(
        """
        Number of appositions to sample for a single pathway.
        """,
        lambda self: self.connection_sample_size)
    cell_sample_size = Field(
        """
        Number of pre/post synaptic cells to sample for pathway measurements.
        """,
        __default_value__=400)

    target = Field(
        """
        A `Mapping` that specifies circuit cells whose network connectivity will
        be analyzed. The default value of an empty `dict` indicates that all
        the circuit cells will be in the target.
        """,
        __default_value__={})
    pathways = Field(
        """
        Pathways to focus on.
        This may be a sequeunce of tuples, a pandas.Series, or a pandas.DataFrame,
        or a callable.
        """,
        __required__=False)
    figsize = Field(
        """
        Size of the figure ... `(width, height)`
        """,
        __default_value__=(None, None))
    context = Field(
        """
        Medium in which the document content is presented to the user.
        """,
        __examples__=["paper", "notebook"],
        __default_value__="notebook")
    font = Field(
        """
        Font family.
        """,
        __default_value__="sans-serif")
    font_scale = Field(
        """
        Overall scale for fonts.
        """,
        __default_value__=2.)
    fontsize = Field(
        """
        Overall size of figure fonts.
        """,
        __default_value__=20)
    title_size = Field(
        """
        Size of the title.
        """,
        __default_value__=30)
    axes_labelsize = Field(
        """
        Size of axes labels.
        """,
        __default_value__="xx-large")
    axes_titlesize = Field(
        """
        Size of axes title.
        """,
        __default_value__="xx-large")
    legend_text_size = Field(
        """
        Size of text in plot legend.
        """,
        __default_value__=32)
    legend_title_size = Field(
        """
        Size of the title of plot legend.
        """,
        __default_value__=42)
    xtick_labelsize = Field(
        """
        How large should the xticks be?
        """,
        __default_value__="small")
    ytick_labelsize = LambdaField(
        """
        How should the yticks be?
        """,
        lambda self: self.xtick_labelsize)
    legend_fontsize = Field(
        """
        Determines how large the legend will be.
        """,
        __default_value__="small")

    def get_rc_params(self):
        """
        Parameters used by matplotlib.
        """
        return {"font.size": self.fontsize,
                "legend.fontsize": self.legend_fontsize,
                "axes.labelsize": self.axes_labelsize,
                "axes.titlesize": self.axes_titlesize,
                "xtick.labelsize": self.xtick_labelsize,
                "ytick.labelsize": self.ytick_labelsize}

    def get_sample_size(self, use_roi_sampling=False):
        """
        Get sample size in context given by the keys of keyword arguments.
        Contexts will evolve and grow with the evolution of this file...
        """
        if use_roi_sampling:
            try:
                return self.sample_size
            except AttributeError:
                return 20
        return 1

    def get_number_bins(self, variable):
        """..."""
        if variable in ("cortical_depth", "depth", 
                        "cortical_height", "height"):
            try:
                return self.n_depth_bins
            except AttributeError:
                try:
                    return self.n_height_bins
                except AttributeError:
                    try:
                        return self.n_bins
                    except AttributeError:
                        pass
                    pass
                pass
            return 50
        raise ValueError("Unknown variable {}".format(variable))
