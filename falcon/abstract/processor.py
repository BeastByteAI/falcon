from falcon.abstract.model import TransformerMixin

from falcon.abstract.task_pipeline import PipelineElement


class Processor(PipelineElement, TransformerMixin):
    """
    Subclass of `PipelineElement`. Used for data pre and post processing (e.g. data scaling).
    """
    pass
