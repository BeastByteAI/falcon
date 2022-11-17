from falcon.tabular.pipelines import SimpleTabularPipeline
from falcon.tabular.learners import SuperLearner
from falcon.tabular.learners.super_learner import _default_estimators

TABULAR_CLASSIFICATION_CONFIGURATIONS = {
    "SuperLearner.mini": {
        "pipeline": SimpleTabularPipeline, 
        "extra_pipeline_options": {
            "learner": SuperLearner, 
            "learner_kwargs": {
                "cv": 10, 
                "base_estimators": _default_estimators['tabular_classification']['mini']
            }
        }
    }, 

    "SuperLearner.mid": {
        "pipeline": SimpleTabularPipeline, 
        "extra_pipeline_options": {
            "learner": SuperLearner, 
            "learner_kwargs": {
                "cv": 5, 
                "base_estimators": _default_estimators['tabular_classification']['mid']
            }
        }
    },

    "SuperLearner.large": {
        "pipeline": SimpleTabularPipeline, 
        "extra_pipeline_options": {
            "learner": SuperLearner, 
            "learner_kwargs": {
                "cv": 3, 
                "base_estimators": _default_estimators['tabular_classification']['large']
            }
        }
    },

    "SuperLearner.xlarge": {
        "pipeline": SimpleTabularPipeline, 
        "extra_pipeline_options": {
            "learner": SuperLearner, 
            "learner_kwargs": {
                "cv": 3, 
                "base_estimators": _default_estimators['tabular_classification']['x-large']
            }
        }
    },

    "SuperLearner": {
        "pipeline": SimpleTabularPipeline, 
        "extra_pipeline_options": {
            "learner": SuperLearner, 
            "learner_kwargs": {}
        }
    }
}

TABULAR_REGRESSION_CONFIGURATIONS = {
    "SuperLearner.mini": {
        "pipeline": SimpleTabularPipeline, 
        "extra_pipeline_options": {
            "learner": SuperLearner, 
            "learner_kwargs": {
                "cv": 10, 
                "base_estimators": _default_estimators['tabular_regression']['mini']
            }
        }
    }, 

    "SuperLearner.mid": {
        "pipeline": SimpleTabularPipeline, 
        "extra_pipeline_options": {
            "learner": SuperLearner, 
            "learner_kwargs": {
                "cv": 5, 
                "base_estimators": _default_estimators['tabular_regression']['mid']
            }
        }
    },

    "SuperLearner.large": {
        "pipeline": SimpleTabularPipeline, 
        "extra_pipeline_options": {
            "learner": SuperLearner, 
            "learner_kwargs": {
                "cv": 3, 
                "base_estimators": _default_estimators['tabular_regression']['large']
            }
        }
    }, 

    "SuperLearner.xlarge": {
        "pipeline": SimpleTabularPipeline, 
        "extra_pipeline_options": {
            "learner": SuperLearner, 
            "learner_kwargs": {
                "cv": 3, 
                "base_estimators": _default_estimators['tabular_regression']['x-large']
            }
        }
    }, 
    
    "SuperLearner": {
        "pipeline": SimpleTabularPipeline, 
        "extra_pipeline_options": {
            "learner": SuperLearner, 
            "learner_kwargs": {}
        }
    }
}