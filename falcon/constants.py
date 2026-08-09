TABULAR_CLASSIFICATION_TASK = "tabular_classification"
TABULAR_REGRESSION_TASK = "tabular_regression"

# Every tag falcon writes into an FNNX manifest is namespaced `<producer>::<name>:<v>`,
# so readers can tell falcon's tags apart from those of any other producer.
DEFAULT_PRODUCER_NAME: str = "falcon.fnnx.ai"
