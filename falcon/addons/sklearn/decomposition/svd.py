from sklearn.decomposition import TruncatedSVD as _TruncatedSVD
from skl2onnx.operator_converters.decomposition import convert_truncated_svd as _convert_truncated_svd
from skl2onnx.shape_calculators.svd import calculate_sklearn_truncated_svd_output_shapes as _calculate_sklearn_truncated_svd_output_shapes
from skl2onnx import update_registered_converter as _update_registered_converter


class ConditionalSVD(_TruncatedSVD):

    def _svd(self, X) -> _TruncatedSVD:
        self._mode = "svd"
        return super().fit_transform(X)
    
    def _identity(self, X):
        self._mode = "identity"
        self.out_dim = X.shape[-1]
        return X
        
    def fit(self, X) -> _TruncatedSVD:
        if X.shape[1] > self.n_components:
            self._svd(X)
        else:
            self._identity(X)
        self.fit_ = True
        return self

    def transform(self, X):
        if self._mode == "svd":
            return super().transform(X)
        else:
            return X

    def fit_transform(self, X, y=None):
        self.fit_ = True
        if X.shape[1] > self.n_components:
            return self._svd(X)
        else:
            return self._identity(X)

def _svd_shape_calc(operator):
    if operator.raw_operator._mode == "svd":
        operator.type = 'SklearnTruncatedSVD'
        _calculate_sklearn_truncated_svd_output_shapes(operator=operator)
    else:
        cls_type = operator.inputs[0].type.__class__
        N = operator.inputs[0].get_first_dimension()
        K = operator.raw_operator.out_dim
        operator.outputs[0].type = cls_type([N, K])


def _svd_converter(scope, operator, container):
    if operator.raw_operator._mode == "svd":
        operator.type = 'SklearnTruncatedSVD'
        _convert_truncated_svd(scope, operator, container)
    else:
        in_name = operator.inputs[0].full_name
        out_name = operator.outputs[0].full_name
        id_name = scope.get_unique_operator_name("identity")
        container.add_node("Identity", in_name, out_name, name=id_name, op_domain="")


_update_registered_converter(
    ConditionalSVD, "FalconConditionalSVD", _svd_shape_calc, _svd_converter
)
