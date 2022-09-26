Search.setIndex({"docnames": ["abstract/index", "abstract/learner", "abstract/model", "abstract/onnx", "abstract/pipeline", "abstract/pipeline_element", "abstract/processor", "abstract/task_manager", "api", "high_level_api", "index", "intro", "tabular/index", "tabular/learners/super_learner", "tabular/pipelines/simple_pipeline", "tabular/processors/label_decoder", "tabular/processors/scaler_and_encoder", "tabular/tab_manager"], "filenames": ["abstract\\index.rst", "abstract\\learner.rst", "abstract\\model.rst", "abstract\\onnx.rst", "abstract\\pipeline.rst", "abstract\\pipeline_element.rst", "abstract\\processor.rst", "abstract\\task_manager.rst", "api.rst", "high_level_api.rst", "index.rst", "intro.rst", "tabular\\index.rst", "tabular\\learners\\super_learner.rst", "tabular\\pipelines\\simple_pipeline.rst", "tabular\\processors\\label_decoder.rst", "tabular\\processors\\scaler_and_encoder.rst", "tabular\\tab_manager.rst"], "titles": ["Abstract", "Learner", "Model", "ONNXConvertible", "Pipeline", "PipelineElement", "Processor", "TaskManager", "API reference", "High level functions", "Welcome to Falcon\u2019s documentation!", "Getting started", "Tabular", "SuperLearner", "SimpleTabularPipeline", "LabelDecoder", "ScalerAndEncoder", "TabularTaskManager"], "terms": {"class": [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 14, 15, 16, 17], "falcon": [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 14, 15, 16, 17], "abstract": [1, 2, 3, 4, 5, 6, 7, 8, 10, 14], "task": [1, 4, 7, 9, 11, 13, 14, 17], "str": [1, 3, 4, 7, 9, 13, 14, 15, 16, 17], "kwarg": [1, 4, 7, 17], "ani": [1, 2, 4, 5, 6, 7, 9, 13, 14, 15, 16, 17], "subclass": [1, 6], "pipelineel": [1, 4, 6, 14], "ar": [1, 7, 9, 14], "awar": 1, "pipelin": [1, 3, 5, 6, 7, 9, 11, 14, 15, 16, 17], "element": [1, 4, 5, 6, 14, 16], "act": 1, "wrapper": 1, "around": 1, "model": [1, 3, 4, 7, 9, 11, 13, 14, 15, 16, 17], "respons": 1, "tune": 1, "hyperparamet": [1, 13], "__init__": [1, 4, 7, 13, 14, 15, 16, 17], "none": [1, 2, 4, 5, 6, 7, 9, 13, 14, 15, 16, 17], "paramet": [1, 2, 4, 5, 6, 7, 9, 13, 14, 15, 16, 17], "current": [1, 7, 9, 11], "ml": [1, 11], "fit": [1, 2, 4, 5, 6, 13, 14, 15, 16], "x": [1, 2, 4, 5, 6, 7, 13, 14, 15, 16], "ndarrai": [1, 2, 4, 5, 6, 13, 14, 15, 16, 17], "dtype": [1, 2, 4, 5, 6, 13, 14, 15, 16, 17], "scalartyp": [1, 2, 4, 5, 6, 14, 15, 16, 17], "y": [1, 2, 4, 5, 6, 13, 14, 16], "npt": [1, 2, 4, 5, 6, 14, 15, 16, 17], "featur": [1, 2, 4, 5, 6, 7, 9, 11, 13, 14, 15, 16, 17], "target": [1, 2, 4, 5, 6, 7, 9, 11, 13, 14, 16, 17], "return": [1, 2, 3, 4, 5, 6, 7, 9, 13, 14, 15, 16, 17], "usual": [1, 2, 4, 5, 6], "type": [1, 2, 3, 4, 5, 6, 7, 9, 13, 14, 15, 16, 17], "fit_pip": [1, 5, 6, 13, 14, 15, 16], "equival": [1, 5, 6, 13, 15, 16], "method": [1, 4, 5, 6, 7, 13, 14, 15, 16], "i": [1, 5, 6, 7, 9, 11, 13, 14, 15, 16, 17], "us": [1, 5, 6, 7, 9, 11, 13, 14, 15, 16, 17], "chainign": [1, 5, 6, 16], "inisd": [1, 5, 6, 16], "dure": [1, 5, 6, 16], "train": [1, 5, 6, 7, 9, 11, 13, 14, 15, 16, 17], "usus": [1, 5, 6, 16], "forward": [1, 5, 6, 13, 15, 16], "predict": [1, 2, 4, 5, 6, 7, 11, 13, 14, 15, 16, 17], "cha": [1, 5, 6], "insid": [1, 5, 6, 7], "infer": [1, 5, 6, 11], "featru": [1, 5, 6, 7, 14], "get_input_typ": [1, 5, 6, 13, 15, 16], "input": [1, 3, 4, 5, 6, 13, 14, 15, 16], "get_output_typ": [1, 5, 6, 13, 15, 16], "output": [1, 3, 4, 5, 6, 13, 14, 15, 16], "base": [2, 3, 4, 5, 7, 13], "all": [2, 3, 4, 5, 7, 11, 14, 17], "pipeline_el": 3, "can": [3, 7, 9], "convert": [3, 7, 13, 15, 16, 17], "onnx": [3, 4, 7, 9, 11, 13, 14, 15, 16, 17], "to_onnx": [3, 13, 15, 16], "tupl": [3, 13, 15, 16, 17], "byte": [3, 4, 7, 13, 14, 15, 16, 17], "int": [3, 13, 15, 16, 17], "list": [3, 9, 13, 14, 15, 16, 17], "option": [3, 4, 7, 9, 13, 14, 15, 16, 17], "serial": [3, 4, 7, 13, 14, 15, 16, 17], "string": [3, 4, 13, 14, 15, 16], "number": [3, 13, 15, 16], "node": [3, 13, 15, 16], "initi": [3, 7, 9, 11, 13, 15, 16, 17], "one": [3, 7, 9, 13, 14, 16, 17], "per": [3, 13, 16], "shape": [3, 13, 15, 16], "serializedmodeltupl": [3, 13, 15, 16], "add_el": [4, 14], "add": [4, 14], "The": [4, 7, 9, 11, 13, 14], "ad": [4, 14], "should": [4, 7, 14, 15, 17], "match": [4, 14], "last": [4, 11, 14, 17], "end": [4, 14], "save": [4, 7, 9, 11, 14, 17], "format": [4, 7, 11, 14, 17], "For": [4, 11, 13, 14], "more": [4, 11, 14], "detail": [4, 9, 11, 14], "pleas": [4, 11, 14], "refer": [4, 10, 11, 14], "document": [4, 14], "taskmanag": [4, 9, 11, 14], "save_model": [4, 7, 14, 17], "default": [4, 7, 9, 13, 14, 15, 16, 17], "data": [6, 7, 9, 11, 14, 16, 17], "pre": [6, 9, 13], "post": 6, "process": 6, "e": [6, 7, 13], "g": [6, 7], "scale": [6, 7, 11, 14], "transform": [6, 15, 16], "self": [6, 7, 16, 17], "pipeline_opt": [7, 9, 17], "dict": [7, 9, 13, 14, 17], "extra_pipeline_opt": [7, 9, 17], "manag": [7, 9, 11, 17], "_summary_": 7, "argument": [7, 9, 11, 14, 15, 16, 17], "pass": [7, 9, 11, 13, 14, 17], "instead": 7, "ones": [7, 9, 17], "addit": [7, 9, 17], "_create_pipelin": [7, 17], "_prepare_data": [7, 17], "prepar": [7, 11, 17], "read": [7, 11, 17], "from": [7, 9, 11, 17], "file": [7, 9, 11, 17], "warn": 7, "clean": [7, 17], "preprocess": 7, "encod": [7, 11, 14, 15, 16], "two": [7, 11], "distinct": 7, "step": [7, 9, 11], "later": 7, "perform": [7, 9, 13, 17], "properti": [7, 17], "default_pipelin": [7, 17], "chosen": 7, "dynam": 7, "default_pipeline_opt": [7, 9, 17], "evalu": [7, 9, 11, 17], "test_data": [7, 9, 11, 17], "metric": 7, "call": [7, 9, 13, 14], "filenam": [7, 17], "either": [7, 9, 11, 17], "onli": [7, 11, 17], "rare": [7, 17], "case": [7, 11, 17], "when": [7, 9, 11, 13, 17], "possibl": [7, 11, 17], "specifi": [7, 11, 17], "disk": [7, 17], "object": [7, 9, 11, 13, 17], "underli": [7, 13, 17], "high": [8, 10, 14], "level": [8, 10, 11, 14], "function": [8, 10, 11], "tabular": [8, 9, 10, 11, 13, 14, 15, 16, 17], "automl": [9, 11], "train_data": [9, 11], "api": [9, 10, 11], "line": [9, 11], "follow": 9, "execut": 9, "2": 9, "3": 9, "1": [9, 14], "If": [9, 17], "test": [9, 11, 17], "set": [9, 13, 17], "provid": [9, 11], "cv": [9, 13, 17], "small": [9, 17], "dataset": [9, 11, 13, 15, 16, 17], "valid": [9, 13], "subset": [9, 13], "big": 9, "after": 9, "re": 9, "whole": 9, "report": [9, 11, 17], "gener": 9, "print": [9, 17], "4": 9, "an": [9, 11, 17], "support": [9, 11], "tabular_classif": [9, 11, 13, 14, 17], "tabular_regress": [9, 11, 13, 14, 17], "classif": [9, 13, 14], "regress": [9, 13], "thi": [9, 11, 14, 15, 17], "path": [9, 17], "csv": [9, 11], "parquet": 9, "panda": [9, 11, 17], "datafram": [9, 11, 17], "numpi": [9, 11, 17], "arrai": [9, 11, 17], "column": [9, 11, 17], "name": [9, 17], "index": [9, 17], "correspond": 9, "given": [9, 17], "These": [9, 17], "overwrit": [9, 17], "attribut": [9, 17], "ignor": [9, 17], "get": 10, "start": 10, "instal": 10, "usag": 10, "power": 11, "machin": 11, "learn": 11, "singl": 11, "code": 11, "simpl": 11, "lightweight": 11, "librari": 11, "design": 11, "peopl": 11, "who": 11, "want": 11, "custom": [11, 13], "instant": 11, "even": 11, "without": 11, "specif": 11, "scienc": 11, "knowledg": 11, "simpli": [11, 14], "give": 11, "your": 11, "which": [11, 13], "you": 11, "do": 11, "rest": 11, "allow": 11, "immedi": 11, "product": 11, "them": 11, "wide": 11, "No": 11, "need": 11, "write": 11, "complic": 11, "anymor": 11, "pip": 11, "git": 11, "http": 11, "github": 11, "com": 11, "okua1": 11, "tabnet": 11, "kera": 11, "easiest": 11, "wai": 11, "highest": 11, "shown": 11, "below": [11, 13], "import": 11, "titan": 11, "label": [11, 14, 15], "futur": 11, "addition": 11, "also": [11, 14], "explicitli": [11, 17], "otherwis": 11, "other": 11, "titanic_test": 11, "sex": 11, "gender": 11, "ag": 11, "surviv": 11, "It": 11, "In": 11, "order": 11, "requir": 11, "might": [11, 14], "relev": 11, "where": [11, 14], "itself": 11, "come": 11, "non": 11, "convent": 11, "sourc": 11, "pd": [11, 17], "df": 11, "read_csv": 11, "while": [11, 14], "enabl": 11, "extrem": 11, "fast": 11, "experement": 11, "doe": [11, 15], "enough": 11, "control": 11, "over": 11, "flexibl": 11, "advanc": 11, "user": 11, "As": 11, "altern": 11, "directli": 11, "helper": 11, "test_df": 11, "pre_ev": [11, 17], "true": [11, 13, 14, 15, 16, 17], "configur": 11, "check": 11, "section": 11, "learner": [13, 14], "base_estim": 13, "callabl": 13, "base_score_threshold": 13, "float": 13, "filter_estim": 13, "bool": [13, 14, 15, 16, 17], "emploi": 13, "stackingmodel": 13, "construct": 13, "meta": 13, "estim": [13, 17], "cross": 13, "threshold": 13, "filter": 13, "fold": [13, 17], "perfom": [13, 17], "float32": [13, 15, 16], "were": 13, "automat": 13, "determin": [13, 17], "size": 13, "balanc": 13, "upsampl": 13, "minor": 13, "float32arrai": [13, 16], "equivalen": 13, "int64arrai": [13, 15], "union": [13, 17], "int64": 13, "its": [13, 15, 16], "mask": [14, 16, 17], "super_learn": 14, "superlearn": 14, "learner_kwarg": 14, "On": 14, "chain": 14, "preprocessor": 14, "super": 14, "integ": [14, 15], "decod": [14, 15], "back": [14, 15], "intern": 14, "numer": [14, 15, 16, 17], "0": 14, "mean": 14, "std": 14, "categor": [14, 15, 16, 17], "hot": 14, "approach": 14, "suitabl": 14, "veri": 14, "cardin": 14, "boolean": [14, 16], "indic": [14, 17], "fals": [14, 16], "consecut": 14, "each": [14, 15, 16], "labeldecod": 14, "appli": [14, 15, 16], "befor": [14, 15], "actual": 14, "occur": 14, "point": 14, "processor": [15, 16], "vice": 15, "versa": 15, "take": 15, "_": [15, 16], "dummi": [15, 16], "__": 15, "sinc": 15, "main": 15, "phase": 15, "noth": 15, "invers": 15, "np": [15, 16], "str_": 15, "els": 15, "origin": [15, 16], "map": [15, 16], "own": [15, 16], "onehotencod": 16, "standardscal": 16, "keep": 16, "compat": 16, "object_": 16, "_description_": 16, "simpletabularpipelin": 17, "ft": 17, "columnslist": 17, "except": 17, "well": 17, "locat": 17, "split": 17, "By": 17, "assum": 17, "creat": 17, "new": 17, "predict_train_set": 17, "obtain": 17, "invok": 17, "procedur": 17, "expect": 17, "avail": 17, "first": 17, "perfrom": 17, "via": 17, "10": 17, "25": 17, "larg": 17}, "objects": {"falcon": [[9, 0, 1, "", "AutoML"], [9, 0, 1, "", "initialize"]], "falcon.abstract": [[1, 1, 1, "", "Learner"], [2, 1, 1, "", "Model"], [3, 1, 1, "", "ONNXConvertible"], [4, 1, 1, "", "Pipeline"], [5, 1, 1, "", "PipelineElement"], [6, 1, 1, "", "Processor"], [7, 1, 1, "", "TaskManager"]], "falcon.abstract.Learner": [[1, 2, 1, "", "__init__"], [1, 2, 1, "", "fit"], [1, 2, 1, "", "fit_pipe"], [1, 2, 1, "", "forward"], [1, 2, 1, "", "get_input_type"], [1, 2, 1, "", "get_output_type"], [1, 2, 1, "", "predict"]], "falcon.abstract.Model": [[2, 2, 1, "", "fit"], [2, 2, 1, "", "predict"]], "falcon.abstract.ONNXConvertible": [[3, 2, 1, "", "to_onnx"]], "falcon.abstract.Pipeline": [[4, 2, 1, "", "__init__"], [4, 2, 1, "", "add_element"], [4, 2, 1, "", "fit"], [4, 2, 1, "", "predict"], [4, 2, 1, "", "save"]], "falcon.abstract.PipelineElement": [[5, 2, 1, "", "fit"], [5, 2, 1, "", "fit_pipe"], [5, 2, 1, "", "forward"], [5, 2, 1, "", "get_input_type"], [5, 2, 1, "", "get_output_type"], [5, 2, 1, "", "predict"]], "falcon.abstract.Processor": [[6, 2, 1, "", "fit"], [6, 2, 1, "", "fit_pipe"], [6, 2, 1, "", "forward"], [6, 2, 1, "", "get_input_type"], [6, 2, 1, "", "get_output_type"], [6, 2, 1, "", "predict"], [6, 2, 1, "", "transform"]], "falcon.abstract.TaskManager": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "_create_pipeline"], [7, 2, 1, "", "_prepare_data"], [7, 3, 1, "", "default_pipeline"], [7, 3, 1, "", "default_pipeline_options"], [7, 2, 1, "", "evaluate"], [7, 2, 1, "", "predict"], [7, 2, 1, "", "save_model"], [7, 2, 1, "", "train"]], "falcon.tabular": [[17, 1, 1, "", "TabularTaskManager"]], "falcon.tabular.TabularTaskManager": [[17, 2, 1, "", "__init__"], [17, 2, 1, "", "_create_pipeline"], [17, 2, 1, "", "_prepare_data"], [17, 3, 1, "", "default_pipeline"], [17, 3, 1, "", "default_pipeline_options"], [17, 2, 1, "", "evaluate"], [17, 2, 1, "", "predict"], [17, 2, 1, "", "predict_train_set"], [17, 2, 1, "", "save_model"], [17, 2, 1, "", "train"]], "falcon.tabular.learners": [[13, 1, 1, "", "SuperLearner"]], "falcon.tabular.learners.SuperLearner": [[13, 2, 1, "", "__init__"], [13, 2, 1, "", "fit"], [13, 2, 1, "", "fit_pipe"], [13, 2, 1, "", "forward"], [13, 2, 1, "", "get_input_type"], [13, 2, 1, "", "get_output_type"], [13, 2, 1, "", "predict"], [13, 2, 1, "", "to_onnx"]], "falcon.tabular.pipelines": [[14, 1, 1, "", "SimpleTabularPipeline"]], "falcon.tabular.pipelines.SimpleTabularPipeline": [[14, 2, 1, "", "__init__"], [14, 2, 1, "", "add_element"], [14, 2, 1, "", "fit"], [14, 2, 1, "", "predict"], [14, 2, 1, "", "save"]], "falcon.tabular.processors": [[15, 1, 1, "", "LabelDecoder"], [16, 1, 1, "", "ScalerAndEncoder"]], "falcon.tabular.processors.LabelDecoder": [[15, 2, 1, "", "__init__"], [15, 2, 1, "", "fit"], [15, 2, 1, "", "fit_pipe"], [15, 2, 1, "", "forward"], [15, 2, 1, "", "get_input_type"], [15, 2, 1, "", "get_output_type"], [15, 2, 1, "", "predict"], [15, 2, 1, "", "to_onnx"], [15, 2, 1, "", "transform"]], "falcon.tabular.processors.ScalerAndEncoder": [[16, 2, 1, "", "__init__"], [16, 2, 1, "", "fit"], [16, 2, 1, "", "fit_pipe"], [16, 2, 1, "", "forward"], [16, 2, 1, "", "get_input_type"], [16, 2, 1, "", "get_output_type"], [16, 2, 1, "", "predict"], [16, 2, 1, "", "to_onnx"], [16, 2, 1, "", "transform"]]}, "objtypes": {"0": "py:function", "1": "py:class", "2": "py:method", "3": "py:property"}, "objnames": {"0": ["py", "function", "Python function"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "property", "Python property"]}, "titleterms": {"abstract": 0, "learner": 1, "model": 2, "onnxconvert": 3, "pipelin": 4, "pipelineel": 5, "processor": 6, "taskmanag": 7, "api": 8, "refer": 8, "high": 9, "level": 9, "function": 9, "welcom": 10, "falcon": 10, "": 10, "document": 10, "get": 11, "start": 11, "instal": 11, "usag": 11, "tabular": 12, "superlearn": 13, "simpletabularpipelin": 14, "labeldecod": 15, "scalerandencod": 16, "tabulartaskmanag": 17}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 56}})