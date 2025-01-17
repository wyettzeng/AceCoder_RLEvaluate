from abc import ABC


class EvalDatasetBaseClass(ABC):
    def load_accuracy(self, model_path: str) -> float:
        raise NotImplementedError()
