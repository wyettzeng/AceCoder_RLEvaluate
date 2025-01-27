import json
from typing import List

from eval_dataset.base_eval_dataset import EvalDatasetBaseClass

"""This Files Contain Functions Which Relates to LiveCodeBench"""


class LiveCodeBenchDataset(EvalDatasetBaseClass):
    def get_saved_inference_file_path(self, model_path: str):
        # the model representation is the last part of the path
        model_repr = model_path[model_path.rfind("/") :]
        dir_path = f"inferenced_output/livecodebench/{model_repr}/"
        dir_path + "Scenario.codegeneration_10_0.2_eval_all.json"

    def load_accuracy(self, model_path: str) -> float:
        """Get the one shot accuracy as a floating point number"""
        # loading the saved inferences
        file_path = self.get_saved_inference_file_path(model_path=model_path)
        with open(file_path, "r") as f:
            strr_input = f.read()
        input_lst = json.loads(strr_input)
        scores = [entry["pass@1"] for entry in input_lst]
        return sum(scores) / len(scores)


if __name__ == "__main__":
    data = LiveCodeBenchDataset()
    model_path = "mistral_instruct_v3_7b"
    prompts = data.load_inference(model_path, 16)
    print(prompts)
