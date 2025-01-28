import json

from eval_dataset.base_eval_dataset import EvalDatasetBaseClass


class EvalPlusDataset(EvalDatasetBaseClass):
    def __init__(self, dataset_name: str):
        super().__init__()
        assert dataset_name in ["mbpp", "mbppplus", "humaneval", "humanevalplus"]
        self.dataset_name = dataset_name

    def get_saved_inference_file_path(self, model_path: str):
        temp = "0"
        if self.dataset_name in ["mbpp", "mbppplus"]:
            dataset_flag = "mbpp"
        else:
            dataset_flag = "humaneval"
        model_flag = model_path.replace("/", "--")
        if model_flag.startswith("--"):
            model_flag = model_flag[2:]
        return f"inferenced_output/evalplus/{dataset_flag}/{model_flag}_vllm_temp_{temp}.0-sanitized.eval_results.json"

    def load_accuracy(self, model_path: str) -> float:
        """Get the one shot accuracy as a floating point number"""
        file_path = self.get_saved_inference_file_path(model_path=model_path)
        with open(file_path, "r") as f:
            strr = f.read()
        inference_dict = json.loads(strr)["pass_at_k"]
        if self.dataset_name in ["mbpp", "humaneval"]:
            return inference_dict["base"]["pass@1"]
        else:
            return inference_dict["plus"]["pass@1"]


if __name__ == "__main__":
    data = EvalPlusDataset("humaneval")
