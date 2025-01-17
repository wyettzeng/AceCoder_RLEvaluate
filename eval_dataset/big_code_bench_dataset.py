import json

from eval_dataset.base_eval_dataset import EvalDatasetBaseClass


class BigCodeBenchDataset(EvalDatasetBaseClass):
    def __init__(self, split: str, subset: str):
        super().__init__()
        self.split = split
        self.subset = subset

    def get_saved_inference_file_path(self, model_path: str):
        extra = "-" + self.subset if self.subset != "full" else ""
        split = self.split
        temperature = "0"
        model_flag = model_path.replace("/", "--")
        file_end = f"--main--bigcodebench{extra}-{split}--vllm-{temperature}-1-sanitized_calibrated_eval_results.json"
        return f"inferenced output/bcb_results/{model_flag}{file_end}"

    def load_accuracy(self, model_path: str) -> float:
        """Get the one shot accuracy as a floating point number"""
        file_path = self.get_saved_inference_file_path(model_path=model_path)
        with open(file_path, "r") as f:
            strr = f.read()
        inference_dict = json.loads(strr)["eval"]
        score_lst = []
        for question_id, inf_lst in inference_dict.items():
            assert len(inf_lst) == 1
            score = 1 if inf_lst[0]["status"] == "pass" else 0
            score_lst.append(score)
        return sum(score_lst) / len(score_lst)


if __name__ == "__main__":
    data = BigCodeBenchDataset(split="complete", subset="hard")
