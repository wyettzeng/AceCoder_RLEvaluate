from typing import Dict

from eval_dataset.base_eval_dataset import EvalDatasetBaseClass
from eval_dataset.big_code_bench_dataset import BigCodeBenchDataset
from eval_dataset.eval_plus_dataset import EvalPlusDataset
from eval_dataset.live_code_bench_dataset import LiveCodeBenchDataset

ALL_EVAL_DATASET: Dict[str, EvalDatasetBaseClass] = {
    "mbpp": EvalPlusDataset("mbpp"),
    "mbpp+": EvalPlusDataset("mbppplus"),
    "humaneval": EvalPlusDataset("humaneval"),
    "humaneval+": EvalPlusDataset("humanevalplus"),
    "lcb": LiveCodeBenchDataset(),
    "bcb_ch": BigCodeBenchDataset(split="complete", subset="hard"),
    "bcb_cf": BigCodeBenchDataset(split="complete", subset="full"),
    "bcb_ih": BigCodeBenchDataset(split="instruct", subset="hard"),
    "bcb_if": BigCodeBenchDataset(split="instruct", subset="full"),
}
