from typing import Dict

from eval_dataset.base_eval_dataset import EvalDatasetBaseClass
from eval_dataset.big_code_bench_dataset import BigCodeBenchDataset
from eval_dataset.eval_plus_dataset import EvalPlusDataset

ALL_EVAL_DATASET: Dict[str, EvalDatasetBaseClass] = {
    "mbpp": EvalPlusDataset("mbpp"),
    "mbpp+": EvalPlusDataset("mbppplus"),
    "humaneval": EvalPlusDataset("humaneval"),
    "humaneval+": EvalPlusDataset("humanevalplus"),
    "bcb_complete_hard": BigCodeBenchDataset(split="complete", subset="hard"),
    "bcb_complete_full": BigCodeBenchDataset(split="complete", subset="full"),
    "bcb_instruct_hard": BigCodeBenchDataset(split="instruct", subset="hard"),
    "bcb_instruct_full": BigCodeBenchDataset(split="instruct", subset="full"),
}
