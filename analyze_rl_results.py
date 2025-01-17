from typing import List

import pandas as pd

from eval_dataset.constants import ALL_EVAL_DATASET

MODELS = [
    "trained_models/lr10_w_E1_kl0.5_CodeRLTrain_llama3_IPS_max2048_____E1",
    "trained_models/lr10_w_E1_kl0.5_CodeRLTrain_llama3_IPS_max2048_____E1/checkpoint-1000",
    "trained_models/lr10_w_E1_kl0.5_CodeRLTrain_llama3_IPS_max2048_____E1/checkpoint-1600",
]


def report_rl_results(models: List[str]):
    results = []
    for dataset_name, dataset in ALL_EVAL_DATASET.items():
        for model in models:
            try:
                acc = dataset.load_accuracy(model)
                results.append(
                    {"model": model, "dataset": dataset_name, "accuracy": acc}
                )
            except Exception as e:
                print(f"failed to retrieve {dataset_name} + {model}")
                # raise e

    df = pd.DataFrame(results)
    df = df.pivot_table(values="accuracy", index="model", columns="dataset")
    print(df)
    df.to_csv("rl_results.csv")
    df.to_excel("rl_results.xlsx")


if __name__ == "__main__":
    report_rl_results(MODELS)
