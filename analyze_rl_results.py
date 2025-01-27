import pandas as pd

from eval_dataset.constants import ALL_EVAL_DATASET

MODELS = [
    "CodeDPO/llama3-RL-both-E2-0117-ckpt1624",
    "CodeDPO/qwen25-coder-base-7b-testcaserm-7b-ppo-binary",
    "CodeDPO/qwen25-ins-7b-testcaserm-7b-reinforce-plus_new_dataset",
    "CodeDPO/qwen25-ins-7b-coderm-7b-ppo",
    "CodeDPO/qwen25-ins-7b-testcaserm-7b-reinforce-plus-binary",
    "CodeDPO/qwen25-ins-7b-coderm-reinforce-plus",
    "CodeDPO/qwen25-ins-7b-testcaserm-7b-reinforce-plus",
]


def report_rl_results():
    results = []
    for dataset_name, dataset in ALL_EVAL_DATASET.items():
        for model_path in MODELS:
            try:
                acc = dataset.load_accuracy(model_path)
                results.append(
                    {
                        "model": model_path,
                        "dataset": dataset_name,
                        "accuracy": acc,
                    }
                )
            except Exception as e:
                print(f"failed to retrieve {dataset_name} + {model_path}: {e}")

    df = pd.DataFrame(results)
    df = df.pivot_table(values="accuracy", index="model", columns="dataset")
    print(df)
    df.to_csv("rl_results.csv")
    df.to_excel("rl_results.xlsx")


if __name__ == "__main__":
    report_rl_results()
