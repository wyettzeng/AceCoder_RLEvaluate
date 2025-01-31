import pandas as pd

from eval_dataset.constants import ALL_EVAL_DATASET

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B",
]


def report_rl_results():
    results = []
    for dataset_name, dataset in ALL_EVAL_DATASET.items():
        for model_path in MODELS:
            try:
                acc = dataset.load_accuracy(model_path)
                results.append(
                    {
                        "model": model_path[model_path.rfind("/") + 1 :],
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
