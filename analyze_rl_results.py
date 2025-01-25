import pandas as pd

from eval_dataset.constants import ALL_EVAL_DATASET

MODELS = {
    "Qwen 7B Instruct (non Coder)": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen RL 01-24": "CodeDPO/qwen25-ins-7b-coderm-reinforce-plus",
    "Qwen Test Case RL 01-25": "CodeDPO/qwen25-ins-7b-testcaserm-7b-reinforce-plus",
}


def report_rl_results():
    results = []
    for dataset_name, dataset in ALL_EVAL_DATASET.items():
        for short_model_name, model_path in MODELS.items():
            try:
                acc = dataset.load_accuracy(model_path)
                results.append(
                    {
                        "model": short_model_name,
                        "dataset": dataset_name,
                        "accuracy": acc,
                    }
                )
            except Exception as e:
                print(f"failed to retrieve {dataset_name} + {short_model_name}")
                # raise e

    df = pd.DataFrame(results)
    df = df.pivot_table(values="accuracy", index="model", columns="dataset")
    print(df)
    df.to_csv("rl_results.csv")
    df.to_excel("rl_results.xlsx")


if __name__ == "__main__":
    report_rl_results()
