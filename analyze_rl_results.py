import pandas as pd

from eval_dataset.constants import ALL_EVAL_DATASET

MODELS = [
    #   "/data/code_llm/trained/qwen25-coder-7b-testcasermbinaryFalse-7b-reinforcepp_new_dataset_hard",
    #   "/data/code_llm/trained/qwen25-coder-7b-testcasermbinaryTrue-7b-reinforcepp_new_dataset_hard-binary",
    #   "Qwen/Qwen2.5-Coder-7B",
    #   "CodeDPO/qwen25-coder-base-7b-testcaserm-7b-new-dataset-hard",
    "/data/code_llm/trained/qwen25-coder-7b-rm-7b-reinforcepp_new_dataset_hard",
    "/data/code_llm/trained/qwen25-base-7b-rm-7b-reinforcepp_new_dataset_hard",
    "CodeDPO/qwen25-ins-7b-coderm_new_margin_scalebt-7b-reinforce-plus-episode_1",
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
