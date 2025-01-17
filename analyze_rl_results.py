import pandas as pd

from eval_dataset.constants import ALL_EVAL_DATASET

MODELS = {
    "01_17": "trained_models/lr10_w_E1_kl0.5_CodeRLTrain_llama3_IPS_max2048_____E1",
    "01_17_chkp_1000": "trained_models/lr10_w_E1_kl0.5_CodeRLTrain_llama3_IPS_max2048_____E1/checkpoint-1000",
    "01_17_chkp_1600": "trained_models/lr10_w_E1_kl0.5_CodeRLTrain_llama3_IPS_max2048_____E1/checkpoint-1600",
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
