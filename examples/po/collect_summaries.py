import pandas as pd
import os
import json
from os.path import join as pjoin


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def collect_summaries(results_path, save_to_path="results_analysis"):
    summaries_dict = {
        "problem_name": [],
        "optimizer": [],
        "regressor": [],
        "training_size": [],
        "degree": [],
        "noise": [],
        "seed": [],
        "regret": [],
        "mse": []
    }

    for r in os.listdir(results_path):
        problem_name, training_size, degree, noise, seed = r.split("-")

        sub_dir = pjoin(results_path, r)

        for file in os.listdir(sub_dir):
            if file.endswith("json") and "Evaluation" in file:

                evaluation_file = load_json(pjoin(sub_dir, file))

                summaries_dict["problem_name"].append(problem_name)
                summaries_dict["optimizer"].append(evaluation_file["optimizer"])
                summaries_dict["regressor"].append(evaluation_file["regressor"])
                summaries_dict["training_size"].append(int(training_size.split("_")[-1]))
                summaries_dict["degree"].append(int(degree.split("_")[-1]))
                summaries_dict["noise"].append(float(noise.split("_")[-1]))
                summaries_dict["seed"].append(int(seed.split("_")[-1]))
                summaries_dict["regret"].append(evaluation_file["regret"])
                summaries_dict["mse"].append(evaluation_file["mse"])

    summaries_df = pd.DataFrame.from_dict(summaries_dict)

    os.makedirs(save_to_path, exist_ok=True)

    summaries_df.to_csv(pjoin(save_to_path, "raw_summaries.csv"), index=False)


if __name__ == "__main__":
    print(os.getcwd())
    collect_summaries("results")
