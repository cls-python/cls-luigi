


if __name__ == "__main__":
    import pandas as pd
    import os
    import json

    os.mkdir("results_summary")
    for dataset in os.listdir("results"):
        results_df = pd.DataFrame()

        imputer_col = []
        scaler_col = []
        feature_preprocessor_col = []
        classifier_col = []
        train_accuracy = []
        test_accuracy = []
        train_balanced_accuracy =[]
        test_balanced_accuracy = []
        last_task_col = []


        out_path = os.path.join("results_summary", f"results_{dataset}.csv")
        for results in os.listdir(os.path.join("./results", dataset)):
            if "run_summary.json" in results:
                with open(os.path.join("results", dataset, results)) as f:
                    results_json = json.load(f)

                pipeline = results_json["pipeline"]
                imputer_col.append(pipeline["imputer"])
                scaler_col.append(pipeline["scaler"])
                feature_preprocessor_col.append(pipeline["feature_preprocessor"])
                classifier_col.append(pipeline["classifier"])

                train_accuracy.append(results_json["accuracy"]["train"])
                test_accuracy.append(results_json["accuracy"]["test"])
                train_balanced_accuracy.append(results_json["balanced_accuracy"]["train"])
                test_balanced_accuracy.append(results_json["balanced_accuracy"]["test"])

                last_task_col.append(results_json["last_task"])

        results_df["imputer"] = imputer_col
        results_df["scaler"] = scaler_col
        results_df["feature_preprocessor"] = feature_preprocessor_col
        results_df["classifier"] = classifier_col
        results_df["train_accuracy"] = train_accuracy
        results_df["test_accuracy"] = test_accuracy
        results_df["train_balanced_accuracy"] = train_balanced_accuracy
        results_df["test_balanced_accuracy"] = test_balanced_accuracy
        results_df["last_task"] = last_task_col

        results_df.sort_values(by="test_balanced_accuracy", ascending=False, inplace=True)

        results_df.to_csv(out_path, index=False)



