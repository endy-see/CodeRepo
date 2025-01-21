import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import os
import math
import re
import argparse
import csv
import random
import time


class PairwiseAgreementEvaluator:
    def __init__(
            self,
            input_path: str,
            output_path_with_nulls: str,
            output_path_no_nulls: str,
            representative_score_path: str,
            methods: str,
            margin: int,
            num_samples: int,
            num_exp: int,
    ) -> None:
        self.methods = methods.split(",")
        self.input_path = input_path
        self.output_path_with_nulls = output_path_with_nulls
        self.output_path_no_nulls = output_path_no_nulls
        self.representative_score_path = representative_score_path
        self.margin = margin
        self.num_samples = num_samples
        self.num_exp = num_exp
        self.results = {"Nulls": {}, "NoNulls": {}}

    def read_tsv(self, file_path):
        return pd.read_csv(file_path, sep="\t")

    def preprocess(
            self, df: pd.DataFrame, df_repre_scores: pd.DataFrame
    ) -> pd.DataFrame:
        df.columns = [
                         "Url",
                         "TrueMarket",
                         "JudgeMarket",
                         "Market",
                         "NormalizedQuery",
                         "Category",
                         "Position",
                         "IsJudged",
                         "Score",
                         "Acceptable",
                         "NA",
                         "Spam",
                         "Escalate",
                         "LpSatAndQcLabel",
                         "LpSatAndQcContinuousScore",
                         "FinalScore",
                         "IsSpamLabel",
                         "Host",
                     ] + self.methods

        df = df[df["NA"] == 0]

        df_processed = df.copy()
        df_processed = df_processed[
            df_processed["IsJudged"] & df_processed["Score"] != -1
            ]
        df_processed = df_processed[df_processed["IsSpamLabel"].isna() == False]
        df_processed["Url"] = df_processed["Url"].apply(lambda v: re.sub(r"#.*", "", v))
        df_processed = df_processed.groupby(["Url", "NormalizedQuery"]).first()

        # print number of nulls
        print("Number of nulls per column:")
        print(df_processed.isnull().sum())

        df_processed.reset_index(inplace=True)

        return df_processed

    def prepare_sub_dfs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare sub dataframes for inter-query and intra-query agreemenr to avoid redundant computation"""

        spammy_df = df[df["IsSpamLabel"]]
        nonspammy_df = df[df["IsSpamLabel"] == False]

        # prepare data for inter-query equal weights agreement
        # populate a dictionary to map from query to a list where first item is a list of
        # row ids for nonspam urls for the query and second item is a list of row ids for spam urls for the query
        queryToRows = {}
        # Group the rows by NormalizedQuery and IsSpamLabel
        groups = df.groupby(["NormalizedQuery", "IsSpamLabel"])
        # (query, SpamLabel) : [row indices] -> query: [[row indices @ label == False], [row indices @ label == True]]
        for k, v in groups.indices.items():
            queryToRows.setdefault(k[0], [[], []])
            queryToRows[k[0]][k[1]] = v

        spam_queries = [
            q for q, v in queryToRows.items() if len(v[0]) > 0 and len(v[1]) > 1
        ]

        return queryToRows, spam_queries, spammy_df, nonspammy_df

    def generate_inter_query_pair(
            self, spammy_df: pd.DataFrame, non_spammy_df: pd.DataFrame
    ) -> None:
        """Calculate agreement between two different queries"""

        inter_query_agrees_spam = spammy_df.sample(n=1, replace=False).iloc[0]
        inter_query_agrees_non_spam = non_spammy_df.sample(n=1, replace=False).iloc[0]
        inter_query_row = {
            "SpamUrl": inter_query_agrees_spam["Url"],
            "NonspamUrl": inter_query_agrees_non_spam["Url"],
        }
        return inter_query_row, inter_query_agrees_spam, inter_query_agrees_non_spam

    def generate_intra_query_pair(
            self, df: pd.DataFrame, spammy_df: pd.DataFrame
    ) -> dict:
        while True:
            intra_query_spam = spammy_df.sample(n=1, replace=False).iloc[0]
            query_non_spam = df[
                df["NormalizedQuery"] == intra_query_spam["NormalizedQuery"]
                ]
            intra_query_non_spam = query_non_spam[
                query_non_spam["IsSpamLabel"] == False
                ]
            if len(intra_query_non_spam) == 0:
                continue
            intra_query_non_spam = intra_query_non_spam.sample(n=1, replace=False).iloc[
                0
            ]
            break

        intra_query_row = {
            "NormalizedQuery": intra_query_spam["NormalizedQuery"],
            "JudgeMarket": intra_query_spam["JudgeMarket"],
            "SpamUrl": intra_query_spam["Url"],
            "NonspamUrl": intra_query_non_spam["Url"],
        }

        return intra_query_row, intra_query_spam, intra_query_non_spam

    def run_single_iteration(
            self,
            df: pd.DataFrame,
            queryToRows: dict,
            spam_queries: list,
            spammy_df: pd.DataFrame,
            nonspammy_df: pd.DataFrame,
            type: str
    ) -> dict:
        """Calculate agreement between within same query"""

        intra_query_list = []
        inter_query_list = []
        intra_query_equal_weight_list = []

        for i in range(self.num_samples):
            ## create intra query pairs
            (
                intra_query_pair,
                intra_query_spam,
                intra_query_non_spam,
            ) = self.generate_intra_query_pair(df, spammy_df)
            (
                intra_query_equal_weight_pair,
                intraquery_equal_query_weight_spam,
                intraquery_equal_query_weight_non_spam,
            ) = self.generate_intraquery_equal_query_weight(
                df, queryToRows, spam_queries
            )
            (
                inter_query_pair,
                inter_query_pair_spam,
                inter_query_pair_non_spam,
            ) = self.generate_inter_query_pair(spammy_df, nonspammy_df)

            for method in self.methods:
                intra_query_pair[method] = (
                                                   intra_query_spam[method] - intra_query_non_spam[method]
                                           ) > self.margin
                intra_query_equal_weight_pair[method] = (
                                                                intraquery_equal_query_weight_spam[method]
                                                                - intraquery_equal_query_weight_non_spam[method]
                                                        ) > self.margin
                inter_query_pair[method] = (
                                                   inter_query_pair_spam[method] - inter_query_pair_non_spam[method]
                                           ) > self.margin

            intra_query_list.append(intra_query_pair)
            intra_query_equal_weight_list.append(intra_query_equal_weight_pair)
            inter_query_list.append(inter_query_pair)

        # write 100 samples from intra_query_list to a intra_query_agreement_samples.tsv. add only Methods[1]
        with open(f"intra_query_agreement_samples_{type}.tsv", "w") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                               "NormalizedQuery",
                               "JudgeMarket",
                               "SpamUrl",
                               "NonspamUrl",
                           ] + self.methods,
                delimiter="\t",
            )
            writer.writeheader()
            for row in intra_query_list:
                writer.writerow(row)

        self.results[type].setdefault("intra_query_agreement", {})
        self.results[type].setdefault("inter_query_agreement", {})
        self.results[type].setdefault("intra_query_equal_weight_agreement", {})

        intra_query_agrees = pd.DataFrame(intra_query_list)
        inter_query_agrees = pd.DataFrame(inter_query_list)
        intra_query_equal_weight_agrees = pd.DataFrame(intra_query_equal_weight_list)

        for method in self.methods:
            intra_query_agrees[method] = intra_query_agrees[method].astype(bool)
            self.results[type]["intra_query_agreement"].setdefault(method, [])
            self.results[type]["intra_query_agreement"][method].append(
                100 * intra_query_agrees[method].sum() / len(intra_query_agrees)
            )

            inter_query_agrees[method] = inter_query_agrees[method].astype(bool)
            self.results[type]["inter_query_agreement"].setdefault(method, [])
            self.results[type]["inter_query_agreement"][method].append(
                100 * inter_query_agrees[method].sum() / len(inter_query_agrees)
            )

            intra_query_equal_weight_agrees[method] = intra_query_equal_weight_agrees[
                method
            ].astype(bool)
            self.results[type]["intra_query_equal_weight_agreement"].setdefault(method, [])
            self.results[type]["intra_query_equal_weight_agreement"][method].append(
                100
                * intra_query_equal_weight_agrees[method].sum()
                / len(intra_query_equal_weight_agrees)
            )

    def generate_intraquery_equal_query_weight(
            self, df, queryToRows, spam_queries, replaceRate=0.2, replaceDelta=10
    ):
        """pairwise agreement within same query, queries with at least one spam equally weighted.
        queries with no spam ignored."""

        query = random.choice(spam_queries)
        intra_query_equal_weight_spam = df.iloc[
            queryToRows[query][1][random.randint(0, len(queryToRows[query][1]) - 1)]
        ]
        intra_query_equal_weight_non_spam = df.iloc[
            queryToRows[query][0][random.randint(0, len(queryToRows[query][0]) - 1)]
        ]
        intra_query_equal_weight_row = {
            "NormalizedQuery": intra_query_equal_weight_spam["NormalizedQuery"],
            "JudgeMarket": intra_query_equal_weight_spam["JudgeMarket"],
            "SpamUrl": intra_query_equal_weight_spam["Url"],
            "NonspamUrl": intra_query_equal_weight_non_spam["Url"],
        }

        return (
            intra_query_equal_weight_row,
            intra_query_equal_weight_spam,
            intra_query_equal_weight_non_spam,
        )

    def calc_auprc(self, df: pd.DataFrame, type: str) -> None:
        """Calculate AUPRC for each method and plot it."""

        self.results[type].setdefault("AUPRC", {})

        plt.figure(figsize=(10, 8))

        for method in self.methods:
            y_test = df["IsSpamLabel"] == True
            y_score = df[method]

            # Create a key for the method if doesn't exist
            self.results[type]["AUPRC"].setdefault(method, [])

            # Calculate AUPRC
            auprc = average_precision_score(y_test, y_score)
            self.results[type]["AUPRC"][method].append(auprc)

            # Compute and plot Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_score)

            plt.step(recall, precision, where='post', label=f'{method} (AUPRC = {auprc:.5f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')

        # Save the plot
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(f'plots/PRC_{type}.png')

        plt.show()

    def process_results(self) -> None:
        processed_results = {}
        # for every key in self.results[type], calculate mean and std using numpy in +\-
        for type in self.results:
            for key in self.results[type]:
                processed_results.setdefault(type, {}).setdefault(key, {})
                for method in self.results[type][key]:
                    if key == "AUPRC":
                        processed_results[type][key][method] = self.results[type][key][method][0]
                    else:
                        processed_results[type][key][method] = (
                                f"{np.mean(self.results[type][key][method]):.2f}"
                                + " +\- "
                                + f"{np.std(self.results[type][key][method]):.2f}"
                        )
        return processed_results

    def calc_metrics(self) -> None:
        df = self.read_tsv(self.input_path)
        df_repre_scores = self.read_tsv('representative_scores.tsv')
        df_processed = self.preprocess(df, df_repre_scores)
        df_processed_with_repre_scores = df_processed.copy()

        # create two versions of the df, one with representative scores and one without (nulls dropped)
        df_processed_with_repre_scores.fillna(
            {x: df_repre_scores[x][0] for x in df_repre_scores.columns}, inplace=True
        )
        queryToRows_repre_scores, spam_queries_repre_scores, spammy_df_repre_scores, non_spammy_df_repre_scores = self.prepare_sub_dfs(
            df_processed_with_repre_scores
        )

        # df_processed.dropna(inplace=True)
        queryToRows, spam_queries, spammy_df, non_spammy_df = self.prepare_sub_dfs(
            df_processed
        )

        for i in range(self.num_exp):
            start = time.time()
            print(f"Experiment {i}")
            self.run_single_iteration(
                df_processed, queryToRows, spam_queries, spammy_df, non_spammy_df, type="NoNulls"
            )

            self.run_single_iteration(
                df_processed_with_repre_scores, queryToRows_repre_scores, spam_queries_repre_scores,
                spammy_df_repre_scores, non_spammy_df_repre_scores, type="Nulls"
            )

            if i == 0:
                self.calc_auprc(df_processed, type="NoNulls")
                self.calc_auprc(df_processed_with_repre_scores, type="Nulls")
            end = time.time()

            print(f"Experiment {i} took {end - start} seconds")

            print(self.results)

        processed_results = self.process_results()

        pd.DataFrame(processed_results["Nulls"]).to_csv(
            self.output_path_with_nulls,
            sep="\t",
            index=self.methods,
            header=True,
            quoting=csv.QUOTE_NONE,
        )

        pd.DataFrame(processed_results["NoNulls"]).to_csv(
            self.output_path_no_nulls,
            sep="\t",
            index=self.methods,
            header=True,
            quoting=csv.QUOTE_NONE,
        )


def main(args):
    onedcg_processor = PairwiseAgreementEvaluator(
        input_path=args.input_path,
        output_path_with_nulls=args.output_path_with_nulls,
        output_path_no_nulls=args.output_path_no_nulls,
        representative_score_path=args.representative_score_path,
        margin=args.margin,
        methods=args.methods,
        num_samples=args.num_samples,
        num_exp=args.num_exp
    )
    onedcg_processor.calc_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path_with_nulls")
    parser.add_argument("--output_path_no_nulls")
    parser.add_argument(
        "--representative_score_path",
        help="tsv that contains representative score with headers",
    )
    parser.add_argument("--methods", help="comma separated list of methods")
    parser.add_argument("--margin", type=float)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--num_exp", type=int, help="number of experiments")

    args, _ = parser.parse_known_args()
    main(args)

