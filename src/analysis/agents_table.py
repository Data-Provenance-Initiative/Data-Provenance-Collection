import pandas as pd

file_path = "src/analysis/data/agents_counter/all_agents_counter.csv"
df = pd.read_csv(file_path)

additional_agents = [
    "GPTBot",
    "ChatGPT-User",
    "Google-Extended",
    "CCBot",
    "ClaudeBot",
    "anthropic-ai",
    "Claude-Web",
    "Amazonbot",
    "FacebookBot",
    "cohere-ai",
    "ia_archiver",
    "Amazonbot",
    "*All Agents*",
]


df["all_percentage"] = (df["all"] / df["observed"] * 100).round(2)
df["some_percentage"] = (df["some"] / df["observed"] * 100).round(2)
df["none_percentage"] = (df["none"] / df["observed"] * 100).round(2)

# changed this to reflect paper, whic shows top 40
top_40_agents = df.sort_values(by="observed", ascending=False).head(40)


additional_df = df[df["agent"].isin(additional_agents)]
combined_df = (
    pd.concat([top_40_agents, additional_df]).drop_duplicates().reset_index(drop=True)
)


latex_table = r"""
\begin{table*}[t!]
\centering
\begin{adjustbox}{width=\textwidth}
\begin{tabular}{l|r|r|r|r|r|r|rr}
\toprule
            \textsc{Agent Name} & \# \textsc{Observed} & \multicolumn{2}{c}{\textsc{All Disallowed}} & \multicolumn{2}{c}{\textsc{Some Disallowed}}  & \multicolumn{2}{c}{\textsc{None Disallowed}}  \\
           & & Count & \% & Count & \% & Count & \% \\
\midrule
"""


for _, row in combined_df.iterrows():
    agent_name = row["agent"]
    observed = int(row["observed"])
    all_count = int(row["all"])
    some_count = int(row["some"])
    none_count = int(row["none"])
    all_percentage = row["all_percentage"]
    some_percentage = row["some_percentage"]
    none_percentage = row["none_percentage"]

    # Highlight the additional agents with a different row color
    if agent_name in additional_agents:
        latex_table += r"    \rowcolor[gray]{0.9}    "

    latex_table += f"{agent_name} & {observed:,} & {all_count} & {all_percentage:.2f}\\% & {some_count} & {some_percentage:.2f}\\% & {none_count} & {none_percentage:.2f}\\% \\\\\n"

latex_table += r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Summary of User Agents with Observed Counts and Disallowed Percentages.}
\end{table*}
"""


print(latex_table)
