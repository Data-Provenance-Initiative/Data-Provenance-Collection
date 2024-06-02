import os
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter, defaultdict


#################################################################
############### Visualization Helpers
#################################################################


def plot_grouped_chart(
    info_groups, 
    group_names, 
    category_key, 
    name_remapper,
    exclude_groups,
    savename
):

    groups = defaultdict(list)
    for group_name in set(group_names) - set(exclude_groups):
        for license_group, dsets_info in info_groups.items():
            count = sum([1 if group_name in cat_to_vals[category_key] else 0 for cat_to_vals in dsets_info.values()])
            if name_remapper:
                groups[name_remapper.get(group_name, group_name)].append(count)
            else:
                groups[group_name].append(count)
    print(groups)

    total_dsets = sum([len(vs) for vs in info_groups.values()])
    custom_colors = ['#e04c71','#e0cd92','#82b5cf']
    groups = {trim_label(k): v for k, v in groups.items() if sum(v)}
    group_order = [k for k, v in sorted(groups.items(), key=lambda x: x[1][0] / sum(x[1]), reverse=False)]
    if len(groups) > 16:
        group_order = group_order[:8] + group_order[-8:]
    return plot_stackedbars(
        groups, None, list(info_groups.keys()),
        custom_colors, group_order, total_dsets, legend=None, savepath=f"paper_figures/altair/{savename}")


def plot_grouped_time_chart(
    info_groups,
    category_key,
    disallow_repeat_dsetnames,
    savename
):
    START_YEAR = 2013
    
    def bucket_time(t):
        if not t:
            return None
        if int(t.split("-")[0]) < START_YEAR:
            return f"< {START_YEAR}"
        else:
            return t.split("-")[0]
            
    ordered_tperiods = [f"< {START_YEAR}"] + [str(x) for x in range(START_YEAR, 2025)]
    groups = defaultdict(list)
    for group_name in ordered_tperiods:
        seenDsets = []
        for license_group, dsets_info in info_groups.items():
            vals = []
            for cat_to_vals in dsets_info.values():
                if disallow_repeat_dsetnames and cat_to_vals["Name"] in seenDsets:
                    continue
                seenDsets.append(cat_to_vals["Name"])
                
                vals.append(1 if group_name == bucket_time(cat_to_vals[category_key]) else 0)
            groups[group_name].append(sum(vals))
            # count = sum([1 if group_name == bucket_time(cat_to_vals[category_key]) else 0 for cat_to_vals in dsets_info.values()])
            # groups[group_name].append(count)
    print(groups)
    custom_colors = ['#e04c71','#e0cd92','#82b5cf']
    return plot_stackedbars(
        groups, None, list(info_groups.keys()),
        custom_colors, ordered_tperiods, 0, legend=None, savepath=f"paper_figures/altair/{savename}")


def plot_license_breakdown(
    infos, 
    license_classes,
    disallow_repeat_dsetnames,
    savename
):
    category_remapper = {
        "All": "Commercial",
        "NC": "Non-Commercial/Academic",
        "Acad": "Non-Commercial/Academic",
        "Custom": "Custom",
    }
    licenses_remapper = {
        "GNU General Public License v3.0": "GNU v3.0",
        "Microsoft Data Licensing Agreement": "Microsoft Data License",
        "Academic Research Purposes Only": "Academic Research Only",
        "Academic Free License v3.0": "AFL v3.0",
    }

    # list of license appearances
    if disallow_repeat_dsetnames:
        license_list = defaultdict(list)
        for cat_to_val in infos.values():
            license_list[cat_to_val["Name"]] = set(license_list[cat_to_val["Name"]]).union(set(cat_to_val["Licenses"]))
        license_list = [l for ll in license_list.values() for l in ll]
    else:
        license_list = [lic for cat_to_val in infos.values() for lic in cat_to_val["Licenses"]] 

    # Remove Unspecified
    license_list = [l for l in license_list if l != "Unspecified"]
    license_counts = Counter(license_list).most_common()
    # print(sum([v for (k, v) in license_counts]))
    # print(license_counts)
    
    def license_to_attributes(license):
        if license == "Custom":
            use_case, attr, sharealike = "Custom", 0, 0
        elif license_classes[license][1] == "?":
            use_case, attr, sharealike = "Non-Commercial/Academic", 1, 1
        else:
            use_case = category_remapper[license_classes[license][0]]
            attr = 1 if int(license_classes[license][1]) else 0
            sharealike = 1 if int(license_classes[license][2]) else 0
        return use_case, attr, sharealike

    license_infos = {}
    for license, count in dict(license_counts).items():
        use_case, attr, sharealike = license_to_attributes(license)
        license_infos[license] = {
            "Count": count, "Requires Attribution": attr, "Requires Share Alike": sharealike,
            "Allowed Use": use_case,
        }
    
    custom_colors = ['#82b5cf','#e04c71','#ded9ca']
    
    plot_seaborn_barchart(
        license_infos, "Licenses", "Count", "Requires Attribution", "Requires Share Alike",
        "Allowed Use", custom_colors, f"paper_figures/{savename}"
    )
    
    total_count = sum([vd["Count"] for vd in license_infos.values()])
    num_attr = sum([vd["Count"] for vd in license_infos.values() if vd["Requires Attribution"] == 1])
    num_sa = sum([vd["Count"] for vd in license_infos.values() if vd["Requires Share Alike"] == 1])
    print(f"Fraction of Total Licenses Requiring Attribution = {round(100 * num_attr / total_count, 2)}%")
    print(f"Fraction of Total Licenses Requiring Share Alike = {round(100 * num_sa / total_count, 2)}%")



# Splitting y-label into multiple lines:
def split_label(label, maxlen=24):
    words = label.split(' ')
    line = []
    new_label = []
    char_count = 0
    for word in words:
        char_count += len(word)
        if char_count > maxlen:
            new_label.append(' '.join(line))
            line = [word]
            char_count = len(word)
        else:
            line.append(word)
    new_label.append(' '.join(line))
    return '\n'.join(new_label)


def trim_label(label, maxlen=20):
    return label if len(label) < maxlen else label[:17] + "..."

def plot_stackedbars(
    data, 
    title, 
    category_names, 
    custom_colors,
    group_order, 
    total_dsets, 
    legend=True, 
    savepath=None
):
    
    # Ensure the color list matches the number of categories
    if len(custom_colors) != len(data[list(data.keys())[0]]):
        raise ValueError("Number of colors does not match number of categories!")
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data, columns=group_order, index=category_names)
    # print(df.columns)
    df = df[group_order].T
    # print(df.columns)
    # df = df[df.columns[bar_order]]
    df.index = df.index.map(split_label)
    
    # Calculate percentages for annotations
    # print(df)
    df_percentage = df.div(df.sum(axis=1), axis=0) * 100
    
    # Melt the dataframe for Altair
    df_melted = df.reset_index().melt(id_vars='index', var_name='category', value_name='value')
    df_melted_percentage = df_percentage.reset_index().melt(id_vars='index', var_name='category', value_name='percentage')
    df_melted['percentage'] = df_melted_percentage['percentage']
    
    order_mapping = {name: i for i, name in enumerate(category_names)}

    # Add an 'order' column based on the 'category' column and our mapping.
    df_melted['order'] = df_melted['category'].map(order_mapping)
    
    # Base chart for bars
    # print(bar_order)
    # print(df_melted.category)
    bars = alt.Chart(df_melted).mark_bar(width=50).encode(
        # y=alt.Y('percentage:Q', stack="normalize", axis=alt.Axis(format='%', labelFontSize=14, titleFontSize=16, title="Percentage (%)"), scale=alt.Scale(domain=[0,1]), order=bar_order),
        x=alt.X('index:N', sort=group_order, title=None, axis=alt.Axis(labelAngle=-25, labelFontSize=14)),
        y=alt.Y('percentage:Q', stack="normalize", sort=category_names, axis=alt.Axis(format='%', labelFontSize=14, titleFontSize=16, title="Percentage (%)", titleFontWeight='normal'), scale=alt.Scale(domain=[0,1])),
        color=alt.Color('category:N', sort=category_names, scale=alt.Scale(range=custom_colors), legend=alt.Legend(title=None) if legend else None),
        order='order:O' 
    )

    # Text annotations inside bars
    text = bars.mark_text(dx=0, dy=-7, align='center', baseline='middle', color='white', fontSize=14).encode(
        text=alt.condition(alt.datum.percentage > 0.05, alt.Text('percentage:Q', format='.1f'), alt.value(''))
    )
    
    # Calculate the totals for each bar
    df_totals = df.sum(axis=1).reset_index()
    df_totals.columns = ['index', 'total']
    df_totals['text_label'] = df_totals.apply(lambda row: f"({row['total']})", axis=1)

    # Totals text above bars
    totals_text = alt.Chart(df_totals).mark_text(dy=-32, align='center', baseline='top', fontSize=14).encode(
        x=alt.X('index:N', sort=category_names, title=None),
        y=alt.value(0),  # Positions text at the top of the bar
        text='text_label:N'
    )

    # Combine all layers
    chart = bars + text + totals_text
    chart = chart.properties(title="" if title is None else title, height=140, width=850)
    
    if savepath:
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        with open(savepath, 'w') as f:
            f.write(chart.to_json())
        # chart.save(savepath)#, format='svg')
    # else:
    return chart

def plot_seaborn_barchart(
    counts, 
    xlabel, 
    ylabel, 
    featureA, 
    featureB, 
    featureC, 
    custom_colors, 
    savepath=None
):
    plt.rcParams['font.family'] = 'Helvetica'
    # Convert counts to a DataFrame
    df = pd.DataFrame({
        xlabel: [split_label(k) for k in counts.keys()],
        ylabel: [v[ylabel] for v in counts.values()],
        featureA: [v[featureA] for v in counts.values()],
        featureB: [v[featureB] for v in counts.values()],
        featureC: [v[featureC] for v in counts.values()],
    })
    
    color_dict = dict(zip(df[featureC].unique(), custom_colors))
    df['color'] = df[featureC].map(color_dict)
    
    df['percentage'] = 100 * df[ylabel] / df[ylabel].sum()

    # sort the DataFrame and select the top categories
    df = df.sort_values(ylabel, ascending=False)[:21]

    # print (df)
    # Create the bar plot
    plt.figure(figsize=(20, 8))
    ax = sns.barplot(x=xlabel, y=ylabel, data=df, width=0.7)  # Adjust the width for increased spacing between bars
    
    # FeatureA edge color and FeatureB denser hatch pattern
    edge_color = 'purple'
    denser_hatch = '||'
    
    for idx, bar in enumerate(ax.patches):
        bar.set_facecolor(df.iloc[idx]['color'])
        if df.iloc[idx][featureA]:
            bar.set_edgecolor(edge_color)
            bar.set_linewidth(2)  # Set edge width for clarity
        if df.iloc[idx][featureB]:
            bar.set_hatch(denser_hatch)
    
    # Custom legend for edge colors and hatches
    legend_patches = [
        Patch(facecolor='gray', edgecolor=edge_color, linewidth=2, label=featureA),
        Patch(facecolor='gray', hatch=denser_hatch, label=featureB, edgecolor='purple'),
        # Rectangle((0, 0), 1, 1, facecolor='gray', hatch=denser_hatch, edgecolor='purple'),  # Custom patch for purple hatch

        # Patch(facecolor='gray', edgecolor=edge_color, linewidth=1.5, hatch=denser_hatch, label=f"{featureA} & {featureB}")
    ]
    # Adding patches for FeatureC colors
    for feature_value, color in color_dict.items():
        legend_patches.append(Patch(facecolor=color, label=f"{featureC}: {feature_value}"))
    ax.legend(handles=legend_patches, loc='upper right', fontsize=20)
    
    # Remove the border around the legend
    legend = ax.get_legend()
    legend.set_frame_on(False)

    # Add text labels
    for idx, bar in enumerate(ax.patches):
        # Adjusted the text positions to display count and percentage values above one another
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.05 * df[ylabel].max()), 
                f"{df.iloc[idx][ylabel]}", 
                ha='center', va='center', color='black', fontsize=18)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.14 * df[ylabel].max()), 
                f"({df.iloc[idx]['percentage']:.1f}%)", 
                ha='center', va='center', color='black', fontsize=18)
        
    ax.set_xlabel('', fontsize=18)
    ax.set_ylabel('', fontsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=18, rotation=65)  # Rotate x-axis labels to 65 degrees
    ax.yaxis.set_tick_params(labelsize=18)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, format='pdf', bbox_inches='tight')
    else:
        plt.show()


def plot_confusion_matrix(
    df,
    yaxis_order=None, 
    xaxis_order=None,
    text_axis=None,
    color_axis=None,
    yaxis_title="",
    xaxis_title="",
    font_size=20, 
    font_style='sans-serif',
    width=400,
    height=400,
):

    # Create the heatmap
    heatmap = alt.Chart(df).mark_rect().encode(
        x=alt.X(f'{xaxis_title}:N', title=xaxis_title, sort=xaxis_order),
        y=alt.Y(f'{yaxis_title}:N', title=yaxis_title, sort=yaxis_order),
        color=alt.Color(f'{color_axis}:Q', scale=alt.Scale(scheme='blues')),
        order="order:Q"
    )

    # circ = heatmap.mark_point().encode(
    #     alt.ColorValue('grey'),
    #     alt.Size('count()').title('Total Tokens')
    # )
    # .transform_filter(
    #     pts
    # )

    text = heatmap.mark_text(
        align='center',
        baseline='middle',
        fontSize=font_size,
        font=font_style,
    ).encode(
        # text=alt.Text('Formatted Percent:Q'),
        text=alt.Text(f'{text_axis}:N'),  # Format the text as "XX.Y"
        # color=alt.value('black'),
        color=alt.condition(
            alt.datum.Percent > 30,
            alt.value('white'),
            alt.value('black')
        )
    )
    
    # Combine heatmap and text annotations, and set font properties
    final_plot = (heatmap + text).properties(
        width=width,
        height=height,
        # title=alt.Title(
        #     text='Example Chart',
        #     fontSize=24,
        #     # fontStyle='italic',
        #     font=font_style
        # ),
    ).configure_axis(
        labelFontSize=font_size,
        labelFont=font_style,
        titleFontSize=font_size,
        titleFont=font_style,
        domain=True
        # labelAngle=30,
    ).configure_axisX(
        labelAngle=20,
        domain=True
    ).configure_axisY(
        domain=True  # Ensure the Y-axis domain line is shown
    ).configure_view(
        stroke='black'  # Add borders around the entire plot
    ).configure_legend(
        labelFontSize=font_size,
        titleFontSize=font_size,
        labelFont=font_style,
        titleFont=font_style
    )
    
    return final_plot



def create_stacked_area_chart(
    df, 
    period_col, 
    status_col, 
    percentage_col, 
    title='', 
    ordered_statuses=None, 
    status_colors=None,
    vertical_line_dates=[],
    width=1200, 
    height=400, 
    font_size=20, 
    font_style='sans-serif',
):
    if ordered_statuses is None:
        ordered_statuses = df[status_col].unique().tolist()
    
    if status_colors is None:
        status_colors = {status: 'gray' for status in ordered_statuses}
    
    # Create the Altair chart
    chart = alt.Chart(df).mark_area().encode(
        x=alt.X(f'{period_col}:T', axis=alt.Axis(format='%Y', title=period_col)),
        y=alt.Y(f'{percentage_col}:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0f', title=percentage_col)),
        color=alt.Color(
            f'{status_col}:N', 
            scale=alt.Scale(domain=list(status_colors.keys()), range=list(status_colors.values())), 
            title=status_col
        ),
        order="order:Q"
    )
    
    if vertical_line_dates:
        for vl_date in vertical_line_dates:
            vl_date = pd.to_datetime(vl_date)
            vertical_line = alt.Chart(pd.DataFrame({period_col: [vl_date]})).mark_rule(
                color='white',
                strokeDash=[5, 5]
            ).encode(
                x=f'{period_col}:T'
            )
            chart = chart + vertical_line

    final_plot = chart.properties(
        title=title,
        width=width,
        height=height
    ).configure_axis(
        labelFontSize=font_size,
        labelFont=font_style,
        titleFontSize=font_size,
        titleFont=font_style,
        domain=True
    ).configure_axisX(
        labelAngle=0,
        domain=True,
        format='%Y',
        tickCount='year',
        labelExpr="timeFormat(datum.value, '%Y')"  # Ensure only year labels are shown
    ).configure_axisY(
        domain=True,
        format='.0f',
        labelExpr="datum.value * 100"  # Scale from 0-1 to 0-100
    ).configure_legend(
        labelFontSize=font_size,
        titleFontSize=font_size,
        labelFont=font_style,
        titleFont=font_style
    )
    
    return final_plot






def plot_robots_time_map_3d_surface_plotly(
    filled_status_summary, agent_groups_to_track, val_key="tokens"
):
    rows = []
    for period, agent_dict in filled_status_summary.items():
        for agent, status_dict in agent_dict.items():
            if agent in agent_groups_to_track:
                for status, url_set in status_dict.items():
                    rows.append(
                        {
                            "period": str(period),
                            "agent": agent,
                            "status": status,
                            val_key: len(url_set),
                        }
                    )
    df = pd.DataFrame(rows)

    filtered_df = df[df["agent"].isin(agent_groups_to_track)]
    filtered_df["period"] = pd.to_datetime(filtered_df["period"], errors="coerce")
    filtered_df = filtered_df.dropna(subset=["period"])
    filtered_df = filtered_df[filtered_df["period"] <= pd.to_datetime("2024-04-30")]
    filtered_df["period"] = filtered_df["period"].dt.strftime("%Y-%m")
    filtered_df = filtered_df.sort_values(by="period")
    filtered_df["year"] = filtered_df["period"].str[:4]
    months_per_year = filtered_df.groupby("year")["period"].nunique().reset_index()
    months_per_year.columns = ["year", "months"]
    normalized_df = filtered_df.merge(months_per_year, on="year")
    normalized_df["normalized_count"] = normalized_df[val_key] / normalized_df["months"]

    pivot_df = normalized_df.pivot_table(
        index=["period", "status"],
        columns="agent",
        values="normalized_count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot_df = pd.melt(
        pivot_df, id_vars=["period", "status"], var_name="agent", value_name="count"
    )

    pivot_df["period"] = pivot_df["period"].astype(str)
    pivot_df["status"] = pivot_df["status"].astype(str)
    pivot_df["agent"] = pivot_df["agent"].astype(str)

    pivot_df["period_code"] = pivot_df["period"].astype("category").cat.codes
    pivot_df["agent_code"] = pivot_df["agent"].astype("category").cat.codes
    pivot_df["status_code"] = pivot_df["status"].astype("category").cat.codes

    periods = pivot_df["period_code"].unique()
    agents = pivot_df["agent_code"].unique()
    statuses = pivot_df["status_code"].unique()

    surface_data = []
    for status in statuses:
        status_df = pivot_df[pivot_df["status_code"] == status]
        z_values = pd.pivot_table(
            status_df,
            values="count",
            index="agent_code",
            columns="period_code",
            fill_value=0,
        ).values
        surface_data.append(z_values)

    fig = go.Figure()

    for idx, status in enumerate(statuses):
        fig.add_trace(
            go.Surface(
                z=surface_data[idx],
                x=periods,
                y=agents,
                colorscale="Viridis",
                name=pivot_df[pivot_df["status_code"] == status]["status"].iloc[0],
            )
        )

    fig.update_layout(
        title="Restriction Status across Agents over Time",
        scene=dict(
            xaxis_title="Period",
            yaxis_title="Agent",
            zaxis_title="Normalized Count",
            xaxis=dict(
                tickvals=pivot_df["period_code"].unique(),
                ticktext=pivot_df["period"].unique(),
            ),
            yaxis=dict(
                tickvals=pivot_df["agent_code"].unique(),
                ticktext=pivot_df["agent"].unique(),
            ),
        ),
        height=800,
        width=1200,
        margin=dict(l=60, r=60, t=80, b=60),
    )

    fig.show()


def plot_robots_time_map_3d_density(
    filled_status_summary, agent_groups_to_track, val_key="count"
):
    rows = []
    for period, agent_dict in filled_status_summary.items():
        for agent, status_dict in agent_dict.items():
            if agent in agent_groups_to_track:
                for status, url_set in status_dict.items():
                    rows.append(
                        {
                            "period": str(period),
                            "agent": agent,
                            "status": status,
                            val_key: len(url_set),
                        }
                    )
    df = pd.DataFrame(rows)

    filtered_df = df[df["agent"].isin(agent_groups_to_track)]
    period_mapping = {
        period: idx for idx, period in enumerate(sorted(filtered_df["period"].unique()))
    }
    agent_mapping = {
        agent: idx for idx, agent in enumerate(sorted(filtered_df["agent"].unique()))
    }
    status_mapping = {
        status: idx for idx, status in enumerate(["no_robots", "none", "some", "all"])
    }

    filtered_df["period_num"] = filtered_df["period"].map(period_mapping)
    filtered_df["agent_num"] = filtered_df["agent"].map(agent_mapping)
    filtered_df["status_num"] = filtered_df["status"].map(status_mapping)

    x = filtered_df["period_num"]
    y = filtered_df["agent_num"]
    z = filtered_df["status_num"]
    values = filtered_df[val_key]

    kde = gaussian_kde([x, y, z], weights=values)
    xi, yi, zi = np.meshgrid(
        np.linspace(x.min(), x.max(), 50),
        np.linspace(y.min(), y.max(), 50),
        np.linspace(z.min(), z.max(), 50),
    )
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
    density = kde(coords).reshape(xi.shape)

    fig = go.Figure(
        data=go.Volume(
            x=xi.flatten(),
            y=yi.flatten(),
            z=zi.flatten(),
            value=density.flatten(),
            isomin=0.1,
            isomax=density.max(),
            opacity=0.1,
            surface_count=20,
            colorscale="Viridis",
        )
    )

    fig.update_layout(
        title=dict(text="Restriction Status across Agents", font=dict(size=24)),
        scene=dict(
            xaxis=dict(
                title="Period",
                tickvals=list(period_mapping.values()),
                ticktext=list(period_mapping.keys()),
                tickangle=45,
                autorange=True,
            ),
            yaxis=dict(
                title="Agent",
                tickvals=list(agent_mapping.values()),
                ticktext=list(agent_mapping.keys()),
                tickangle=-45,
                autorange=True,
            ),
            zaxis=dict(
                title="Status",
                tickvals=list(status_mapping.values()),
                ticktext=list(status_mapping.keys()),
            ),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        height=800,
        width=1200,
    )

    fig.show()


def plot_company_comparisons_altair(
    df,
    vertical_line_dates=[],
    color_mapping={}
):
    # Assuming you have your data in a DataFrame called 'df'
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    
    # Convert the 'Period' column to datetime
    # df['period'] = pd.to_datetime(df['period'], format='%Y-%m')
    df['period'] = df['period'].dt.to_timestamp()
    
    # Calculate the percentage of tokens for the 'Restrictive' status
    total_tokens = df.groupby(['period', 'agent'])['tokens'].sum().reset_index()
    restrictive_tokens = df[df['status'] == 'all'].groupby(['period', 'agent'])['tokens'].sum().reset_index()
    data = pd.merge(total_tokens, restrictive_tokens, on=['period', 'agent'], how='left').fillna(0)
    data['percent_Restrictive'] = (data['tokens_y'] / data['tokens_x']) * 100
    data = data[['period', 'agent', 'percent_Restrictive']]
    
    # Create the color scale based on the provided color mapping
    color_scale = alt.Scale(domain=list(color_mapping.keys()), range=list(color_mapping.values()))

    # Create the Altair chart
    chart = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X('yearmonth(period):T', title='Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('percent_Restrictive:Q', title='Percentage of Tokens'),
        color=alt.Color('agent:N', scale=color_scale, legend=alt.Legend(title='Agent', orient='bottom')),
        # color=alt.Color('agent:N', scale=color_scale, legend=alt.Legend(title='Agent')),
        tooltip=['agent', alt.Tooltip('percent_Restrictive:Q', format='.2f')]
    )
    
    if vertical_line_dates:
        for vl_date in vertical_line_dates:
            vl_date = pd.to_datetime(vl_date)
            vertical_line = alt.Chart(pd.DataFrame({'period': [vl_date]})).mark_rule(
                color='white',
                strokeDash=[5, 5]
            ).encode(
                x='period:T'
            )
            chart = chart + vertical_line

    chart = chart.properties(
        width=1000,
        height=200
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )
    
    return chart
