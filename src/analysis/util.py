import os
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


############################################################
###### Text Pinetuning Analysis Functions
############################################################


CONTENT_DOMAIN_MAPPING = {
    'News': ['News', 'Current affairs, sports', 'News archives', 'Newsletter', 'Articles'],
    'Education/Knowledge': ['Education/Knowledge', 'info about schools', 'Education/Knowledge', 'Exams', 'Company website and services'],
    'Entertainment': [
        'Entertainment', 'Anime', 'Sports', 'Gaming', 'Sport', 'Sports/games', 'University sports', 'Game website', 'Adult stories website',
        'Porn website', 'Escort procurement'
    ],
    'Business/Finance/Organizational': [
        'Business/Finance', 'Finance', 'Business directory', 'B2B marketplace', 'Business finder',
        'Personal business', 'Management Consultant', 'Company portal', 'Company website',
        'Fundraising portal', 'Marketing', 'Non-profit agency', 'Non Profit', 'Non-profit/Charity',
        'Nonprofit organization ', 'Non-Profit', 'Non Profit, Recycling', 'Freelancer designer', 
        'Freelancer designer & developer', 'Application website', 'Organizations', 'Charity', 'Donation', 
        'Fundraising/Donation', 'Organization description', 'Insurance company website',
        'Security', 'jobs'
    ],
    'Technology/Code': [
        'Technology/Code', 'Search engine', 'Search engine portal', 'Internet RFC/STD/FYI/BCP document archives', 'Search tool',
        'Web ring', 'Mailing list', 'directory / webring'
    ],
    'E-Commerce': ['E-Commerce', 'Ecommerce', 'E-shop page', 'Online shop'],
    'Academic': ['Academic', 'Social sciences', 'Science', 'Ideologies and thought', 'Religion and Philosophy', 'ancestry analysis'],
    'General': [
        'General', 'Prototype', 'Forum', 'Encyclopedia'],
    'Cultural/Artistic': [
        'Cultural/Artistic', 'Painting', 'Personal website', 'personal website', 'Personal page', 'personal webpage', 'reddit posts',
        'Music',
    ],
    'Social/Lifestyle': [
        'Social/Lifestyle', 'Social', 'Lifestyle', 'Home improvement / design', 'Fitness',
        'Home improvement', 'Human Rights / NGO', 'Women Empowerment', 'geographic and demographic analytics',
        'Food', 'Recipes', 'Beverage', 'Menu'
    ],
    'Reviews': ['Reviews', 'Review portal', 'Images database', 'digital archive'],
    'Government/Policy': ['Government/Policy', 'Politics', 'Military', 'Employment', 'Service'],
    'Biomedical/Health': ['Biomedical/Health', 'Biology', 'Medical'],
    'Books': ['Books', 'Documents'],
    'Legal': ['Legal'],
    'Religion': ['Religion', 'Religion/Christianity', 'Religious website'],
    'Travel': ['Travel', 'Travel and tourism', 'Travel information', 'Trips'],
    'Blog': ['Blog', 'Blogs'],
}
CONTENT_DOMAIN_INVERSE_MAPPING = {v.lower().strip(): k.lower().strip() for vs in CONTENT_DOMAIN_MAPPING.items() for v in vs}

WEBSITE_SERVICE_MAPPING = {
    'News/Periodicals': [
        'Australian news and financial site providing video, audio, and stories',
        'Broadcasting and radio shows while communicating with listeners',
        'Contains a wide variety of different books, audio, and articles',
        'Daily tips',
        'Electronic publishing sit containing a variety of content',
        'Jamaica news, lifestyle, and entertainment articles',
        'Journal',
        'Lifestyle magazine',
        'News and articles for comic books',
        'News and broadcasting site providing video, audio, and stories',
        'News and entertainment site for pop culture',
        'News and lifestyle site for Kitsap providing video, audio, and stories',
        'News and lifestyle site providing video, audio, and stories',
        'News and technology site',
        'News articles about different social trends',
        'News articles and reviews for biking',
        'News site providing video and stories',
        'News site providing video, audio, and stories',
        'Periodicals',
        'Radio Station'
    ],
    'Organization/personal website': [
        'A member and association organization for bartenders',
        'A website where users can upload and download books and other publications',
        'Artist website',
        'Astrology website',
        'Business / Finance',
        'Business directory',
        'Camera and Security Systems',
        'Charity',
        'CharityAutism',
        'Chiropractic, Nutrition, therapies services',
        'Company locations for firestone automotive parts and services',
        'Company website',
        'Dental Health',
        'Employee Training',
        'Finance advise & services',
        'Financial Planning',
        'Financial information',
        'Find and hire freelancers',
        'Find companies for different home services, cleaning, and other needs.',
        'Information about the services offered by the manufacturing company',
        'Insurance Services',
        'Internation Organization website',
        'Local sport league organization website',
        'NGO',
        'Non Profit',
        'Non-profit',
        'Nonprofit Website',
        'Nonprofit website',
        'Organization',
        'Organization website',
        'Packaging and supplies services',
        'Personal Blog',
        'Personal Injury LawGraphics and web designing',
        'Personal Website',
        'Personal website',
        'Property Selling',
        'Religious organization website',
        'Restaurant',
        'Road Running',
        'Social/Human rights',
        'Sports website for a university',
        'Tech storage solutions',
        'investment management',
        'nonprofit organization',
        'personal site',
        'personal webpage',
        'personal website',
        'school website'
    ],
    'Encyclopedia/Database': [
        '& Encyclopedia',
        'Article directory',
        'Collection of photography guides',
        'Compilation of books, historic and other academic works at Tufts',
        'Compilation of different medical research papers',
        'Compilation of different podcasts',
        'Compilation of different sermons',
        'Compilation of tumblr blogs',
        'Contains a repository of different fantasy novels and books',
        'Database of sermons',
        'Education/Knowledge',
        'Encyclopedia',
        'Encyclopedia/Database',
        'Information',
        'Internet RFC/STD/FYI/BCP document archives',
        'Law resources',
        'Legal Research Database',
        'Legal database',
        'Mail archive',
        'Oncology research papers and archives',
        'Portal',
        'Religion-based library',
        'Search',
        'Search engine',
        'Search for books',
        'Search tool',
        'Tracking GIT changes for GNU OS',
        'User can access coupon codes for different services',
        'Wiki site for creative writing',
        'Wiki site for fan translation',
        'Wikipedia skin for easier reading',
        'jobs database',
        'online library',
        'patents website',
        'tool for web traffic overview'
    ],
    'Ecommerce': [
        'A website for a software company providing coding and other services',
        'Book self-publishing website',
        'Bookings website for sailing trips',
        'Business Apps',
        'Buy and sell different books online',
        'Computer repairing',
        'Coupon codes',
        'Designing and editing',
        'E-Commerce',
        'Ecommerce',
        'Find local apartments to rentTravel and Navigation Services',
        'Hair Treatment',
        'Hotel',
        'Paywalled business analysis',
        'Physical store directory',
        'Plant and gardening store',
        'Streaming services',
        'Travel Guide',
        'Travel bookings',
        'Travel guide',
        'Travel website where users can book flights, transportation, hotels, and other accommodations',
        'Wholesale',
        'booking platform',
        'gambling website',
        'independent sellers platform',
        'product website',
        'travel website with curated blogs'
    ],
    'Academic': [
        'A wide variety of technological articles and newsletters',
        'Academic',
        'Academic opportunity search',
        'Clinical and medical podcasts, blogs, news, and videos',
        'Diabetes medical information and articles',
        'Education',
        'Educational resources + review aggregator',
        'Educational study tools',
        'Financial reports, filings, news, and transcripts',
        'Journal,health',
        'Medical and biotech research for immunology',
        'Provides testing for dna, drugs, and other things',
        'USGPO articles and information',
        'academic journal website',
        'education/ courses'
    ],
    'Social Media/Forums': [
        'A forum for obisidian games where users can discuss games and community announcements',
        'A forum for relationship and dating advice with some crafted articles',
        'A forum for users to ask questions about metalworking and model engineering',
        'A forum where users can ask questions about the unity game engine',
        'Advice',
        'Also contains a forum',
        'Also forum/social',
        'An internet archive capture of a gaming article',
        'Blog posts portal',
        'Blogs/forum for building things',
        'Contains user uploaded videos',
        'Crowdfunding platform',
        'Eritrean analysis forum',
        'Fan art website',
        'Find local area businesses',
        'Find user reviews for different products',
        'Forum for different rockstar videogames',
        'Forum for mobile developers and others to ask questions about phones and accessories.',
        'Forum for tech news, computers, and other forms of hardware',
        'Forum for users to discuss body building, fitness, and trt',
        'Forum for users to discuss lucid dreaming experiences',
        'Forums',
        'Forums and community for mothers',
        'Forums for discussing the video game warframe',
        'Game news forum',
        'Gaming community forum where users can comment on gaming videos and have discussions',
        'Gaming website',
        'Health articles for different chronic illnesses and community forums',
        'Information about pregnancy and community forums',
        'Personal blog',
        'Question/Answering portal',
        'Review',
        'Review portal',
        'Review site',
        'Review site / photo aggregator',
        'Reviews',
        'Social Media',
        'Social Media for tech',
        'Social Media/Forums',
        'Social media / forums',
        'Social media book reviews website',
        'Social media/forums',
        'Stack exchange forum for physics questions and answers',
        'Tech product discussion',
        'The 4chan forum',
        'User forums',
        'Users can find local caregivers for different needs',
        'Users rate different movies, tv, and video games',
        'Users talk about different unsolved mysteries and related news',
        'Video game community and forums',
        'Video game, community forum, and wikipedia',
        'and forum',
        'content and forum',
        'forum',
        'forums',
        'law (student) network',
        'old forum no longer available',
        'online community',
        'reddit posts',
        'review platform',
        'social media'
    ],
    'Government': [
        'Articles and comments about Australian government and politics',
        'Government',
        'Government records',
        'Government website containing agenda and publications',
        'Governmental information tracking',
        'sustainability website with news and tips'
    ],
    'Blog': [
        '(single-poster) Blog',
        'Adult stories',
        'Archive of technology blog posts',
        'Blog',
        'Blog directory',
        'Blog platform',
        'Blog system',
        'Blog,Podcast',
        'Blog?',
        'Blogs',
        'Christian opinion website',
        'Climate Blog',
        'Entertaiment blog',
        'Entertainment Blog',
        'Entertainment biographies',
        'Fan fiction for fantasy',
        'Lifestyle',
        'Lifestyle blog',
        'Lifestyle portal',
        'Music and information about a Dutch band',
        'Personal blog',
        'Poetry sharing, reading, and articles',
        'Religious blog style website',
        'Self help site',
        'Technology blog',
        'blog ',
        'blog site for gear review',
        'blog site of a religion group',
        'blog style website for food',
        'blogs',
        'novels website',
        'personal blog'
    ],
    'Periodicals/News': [
        'A news and lifestyle website for Cape Cod and state of Massachusetts',
        'Argentinian racing content',
        'Books,ebooks,magazine',
        'Contains articles about current events in Sarasota',
        'Contains articles about current events in Utah',
        'Current events in Utah and news articles',
        'Energy price news',
        'India news articles and videos',
        'Indian consumer technology news',
        'Lifestyle articles portal',
        'Local news articles for Colorado',
        'Middle East news articles',
        'Mix of news and fan content',
        'News and Search Services',
        'News and insights',
        'News and reviews portal',
        'News articles for Ghana',
        'News articles, podcasts, and radio about current events',
        'News collection',
        'News for tabletop and rpg games and community forums',
        'News site',
        'News site for different medical and life sciences articles',
        'News website and blog',
        'News, weather, and general info',
        'Opinion pieces on news stories',
        'Periodical',
        'Podcasts',
        'Regional sports news',
        'TV Station',
        'Technology newsletter and articles',
        'Telegraph India news articles',
        'Video game news',
        'Video game reviews',
        'entertainment tv channel',
        'non-official news media',
        'non-periodical news',
        'religious magazine'
    ]
}
WEBSITE_SERVICE_INVERSE_MAPPING = {v.lower().strip(): k.lower().strip() for vs in WEBSITE_SERVICE_MAPPING.items() for v in vs}






############################################################
###### Text Finetuning Analysis Functions
############################################################

def check_datasummary_in_constants(rows, all_constants):
    """Tests your data summary rows to see if all the values are in the constants.

    If not, it will print out the missing values, and which data collections they came from.
    """
    CREATOR_TO_COUNTRY = {v: k for k, vs in all_constants["CREATOR_COUNTRY_GROUPS"].items() for v in vs}
    CREATOR_TO_GROUP = {v: k for k, vs in all_constants["CREATOR_GROUPS"].items() for v in vs}
    TASK_TO_GROUP = {v: k for k, vs in all_constants["TASK_GROUPS"].items() for v in vs}
    LANG_TO_GROUP = {v: k for k, vs in all_constants["LANGUAGE_GROUPS"].items() for v in vs}
    SOURCE_TO_GROUP = {v: k for k, vs in all_constants["DOMAIN_GROUPS"].items() for v in vs}
    LICENSE_CLASSES = list(all_constants["LICENSE_CLASSES"].keys()) + list(all_constants["CUSTOM_LICENSE_CLASSES"].keys())

    def check_entities(collection_id, vals, const_map, miss_dict):
        for v in vals:
            if v not in const_map:
                miss_dict[v].add(collection_id)
                
    # Category --> missing entry --> list of Collection IDs where this comes from.
    missing_metadata = defaultdict(lambda: defaultdict(set))
    for row in rows:

        row_licenses = [lic["License URL"] if lic["License"] == "Custom" else lic["License"] for lic in row["Licenses"]]
        check_entities(row["Collection"], row_licenses, 
                       LICENSE_CLASSES, missing_metadata["License Classes"])

        check_entities(row["Collection"], row.get("Creators", []), 
                       CREATOR_TO_GROUP, missing_metadata["Creator Groups"])

        check_entities(row["Collection"], row.get("Creators", []), 
                       CREATOR_TO_COUNTRY, missing_metadata["Creator Countries"])

        check_entities(row["Collection"], row.get("Task Categories", []), 
                       TASK_TO_GROUP, missing_metadata["Task Categories"])

        check_entities(row["Collection"], row.get("Text Sources", []), 
                       SOURCE_TO_GROUP, missing_metadata["Text Sources"])

        check_entities(row["Collection"], row.get("Languages", []), 
                       LANG_TO_GROUP, missing_metadata["Languages"])
        
    for category, missing_info in missing_metadata.items():
        if len(missing_info) == 0:
            print(f"No missing info for {category}!")
            # print()
        else:
            print(f"There is metadata missing from the constants/ files for {category}:")
            print("Please check if you can modify the name of the entity (in data summary) to exactly match the entity as written in the constants files -- so we don't have multiple versions.")
            print("If it is not in the constants file in any form, then add it to the constants file.")
            print()
            for x, collections in missing_info.items():
                print(x + f"   |   Appears in: {collections}")
        print()


def extract_info(rows, all_constants):
    """Interpret the categories across all data summary rows.

    Returns:
        Dict: {dataset_uid --> {attribute --> value}}

        The value can be a list (e.g. tasks/sources/creators), float (e.g. num exs), or string (license class)
    """
    CREATOR_TO_COUNTRY = {v: k for k, vs in all_constants["CREATOR_COUNTRY_GROUPS"].items() for v in vs}
    CREATOR_TO_GROUP = {v: k for k, vs in all_constants["CREATOR_GROUPS"].items() for v in vs}
    TASK_TO_GROUP = {v: k for k, vs in all_constants["TASK_GROUPS"].items() for v in vs}
    LANG_TO_GROUP = {v: k for k, vs in all_constants["LANGUAGE_GROUPS"].items() for v in vs}
    SOURCE_TO_GROUP = {v: k for k, vs in all_constants["DOMAIN_GROUPS"].items() for v in vs}

    # {dataset_uid --> {attribute --> value}}
    dataset_infos = {}
    for row in rows:
        dataset_uid = row["Unique Dataset Identifier"]
        if dataset_uid not in dataset_infos:
            dataset_infos[dataset_uid] = {
                "Name": row["Dataset Name"],
                "Sources": set(),
                "Domains": set(),
                "Synthetic": False,
                "Licenses": set(),
                "License Use (DataProvenance)": None,
                "License Attribution (DataProvenance)": None,
                "License Share Alike (DataProvenance)": None,
                "License Use (HuggingFace)": None,
                "License Use (GitHub)": None,
                "License Use (PapersWithCode)": None,
                "Creators": set(),
                "Creator Groups": set(),
                "Creator Countries": set(),
                "Input Text Lengths": 0,
                "Target Text Lengths": 0,
                "Num Exs": 0,
                "Dialog Turns": 0,
                "Tasks": set(),
                "Task Groups": set(),
                "Languages": set(),
                "Language Groups": set(),
                "Preparation Times": None
            }
        
        info = dataset_infos[dataset_uid]

        # Update sets
        info["Sources"].update(row.get("Text Sources", []))
        info["Domains"].update({SOURCE_TO_GROUP[x] for x in row.get("Text Sources", [])})
        info["Licenses"].update([lic_info["License"] for lic_info in row.get("Licenses", [])])
        info["Creators"].update(row.get("Creators", []))
        info["Creator Groups"].update([CREATOR_TO_GROUP[x] for x in row.get("Creators", [])])
        info["Creator Countries"].update([CREATOR_TO_COUNTRY[x] for x in row.get("Creators", [])])
        info["Tasks"].update(row.get("Task Categories", []))
        info["Task Groups"].update([TASK_TO_GROUP[x] for x in row.get("Task Categories", [])])
        info["Languages"].update(row.get("Languages", []))
        info["Language Groups"].update([LANG_TO_GROUP[x] for x in row.get("Languages", [])])

        # Update numeric values
        text_metrics = row.get("Text Metrics", {})
        info["Input Text Lengths"] += text_metrics.get("Mean Inputs Length", 0)
        info["Target Text Lengths"] += text_metrics.get("Mean Targets Length", 0)
        info["Num Exs"] += text_metrics.get("Num Dialogs", 0)
        info["Dialog Turns"] += text_metrics.get("Mean Dialog Turns", 0)

        # Synthetic data flag
        info["Synthetic"] = info["Synthetic"] or (len(row.get("Model Generated", [])) > 0)

        # Update single values, assuming they are consistent across duplicated datasets
        lic_keys = ["License Use (DataProvenance)", "License Use (DataProvenance)", "License Use (DataProvenance)", "License Use (DataProvenance)", \
                    "License Attribution (DataProvenance)", "License Share Alike (DataProvenance)", "Preparation Times"]
        for field in lic_keys:
            if row.get(field) is not None:
                if info[field] is not None and info[field] != row.get(field):
                    raise ValueError(f"Inconsistent values for {field} in dataset '{dataset_uid}'")
                info[field] = row.get(field)

        s2_time = row.get("Inferred Metadata", {}).get("S2 Date") or "3000"
        pwc_time = row.get("Inferred Metadata", {}).get("PwC Date") or "3000"
        hf_time = row.get("Inferred Metadata", {}).get("Github Date") or "3000"
        gh_time = row.get("Inferred Metadata", {}).get("HF Date") or "3000"
        earlier_time = min([s2_time, pwc_time, hf_time, gh_time])
        info["Preparation Times"] = None if earlier_time == "3000" else earlier_time

    # Convert set to list to finalize
    set_keys = ["Sources", "Domains", "Licenses", "Creators", "Creator Groups", "Creator Countries", "Tasks", "Task Groups", "Languages", "Language Groups"]
    for dataset_uid, info in dataset_infos.items():
        for key in set_keys:
            info[key] = list(info[key])
            
    return dataset_infos


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