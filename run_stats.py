import re
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as smm

def cohens_d(list1, list2):
    n1, n2 = len(list1), len(list2)
    s1, s2 = np.nanvar(list1, ddof=1), np.nanvar(list2, ddof=1)
    pooled_var = ((n1-1) * s1 + (n2-1) * s2) / (n1 + n2 - 2)
    return (np.nanmean(list1) - np.nanmean(list2)) / np.sqrt(pooled_var)

def print_stats(name, filtered_data):
    av = np.mean(filtered_data)
    std = np.std(filtered_data)
    stat, pval = stats.shapiro(filtered_data)
    
    print(f"{name}, num elements: {len(filtered_data)}, mean: {av}, std: {std}, normal (above 0.5 is normal): {pval}")

# trying median absolute deviation
def remove_outliers(data, name):  # takes in a list or pandas Series
    # for col in names:
    #     print(col)
    data = np.array(data) #[indices.astype(int)]
    median = np.median(data)
    mad_list = list(np.abs(data - median))
    mad_list.sort()
    
    mad = np.median(np.abs(data - median))  # Median Absolute Deviation
    # print(median, mad, mad_list)
    threshold = 3  # Equivalent to 3 standard deviations
    filtered_data = data[np.abs(data - median) <= threshold * mad]
    new_list = []
    for el in data:
        delta = np.abs(el - median)
        if delta <= (threshold * mad):
            new_list.append(el)
    # print(new_list, median, mad)
    print_stats(name, filtered_data)
    
    return filtered_data


def calculate_stats(df, relevant_columns):
    all_metrics = {
        'fc' : 'Fixation Count',
        'fd' : 'Cumulative Fixation Duration',
        'avfd' : 'Average Fixation Duration',
        'ccd' : 'Code Comp. Duration',
        'prose' : 'Prose Duration',
        'writing' : 'Writing Duration',
        'neg' : 'Negative Valence',
        'pos' : 'Positive Valence',
        'compound' : 'Compound Metric'
    }
    
    i = 0
    while i < len(relevant_columns):
        metric = re.split('_', relevant_columns[i])[-1]
        title = all_metrics[metric]
        print(title)
        
        neutral_temp = df[relevant_columns[i]].dropna()
        # print(f"Neutral: {list(neutral_temp)}")
        neutral = remove_outliers(neutral_temp, relevant_columns[i])
        i += 1
        
        lemon_temp = df[relevant_columns[i]].dropna()
        # print(f"Lemon: {list(lemon_temp)}")
        lemon = remove_outliers(lemon_temp, relevant_columns[i])
        i += 1
        
        fish_temp = df[relevant_columns[i]].dropna()
        # print(f"Fish: {list(fish_temp)}")
        fish = remove_outliers(fish_temp, relevant_columns[i])
        i += 1
        
        print(f"Neutral: {list(neutral)}\nLemon: {list(lemon)}\nFish: {list(fish)}")
        # print(f"\nNeutral: {neutral}\nLemon: {[lemon]}\nFish: {fish}\n")
        
        # ANOVA
        anova_stat, anova_pval = stats.f_oneway(neutral, fish, lemon)
        print(f"ANOVA F: {anova_stat}, P VAL: {anova_pval}")
        
        # T-Tests
        nf_stat, nf_pval = stats.ttest_ind(neutral, fish)
        nl_stat, nl_pval = stats.ttest_ind(neutral, lemon)
        fl_stat, fl_pval = stats.ttest_ind(fish, lemon)
        
        print(f"Neutral v Fish | T-stat: {nf_stat}, p-val: {nf_pval}, Effect Size: {cohens_d(neutral, fish)}")
        print(f"Neutral v Lemon | T-stat: {nl_stat}, p-val: {nl_pval}, Effect Size: {cohens_d(neutral, lemon)}")
        print(f"Fish v Lemon | T-stat: {fl_stat}, p-val: {fl_pval}, Effect Size: {cohens_d(fish, lemon)}\n")
        
        
def make_three_group_lists(df, important_col):
    fish_idx = np.where(df['CONDITION'] == 'bad')
    lemon_idx = np.where(df['CONDITION'] == 'good')
    neutral_idx = np.where(df['CONDITION'] == 'neutral')
    
    fish_list = df.loc[fish_idx, important_col]
    lemon_list = df.loc[lemon_idx, important_col]
    neutral_list = df.loc[neutral_idx, important_col]
    
    return fish_list, lemon_list, neutral_list
        

def three_lists_stats(fish_list, lemon_list, neut_list, name, remove=True):
    fish_list = fish_list.replace(' ', np.nan).dropna()
    lemon_list = lemon_list.replace(' ', np.nan).dropna()
    neut_list = neut_list.replace(' ', np.nan).dropna()
    print(name)
    # neut_list = neut_list.dropna()
    # print(type(neut_list[0]))
    fish = [float(el) for el in fish_list]
    lemon = [float(el) for el in lemon_list]
    neutral = [float(el) for el in neut_list]
    # list(fish_list.dropna())
    # l = list(lemon_list.dropna())
    # n = list(neut_list.dropna())
    fish.sort()
    lemon.sort()
    neutral.sort()
    
    print(f"FISH: {fish}")
    print(f"LEMON: {lemon}")
    print(f"NEUTRAL: {neutral}")
    
    if remove:
        neutral = remove_outliers(neutral, 'neutral')
        lemon = remove_outliers(lemon, 'lemon')
        fish = remove_outliers(fish, 'fish')
    # else:
    #     neutral = n.dropna()
    #     lemon = l.dropna()
    #     fish = f.dropna()
        
        print_stats('neutral', neutral)
        print_stats('lemon', lemon)
        print_stats('fish', fish)
        
    # ANOVA
    anova_stat, anova_pval = stats.f_oneway(neutral, fish, lemon)
    print(f"ANOVA F: {anova_stat}, P VAL: {anova_pval}")
    
    # T-Tests
    # for i in range(len(neutral)):
    #     print(type(i))
    # for i in range(len(fish)):
    #     print(type(i))
    # for i in range(len(lemon)):
    #     print(type(i))
    nf_stat, nf_pval = stats.ttest_ind(neutral, fish)
    nl_stat, nl_pval = stats.ttest_ind(neutral, lemon)
    fl_stat, fl_pval = stats.ttest_ind(fish, lemon)
    
    print(f"Neutral v Fish | T-stat: {nf_stat}, p-val: {nf_pval}, Effect Size: {cohens_d(neutral, fish)}")
    print(f"Neutral v Lemon | T-stat: {nl_stat}, p-val: {nl_pval}, Effect Size: {cohens_d(neutral, lemon)}")
    print(f"Fish v Lemon | T-stat: {fl_stat}, p-val: {fl_pval}, Effect Size: {cohens_d(fish, lemon)}\n")
        
