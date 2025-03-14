import re
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as smm

def cohens_d(list1, list2):
    n1, n2 = len(list1), len(list2)
    s1, s2 = np.nanvar(list1, ddof=1), np.nanvar(list2, ddof=1)
    pooled_var = ((n1-1) * s1 + (n2-1) * s2) / (n1 + n2 - 2)
    return (np.nanmean(list1) - np.nanmean(list2)) / np.sqrt(pooled_var)

def hedges_g(group1, group2):
    g = pg.compute_effsize(group1, group2, eftype='hedges')
    return g

def print_stats(name, data):
    av = np.mean(data)
    std = np.std(data)
    stat, pval = stats.shapiro(data)
    
    print(f"{name}, num elements: {len(data)}, mean: {av}, std: {std}, normal (above 0.5 is normal): {pval}")
    return pval

def run_stats_procedure(lemon, neutral, fish, title):
    print(f"\nNeutral: {sorted(list(neutral))}\nLemon: {sorted(list(lemon))}\nFish: {sorted(list(fish))}\n")
    # ANOVA
    norm_neutral_p = print_stats('Neutral', neutral)
    norm_lemon_p = print_stats('Lemon', lemon)
    norm_fish_p = print_stats('fish', fish)
    
    # calculating homogeneity of variance
    var_stat, var_p = stats.levene(lemon, neutral, fish)
    
    if var_p < 0.05 or norm_neutral_p < 0.05 or norm_lemon_p < 0.05 or norm_fish_p < 0.05:
        test = 'kruskal'
        group_stat, group_p = stats.kruskal(lemon, neutral, fish)
        print(f"Kruskal-Wallis: H {group_stat}, P VAL: {group_p}\n")
    else:
        test = 'anova'
        group_stat, group_p = stats.f_oneway(neutral, fish, lemon)
        print(f"ANOVA F: {group_stat}, P VAL: {group_p}\n")
    
    output[title] = {
        'neutral': sorted(neutral),
        'lemon': sorted(lemon),
        'fish': sorted(fish)
    }
    box_plots_prefiltering(title, neutral, lemon, fish)
    
    if group_p < 0.05:
        if test == 'anova':
            # T-Tests
            nf_stat, nf_pval = stats.ttest_ind(neutral, fish)
            nl_stat, nl_pval = stats.ttest_ind(neutral, lemon)
            fl_stat, fl_pval = stats.ttest_ind(fish, lemon)
            p_vals = [nf_pval, nl_pval, fl_pval]
            print(f"neutral v fish | t-stat: {nf_stat}, p-val: {nf_pval}, effect size: {cohens_d(neutral, fish)}, other effect size: {hedges_g(neutral, fish)}")
            print(f"neutral v lemon | t-stat: {nl_stat}, p-val: {nl_pval}, effect size: {cohens_d(neutral, lemon)}, other effect size: {hedges_g(neutral, lemon)}")
            print(f"fish v lemon | t-stat: {fl_stat}, p-val: {fl_pval}, effect size: {cohens_d(fish, lemon)}, other effect size: {hedges_g(fish, lemon)}\n")
            
        elif test == 'kruskal':
            nf_stat, nf_pval = stats.mannwhitneyu(neutral, fish, alternative='two-sided')
            nl_stat, nl_pval = stats.mannwhitneyu(neutral, lemon, alternative='two-sided')
            fl_stat, fl_pval = stats.mannwhitneyu(fish, lemon, alternative='two-sided')
            p_vals = [nf_pval, nl_pval, fl_pval]
            print(f"neutral v fish | U-stat: {nf_stat}, p-val: {nf_pval}, effect size: {cohens_d(neutral, fish)}, other effect size: {hedges_g(neutral, fish)}")
            print(f"neutral v lemon | U-stat: {nl_stat}, p-val: {nl_pval}, effect size: {cohens_d(neutral, lemon)}, other effect size: {hedges_g(neutral, lemon)}")
            print(f"fish v lemon | U-stat: {fl_stat}, p-val: {fl_pval}, effect size: {cohens_d(fish, lemon)}, other effect size: {hedges_g(fish, lemon)}\n")
            
    return output


def box_plots_prefiltering(title, neutral, lemon, fish):
    colors = ['#F0E68C', '#708090', '#FFB3BA']  # Slate Gray for Neutral, Softer yellow for Lemon, and Pink for Fish

    # Creating the box and whisker plot
    plt.figure(figsize=(11, 6))
    box = plt.boxplot(
        [lemon, neutral, fish], 
        labels=['Lemon', 'Neutral', 'Fish'], 
        patch_artist=True, 
        boxprops=dict(facecolor='lightgray', color='black'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )

    # Set individual colors for the boxes
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Adding title and labels
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Condition", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(f"figures/{title}.png", dpi=150)


def calculate_stats(df, relevant_columns, remove=True):
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
    
    output = {}
    
    i = 0
    while i < len(relevant_columns):
        metric = re.split('_', relevant_columns[i])[-1]
        title = all_metrics[metric]
        print(title)
        
        neutral = df[relevant_columns[i]].dropna()
        i += 1
        
        lemon = df[relevant_columns[i]].dropna()
        i += 1
        
        fish = df[relevant_columns[i]].dropna()
        i += 1
       
        output = run_stats_procedure(lemon, neutral, fish, title)
                        
    return output
        
        
def make_three_group_lists(df, important_col):
    fish_idx = np.where(df['CONDITION'] == 'bad')
    lemon_idx = np.where(df['CONDITION'] == 'good')
    neutral_idx = np.where(df['CONDITION'] == 'neutral')
    
    fish_list = df.loc[fish_idx, important_col]
    lemon_list = df.loc[lemon_idx, important_col]
    neutral_list = df.loc[neutral_idx, important_col]
    
    return fish_list, lemon_list, neutral_list
        

def three_lists_stats(fish_list, lemon_list, neut_list, name):
    fish_list = fish_list.replace(' ', np.nan).dropna()
    lemon_list = lemon_list.replace(' ', np.nan).dropna()
    neut_list = neut_list.replace(' ', np.nan).dropna()
    print(name)
    fish = [float(el) for el in fish_list]
    lemon = [float(el) for el in lemon_list]
    neutral = [float(el) for el in neut_list]
    fish.sort()
    lemon.sort()
    neutral.sort()
   
    output = run_stats_procedure(lemon, neutral, fish, name)
              
    return output
