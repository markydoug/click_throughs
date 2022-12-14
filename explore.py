import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.ticker as mtick

days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']

def click_percentage(train):
    #split data for plotting
    train_click = train[train.click == 1]
    train_no_click = train[train.click == 0]

    #create the percentages
    values = [len(train_click.click), len(train_no_click.click)] 

    # generate and show pie chart
    plt.style.use('seaborn')
    plt.pie(values, labels=["Click", "Didn't Click"] , autopct='%.2f%%', colors=['#ffc3a0', '#c0d6e4'], textprops={'fontsize': 14})
    plt.title('Click-throughs make up 17% of the Train Data', size=20)
    plt.show()

def hour_click_through_viz(train):
    #set font size
    sns.set(font_scale=1.5)
    #set graph style
    sns.set_style('white')
    
    #set size of the graphs
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))
    fig.tight_layout(pad=4.0)

    sns.countplot(data=train, x='hour_of_day', palette='colorblind', ax=ax1)
    ax1.set_title("Total Instances by Hour of the Day (24hrs)")
    ax1.set_ylabel('Total number of Clicks')
    ax1.set_xlabel('Hour of the day (24hrs)')

    click_mean = pd.DataFrame(train.groupby('hour_of_day').click.mean())*100
    sns.lineplot(data=click_mean, x='hour_of_day', y='click', ax=ax2)
    ax2.set_title('Percentage of Clicks Per Hour')
    fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax2.yaxis.set_major_formatter(xticks)
    ax2.set_ylabel('Percentage of Actual Click Throughs')
    ax2.set_xlabel('Hour of the day (24hrs)')

    plt.show()

def chi_square_matrix(data, feature, target):
    info = []

    for i in data[feature].unique():
        observed = pd.crosstab(data[feature]==i, data[target])
        #run χ^2 test
        chi2, p, degf, expected = stats.chi2_contingency(observed)

        output = {
            feature: i,
            "χ^2": chi2,
            "p-value": p
        }
        
        info.append(output)
        
    df = pd.DataFrame(info)
    df = df.sort_values(by=['p-value','χ^2'], ascending=[True,False])

    df = df.set_index(feature)

    return df

def day_click_through_viz(train):
    #set font size
    sns.set(font_scale=1.5)
    #set graph style
    sns.set_style('white')

    #set size of the graphs
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))
    fig.tight_layout(pad=4.0)

    sns.countplot(data=train, x='day_of_week', palette='colorblind', ax=ax1, order=days)
    ax1.set_title("Total Instances by Day of the Week")
    ax1.set_ylabel('Total number of Clicks')
    ax1.set_xlabel('Day of the Week')

    click_mean = pd.DataFrame(train.groupby('day_of_week').click.mean())
    click_mean.index = pd.Categorical(click_mean.index,
                                categories=days,
                                ordered=True)
    sns.lineplot(data=click_mean, x=click_mean.index, y=click_mean.click*100)
    ax2.set_title('Percentage of Clicks Per Hour')
    fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax2.yaxis.set_major_formatter(xticks)
    ax2.set_ylabel('Percentage of Actual Click Throughs')
    ax2.set_xlabel('Day of the Week')

    plt.show()

def banner_pos_viz(train):
    #set font size
    sns.set(font_scale=1.5)
    #set graph style
    sns.set_style('white')
        
    #set size of the graphs
    fig, ax = plt.subplots(1,1, figsize=(20,8))
    fig.tight_layout(pad=4.0)

    sns.countplot(data=train, x='banner_pos', hue='click', ax=ax)
    ax.set_title("Clicks by Banner Position")
    ax.set_ylabel('Total number of Clicks')
    ax.set_xlabel('Day of the Week')
    ax.legend(["Didn't Click", 'Click'])

    plt.show()

def continuous_vars_ttest(train):
    annon_col = [f'C{i}' for i in range(14,22)]

    train_click = train[train.click == 1]
    train_no_click = train[train.click == 0]

    stuff = []
    for col in annon_col:
        stat, p = stats.ttest_ind(train_click[col], train_no_click[col])
        output = {
            "Feature": col,
            "t-stat": stat,
            "p-value": p
        }
            
        stuff.append(output)
            
    df = pd.DataFrame(stuff)

    return df.set_index('Feature')