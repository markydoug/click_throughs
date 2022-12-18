# Click Through Prediction
## Will they click our ad?

## Executive Summary
Through exploration of the data, it was found that certain hours of the day and certain days of the week have a relationship with click-throughs. Also, the anonymized continuous features were seen to have different means when someone clicked and when someone didn't click so those were put into my model. I ran several different models and found that Random Forest could beat baseline. Until I can look more into other features, we can use this model to predict the click-through rate of future advertisements.

## Project Description
In online advertising, click-through rate (CTR) is a very important metric for evaluating ad performance. In this project I will try to predict if someone will click-through based on information provided from the website.

## Project Goals
* Identify key features that can be used to create an effective predictive model.
* Use classification models to make click-through predictions.
* Use findings to make recommendations and establish a foundation for future work to improve model's performance.

## Initial Thoughts
My initial hypothesis is that those who are browsing the website in the middle of the night are more likely to click the advertisement.

## The Plan
* Aqcuire the data from [here](https://github.com/interviewquery/takehomes/tree/oreilly_1/oreilly_1)

* Prepare data
    * Checked for nulls, there were none
    * Changed the ```hour``` column to datetime
    * Created ```day_of_week``` and ```hour_of_day``` based off of ```hour```

* Explore data in search of drivers of churn
    * Answer the following initial questions
        * What percentage of instances result in a click-through?
        * Does hour of the day have a relationship with clicks?
        * Does day of the week have a relationship with clicks?
        * Does banner position have a relationship with clicks?

* Develop a model to predict the value of a house
    * Use drivers identified through exploration to build different predictive models
    * Evaluate models on train and validate data
    * Select best model based on highest accuracy
    * Evaluate the best model on the test data

* Draw conclusions

## Data dictionary
| Feature | Definition | Type |
|:--------|:-----------|:-------
|**id**| Advertisement instance unique id | *int*|
| **hour** | The date and hour of day |*datetime*|
| **C1** | Anonymized categorical variable |*int*|
|**banner_pos**| Location of ad on the page | *int*|
|**hour_of_day**| Hour of the day when the ad was displayed | *str*|
|**day_of_week**| Day of the week when the ad was displayed | *str*|
|**C14-C21**|Anonymized continuous variables | *int*|
|**Target variable**
|**click**| Did they click? (1-Yes, 0-No) | *int* |


## Steps to Reproduce
1. Clone this repo
2. Run ```final_project.ipynb``` notebook.

## Conclusion
### Summary
* ```hour_of_day``` (0, 1, 8, 9, 11, 14, 15, 16, 18, 19, 20, 21) probably have a relationship with```clicks```.
* Each ```day_of_week``` probably has a relationship with ```click```.
* ```banner_pos``` (0, 1, 2, 7) probably have a relationship with ```click```.
* Proabably a difference in means between each anonymized continuous feature (```C14```-```C21```) for those who click and don't click.

### Recommendations
* We need to invest in some cloud computing in order to run statisical tests and models on such large data sets.
* Even though my best model only performs 0.2% better than baseline, we can use it for now as we continue to improve our model.

### Next Steps
* In the next iteration:
    * Use cloud computing to look more into all the different features.
    * Try hashing some of the features to see if that will improve my models.