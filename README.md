# Harnessing Neural Networks for Airbnb Home Price Predictions

## Introduction

In the short-term rental economy, Airbnb stands at the forefront, revolutionizing how people travel and experience new destinations through their online marketplace for listings. This project dives into analyzing a dataset of Airbnb listings from Washington, DC, gathered in July 2023. Our goal is to construct a series of predictive models that accurately forecast the pricing of these listings. Utilizing neural network and regression analysis, we aim to decode the factors that influence listing prices on Airbnb's platform. This endeavor not only seeks to enhance our understanding of the rental marketplace dynamics but also to equip potential hosts with the knowledge to competitively price their listings, thereby optimizing their earnings and ensuring a better match between hosts and guests in this bustling digital ecosystem.

## Problem Statement

Airbnb faces the intricate challenge of maintaining a balanced and competitive marketplace that caters effectively to both hosts and guests. The complexity of pricing listings, influenced by various factors such as location, amenities, and time of year, poses a significant obstacle for hosts aiming to optimize their income while ensuring occupancy. Mispriced listings could lead to missed opportunities for hosts and an inefficient marketplace. Through our analysis, we aim to employ neural network models to predict listing prices accurately, identify the critical determinants of price setting, and furnish hosts with actionable insights to strategically price their properties.

## Processing Our Data

To start our model building process, we began with the examination of our dataset. We refined our Airbnb data through a preprocessing stage to prime it for our analysis. First, we converted and categorized key attributes — for instance, transforming 'bathrooms' from textual to numerical data — thereby ensuring each variable numerical and categorical variable was prepared for our analytical models. We converted essential categorical variables such as 'neighborhood,' 'superhost,' and 'room_type' into dummy variables to allow our models to accurately interpret these variables. Additionally, we incorporated a temporal dimension by calculating the 'age' of listings, providing deeper insights into our analysis.

Following the initial data preprocessing, we allocated 70% of our dataset to training and the remaining 30% to testing, ensuring a balanced approach to model training and evaluation. To address missing values, we imputed numerical gaps with median values and filled categorical voids with the most common occurrences, preserving the dataset's consistency. Additionally, we normalized all numerical variables to a uniform scale, a step in our process aimed at optimizing the performance of our neural network models. In the final phase of data preparation, we streamlined our dataset by renaming columns to eliminate spaces, facilitating a smoother model training and analysis process. This step, along with thorough cleaning, feature engineering, normalization, and imputation, ensured our models received high-quality data. Such rigorous preprocessing directly enhanced the accuracy and reliability of our predictive models, allowing us to forecast Airbnb listing prices with confidence.

## Model Training, Evaluation & Selection

We designed four distinct neural network models, each tailored to parse the intricacies of the Airbnb market. The development of NN2 introduces a nuanced architecture with two hidden layers, hosting three and two neurons, and employs the rectified linear unit (ReLU) as its activation mechanism. Aligning with NN1 in fundamental configurations like stepmax and linear output adjustments, NN2 sets itself apart with ReLU activation and a specific hidden layer configuration (hidden=c(3,2)). Post-training, NN2 undergoes evaluations akin to NN1, including structure visualization and performance metrics analysis. Adjustments are made to NN2’s test data predictions for price scale comparison, with visual analysis underscoring its enhanced precision in price forecasting, a testament to its complex architecture and selected activation strategy.

The third and fourth models, NN_caret1 and NN_caret2, were developed using the 'caret' package, which streamlines model training and evaluation. NN_caret1 was calibrated with a default tuning setup, while NN_caret2 was fine-tuned using a custom grid to determine the optimal combination of neurons and decay. To assess the performance of each model, we applied 10-fold cross-validation, ensuring robust validation through multiple training and evaluation cycles.

Upon review of the RMSE metrics and analysis from the plotted prediction vs actual price graphs, we observed that both NN1 and NN_caret1 models exhibited superior performance with lower root mean square error (RMSE) and higher R-squared values, indicating a tighter fit to the actual listing prices. The comparative performance of these models can be visually inspected in Figures 3 and 4 of the Appendix, which depict the scatter plots of predicted versus actual prices for NN_caret1 and NN1, respectively. NN1, in particular, displayed a balance of complexity and performance with an RMSE of approximately 99.49 and an R-squared of 0.464. NN_caret1 followed closely, with an RMSE of around 99.04 and an R-squared of 0.469, suggesting its predictions were marginally more aligned with the actual prices.

## Regression comparison

In a prior project, we performed a linear regression analysis on the same dataset. Upon revisiting our analytical findings, the contrast between our neural network models and the regression model we previously ran is stark. The regression model, while providing a transparent and easily interpretable framework, fell short in predictive accuracy, as evidenced by a higher RMSE of 241.636. This can be visually confirmed in Figure 5 of the Appendix, which contrasts the actual prices with those predicted by the regression model.

In contrast, our neural network models, particularly NN_caret1, displayed superior performance with a significantly lower RMSE and higher R-squared value, indicating greater predictive reliability and a closer fit to the actual data. The regression model, with its linear approach, simplifies the intricate dynamics of Airbnb's pricing, failing to account for the complex, nonlinear relationships between factors like neighborhood desirability and property features. This limitation is where neural network models excel, harnessing their capacity to learn from these complex interactions and offering a more potent predictive tool. Yet, their sophistication comes with a trade-off in interpretability—the intricate workings of neural networks often remain opaque, making it challenging to pinpoint the effect of individual factors. Despite this, when it comes to the reliability of predictions, neural networks stand out for stakeholders in the competitive Airbnb market, providing insights into pricing strategies. At the same time, the clear and direct correlations presented by regression models retain their appeal for those who prioritize transparency and straightforward recommendations over predictive depth.

## Conclusion: Managerial Insights

In conclusion, our comprehensive analysis reveals that neural network models, particularly NN_caret1, are superior in predicting Airbnb listing prices, outperforming traditional linear regression models. The key factors impacting price include the number of accommodations, bathroom facilities, and the host's status as a superhost, all of which are captured with greater nuance and depth by neural networks. Despite the less transparent nature of neural networks compared to regression models, their ability to process complex, non-linear relationships between variables provides a more detailed and accurate pricing strategy for Airbnb hosts. These insights suggest that hosts can optimize their earnings by focusing on improving their status to superhost and investing in the quality and number of amenities offered, which are more heavily weighted in pricing decisions. For guests, this translates into a better understanding of what factors into the cost of their stays, ensuring a more informed decision-making process when selecting listings.