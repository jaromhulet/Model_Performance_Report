# Model_Performance_Report
The process of understanding a model's predictive performance is much more involved than looking at the total RMSE for a test dataset.  The information loss required to get to a single number is extensive.  Single number summaries are a good place to start, but are very far from giving a good picture of how the model is performing.  

This code is designed to help model developers quickly and systematically get a better understanding of how their models perform.  Visualizations and tables are primarily used.  The vision is to create a collection of methods that the user can call to quickly generate a series of charts and tables.

The package performs two primary functions: (1) create performance tables/visualizations by user defined cuts and (2) create residual plots by user definted cuts.  This package allows the user to cut the data as few or as many times as desired while at the same time observing performs by cutting by any variable.
