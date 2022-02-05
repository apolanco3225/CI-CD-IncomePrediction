# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This is a logistic regression classifier that uses default configuration.

## Intended Use

Predict weather a person earns more than 50 K USD a year or not using features taken from a census.

## Training Data

Source of data https://archive.ics.uci.edu/ml/datasets/census+income ; 80% of the data is used for training using strtified KFold.

## Evaluation Data

Source of data https://archive.ics.uci.edu/ml/datasets/census+income ; 20% of the data is used to evaluate the model.

## Metrics

Binary accuracy was used to measure the performance.

## Ethical Considerations

Metrics were applied on different slices of data looking for evidence of discrimination of some groups.


## Caveats and Recommendations

Misleading behaveour based on race and gender.




