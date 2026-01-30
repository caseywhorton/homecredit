# Project Log

## Date: 2025-10-20 (60 minutes)

Created code snippets in Jupyter notebooks for creating bucketed categorical features.  
Created transformation pipeline logic.  
Saved preprocessing pipeline to `artifacts` directory.  
Added `artifacts` directory to dvc tracking.

## Date: 2025-10-21 (60 minutes)

Trialed a few ML algorithms on dataset.  
Features have Nulls/NaN.  
Used a random forest classifier for first iteration, massively overfit to the data.  
Will look at the NaN in the dataset again and clean it further.

## Date: 2025-10-27 (30 minutes)

Looked at decision trees - overfitting or underfitting.  
Used a random forest on 3 input features, underfitting.  
Added a vibe-coded SHAP value section, still evaluating.

## Date: 2025-11-04 (30 minutes)

Reperformed SHAP value section  
Saved new preprocessing artifact  
Saved model artifact  

## Date: 2025-11-05 (45 minutes)

Pushed dvc tracked files to dagshub
Read about the tracking  
Added a preprocessing.py file under a utils directory

## Date: 2025-11-06 (30 minutes)

Adding more functions to the utils directory.  
Researched next move to ML flow after running a flow.  

## Date: 2025-11-10  (120 minutes)

Added a training module.  
Added a parameters YAML file to parameterize training.  
Retrained model with same parameters and pushed changes to DVC.  
Converted flow to MLFlow.

## Date 2025-12-03 (45 minutes)

Added extra categorical features for employment and education type.  
Added images showing separation for different education and employment types.  
Did not show model improvement.  
Looked at the credit card history again, planning on tackling utilization next time.  

## Date 2026-01-05 (30 minutes)

Added the max utilization feature.
Measures the maximum credit utilization seen in customer's history.
Added AUC and ROC curve to ML Flow.

## Date 2026-01-21 (30 minutes)

Added days past due (DPD) and month on book (MOB) features.  
Ran a few experiments, no dramatic change but a new lead model.  
Able to order the experiment runs by the test AUC which shows best models.  

## Date 2026-01-22 (30 minutes)

Add a few features for the max installment date and what I believe to be a minimum payment.  
Starting to see some overfitting, so looking to keep adding features.  

## Date 2026-01-26 (30 minutes)

Updated the readme, preprocessing and trian scripts for formatting.  
Create a to-do-list.  

## Date 2026-01-28

Added cross validation portion to model training.  
Log the cross validation mean results to ML flow.  