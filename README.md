# Time-to-Default Modelling on Lending Club Loans: Cox Proportional Hazards vs Random Survival Forest
** Data Source ** : https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv?select=loan.csv

## Project Overview
Credit risk is often considered a binary classification task where the target variable is to predict who will default. However, in real cases, decisions depend on predicting when default will happen allowing business to account for customers who are yet to default by the end of the observation time (censoring). This research benchmarks an interpretable cox proportional hazard model against a Random Survival Forest model under a time based train-test split. The results show moderate discrimination (C-index 0.67 Cox vs 0.66 RSF) and strong early targeting value (top-10% lift ≈1.87× at 12 months for Cox; ≈1.98× for RSF), while calibration worsens as the horizon lengthens. These findings support survival modelling as a practical framework for early-default targeting, while highlighting the need for horizon-specific calibration for long-run PD estimation.

## Project Objectives
- Analyze borrower default risk over time
- Compare statistical and machine learning survival models
- Communicate model outputs through visual and interpretable results

## Key Contributions
- Implemented Cox Proportional Hazards model for interpretable survival analysis
- Applied Random Survival Forest to capture nonlinear risk patterns
- Compared model outputs and evaluated differences in predictive behavior
- Interpreted model results to derive meaningful insights on borrower risk

## Tech Stack
- Python (pandas, numpy, sklearn)
- Survival Analysis (Cox PH, Random Survival Forest)
- Data Visualization (Python, Power BI)

## Visualization & Interpretation
This project focuses on communicating model results through visual analysis.

- Survival curves were used to understand how default risk evolves over time  
- Hazard ratios from the Cox model were interpreted to identify key risk factors  
- Model comparison plots highlight differences between linear and nonlinear approaches  

Key interpretation:
- Survival analysis provides **time-aware risk insights**, unlike standard classification models  
- Random Survival Forest captures complex patterns that may not be visible in linear models  
- Visual outputs support better understanding of borrower risk dynamics  

## Presentation (Visualization Output)
The full analysis and visual results are presented in the slides below: [https://github.com/Julie-Montague/Survival-Analysis-Modelling/blob/main/survival_analysis_report_PPT.pdf]

The presentation includes:
- Model comparison visuals
- Survival probability analysis
- Interpretation of risk over time

## Results
- Identified key drivers of default risk over time  
- Demonstrated differences between Cox PH and Random Survival Forest  
- Showed how survival models enhance interpretability of risk  

## Reproducibility
All scripts, outputs, and analysis steps are included in this repository.
