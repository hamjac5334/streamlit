# Who Uses LinkedIn? Predicting LinkedIn Usage from Demographics

Built an interactive Streamlit app that uses a logistic regression model to predict whether a person uses LinkedIn from six demographic traits, with live charts showing how the prediction changes with age and income.

**Live app:** [LinkedIn User Predictor]([ADD STREAMLIT LINK])

**Code:** [`app.py`]([ADD REPO LINK])

A marketing team enters a person's profile (income, education, age, gender, marital status, and parental status) and gets back a classification, the probability that person uses LinkedIn, and how that probability compares to the average survey respondent. Two "what-if" charts hold the profile fixed and vary only age or only income, which shows which audiences are most worth targeting.



## The business question

**Client:** Marketing team planning LinkedIn ad spend

- **Situation.** LinkedIn is a key channel for reaching professional audiences, but not every demographic segment uses it equally.
- **Complication.** Without a clear picture of who actually uses the platform, ad budgets get spread across segments that are unlikely to be reached.
- **Question.** Given a person's demographic profile, how likely are they to be a LinkedIn user, and which traits move that likelihood the most?



## How the app works

1. **Enter a profile** in the sidebar: household income (9 brackets), education (8 levels), age (18–98), gender, marital status, and whether they're a parent of a child under 18.
2. **Get a prediction:** a LinkedIn user / non-user classification plus the predicted probability, shown against the survey average.
3. **Explore the drivers:**
   - **Age curve.** The same profile scored at every age from 18 to 98.
   - **Income bars.** The same profile scored across all 9 income brackets.
   - **Survey context.** An expandable panel showing the respondent count, the overall LinkedIn usage rate, and the usage rate at each education level.



## Approach

1. **Data.** Survey responses on social media usage (`social_media_usage.csv`), with LinkedIn usage recorded in the `web1h` field.
2. **Cleaning.**
   - Target `sm_li` = 1 if the respondent uses LinkedIn, 0 otherwise.
   - Income kept only for valid codes 1–9 and education only for 1–8. Age is capped at 98.
   - Parent, married, and female recoded as binary flags.
   - Refused or don't-know responses set to missing, then dropped.
3. **Features.** `income`, `education`, `parent`, `married`, `female`, `age`.
4. **Modeling.** Logistic regression with `class_weight="balanced"` to correct for LinkedIn users being the minority class.
5. **Translation.** The model runs live in Streamlit. Data loading and model training are cached, so predictions update instantly as the inputs change.

**Model performance.** [ADD: accuracy, precision, recall, and F1 on a held-out test set, and the key coefficients]



## Key takeaways

- [ADD: e.g., Income and education are the strongest positive predictors of LinkedIn use]
- [ADD: e.g., Probability declines steadily with age, holding everything else constant]
- [ADD: any notable effect of gender, marital status, or parental status]



## Limitations

- **Survey data.** Usage is self-reported and reflects a single point in time.
- **Linear assumptions.** Logistic regression assumes each feature has a consistent effect on the log-odds. For example, the age effect can't bend or plateau.
- **Ordinal features treated as numeric.** Income and education brackets are modeled as evenly spaced numbers, even though the bracket widths differ.
- **Full-sample training.** The deployed model is trained on all cleaned data, so its performance should be evaluated separately on a held-out split.
- **Limited features.** Six demographics leave out occupation, industry, and location, all likely strong predictors.



## What I'd do with more time

- Add a train/test split and show model metrics and a confusion matrix in the app.
- One-hot encode income and education to capture non-linear bracket effects.
- Compare against tree-based models (random forest, gradient boosting).
- Add feature-importance and coefficient visuals for the marketing team.
- Add occupation, region, and urban/rural features.



## Tools and stack

- **Python:** pandas, numpy, scikit-learn
- **Modeling:** LogisticRegression (sklearn) with class balancing
- **App:** Streamlit (sidebar inputs, metrics, line and bar charts, cached data and model)
- **Deliverables:** Deployed Streamlit web app

**Run locally:**
pip install streamlit pandas numpy scikit-learn
streamlit run app.py
(Keep `social_media_usage.csv` in the same folder as `app.py`.)
