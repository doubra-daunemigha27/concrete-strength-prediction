# Concrete Strength Prediction

Predicting concrete compressive strength (MPa) from its ingredients and curing age, built end to end on Databricks.

**Tools:** Python, pandas, PySpark, Spark SQL, Delta Lake, scikit learn, Plotly, Databricks (Unity Catalog)

## The question

Concrete is tested for strength by crushing samples in a lab, which takes days or weeks. Can a model predict strength from the mix recipe alone, and which ingredients matter most?

## Pipeline

raw Excel file → cleaned Delta table → Spark SQL exploration → machine learning model → predictions saved back as a Delta table

1. **Load and clean:** read the original `.xls` file from a Unity Catalog volume, renamed messy column names, added a permanent `sample_id`, confirmed zero missing values, and found 25 exact duplicate rows
2. **Store:** saved the clean data as a Delta table (`concrete_clean`)
3. **Explore with Spark SQL:** validated row counts, checked how many mixes use fly ash (464 of 1,030), looked at average strength by curing age, and ranked each input by its correlation with strength
4. **Model:** compared a linear regression baseline against a random forest, using an 80/20 split plus 5 fold cross validation
5. **Evaluate:** checked feature importance, plotted predicted vs actual, and analyzed where the model misses
6. **Write back:** saved test predictions and errors as a Delta table (`concrete_predictions`) and queried the riskiest misses with SQL

## Results

| Model | Avg R² (5 fold CV) | MAE (test set) |
|---|---|---|
| Linear regression (baseline) | 0.60 | 7.75 MPa |
| **Random forest** | **0.91** | **3.78 MPa** |

The random forest beat the baseline on every fold.

## Key findings

* **Age and cement drive strength,** together accounting for about two thirds of the model's decisions
* **Correlation undersold age.** In a simple correlation check, cement led (0.50) and age ranked only third (0.33). The random forest ranked age first, because age affects strength along a curve, not a straight line.
* **Strength grows fast in the first month, then levels off.** That curve is why the linear model struggles and the random forest does well. 28 days is the most common test age, matching the industry standard.
* **Errors are balanced** (100 overpredicted, 106 underpredicted), but the worst misses **overpredict by up to 17 MPa,** which is the risky direction for safety
* **The biggest misses come from near identical recipes** around 313 kg of cement, which look like repeated tests of the same mix and are worth reviewing with a materials expert

## Limitation

The data has 25 exact duplicate rows (the same recipe tested more than once). Because the split was random, a repeated mix can land in both the training and test sets, which may make scores look slightly better than they would on brand new recipes.

## Lesson learned

Spark returned rows in a different order than pandas, so I added a permanent `sample_id` to keep results reproducible across steps.

## Next steps

* Remove or group duplicate mixes before splitting, then retest
* Engineer features like the water to cement ratio
* Track experiments with MLflow
* Try gradient boosting models
* Apply the same approach to other materials lab data, such as polymers

## Data

Concrete Compressive Strength dataset by I. C. Yeh, from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) (CC BY 4.0). 1,030 mixes, 8 inputs (7 ingredients in kg per m³ plus age in days), 1 target (strength in MPa).

Original paper: Yeh, I. C. (1998). Modeling of strength of high performance concrete using artificial neural networks. *Cement and Concrete Research*, 28(12), 1797 to 1808.

## How to run

1. Download `Concrete_Data.xls` from the UCI link above (or use the copy in `/data`)
2. Upload it to a Unity Catalog volume in Databricks, or update the file path in the first code cell
3. Import `concrete_strength_project.ipynb` into Databricks and run all cells

## Author

**Doubra Daunemigha** · [LinkedIn](https://www.linkedin.com/in/doubra-daunemigha) · d.daunemigha.it@gmail.com
