# Model Describer

## Simple code to make 'black box' machine learning models interpretable to humans. 

Model-describer makes it possible for everyday humans to understand 'black-box' machine learning models in two key ways:

- model-describer helps us understand how the model 'believes' different groups behave within the model

- model-describer helps makes it clear where the model is making particularly accurate or inaccurate predictions

### Steps to follow

- Download the source code and import it as follows.

- It can directly run on jupyter notebooks.

- You can interpret for both regression and classification models.

## For Regression model

```python
from error_viz import ErrorVizRegression as EVR

evr = EVR(df,modelobj,target = ydepend, model_df = model_df)
```

### for continuous variables

```python
evr.create_combined_cont_plot(
    feature = 'km_driven', # your custom variable
    fill_nulls = False,
    groupbyvar = 'fuel', #Leaving this variable as None results in a simple plot without any grouping
)
```








