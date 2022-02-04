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
    feature = 'km_driven', #your custom variable
    fill_nulls = False,
    groupbyvar = 'fuel', #Leaving this variable as None results in a simple plot without any grouping
)
```

![fig1](https://user-images.githubusercontent.com/80465899/152479353-f3eca4ee-7903-45d4-b83a-fff5e803aa3b.png)

### for categorical variables

```python
evr.create_combined_cat_plot(
    feature = 'seller_type', #your custom variable
    groupbyvar = 'owner' #Leaving this variable as None results in a simple plot without any grouping
) 
```

![fig2](https://user-images.githubusercontent.com/80465899/152479531-2b47bf06-088b-4a92-a309-84386d9f1e39.png)

## For Classification model

```python
from error_viz import ErrorVizClassification as EVC

evc = EVC(df,modelobj,target = ydepend, model_df = model_df)
```

### for continuous variables

```python
evc.create_combined_cont_plot(
    feature = 'km_driven', #your custom variable
    fill_nulls = True,
    groupbyvar = 'fuel', #Leaving this variable as None results in a simple plot without any grouping
)
```

![fig3](https://user-images.githubusercontent.com/80465899/152480632-d33879d0-f0a3-4de3-aa05-bec235f8eeda.png)

### for categorical variables

```python
evr.create_combined_cat_plot(
    feature = 'seller_type', #your custom variable
    groupbyvar = 'owner' #Leaving this variable as None results in a simple plot without any grouping
) 
```

![fig4](https://user-images.githubusercontent.com/80465899/152480867-a9407a70-0d2f-4d24-9d6e-63f1b3b91d55.png)

## Requirements

- numpy 
- pandas 
- plotly
- sklearn

