# Bayesian Committee Machine for Distributed Gaussian Process Regression


This project combines the **Bayesian Committee Machine (BCM)** model with **Gaussian Process Regression (GPR)** to provide an efficient and scalable regression method on large datasets. BCM divides large datasets into smaller subsets and trains a separate GPR model for each subset. This approach reduces the computational challenges of processing large datasets with a single GPR model and at the same time combines the uncertainties of the predictions, resulting in more reliable results.

#### Technical Specifications:
1. **Bayesian Committee Machine (BCM) Approach**:
   - BCM divides the dataset into `n_subsets`. A Gaussian Process Regressor (GPR) is trained on each subset.
   - After training, the predictions of each subset are aggregated and each prediction is weighted according to the model's uncertainties (variance). In this way, the predictions of the models with higher reliability receive more weight.
   - The combined forecast is calculated as the weighted average of the outputs of all subset models based on their variance. The combined variance is also determined by the sum of the inverses of all variances.

2. **Gaussian Process Regressor (GPR)**:
   - Gaussian Process Regressor (GPR) is a powerful method for performing regression on data using a Bayesian approach. GPR estimates the possible values of a function along with an estimate of uncertainty.
   - The GPR model is structured with the following parameters:
     - `kernel`: User-definable kernel function. The default kernel in the project is set to a combination of **RBF (Radial Basis Function)** and **Constant Kernel**.
     - `alpha`: A small value added to the kernel matrix diagonal. This is used to improve numerical stability and to model noise.
     - `n_restarts_optimizer`: The number of restarts to perform optimization of the model. Multiple optimizations are performed to increase the convergence of the kernel hyperparameters to the global optimum.
     - `normalize_y`: Determines whether to normalize the target values.
     - `random_state`: The seed value used for random number generation, providing reproducible results.

3. **Kernel Functions**:
   The Gaussian Process used with the BCM model is based on kernel functions. The kernel is the function that measures the similarities between the data and determines how the Gaussian Process estimates a function. The project supports the following kernel functions:
   - **RBF Kernel (Radial Basis Function)**: One of the most widely used kernel functions that works based on the distance between data.
   - **Matern Kernel**: A kernel that works similarly to RBF, but offers a more flexible structure and can model the roughness of the data.
   - **Rational Quadratic Kernel**: A generalization of the RBF kernel, used to model variations at various scales.
   - **Dot Product Kernel**: Used to model linear relationships.

4. **Model Training and Forecasting**:
   - **fit(X, y)**: Given the input data (`X`) and target values (`y`) to the model, the data is divided into subsets and the Gaussian Process Regressor is trained on each subset.
   - **predict(X)**: Predictions are made on the input data. Each subset model generates its own prediction and uncertainty. These predictions are then combined by weighting them according to the uncertainties (variance).

### Code Structure:

- `split_data(X, y)`: Splits the data into `n_subsets` number of subsets. The data is randomly sorted and assigned to subsets.
- `train_models()`: Trains the Gaussian Process model for each subset.
- `predict(X)`: Takes the predictions of the trained models and combines the predictions with a weighting method based on variances. It also calculates the uncertainties of the predictions.
- `fit(X, y)`: Splits the data and trains Gaussian Process models. The model can be flexibly configured with user-provided hyperparameters, kernel and model configuration options.

### DataSet
The following code generates a complex zigzag pattern with increased noise and variability. This synthetic dataset is used to test the performance of the BCM models.
```python
import numpy as np
import pandas as pd

# Creating a more complex zigzag pattern while maintaining less density
# Generate a larger sample size with an extended range and complex pattern
increased_complexity_sample_size = 300

# Generate the base x values for the larger dataset with extended range
x_increased_complexity = np.linspace(0, 20, increased_complexity_sample_size)

# Create a more complex zigzag pattern for y with increased variation and less density
y_increased_complexity = (
    np.sin(x_increased_complexity * 2 * np.pi / 5) * 8 + 
    np.cos(x_increased_complexity * 3 * np.pi / 4) * 5 + 
    np.random.normal(0, 1.5, increased_complexity_sample_size)  # Increased noise for more variability
)

# Create a DataFrame with the new more complex zigzag pattern
increased_complexity_zigzag_data = pd.DataFrame({'X': x_increased_complexity, 'Y': y_increased_complexity})

# Saving the increased complexity zigzag data to a CSV file
increased_complexity_zigzag_data.to_csv('increased_complexity_zigzag_data.csv', index=False)

```
![image](https://github.com/user-attachments/assets/937da2f9-5032-4097-b95f-8d49b66c3508)

| Model             | RMSE         | MSE           |
|-------------------|--------------|---------------|
| Linear Regression | 6.475503     | 41.932141     |
| KNN               | 1.679772     | 2.821632      |
| GBM               | 1.103123     | 1.216879      |
| BCM RBF(kernel)   | 0.000019     | 3.529085e-10  |

Similar performances on other datasets.

> **Note**: Rational Quadratic Kernel and Dot Product Kernel can also give warnings.

>**Develop**: I will continue to develop this library if requested, I will develop it for my own use, there are still things I want to do. If there is demand, it will be developed in a more optimized and cleaner way with a general library logic.

>**ChatBot**: If there is enough support for this library, I plan to open a Bayesian based chatbot library to the general public.
