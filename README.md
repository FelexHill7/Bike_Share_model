# Bike Share Regression

Target: `cnt` (total bikes rented per hour). This is a supervised regression problem.

Data: UCI Bike Sharing dataset (`fetch_ucirepo(id=275)`), 17,379 hourly rows from 2011-2012. Licence CC BY 4.0, citation: Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5W894>

Course: CSSD 3131 (York University). Assignment 2 is due Oct 20, with the oral evaluation on Oct 22.

## Notebook structure

| Part | Contents |
| --- | --- |
| 1. Dataset and problem | Source, licence, citation, problem statement, cost $J(\mathbf{w})$ and evaluation measures |
| 2. Exploration | Summary statistics, daily trend, rentals against every feature, hour × working-day plot, correlations |
| 3. Preparation | Cleaning table (incl. condition number of $X^TX$ for dropping `atemp`), scaling, train/validation/test split (60/20/20) |
| 4. Initial model | Multiple linear regression baseline: normal equation and gradient descent (cost curve for 3 learning rates) |
| 5. Complexity | Polynomial degree 1-4, learning curve of the baseline, random forest tree depth |
| 6. Improving | Ridge (λ on the validation set), sinusoidal features (K on the validation set), gradient boosting with early stopping, random-search tuning |
| 7. Analysis | Learning curves, two-week plot, time-based split, results table and chart |
| 8. Conclusions | Findings, what worked and didn't, what I learned, limitations |

## Results

| Model | RMSE (random) | MAE (random) | R² (random) | RMSE (time) | MAE (time) | R² (time) |
| --- | --- | --- | --- | --- | --- | --- |
| Linear regression (baseline) | 139.3 | 104.8 | 0.387 | 183.5 | 138.6 | 0.307 |
| Polynomial degree 4 + Ridge (λ = 100) | 102.2 | 71.9 | 0.670 | 286.2 | 205.3 | −0.684 |
| Sinusoidal regression (K = 5) | 75.1 | 43.8 | 0.822 | 78.5 | 50.0 | 0.873 |
| Random forest | 42.0 | 24.6 | 0.944 | 70.1 | 45.4 | 0.899 |
| **Gradient boosting** | **38.5** | **23.3** | **0.953** | **64.8** | **42.4** | **0.914** |
| Gradient boosting (tuned) | 38.0 | 22.5 | 0.954 | 65.5 | 42.3 | 0.912 |

Gradient boosting is the best model on both splits.

## Key findings for the report and the oral

- **Baseline underfits (high bias):** train RMSE 142.4 ≈ test 139.3, and the learning curve plateaus at about 142. More data won't help.
- **Gradient descent = normal equation:** with α = 0.1 the cost reaches 10,136.1 in about 50 iterations, and the weights match the direct solution to within 0.0025. α = 0.01 hasn't converged after 500 iterations.
- **Polynomial degree shows overfitting:** degree 4 (1,364 columns) has train RMSE 99.4 but validation RMSE 425, with weights up to 3,207 (Lecture 7 / 8 picture).
- **Ridge fixes it:** validation error is U-shaped in λ (best λ = 100, validation 101.7), and the largest weight shrinks from 3,207 to 105.
- **Feature engineering beats more features:** 31 sine/cosine columns (RMSE 75.1) beat 1,364 regularized polynomial columns (RMSE 102.2).
- **Random forest:** overfits (train 15.9 vs test 42.0), but test error stays flat as depth grows because averaging trees cancels variance.
- **Gradient boosting:** early stopping halted at iteration 193 of 500; smaller train-test gap than the forest. Tuning gains only about 1%.
- **Time split:** hours aren't i.i.d., so the random split is optimistic. Tree models' RMSE rises about 67%; the sinusoidal model only about 5%. The Ridge polynomial collapses (predicts 90 bikes per hour for November against 213 actual, 702 negative predictions) because its polynomial terms extrapolate outside the training range.

## Course concepts used

| Lecture | Concept | Where |
| --- | --- | --- |
| Lec 3 (linear algebra) | Vectorization, rank / singular $X^TX$ | Part 4 ($\hat{\mathbf{y}} = X\mathbf{w}$), Part 3 (`atemp` condition number) |
| Lec 5 (calculus) | Gradient, chain rule | Part 4 gradient descent |
| Lec 6 (regression I) | ERM, i.i.d., cost $J$, gradient descent, learning rate | Parts 1, 3, 4, 7 |
| Lec 7 (regression II) | Multiple linear regression, normal equation, polynomial features $\Psi(X)$, degree M | Parts 4, 5 |
| Lec 8 (overfitting) | Under/overfitting, bias-variance tradeoff, learning curves, generalization, baseline, early stopping, feature engineering | Parts 4-7 |
| Lec 9 (regularization) | Ridge (L2), λ, hyperparameters vs parameters, train/validation/test, grid vs random search, powers of ten | Parts 3, 6 |

## Models tried and dropped

| Model | R² (random) | Why it was dropped |
| --- | --- | --- |
| Linear regression, one-hot | 0.682 | Each hour gets its own coefficient, but the effects are additive, so there's no hour × working-day interaction. Gives 321 negative predictions. |
| Periodic spline regression + interactions | 0.922 | Scored **R² −5.67 on the time split**. Its month × year interactions extrapolated wildly, predicting up to 3,421 bikes in an hour. |

## To do

1. Upload to Google Colab, Run All, download with outputs saved
2. Rename to `A2_Felex_Hill_221324454.ipynb` if Felex is the first name (brief: `A2_FirstName_LastName_StudentID`)
3. Write the PDF report (the markdown cells are the outline for each part)
4. Prepare for the oral: bias vs variance diagnosis, normal equation vs gradient descent, why λ is chosen on validation, why Ridge fails on the time split, permutation importance

## Report caveats

- **R² across splits:** the sinusoidal model's R² rises on the time split (0.822 → 0.873) while its RMSE grows (75.1 → 78.5), because the Aug-Dec test period has more variance (std 220.5 vs 178.0). Compare with RMSE and MAE.
- **Negative predictions:** baseline 158, Ridge 190, gradient boosting 50 test hours. A log target (as in the sinusoidal model) prevents this.
- **Permutation importance can exceed 1** (`hr` is about 1.4), because R² drops below 0 when a key feature is shuffled.
- **Leakage:** `casual` and `registered` are excluded because they add up to `cnt`.

## Possible extensions

- **Lag features** (previous hour's count, same hour yesterday): random forest R² 0.944 → 0.960, but this changes the task to next-hour forecasting.
- **L1 (Lasso) / Elastic Net** on the polynomial features, to see which terms matter.
- **`TimeSeriesSplit`** cross-validation for tuning.
