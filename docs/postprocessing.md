# Postprocessing

## Feature Neutralization

`FeatureNeutralizer` provides classic feature neutralization by subtracting a linear model's influence, ensuring that predictions are not overly influenced by a specific set of features.

### Why?
- **Reduce Overfitting**: By neutralizing predictions, you can potentially reduce the risk of overfitting to specific feature characteristics.
- **Control Feature Influence**: Allows you to have a granular control on how much influence a set of features can exert on the final predictions. 
- **Enhance Model Robustness**: By limiting the influence of potentially noisy or unstable features, you might improve the robustness of your model's predictions across different data periods.

### Quickstart

Make sure to pass both the features to use for penalization as a `pd.DataFrame` and the accompanying era column as a `pd.Series` to the `predict` method.
```python
import pandas as pd
from numerblox.neutralizers import FeatureNeutralizer

predictions = pd.Series([0.24, 0.87, 0.6])
feature_data = pd.DataFrame([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
era_data = pd.Series([1, 1, 2])

neutralizer = FeatureNeutralizer(pred_name="prediction", proportion=0.5, cuda=False)
neutralizer.fit()
neutralized_predictions = neutralizer.predict(X=predictions, features=feature_data, eras=era_data)
```

### Note
Neutralization can be run on the GPU by setting `cuda=True`. When setting this ensure you have CuPy installed.

```bash
pip install cupy
```


## FeaturePenalizer

`FeaturePenalizer` neutralizes predictions using TensorFlow based on provided feature exposures. It's designed to integrate seamlessly with scikit-learn.

### Why?
- **Limit Feature Exposure**: Ensures that predictions are not excessively influenced by any individual feature, which can help in achieving more stable predictions.
- **Enhanced Prediction Stability**: By penalizing high feature exposures, it might lead to more stable and consistent predictions across different eras or data splits.
- **Mitigate Model Biases**: If a model is relying too heavily on a particular feature, penalizing can help in balancing out the biases and making the model more generalizable.

### Quickstart

Make sure to pass both the features to use for penalization as a `pd.DataFrame` and the accompanying era column as a `pd.Series` to the `predict` method.
```python
from numerblox.penalizers import FeaturePenalizer

predictions = pd.Series([0.24, 0.87, 0.6])
feature_data = pd.DataFrame([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
era_data = pd.Series([1, 1, 2])

penalizer = FeaturePenalizer(max_exposure=0.1, pred_name="prediction")
penalizer.fit(X=predictions)
penalized_predictions = penalizer.predict(X=predictions, features=feature_data, eras=era_data)
```