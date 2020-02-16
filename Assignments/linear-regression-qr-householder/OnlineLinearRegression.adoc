= Online Linear Regression

Implement online linear regression using only python standard library and numpy.
From numpy only elementary operations are allowed (addition, multiplication, indexing,...), definitely not something from `numpy.linalg`.
Preferred solution is to use QR decomposition, however, any other algorithm with similar or better complexity is accepted and appreciated.

Your implementation should have the following interface:

[source,python]
```
class LinearRegression:

    def fit(self, X: Tuple[float], y: float) -> LinearRegression:
        ...
    def predict(self, X: Tuple[float]) -> float:
        ...
```
