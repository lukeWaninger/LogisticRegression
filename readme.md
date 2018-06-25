## Logistic Regression

The file logistic_regression.py is my implementation of a logistic regression classifier with L2 regularization. The
optimization problem is solved using the fast-gradient descent algorithm.
The classifier uses the inverse of the Lipshitz constant to calculate the initial learning rate and uses backtracking
line search to update the learning rate through successive iterations.

Jupyter notebook demos for a simulated dataset and the well known Digits dataset are included as well as a performance
comparison to scikit-learn.

#### notes
* Training the Digits dataset using the HIGH image resolution setting will require 4.5gb of ram for computation.
* The Digits demo by default uses all CPU cores available. To lower this setting, change the cpu_count variable in the
first cell of the demo.

