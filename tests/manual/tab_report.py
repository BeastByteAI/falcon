from falcon.tabular import reporting
import numpy as np

y = np.random.randint(0, 3, size=100)
y_hat = np.random.randint(0, 3, size=100)

reporting.print_classification_report(y, y_hat)
