import pandas as pd
import numpy as np

from gaussian_nb import GaussianNaiveBayes

df = pd.DataFrame()
df['x1'] = np.random.randint(1,200,100)
df['x2'] = np.random.randint(1,5,100)
df['x3'] = np.random.randint(1,200,100)

classes = np.random.randint(1,4,100)

nb = GaussianNaiveBayes(df, classes, categoricals=[False, True, False], debug_mode=True)

nb.train()

test = pd.DataFrame()
test['x1'] = [150]
test['x2'] = [3]
test['x3'] = [88]

predictions = nb.predict(test)
print(predictions)
#print(len(classes))
#print(nb._class_probabilities)