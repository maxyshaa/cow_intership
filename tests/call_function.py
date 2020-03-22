#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.datasets import load_boston


dataset = load_boston()
X = dataset.data
y = dataset.target[:,np.newaxis]

linear_regression(X, y)

