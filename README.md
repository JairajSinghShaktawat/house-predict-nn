# House Price Predictions With Linear Regression

### Overview
Predicting house prices using the Ames housing dataset (http://ww2.amstat.org/publications/jse/v19n3/decock.pdf). First, I did some data preprocessing by dropping unimportant features, converting categorial variables into a numerical representation via a dictionary  mapping, and then normalizing inputs. 

Next, I used linear regression to predict prices. I ended up with similar accuracy (~76%) on test datasets using unregularized linear regression, L1-regularized regression, and L2-regularized regression. I also tried training on a polynomial feature expansion. 
