# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Person or organization developing model: Carlos Miranda, Udacity Student
- Model date: 2022/02/27
- Model version: 1
- Model type: scikit-learn RandomForestClassifier
- Information about training algorithms, parameters, fairness constraints or other applied approaches, and features: Basic K-Fold cross-validation with default parameters. The dataset contains personal information such as workclass, education, marital status, race, sex, which can be freely exploited by the model.
- License: MIT
- Where to send questions or comments about the model: GitHub issues

## Intended Use

Educational purposes.

## Training Data

Training data can be found in data/census_clean.csv.

## Evaluation Data

Testing data can be found in data/census_clean.csv.

## Metrics
Metrics include precision, recall and fbeta. The scores given here are averaged across 5 folds:

| Fold  	| fbeta               	| precision         	| recall              	|
|-------	|---------------------	|-------------------	|---------------------	|
| train 	| 0.99995 (± 4e-05)   	| 0.99994 (± 8e-05) 	| 0.99997 (± 6e-05)   	|
| test  	| 0.67346 (± 0.01006) 	| 0.7325 (± 0.0166) 	| 0.62347 (± 0.01177) 	|

### Slice metrics: Race

| Value              	| Precision 	| Recall  	| FBeta   	|
|--------------------	|-----------	|---------	|---------	|
| White              	| 0.95447   	| 0.92497 	| 0.93949 	|
| Black              	| 0.95503   	| 0.93282 	| 0.94379 	|
| Asian-Pac-Islander 	| 0.95038   	| 0.90217 	| 0.92565 	|
| Amer-Indian-Eskimo 	| 0.91429   	| 0.88889 	| 0.90141 	|
| Other              	| 1.00000   	| 0.88000 	| 0.93617 	|

### Slice metrics: Sex

| Value  	| Precision 	| Recall  	| FBeta   	|
|--------	|-----------	|---------	|---------	|
| Male   	| 0.95474   	| 0.92765 	| 0.94100 	|
| Female 	| 0.95183   	| 0.90500 	| 0.92783 	|

### Slice metrics: Education

| Value        	| Precision 	| Recall  	| FBeta   	|
|--------------	|-----------	|---------	|---------	|
| Bachelors    	| 0.95266   	| 0.95137 	| 0.95202 	|
| HS-grad      	| 0.95405   	| 0.88000 	| 0.91553 	|
| 11th         	| 1.00000   	| 0.90000 	| 0.94737 	|
| Masters      	| 0.97167   	| 0.96559 	| 0.96862 	|
| 9th          	| 1.00000   	| 0.77778 	| 0.87500 	|
| Some-college 	| 0.94364   	| 0.89329 	| 0.91778 	|
| Assoc-acdm   	| 0.92366   	| 0.91321 	| 0.91841 	|
| Assoc-voc    	| 0.94220   	| 0.90305 	| 0.92221 	|
| 7th-8th      	| 1.00000   	| 0.87500 	| 0.93333 	|
| Doctorate    	| 0.96393   	| 0.96078 	| 0.96236 	|
| Prof-school  	| 0.96984   	| 0.98818 	| 0.97892 	|
| 5th-6th      	| 1.00000   	| 0.75000 	| 0.85714 	|
| 10th         	| 0.91045   	| 0.98387 	| 0.94574 	|
| 1st-4th      	| 1.00000   	| 0.83333 	| 0.90909 	|
| Preschool    	| 1.00000   	| 1.00000 	| 1.00000 	|
| 12th         	| 1.00000   	| 0.81818 	| 0.90000 	|

### Slice metrics: Native country

| Value                      	| Precision 	| Recall  	| FBeta   	|
|----------------------------	|-----------	|---------	|---------	|
| United-States              	| 0.95409   	| 0.92456 	| 0.93909 	|
| Cuba                       	| 0.96000   	| 0.96000 	| 0.96000 	|
| Jamaica                    	| 0.90909   	| 1.00000 	| 0.95238 	|
| India                      	| 0.94444   	| 0.85000 	| 0.89474 	|
| ?                          	| 0.97857   	| 0.93836 	| 0.95804 	|
| Mexico                     	| 1.00000   	| 0.84848 	| 0.91803 	|
| South                      	| 0.88889   	| 1.00000 	| 0.94118 	|
| Puerto-Rico                	| 1.00000   	| 1.00000 	| 1.00000 	|
| Honduras                   	| 1.00000   	| 1.00000 	| 1.00000 	|
| England                    	| 0.93548   	| 0.96667 	| 0.95082 	|
| Canada                     	| 1.00000   	| 0.97436 	| 0.98701 	|
| Germany                    	| 0.93182   	| 0.93182 	| 0.93182 	|
| Iran                       	| 0.94444   	| 0.94444 	| 0.94444 	|
| Philippines                	| 0.96364   	| 0.86885 	| 0.91379 	|
| Italy                      	| 0.95833   	| 0.92000 	| 0.93878 	|
| Poland                     	| 1.00000   	| 1.00000 	| 1.00000 	|
| Columbia                   	| 0.66667   	| 1.00000 	| 0.80000 	|
| Cambodia                   	| 1.00000   	| 0.57143 	| 0.72727 	|
| Thailand                   	| 1.00000   	| 1.00000 	| 1.00000 	|
| Ecuador                    	| 1.00000   	| 0.50000 	| 0.66667 	|
| Laos                       	| 0.50000   	| 0.50000 	| 0.50000 	|
| Taiwan                     	| 1.00000   	| 1.00000 	| 1.00000 	|
| Haiti                      	| 1.00000   	| 1.00000 	| 1.00000 	|
| Portugal                   	| 1.00000   	| 1.00000 	| 1.00000 	|
| Dominican-Republic         	| 1.00000   	| 1.00000 	| 1.00000 	|
| El-Salvador                	| 1.00000   	| 0.88889 	| 0.94118 	|
| France                     	| 0.91667   	| 0.91667 	| 0.91667 	|
| Guatemala                  	| 1.00000   	| 0.33333 	| 0.50000 	|
| China                      	| 0.95000   	| 0.95000 	| 0.95000 	|
| Japan                      	| 0.95833   	| 0.95833 	| 0.95833 	|
| Yugoslavia                 	| 0.85714   	| 1.00000 	| 0.92308 	|
| Peru                       	| 0.66667   	| 1.00000 	| 0.80000 	|
| Outlying-US(Guam-USVI-etc) 	| 1.00000   	| 1.00000 	| 1.00000 	|
| Scotland                   	| 1.00000   	| 1.00000 	| 1.00000 	|
| Trinadad&Tobago            	| 0.50000   	| 0.50000 	| 0.50000 	|
| Greece                     	| 1.00000   	| 0.87500 	| 0.93333 	|
| Nicaragua                  	| 1.00000   	| 1.00000 	| 1.00000 	|
| Vietnam                    	| 1.00000   	| 0.60000 	| 0.75000 	|
| Hong                       	| 0.85714   	| 1.00000 	| 0.92308 	|
| Ireland                    	| 1.00000   	| 1.00000 	| 1.00000 	|
| Hungary                    	| 0.75000   	| 1.00000 	| 0.85714 	|
| Holand-Netherlands         	| 1.00000   	| 1.00000 	| 1.00000 	|

## Ethical Considerations

This model should not be used in any serious scenario.

## Caveats and Recommendations
One can see the model seems to overfit, thus it would be necessary to enhance the training procedure in order for the model to generalize better.
