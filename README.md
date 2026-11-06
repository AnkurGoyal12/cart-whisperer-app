# Cart Whisperer: predicting purchase intent from website visits

## a. Problem statement

Online stores get many visits but many of them end without a purchase. That means huge cost to retailer in terms of lost revenue and wasted marketing spend. Given what we can observe about a visit (pages viewed, time spent, bounce and exit behaviour, traffic source, month), can we predict whether a visit will end in a purchase so that retailer can maximise the revenue from online stores. In most cases, conversion ratio is very small making most of the datasets skewed. Selecting the right model and performance metrics becomes extremely important to take decisions for desired results. A reliable prediction of a real buyer can enable retailer to offer discounts for a customer who is likely to buy and not waste an offer on one that won’t. A model that predicts revenue as a binary classifier with multiple visit characteristics (features) is needed.

## b. Dataset description

Multivariate dataset "Online Shoppers Purchasing Intention Dataset" is taken from https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset.

The dataset consists of feature vectors belonging to 12,330 sessions.
The 17 features are a mix of numeric, text and true/false with no missing values.

| Variable | Group | Data type | Preparation in my pipeline |
|---|---|---|---|
| `Administrative` | Page visits | integer | scaled |
| `Administrative_Duration` | Page visits | decimal | scaled |
| `Informational` | Page visits | integer | scaled |
| `Informational_Duration` | Page visits | decimal | scaled |
| `ProductRelated` | Page visits | integer | scaled |
| `ProductRelated_Duration` | Page visits | decimal | scaled |
| `BounceRates` | Visit behaviour | decimal | scaled |
| `ExitRates` | Visit behaviour | decimal | scaled |
| `PageValues` | Visit behaviour | decimal | scaled |
| `SpecialDay` | Visit context | decimal | scaled |
| `Month` | Visit context | text | one-hot encoded |
| `VisitorType` | Visit context | text | one-hot encoded |
| `Weekend` | Visit context | true/false | one-hot encoded |
| `OperatingSystems` | Technical | integer code | one-hot encoded |
| `Browser` | Technical | integer code | one-hot encoded |
| `Region` | Technical | integer code | one-hot encoded |
| `TrafficType` | Technical | integer code | one-hot encoded |
| `Revenue` | **Target** | true/false (converted to 1/0) | not a feature |

As can be seen from below table, only 15.5% visits translate into purchase making the dataset skewed. Metrics like accuracy can be misleading in this case.

| `Revenue` value | Meaning | Visits | Share |
|---|---|---|---|
| False (0) | visit ended without a purchase | 10,422 | 84.5% |
| True (1) | visit ended in a purchase | 1,908 | 15.5% |
| **Total** | | **12,330** | **100%** |

Stratified sampling is needed to carve out test data so that buyer vs non-buyer proportion remains intact in training and test data and model trained on training data gives fair measure of model performance.


| Set | Visits | Buyers | Non-buyers | Buyer share |
|---|---|---|---|---|
| Training (80%) | 9,864 | 1,526 | 8,338 | 15.47% |
| Test (20%) | 2,466 | 382 | 2,084 | 15.49% |

### Data Preprocessing 

1. One hot encoding - Features like OperatingSystems, Browser, Region, TrafficType are one-hot encoded as they are numeric values but are not ordinal. These are categorical features and one-hot encoding gives each code its own yes/no column, so the models treat them as categories. Month, VisitorType and Weekend are text and true/false columns converted using one-hot encoding

2. Data scaling - BounceRates runs from 0 to 0.2, while ProductRelated_Duration runs up to about 64,000. Models that measure distances or use gradients, such as Logistic Regression, KNN and SVM, would let the large-number columns dominate. That is why the numeric columns are scaled.

3. The duration columns and PageValues have a mean far below their maximum (for example, a mean of about 1,195 against a maximum of about 63,974), so a few visits are very unusual.

4. Data leakage - the rules are learned from training data only, then applied to the test data, which keeps the test scores real.

## c. GitHub repository

Repository: https://github.com/AnkurGoyal12/cart-whisperer-app

Live app: https://cart-whisperer.streamlit.app

## d. Models used

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8840 | 0.8866 | 0.7353 | 0.3927 | 0.5119 | 0.4817 |
| Decision Tree | 0.8658 | 0.7271 | 0.5726 | 0.5262 | 0.5484 | 0.4703 |
| KNN | 0.8767 | 0.8026 | 0.6696 | 0.4031 | 0.5033 | 0.4562 |
| Naive Bayes | 0.2786 | 0.7692 | 0.1753 | 0.9869 | 0.2977 | 0.1471 |
| Support Vector Machine | 0.8954 | 0.8739 | 0.7366 | 0.5052 | 0.5994 | 0.5543 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best AUC (0.8866) shows the model ranks well, but the low recall (0.3927) shows it missed most buyers, because it only says “buyer” above a 0.5 probability, which hurts recall on skewed data like this. With few buyers in the data, Logistic Regression’s probabilities for real buyers often stay below the 0.5 cut-off, so it misses many of them. Weighting the rare class more or lowering the cut-off is a possible way to improve recall. Precision of  0.7353 means that out of 204 visits the model flagged as buyers, 150 really bought, so 26.5% offers would be wasted. High precision with low recall describes a cautious model that predicts “buyer” when it’s quite sure. It prefers to stay quiet rather than risk a wrong call. This saves retailer from spending budget on non buyers but it comes at cost of missing a lot of real buyers |
| Decision Tree | This model strikes best balance with precision and recall which makes it most even model in the table however both values are moderate. The tree finds 201 of the 382 buyers, but 150 of its flags are false alarms, which is more than any model except Naive Bayes. Decision tree gives a decent F1 (0.5484, second only to SVM), but it doesn’t beat the others on AUC or accuracy. The Decision Tree has the lowest AUC in the table (0.7271) and the lowest accuracy of the four usable models (0.8658). That says it's ranking of visits is the least reliable. Tree is unrestricted, so its probabilities are coarse and its ranking is weak. And as an improvement idea, limiting its depth or pruning could help.|
| KNN | KNN looks at its 5 most similar visits in the training data and takes a majority vote. For a buyer to be labelled “buyer”, at least 3 of those 5 neighbours have to be buyers. With buyers only about 15.5% of the data, even a visit that really is a buyer usually has mostly non-buyers around it. The vote goes to “no buyer”, so many real buyers are missed. That is why recall is low (0.4031), even though precision is decent (0.6696): when 3 or more neighbours are buyers, the call is usually right. There’s a second possible factor. After encoding, each visit has 75 columns. In so many dimensions, “nearest” becomes less meaningful, because distances between visits start to look alike. |
| Naive Bayes | Accuracy (0.2786) and precision (0.1753) are indeed the worst in the table. Its recall is the highest in the table (0.9869). It finds 377 of the 382 buyers. So it’s not a model that fails at everything. It fails in a specific way. It also flags 1,774 non-buyers as buyers, which is why precision collapses. That would mean retailer spending lot of discount offers on non buyers. AUC is mid-table and clearly better than a coin flip (0.5). Naive Bayes’s ranking of visits is still somewhat useful, and the real damage is done by the 0.5 cut-off, where its overconfident probabilities push almost every visit above the line. I also tried The Multinomial experiment and it swung the other way (recall 0.1099, accuracy 0.8398), so it didn’t help.|
| Support Vector Machine | SVM leads on accuracy, F1 and MCC. The soft margin lets it accept some mistakes in the overlap between buyers and non-buyers instead of being forced to separate them perfectly. The RBF kernel let's it draw a curved, non-linear boundary, while Logistic Regression is limited to a straight one. If buyers and non-buyers mix in a more complex pattern, a flexible boundary can fit it better. Scaling helps too, since SVMs are sensitive to scale and I standardized the numeric columns. |
| Overall winner for this dataset | SVM has the best accuracy (0.8954), F1 (0.5994) and MCC (0.5543), and the best precision by a hair (0.7366). F1 and MCC are fairer than accuracy on skewed data. RBF kernel let's it draw a non linear decision boundary helping in better classification. If catching buyers mattered most, the Decision Tree catches slightly more buyers but raises far more false alarms. |


