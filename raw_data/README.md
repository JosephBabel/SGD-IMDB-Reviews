# Instructions for Classifying Train and Test Data
We need an equal amount of data for each classification in both our training and test set.
To classify 300 reviews for train and 300 reviews for test there will be
60 reviews for each classification (mostly negative, slightly negative, neurtral, slightly positive, mostly positive).

### Task Distribution
#### Cameron:
Will go through the order in train_negative_order.txt and classify each review in the aclIMDB/train/neg folder into 3 folders (0_mostly_negative, 1_slightly_negative, 2_neutral)
until there are 60 reviews in labeled_data/train/0_mostly-negative folder, 60 reviews in labeled-data/train/1_slightly_negative folder, and 30 reviews in labeled_data/train/2_neutral folder.

Will also go through the order in test_negative_order.txt and classify each review in the aclIMDB/test/neg folder into 3 folders (0_mostly_negative, 1_slightly_negative, 2_neutral)
until there are 60 reviews in labeled_data/test/0_mostly_negative folder, 60 reviews in labeled_data/test/1_slightly_negative folder, and 30 reviews in labeled_data/test/2_neutral folder.

#### Joseph:
Will go through the order in train_positive_order.txt and classify each review in the aclIMDB/train/pos folder into 3 folders (4_mostly_positive, 3_slightly_positive, 2_neutral)
until there are 60 reviews in labeled_data/train/4_mostly_positive folder, 60 reviews in labeled_data/train/3_slightly_positive folder, and 30 reviews in labeled_data/train/2_neutral folder.

Will also go through the order in test_positive_order.txt and classify each review in the aclIMDB/test/pos folder into 3 folders (4_mostly_positive, 3_slightly_positive, 2_neutral)
until there are 60 reviews in labeled_data/test/4_mostly_positive folder, 60 reviews in labeled_data/test/3_slightly_positive folder, and 30 reviews in labeled_data/test/2_neutral folder.

### Classification Rules
When we combine our data we will have 60 reviews for each classification for both train and test data and 600 labled reviews in total.

Reviews will be classified by these rules:

**Mostly-Negative**		- Entirely criticism with no or very little praise

**Slightly-Negative**	- More criticism than praise

**Neutral**				- Equal amount of praise and criticism

**Slightly-Positive**	- More praise than criticism

**Mostly-Positive**		- Entirely praise with no or very little criticism
