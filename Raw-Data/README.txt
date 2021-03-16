We need an equal amount of data for each classification in both our training and test set
We don't need to worry about our test set at the moment, but to classify 300 reviews we will need
60 reviews for each classification (mostly negative, slightly negative, neurtral, slightly positive, mostly positive)

You will go through the order in Train-Negative-Order.txt and classify each review in the aclIMDB/train/neg folder into 3 folders (Mostly-Negative, Slightly-Negative, Neutral)
until there are 60 reviews in Labeled-Data/Train/Mostly-Negative folder, 60 reviews in Labeled-Data/Train/Slightly-Negative folder, and 30 reviews in Labeled-Data/Train/Neutral folder.

I will go through the order in Train-Positive-Order.txt and classify each review in the aclIMDB/train/pos folder into 3 folders (Mostly-Positive, Slightly-Positive, Neutral)
until there are 60 reviews in Labeled-Data/Train/Mostly-Positive folder, 60 reviews in Labeled-Data/Train/Slightly-Positive folder, and 30 reviews in Labeled-Data/Train/Neutral folder.

When we combine our data we will have 60 reviews for each classification.

Reviews will be classified by these rules:

Mostly-Negative		- Entirely criticism with no or very little praise

Slightly-Negative	- More criticism than praise

Neutral				- Equal amount of praise and criticism

Slightly-Positive	- More praise than criticism

Mostly-Positive		- Entirely praise with no or very little criticism
