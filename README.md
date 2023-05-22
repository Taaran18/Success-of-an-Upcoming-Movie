# Success of an Upcoming Movie

Text Classification using Logistic Regression

## Description

This project focuses on classifying text data into two categories: "Disaster" and "Not Disaster" using the Logistic Regression algorithm. The project utilizes the scikit-learn library for text preprocessing, feature extraction, model training, and evaluation.

## Usage

The project performs the following steps:

1. Import the necessary libraries, including pandas, numpy, and scikit-learn modules.

2. Load the training and test datasets using `pd.read_csv()`.

3. Split the training data into features (`X`) and target (`y`).

4. Split the training data into training and validation sets using `train_test_split()`.

5. Create a pipeline for preprocessing and model training using `Pipeline()`:

   - `TfidfVectorizer()` is used for text feature extraction.
   - `LogisticRegression()` is used as the classification model.

6. Fit the pipeline on the training data using `pipeline.fit()`.

7. Display the pipeline and its transformers/estimators in a DataFrame.

8. Evaluate the model on the validation set by predicting the target using `pipeline.predict()`.

9. Calculate the classification report using `classification_report()`.

10. Calculate the confusion matrix using `confusion_matrix()` and display it in a DataFrame.

11. Calculate precision, recall, F1-score, and support.

12. Create a DataFrame to store the evaluation results.

13. Apply the pipeline on the test data by predicting the target.

14. Prepare the submission file by updating the `target` column of the sample submission DataFrame.

15. Display the head of the submission DataFrame.

## Results

The project outputs the following results:

- Pipeline: A DataFrame showing the pipeline steps and their corresponding transformers/estimators.
- Classification Report: A report showing precision, recall, F1-score, and support for each class.
- Confusion Matrix: A DataFrame representing the confusion matrix.
- Results: A DataFrame containing precision, recall, F1-score, and support for each class.
- Submission: The head of the submission DataFrame, which can be used for submission in a competition or further analysis.

## Contributing

Contributions to this project are welcome. You can contribute by following these steps:

1. Fork the repository.

2. Create a new branch:

   ```shell
   git checkout -b feature/your-feature
   ```

3. Make your changes and commit them:

   ```shell
   git commit -m "Add your message here"
   ```

4. Push to the branch:

   ```shell
   git push origin feature/your-feature
   ```

5. Open a pull request.

## License

This project is licensed

 under the [MIT License](LICENSE).

## Contact

For any questions or suggestions, please feel free to contact [Taaran Jain](mailto:taaranjain16@gmail.com).
