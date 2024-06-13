 # Chronological Attribution of Ancient Texts Using Deep Neural Networks

 ## Project Overview

 This project aims to predict the exact dating of ancient inscriptions using text data through the design, implementation, and evaluation of deep neural network models. The dataset utilized comprises inscriptions from the I.PHI, the largest available dataset of ancient Greek inscriptions.

 ## Installation

 To set up the project, follow these steps:

 1. Clone the repository:
     ```sh
     git clone https://github.com/johnvelgakis/ChronologicalAttribution.git
     ```
 2. Navigate to the project directory:
     ```sh
     cd ChronologicalAttribution
     ```
 3. Install the required dependencies:
     ```sh
     pip install -r requirements.txt
     ```

 ## Usage

 To run the project, you can use the provided Jupyter notebooks and scripts. Below are some example commands:

 1. **Data Exploration**:
     ```sh
     python data_explore.py
     ```

 2. **Training the Model**:
     Open `ChronologicalAttribution.ipynb` and follow the instructions to preprocess the data, train the model, and evaluate the results.

 ## Project Structure

 - `data_explore.py`: Script for data exploration and preprocessing.
 - `utils.py`: Utility functions used across the project.
 - `ChronologicalAttribution.ipynb`: Jupyter notebook for training and evaluating the model.
 - `topologies_RMSE.csv`: CSV file containing RMSE values for different model topologies.
 - `README.md`: Project documentation (this file).

 ## Data Preprocessing

# Data preprocessing is a crucial step in preparing the dataset for training machine learning models. For this project, the following steps were performed:

 1. **Data Exploration and Analysis**:
     - The dataset consists of 2802 observations (inscriptions) with features such as `id`, `text`, `metadata`, `region_main_id`, `region_main`, `region_sub_id`, `region_sub`, `date_str`, `date_min`, `date_max`, `date_circa`.
     - The inscriptions span from -720 to 1453.
     - The dataset contains 24679 unique words (tokens). The average number of words per inscription is calculated, and frequency distributions for unigrams and bigrams are analyzed.

 2. **Data Cleaning**:
     - Ensured that there are no missing values or duplicate entries.

 3. **Text Vectorization**:
     - **Tokenization**: Split the text of each inscription into individual words or tokens.
     - **TF-IDF Vectorization**: The frequency of each token in an inscription is calculated and then scaled by the inverse document frequency.

 4. **Normalization**:
     - The text vectors are normalized to have values between 0 and 1 using the L2 norm.
     - The date labels are normalized to lie within the range of -1 to 1 using the `MaxAbsScaler`.

 5. **Data Splitting**:
     - The dataset is split into training, validation, and test sets (70:30 ratio). The training set is further divided into training and validation sets (80:20 ratio).

 6. **Cross-Validation**:
     - A 5-fold cross-validation is used to validate the model's performance and avoid overfitting.

 7. **Feature Selection**:
     - The TF-IDF vectors of the text are used as input features, and the date ranges (`date_min`, `date_max`) are used as output labels.

 ## Model Training and Evaluation

 The model training and evaluation process includes:

 1. **Model Selection**:
     - Various neural network topologies were experimented with, including different numbers of hidden layers and nodes.

 2. **Training**:
     - The models were trained using the backpropagation algorithm.

 3. **Evaluation**:
     - Models were evaluated based on the Root Mean Square Error (RMSE) metric.
     - A custom RMSE function, `rmse_custom`, was defined to handle the specific requirements of predicting date ranges:

       If \([\hat{y}_{\text{min}}, \hat{y}_{\text{max}}] \subseteq [y_{\text{min}}, y_{\text{max}}]\), then
$$
\text{rmse\_custom} = 0
$$

Otherwise,
$$
\text{rmse\_custom} = \sqrt{\frac{1}{2} (y_{\text{min}} - \hat{y}_{\text{min}})^2}
$$

     

 4. **Hyperparameter Tuning**:
     - Various hyperparameters, such as the number of hidden layers, number of neurons, learning rate, and batch size, were tuned to optimize the model's performance.

 5. **Early Stopping**:
     - Early stopping was used to prevent overfitting during training. The training process was halted if the validation RMSE did not improve after a certain number of epochs (patience).

 6. **Results Comparison**:
     - The performance of different model architectures was compared using RMSE.
     - The best-performing model architecture was selected based on the lowest validation RMSE.

 ## Contributing

 If you wish to contribute to this project, please follow these steps:

 1. Fork the repository.
 2. Create a new branch (`git checkout -b feature-branch`).
 3. Commit your changes (`git commit -am 'Add new feature'`).
 4. Push to the branch (`git push origin feature-branch`).
 5. Create a new Pull Request.

 ## License

 This project is licensed under the MIT License. See the `LICENSE` file for more details.
