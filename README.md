
# DataNavigator

This project is a web-based application built using Flask that allows users to upload datasets, perform various preprocessing steps, and visualize data correlations. The application provides a user-friendly interface for data analysis, making it easier for non-technical users to interact with and explore datasets.


## Features

- **Upload Dataset**: Users can upload CSV files containing datasets.
- **Data Preprocessing**: Select columns to drop, and choose a target column for analysis.
- **Encoding Options**: Option to encode categorical columns using OrdinalEncoder or OneHotEncoder.
- **Correlation Analysis**: Generate and visualize a correlation heatmap using Seaborn.
- **Box Plot Visualization**: Generate box plots for the top correlated features.
- **Outlier Removal**: Identify and remove outliers using the Interquartile Range (IQR) method.
- **Logistic Rgression**: Apply Logistic Rgression to the processed dataset.

## Project Structure
- `app.py`: The main Flask application file that handles routing and backend logic.
- `flask_session/`: Directory for managing Flask session data.
- `static/`: Directory for storing static assets like CSS, JavaScript, and images.
- `templates/`: Directory for storing HTML templates used in the web app.
- `uploads/`: Directory where uploaded datasets are temporarily stored.
- `venv/`: Virtual environment containing project dependencies (ignored by Git).
## Installation
To run this project locally, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/gautam0406/data-navigator-v1.git
    cd data-navigator-v1
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the Application**:
    ```bash
    python app.py
    ```
    The application will be accessible at `http://127.0.0.1:5000`.
## Usage

1. Open the web application in your browser.
2. Upload a CSV dataset.
3. Explore the dataset and perform preprocessing:
   - Select columns to drop.
   - Choose a target column.
   - Apply encoding to categorical data.
4. Generate and view the correlation heatmap.
5. Visualize box plots for top correlated features.
6. Remove outliers from the dataset.
7. Apply Logistic Rgression.
## Dependencies
- Flask
- Pandas
- Plotly
- Scikit-learn

These dependencies are listed in the `requirements.txt` file.