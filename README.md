
# Project: Wheat Price Forecasting System

---

## Overview

This project includes a complete workflow for predicting wheat prices using historical data, machine learning models, and a Streamlit dashboard for visualization. The application handles data ingestion, forecasting model creation, and dashboard deployment.

---

## Project Files

### 1. `setup.sh`
This Bash script automates database setup using MySQL. It performs the following tasks:
- Drops the `food_data` database if it exists.
- Creates the `food_data` database.
- Grants all privileges on the `food_data` database to the root user (`root`).

#### **How to Run:**
```bash
chmod +x setup.sh
./setup.sh
```

---

### 2. `Data_Ingestion.ipynb`
This Jupyter Notebook handles the entire data pipeline:
- **Data Source:** Fetches data from a specified URL.
- **Database Operations:** Inserts the fetched data into the MySQL database.
- **ETL Job:** Extracts, transforms, and loads data into the required format for forecasting.

---

### 3. `ForeCasting.ipynb`
This Jupyter Notebook focuses on building a predictive model:
- **Model Training:** Uses machine learning algorithms (Random Forest and Linear Regression) to train models on historical data.
- **Model Export:** Saves the trained model for use in the application.

---

### 4. `application.py`
This is the main Streamlit application that:
- Connects to the MySQL database.
- Loads the trained model created in `ForeCasting.ipynb`.
- Provides an interactive web interface where users can:
  - View historical wheat prices.
  - See future price predictions using the trained model.
  - Select prediction durations (short-term or long-term).
  - Explore interactive data visualizations.

#### **How to Run:**
```bash
streamlit run application.py
```

---

## Installation Guide

1. Clone the repository:
   ```bash
   git clone git@github.com:vinaya-thimmappa/AMPBA_Foundation_Project.git
   cd AMPBA_Foundation_Project
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the MySQL database:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

5. Run the Streamlit application:
   ```bash
   streamlit run application.py
   ```

---

## System Requirements
- Python 3.8 or higher
- MySQL Server
- Required Python Libraries:
  - `streamlit`
  - `mysql-connector-python`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

---

## Future Enhancements
- Add a CI/CD pipeline for automated deployments.
- Enable Docker-based deployment.
- Enhance forecasting with advanced models like LSTM.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contributors
- **Project Owner:** 
- **Contributors:** Open for collaboration! Feel free to submit pull requests or issues.
