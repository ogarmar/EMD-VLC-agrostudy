# EDM TASK: Valencia Agri-Insights Pro

**Valencia Agri-Insights Pro** is an advanced interactive web application built with Streamlit for sustainable urban agricultural management in the city of Valencia. The app leverages open data sources to provide actionable insights, predictive analytics, and recommendations for crop planning, cost analysis, and continuous monitoring.

## Project Overview

This project addresses urban challenges related to agricultural land use, water management, and crop diversification using open data from Valencia. It integrates state-of-the-art data science methodologies, including predictive modeling, AutoML, recommendation systems, and model monitoring, to support decision-making for city planners, farmers, and stakeholders.

## Features

- **Interactive Dashboard:** Visualizes trends in cultivated area, crop group distributions, and key agricultural metrics by year.
- **Predictive Models:** Includes both a default Random Forest regressor and an AutoML pipeline (FLAML) to predict cultivated area based on crop type, irrigation, and other features.
- **Recommendation Systems:** Offers hybrid recommendations for crop diversification and similarity-based suggestions, as well as risk analysis for campaign planning.
- **Cost and Utility Analysis:** Calculates water costs, simulates utility and profitability for different crops, and visualizes cost distributions.
- **Model Evaluation:** Provides ROC analysis, calibration curves, and illustrative model comparisons, including simulated model fusion.
- **Continuous Monitoring:** Tracks prediction errors over time, generates alerts for high-error cases, and allows users to download quality reports.
- **User-Friendly Interface:** Built with Streamlit for ease of use and rapid interaction.

## Methodology

The application employs a variety of data science and machine learning techniques covered in the course:

- **Predictive Modeling:** Regression models (Random Forest, AutoML/FLAML) for yield prediction.
- **Model Evaluation:** Uses MAE, R², ROC analysis, and calibration curves to assess model performance.
- **AutoML:** Integrates FLAML for automated model selection and hyperparameter tuning.
- **Recommendation Systems:** Implements both content-based and hybrid recommenders for crop selection.
- **Cost-Based Evaluation:** Analyzes water costs and simulates utility functions for economic assessment.
- **Model Fusion:** Demonstrates the benefits of combining multiple models.
- **Monitoring & Reporting:** Logs predictions, monitors errors, generates alerts, and produces downloadable reports.
- **Web App Deployment:** The solution is implemented as a web application for accessibility and real-world usability.

## Data Sources

- [Valencia Open Data Portal Superficie Cultivada]((https://valencia.opendatasoft.com/explore/dataset/superficie-cultivada-cultivos/table/))


## How to Run

1. **Install Requirements**
2. 
-pip install -r requeriments.txt

3. **Run the Application**
4. 
-streamlit run app.py

5. **Access the App**
Open your browser at [http://localhost:8501](http://localhost:8501)

## Team

Developed by: Yasmin Serena Diaconu, Óscar García Martínez, Víctor Mañez Poveda

---

**This project demonstrates the integration of predictive analytics, recommendation systems, cost-based evaluation, and model monitoring in a real-world urban data science scenario.**
