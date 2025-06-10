import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from pathlib import Path
from datetime import datetime
import time 
import random 
import logging

from flaml import AutoML 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="Valencia Agri-Insights Pro", page_icon="🌾")

@st.cache_data
def load_data(file_path):
    logger.info(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.lower()
        
        for col in ['secano_hectareas', 'regadio_hectareas', 'superficie_cultivada_hectareas']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['año'] = df['año'].astype(int)
        df['grupo_de_cultivo'] = df['grupo_de_cultivo'].str.strip()
        df['cultivo'] = df['cultivo'].str.strip()
        
        df['proporcion_regadio'] = df['regadio_hectareas'] / df['superficie_cultivada_hectareas'].replace(0, 1)
        df['proporcion_regadio'] = df['proporcion_regadio'].fillna(0)
        
        logger.info("Data preprocessing complete.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

class CropYieldModel:
    def __init__(self):
        self.model = None
        self.le_group = LabelEncoder()
        self.le_crop = LabelEncoder()
        logger.info("CropYieldModel initialized.")
        
    def train(self, df):
        logger.info("Starting default model training.")
        try:
            df_train = df.copy()
            df_train['grupo_encoded'] = self.le_group.fit_transform(df_train['grupo_de_cultivo'])
            df_train['cultivo_encoded'] = self.le_crop.fit_transform(df_train['cultivo'])
            logger.info("Label encoding complete for training data.")
            
            X = df_train[['año', 'grupo_encoded', 'cultivo_encoded', 'secano_hectareas', 'regadio_hectareas']]
            y = df_train['superficie_cultivada_hectareas']
            logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logger.info(f"Train/test split complete. X_train shape: {X_train.shape}")
            
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            logger.info("RandomForestRegressor fitted.")
            
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logger.info(f"Default model evaluation: MAE={mae:.2f}, R2={r2:.2f}")
            
            return mae, r2, y_test, y_pred
        except Exception as e:
            logger.error(f"Training error for Default Model: {e}", exc_info=True)
            st.error(f"Training error for Default Model: {e}")
            return None, None, None, None
    
    def predict(self, year, group, crop, secano, regadio):
        logger.info(f"Attempting prediction for: {group}/{crop} in {year} (S:{secano}, R:{regadio})")
        try:
            if self.model is None:
                logger.warning("Attempted prediction but main Crop Yield Model is not trained.")
                st.warning("The main Crop Yield Model is not trained yet. Please train it first.")
                return 0
            
            if not hasattr(self.le_group, 'classes_') or not hasattr(self.le_crop, 'classes_'):
                logger.error("Label encoders are not fitted before prediction.")
                st.error("Label encoders are not fitted. Please run 'Train Default Yield Model' first.")
                return 0

            if group not in self.le_group.classes_:
                logger.warning(f"Crop group '{group}' not recognized by the model.")
                st.warning(f"Crop group '{group}' not recognized by the model. Cannot predict.")
                return 0
            if crop not in self.le_crop.classes_:
                logger.warning(f"Crop '{crop}' not recognized by the model.")
                st.warning(f"Crop '{crop}' not recognized by the model. Cannot predict.")
                return 0

            group_enc = self.le_group.transform([group])[0]
            crop_enc = self.le_crop.transform([crop])[0]
            
            input_data = pd.DataFrame({
                'año': [year],
                'grupo_encoded': [group_enc],
                'cultivo_encoded': [crop_enc],
                'secano_hectareas': [secano],
                'regadio_hectareas': [regadio]
            })
            
            prediction = self.model.predict(input_data)[0]
            logger.info(f"Prediction successful: {prediction:.2f} ha")
            return prediction
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            st.error(f"Prediction error: {e}")
            return 0

class HybridRecommender:
    def __init__(self, df):
        self.df = df
        logger.info("HybridRecommender initialized. Calculating crop similarity.")
        self.crop_similarity = self._calculate_similarity()
        logger.info("Crop similarity calculation complete.")
        
    def _calculate_similarity(self):
        crop_features = self.df.groupby('cultivo').agg({
            'grupo_de_cultivo': 'first',
            'secano_hectareas': 'mean',
            'regadio_hectareas': 'mean',
            'superficie_cultivada_hectareas': 'mean'
        })
        logger.info(f"Crop features for similarity: {crop_features.shape}")
        
        features_encoded = pd.get_dummies(crop_features, columns=['grupo_de_cultivo'])
        
        features_normalized = features_encoded.copy()
        for col in features_encoded.columns:
            col_min = float(features_encoded[col].min())
            col_max = float(features_encoded[col].max())

            if (col_max - col_min) > 1e-9:
                features_normalized[col] = (features_encoded[col] - col_min) / (col_max - col_min)
            else:
                features_normalized[col] = 0.0
        logger.info("Crop features normalized.")
        
        similarity_matrix = np.dot(features_normalized, features_normalized.T)
        
        sim_min = similarity_matrix.min()
        sim_max = similarity_matrix.max()
        if (sim_max - sim_min) > 1e-9:
            similarity_matrix = (similarity_matrix - sim_min) / (sim_max - sim_min)
        else:
            similarity_matrix = np.zeros_like(similarity_matrix)
        logger.info("Similarity matrix calculated and normalized.")

        return pd.DataFrame(
            similarity_matrix,
            index=features_encoded.index,
            columns=features_encoded.index
        )
    
    def recommend(self, base_crop, n=5):
        logger.info(f"Request for recommendations for base crop: {base_crop}")
        try:
            if base_crop not in self.crop_similarity.index:
                logger.warning(f"Base crop '{base_crop}' not found in similarity matrix.")
                st.warning(f"Base crop '{base_crop}' not found for similarity recommendations.")
                return []
                
            similar = self.crop_similarity[base_crop].sort_values(ascending=False).iloc[1:n+1]
            logger.info(f"Found {len(similar)} similar crops for {base_crop}.")
            return list(similar.index)
        except Exception as e:
            logger.error(f"Error in recommendation: {e}", exc_info=True)
            st.error(f"Error in recommendation: {e}")
            return []

def calculate_water_cost(df, cost_per_m3=0.45, consumption_per_ha=5000):
    logger.info("Calculating water costs.")
    df_copy = df.copy() # Ensure working on a copy
    df_copy.loc[:, 'coste_agua'] = df_copy['regadio_hectareas'] * consumption_per_ha * cost_per_m3
    return df_copy

def plot_calibration_curve(y_true, y_pred, n_bins=10):
    logger.info("Generating calibration curve.")
    y_true_binary = (y_true > np.median(y_true)).astype(int)
    y_pred_scaled = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))

    if len(np.unique(y_true_binary)) < 2:
        logger.warning("Cannot plot calibration curve: 'y_true_binary' has only one class.")
        st.warning("Cannot plot calibration curve: 'y_true_binary' has only one class.")
        return go.Figure()

    prob_true, prob_pred = calibration_curve(y_true_binary, y_pred_scaled, n_bins=n_bins)
    logger.info("Calibration curve data prepared.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prob_pred, 
        y=prob_true,
        mode='lines+markers',
        name='Model'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Ideal',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title='Calibration Curve (Simulated for Regression Confidence)',
        xaxis_title='Average Prediction',
        yaxis_title='Fraction of Positives',
        showlegend=True
    )
    return fig

def plot_roc_analysis(df, threshold=0.5):
    logger.info("Generating ROC analysis.")
    df_copy = df.copy() 

    # FIX for SettingWithCopyWarning using .loc
    max_superficie = df_copy['superficie_cultivada_hectareas'].max()
    if max_superficie > 0:
        df_copy.loc[:, 'riesgo'] = np.where(
            df_copy['superficie_cultivada_hectareas'] > threshold * max_superficie, 
            1, 
            0
        )
    else:
        df_copy.loc[:, 'riesgo'] = 0 
    logger.info("Risk variable calculated for ROC analysis.")
    
    np.random.seed(42)
    # FIX for SettingWithCopyWarning using .loc
    df_copy.loc[:, 'prediccion_riesgo'] = np.random.rand(len(df_copy))
    logger.info("Dummy risk prediction generated.")
    
    thresholds = np.linspace(0, 1, 100)
    tpr = []
    fpr = []
    
    if df_copy['riesgo'].sum() == 0 or (len(df_copy) - df_copy['riesgo'].sum()) == 0:
        logger.warning("Cannot plot ROC curve: Not enough variation in 'riesgo' (risk) variable.")
        st.warning("Cannot plot ROC curve: Not enough variation in 'riesgo' (risk) variable.")
        return go.Figure()
        
    for thresh in thresholds:
        tp = ((df_copy['prediccion_riesgo'] >= thresh) & (df_copy['riesgo'] == 1)).sum()
        fp = ((df_copy['prediccion_riesgo'] >= thresh) & (df_copy['riesgo'] == 0)).sum()
        fn = ((df_copy['prediccion_riesgo'] < thresh) & (df_copy['riesgo'] == 1)).sum()
        tn = ((df_copy['prediccion_riesgo'] < thresh) & (df_copy['riesgo'] == 0)).sum()
        
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    logger.info("TPR and FPR calculated for ROC curve.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    
    fig.update_layout(
        title='ROC Curve - Risk Analysis',
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        showlegend=True
    )
    return fig

def recommend_diversification(df, selected_year):
    logger.info(f"Generating diversification recommendations for year: {selected_year}")
    current_cultivos = set(df[df['año'] == selected_year]['cultivo'].unique())
    all_cultivos = set(df['cultivo'].unique())
    potential_cultivos = list(all_cultivos - current_cultivos)
    
    if potential_cultivos:
        logger.info(f"Found {len(potential_cultivos)} potential crops for diversification.")
        return f"**Diversification suggestions:** {', '.join(potential_cultivos[:5])}"
    logger.info("No diversification suggestions found.")
    return "Excellent crop diversity!"

def setup_monitoring():
    logger.info("Setting up monitoring directory and file.")
    Path('monitoring').mkdir(exist_ok=True)
    if not os.path.exists('monitoring/predictions.csv'):
        pd.DataFrame(columns=['timestamp', 'cultivo', 'predicted', 'actual', 'error']).to_csv('monitoring/predictions.csv', index=False)
        logger.info("monitoring/predictions.csv created.")
    else:
        logger.info("monitoring/predictions.csv already exists.")

def log_prediction(cultivo, predicted, actual):
    logger.info(f"Logging prediction: Cultivo='{cultivo}', Predicted={predicted:.2f}, Actual={actual:.2f}")
    error = abs(predicted - actual)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_entry = pd.DataFrame({
        'timestamp': [timestamp],
        'cultivo': [cultivo],
        'predicted': [predicted],
        'actual': [actual],
        'error': [error]
    })
    
    if os.path.exists('monitoring/predictions.csv'):
        existing = pd.read_csv('monitoring/predictions.csv')
        updated = pd.concat([existing, new_entry], ignore_index=True)
        updated.to_csv('monitoring/predictions.csv', index=False)
    else:
        new_entry.to_csv('monitoring/predictions.csv', index=False)
    logger.info("Prediction logged successfully.")

def check_for_alerts(threshold=0.2):
    logger.info("Checking for monitoring alerts.")
    if os.path.exists('monitoring/predictions.csv'):
        df = pd.read_csv('monitoring/predictions.csv')
        if df.empty:
            logger.info("No monitoring data to check for alerts.")
            return None
        
        df['error_ratio'] = df['error'] / df['actual'].replace(0, 1)
        high_errors = df[df['error_ratio'] > threshold]
        
        if not high_errors.empty:
            last_alert = high_errors.iloc[-1]
            logger.warning(f"ALERT: High error detected. Last alert for {last_alert['cultivo']}.")
            return f"ALERT: High error in {last_alert['cultivo']} (Error: {last_alert['error']:.2f} ha)"
    logger.info("No recent alerts detected.")
    return None

@st.cache_resource
def get_yield_model():
    logger.info("Retrieving or initializing CropYieldModel (cached).")
    return CropYieldModel()

@st.cache_resource
def get_recommender(df_arg):
    logger.info("Retrieving or initializing HybridRecommender (cached).")
    return HybridRecommender(df_arg)

def automl_training_flaml(df):
    logger.info("Starting FLAML AutoML training process.")
    try:
        df_ml = df.copy()

        features = ['año', 'grupo_de_cultivo', 'cultivo', 'secano_hectareas', 'regadio_hectareas']
        target = 'superficie_cultivada_hectareas'
        
        for col in features:
            if col not in df_ml.columns:
                logger.error(f"Missing required feature column for FLAML: {col}")
                return False, f"Missing required feature column: {col}", None

        if len(df_ml) < 20: 
            logger.warning(f"Not enough data for FLAML training: {len(df_ml)} rows found.")
            return False, "Not enough data for FLAML to train effectively. Need at least 20 rows.", None
        
        if df_ml[target].nunique() < 2:
            logger.warning(f"Target variable '{target}' has no variation ({df_ml[target].nunique()} unique values).")
            return False, f"The target variable '{target}' has no variation. FLAML cannot train a regressor.", None

        df_ml.loc[:, 'grupo_de_cultivo'] = df_ml['grupo_de_cultivo'].astype('category')
        df_ml.loc[:, 'cultivo'] = df_ml['cultivo'].astype('category')
        logger.info("Categorical features converted for FLAML.")

        X = df_ml[features]
        y = df_ml[target]
        logger.info(f"FLAML training data X shape: {X.shape}, y shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"FLAML Train/test split complete. X_train shape: {X_train.shape}")

        if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
            logger.error("FLAML data splitting resulted in empty sets.")
            return False, "Data splitting resulted in empty training or testing sets. Please check your dataset size.", None

        st.info("FLAML is searching for the best model. This can take a while...")
        logger.info("Initializing FLAML AutoML and starting fit.")

        automl = AutoML()
        automl.fit(
            X_train, 
            y_train, 
            task='regression', 
            time_budget=120, # 2 minutes (120 seconds), adjust as needed
            metric='mae', 
            log_file_name="flaml_training.log"
        )
        logger.info("FLAML AutoML fit finished.")
        
        # Check if a model was actually found and fitted by FLAML
        if automl.model is None: 
            logger.error("FLAML.model is None after fitting. No model was found or fitted.")
            return False, "FLAML did not find or fit a suitable model within the given time budget. Try increasing 'time_budget' or check data for issues.", None
        
        # Verify the predict method exists before calling it (defensive check)
        if not hasattr(automl, 'predict') or not callable(automl.predict):
            logger.error("FLAML model object does not have a callable 'predict' method after training.")
            return False, "FLAML model object does not have a callable 'predict' method after training.", None

        logger.info("Making predictions with FLAML best model.")
        y_pred = automl.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"FLAML model evaluation: MAE={mae:.2f}, R2={r2:.2f}")
        
        Path('models').mkdir(exist_ok=True)
        joblib.dump(automl, 'models/flaml_best_model.pkl') 
        logger.info("FLAML best model saved to models/flaml_best_model.pkl")

        return True, f"FLAML found the best pipeline! MAE: {mae:.2f}, R²: {r2:.2f}", automl
    except Exception as e:
        logger.error(f"Critical error in AutoML with FLAML: {e}", exc_info=True)
        return False, f"Error in AutoML with FLAML: {e}", None

def main():
    logger.info("Streamlit app started.")
    df = load_data('data/superficie-cultivada-cultivos.csv')
    
    if df.empty:
        st.warning("Could not load data. Please check the file path and content.")
        logger.error("Dataframe is empty, stopping app execution.")
        return
    
    setup_monitoring()
    
    yield_model = get_yield_model() 
    recommender = get_recommender(df)

    st.sidebar.header("Global Configuration")
    if not df['año'].empty:
        selected_year = st.sidebar.selectbox("Reference Year", sorted(df['año'].unique()), index=len(df['año'].unique())-1, key='sidebar_ref_year')
    else:
        selected_year = datetime.now().year
        st.sidebar.warning("No year data available, defaulting to current year.")
        logger.warning("No year data in DataFrame, defaulting sidebar year to current year.")

    cost_per_m3 = st.sidebar.slider("Water Cost (€/m³)", 0.1, 1.0, 0.45, 0.05, key='sidebar_water_cost')
    water_per_ha = st.sidebar.slider("Water Consumption (m³/ha)", 1000, 10000, 5000, 100, key='sidebar_water_cons')
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "🔮 Predictions", 
        "💡 Recommendations", 
        "💰 Cost Analysis", 
        "📈 Model Evaluation",
        "🔍 Monitoring"
    ])
    
    with tab1:
        st.header("Valencia Agricultural Analysis")
        logger.info(f"Displaying Dashboard tab for year: {selected_year}")
        
        df_selected_year = df[df['año'] == selected_year]

        if df_selected_year.empty:
            st.info(f"No data available for the year {selected_year}. Please select another year.")
            logger.info(f"No data for selected year {selected_year} in dashboard.")
        else:
            total_area = df_selected_year['superficie_cultivada_hectareas'].sum()
            unique_crops = df_selected_year['cultivo'].nunique()
            irrigation_perc = (df_selected_year['regadio_hectareas'].sum() / total_area * 100) if total_area > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Area", f"{total_area:,.0f} ha")
            col2.metric("Unique Crops", unique_crops)
            col3.metric("Irrigated Area", f"{irrigation_perc:.1f}%")
            logger.info(f"Dashboard metrics: Total Area={total_area}, Unique Crops={unique_crops}, Irrigated %={irrigation_perc}")
            
            st.subheader("Evolution of Cultivated Area")
            df_year = df.groupby('año')['superficie_cultivada_hectareas'].sum().reset_index()
            fig_trend = px.line(df_year, x='año', y='superficie_cultivada_hectareas', 
                                 title='Total Cultivated Area per Year',
                                 markers=True)
            st.plotly_chart(fig_trend, use_container_width=True)
            
            st.subheader(f"Distribution by Crop Group ({selected_year})")
            df_group = df_selected_year.groupby('grupo_de_cultivo')['superficie_cultivada_hectareas'].sum().reset_index()
            fig_group = px.pie(df_group, values='superficie_cultivada_hectareas', names='grupo_de_cultivo',
                               title=f'Distribution by Group in {selected_year}')
            st.plotly_chart(fig_group, use_container_width=True)
    
    with tab2:
        st.header("Predictive Models")
        
        if st.button("Train Default Yield Model (Random Forest)", key='train_default_model_btn'):
            logger.info("User clicked 'Train Default Yield Model'.")
            with st.spinner("Training default model..."):
                mae, r2, y_test, y_pred = yield_model.train(df)
                
            if mae is not None:
                st.success(f"Default Random Forest Model trained! MAE: {mae:.2f}, R²: {r2:.2f}")
                
                if y_test is not None and y_pred is not None and len(y_test) > 0:
                    fig_pred = px.scatter(
                        x=y_test, 
                        y=y_pred,
                        labels={'x': 'Actual Value', 'y': 'Prediction'},
                        title='Predictions vs Actual Values (Default RF Model)'
                    )
                    fig_pred.add_trace(go.Scatter(
                        x=[min(y_test), max(y_test)], 
                        y=[min(y_test), max(y_test)],
                        mode='lines',
                        name='Ideal Line'
                    ))
                    st.plotly_chart(fig_pred, use_container_width=True)
                    logger.info("Default model prediction plot displayed.")
                else:
                    st.info("Not enough test data to plot predictions after training.")
                    logger.info("Not enough test data for default model prediction plot.")
        
        st.subheader("AutoML with FLAML")
        st.info("FLAML will automatically train and select the best model. It's designed for speed and efficiency.")
        if st.button("Run AutoML (FLAML)", key='run_automl_flaml_btn'):
            logger.info("User clicked 'Run AutoML (FLAML)'.")
            with st.spinner("Running FLAML to find the best pipeline. This can take significant time..."):
                success, message, flaml_automl = automl_training_flaml(df) 
            if success:
                st.success(message)
                st.info("FLAML model saved to the 'models' directory.")
                st.session_state['flaml_automl'] = flaml_automl
                logger.info("FLAML AutoML training successful and model stored in session_state.")
            else:
                st.error(message)
                logger.error(f"FLAML AutoML training failed: {message}")
        
        st.subheader("Crop Simulator")
        col1, col2 = st.columns(2)
        
        with col1:
            min_year_data = df['año'].min() if not df['año'].empty else 2023
            year = st.number_input("Year", min_value=min_year_data, max_value=2030, value=selected_year, key='sim_year_input')
            
            crop_group_options = df['grupo_de_cultivo'].unique().tolist()
            if len(crop_group_options) > 0:
                crop_group = st.selectbox("Crop Group", crop_group_options, key='sim_crop_group_select')
            else:
                crop_group = None
                st.warning("No crop groups available in data.")
                logger.warning("No crop groups available for simulator.")
            
        with col2:
            crop_options = df[df['grupo_de_cultivo'] == crop_group]['cultivo'].unique().tolist() if crop_group else []
            if len(crop_options) > 0:
                crop = st.selectbox("Crop", crop_options, key='sim_crop_select')
            else:
                crop = None
                st.warning("No crops available for selected group.")
                logger.warning("No crops available for selected group in simulator.")
            
            secano = st.number_input("Rainfed (ha)", min_value=0.0, value=10.0, key='sim_secano_input')
            regadio = st.number_input("Irrigated (ha)", min_value=0.0, value=50.0, key='sim_regadio_input')
        
        if st.button("Predict Area using Default Model", key='predict_area_btn'):
            logger.info("User clicked 'Predict Area using Default Model'.")
            if yield_model.model is not None and crop_group is not None and crop is not None:
                prediction = yield_model.predict(year, crop_group, crop, secano, regadio)
                if prediction is not None and prediction > 0:
                    st.metric("Predicted Cultivated Area", f"{prediction:.2f} ha")
                    log_prediction(crop, prediction, secano + regadio) 
                    logger.info(f"Default model prediction displayed and logged for {crop}.")
                elif prediction is not None:
                    st.metric("Predicted Cultivated Area", f"{prediction:.2f} ha")
                    log_prediction(crop, prediction, secano + regadio) 
                    logger.info(f"Default model prediction (<=0) displayed and logged for {crop}.")
            else:
                st.warning("Please train the default model and select valid crop group/crop first.")
                logger.warning("Cannot predict: Default model not trained or invalid crop selection.")
    
    with tab3:
        st.header("Recommendation Systems")
        logger.info("Displaying Recommendations tab.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Crop Diversification")
            st.markdown(recommend_diversification(df, selected_year))
            
            st.subheader("Risk Recommendations")
            df_selected_year = df[df['año'] == selected_year]
            if not df_selected_year.empty:
                top_risk_crops = df_selected_year.nlargest(5, 'superficie_cultivada_hectareas')['cultivo'].tolist()
                st.info(f"Crops with largest area (potential market risk): {', '.join(top_risk_crops)}")
                logger.info(f"Displayed top risk crops for {selected_year}.")
            else:
                st.info("No data for risk recommendations in the selected year.")
                logger.info(f"No data for risk recommendations for {selected_year}.")
            
        with col2:
            st.subheader("Similarity-Based Recommendations")
            base_crop_options = df['cultivo'].unique().tolist()
            if len(base_crop_options) > 0:
                base_crop = st.selectbox("Select a base crop", base_crop_options, key='reco_base_crop_select')
            else:
                base_crop = None
                st.warning("No crops available for similarity recommendations.")
                logger.warning("No crops available for similarity recommendations selectbox.")

            if base_crop:
                recommendations = recommender.recommend(base_crop)
                if recommendations:
                    st.write(f"Crops similar to **{base_crop}**:")
                    for rec in recommendations:
                        st.info(f"- {rec}")
                    logger.info(f"Displayed {len(recommendations)} similarity recommendations for {base_crop}.")
                else:
                    st.warning("No recommendations found for this crop.")
                    logger.info(f"No similarity recommendations found for {base_crop}.")
            else:
                st.info("Select a base crop to get recommendations.")
            
            st.subheader("Ranking for Agricultural Campaigns")
            df_selected_year = df[df['año'] == selected_year]
            if not df_selected_year.empty:
                top_crops = df_selected_year.groupby('cultivo')['superficie_cultivada_hectareas'].sum().nlargest(5).reset_index()
                st.dataframe(top_crops)
                logger.info(f"Displayed top crops for campaigns for {selected_year}.")
            else:
                st.info("No data for campaign ranking in the selected year.")
                logger.info(f"No data for campaign ranking for {selected_year}.")

    with tab4:
        st.header("Cost and Utility Analysis")
        logger.info(f"Displaying Cost and Utility Analysis tab for year: {selected_year}")
        
        df_costs = calculate_water_cost(df[df['año'] == selected_year], cost_per_m3, water_per_ha)
        
        if df_costs.empty:
            st.info(f"No cost data available for the year {selected_year}. Please select another year.")
            logger.info(f"No cost data for {selected_year}.")
        else:
            total_water_cost = df_costs['coste_agua'].sum()
            
            st.metric("Total Water Cost", f"€{total_water_cost:,.2f}")
            logger.info(f"Total water cost: {total_water_cost:.2f}.")
            
            st.subheader("Cost Distribution by Group")
            df_cost_group = df_costs.groupby('grupo_de_cultivo')['coste_agua'].sum().reset_index()
            fig_cost = px.bar(df_cost_group, x='grupo_de_cultivo', y='coste_agua',
                              title='Water Cost by Crop Group',
                              labels={'coste_agua': 'Cost (€)', 'grupo_de_cultivo': 'Crop Group'})
            st.plotly_chart(fig_cost, use_container_width=True)
            
            st.subheader("Cost-Benefit Analysis")
            st.info("Below is a hypothetical utility analysis based on water costs and estimated yield")
            
            df_utility = df_costs.copy()
            if not df_utility.empty:
                df_utility['rendimiento_kg_ha'] = np.random.uniform(1000, 5000, len(df_utility))
                df_utility['precio_kg'] = np.random.uniform(0.5, 3.0, len(df_utility))
                df_utility['ingresos'] = df_utility['superficie_cultivada_hectareas'] * df_utility['rendimiento_kg_ha'] * df_utility['precio_kg']
                df_utility['utilidad'] = df_utility['ingresos'] - df_utility['coste_agua']
                
                st.dataframe(df_utility[['cultivo', 'utilidad']].sort_values('utilidad', ascending=False).head(10))
                logger.info("Utility analysis displayed.")
            else:
                st.info("Cannot perform utility analysis as there is no data for the selected year.")
                logger.info("No data for utility analysis.")
    
    with tab5:
        st.header("Predictive Model Evaluation")
        logger.info("Displaying Model Evaluation tab.")
        
        st.subheader("ROC Analysis for Risk Model")
        fig_roc = plot_roc_analysis(df)
        st.plotly_chart(fig_roc, use_container_width=True)
        
        st.subheader("Calibration Curve")
        if len(df['superficie_cultivada_hectareas'].unique()) > 1 and len(df) > 1:
            y_true_for_calib = df['superficie_cultivada_hectareas']
            y_pred_for_calib = (df['superficie_cultivada_hectareas'] * np.random.uniform(0.8, 1.2, len(df))).values
            fig_calib = plot_calibration_curve(y_true_for_calib, y_pred_for_calib)
            st.plotly_chart(fig_calib, use_container_width=True)
            logger.info("Calibration curve displayed.")
        else:
            st.info("Not enough unique values or data points in 'superficie_cultivada_hectareas' for calibration curve (simulated).")
            logger.info("Not enough data for calibration curve.")

        st.subheader("Model Comparison (Illustrative)")
        st.info("FLAML automatically compares and selects the best models. The actual results will be in the 'Predictions' tab.")
        model_results_illustrative = pd.DataFrame({
            'Model': ['FLAML (Best Model)', 'Random Forest', 'Gradient Boosting'],
            'MAE': ['Varies by FLAML Run', '125.4', '118.7'],
            'R²': ['Varies by FLAML Run', '0.89', '0.91'],
            'Training Time (s)': ['Varies by FLAML Run', '12.4', '8.7']
        })
        for col in ['MAE', 'R²', 'Training Time (s)']:
            model_results_illustrative[col] = model_results_illustrative[col].astype(str)

        st.dataframe(model_results_illustrative)
        
        st.subheader("Model Fusion")
        st.info("Model fusion combines predictions from multiple algorithms to improve overall accuracy")
        fig_ensemble = go.Figure()
        fig_ensemble.add_trace(go.Bar(name='Individual Model', x=model_results_illustrative['Model'], y=[120, 125.4, 118.7]))
        fig_ensemble.add_trace(go.Bar(name='Model Fusion (Simulated)', x=['FLAML (Best Model)', 'Random Forest', 'Gradient Boosting'], y=[110.0, 110.0, 110.0]))
        fig_ensemble.update_layout(barmode='group', title='Comparison: Individual Models vs Fusion (Simulated)')
        st.plotly_chart(fig_ensemble, use_container_width=True)
    
    with tab6:
        st.header("Model Monitoring")
        logger.info("Displaying Model Monitoring tab.")
        
        alert = check_for_alerts()
        if alert:
            st.warning(alert)
            logger.warning(f"Monitoring alert displayed: {alert}")
        else:
            st.success("No recent alerts detected")
            logger.info("No monitoring alerts to display.")
        
        if os.path.exists('monitoring/predictions.csv'):
            df_monitor = pd.read_csv('monitoring/predictions.csv')
            
            if not df_monitor.empty:
                st.subheader("Prediction Error Evolution")
                df_monitor['timestamp'] = pd.to_datetime(df_monitor['timestamp'])
                fig_error = px.line(df_monitor, x='timestamp', y='error', 
                                     color='cultivo',
                                     title='Prediction Error Over Time')
                st.plotly_chart(fig_error, use_container_width=True)
                logger.info("Prediction error evolution plot displayed.")
                
                st.subheader("Error Distribution")
                fig_hist = px.histogram(df_monitor, x='error', 
                                         nbins=20,
                                         title='Prediction Error Distribution')
                st.plotly_chart(fig_hist, use_container_width=True)
                logger.info("Error distribution plot displayed.")
                
                if st.button("Generate Quality Report", key='generate_report_btn'):
                    logger.info("User clicked 'Generate Quality Report'.")
                    report = f"""
                    # Model Quality Report
                    **Date:** {datetime.now().strftime("%Y-%m-%d")}
                    **Total predictions:** {len(df_monitor)}
                    **Average error:** {df_monitor['error'].mean():.2f} ha
                    **Maximum error:** {df_monitor['error'].max():.2f} ha
                    **Cultivo con mayor error:** {df_monitor.loc[df_monitor['error'].idxmax()]['cultivo']}
                    """
                    st.download_button(
                        "Download Report",
                        report,
                        "quality_report.md",
                        "text/markdown"
                    )
                    logger.info("Quality report generated and offered for download.")
            else:
                st.info("No monitoring data available yet. Make some predictions to see data here.")
                logger.info("No monitoring data available.")
        else:
            st.info("No monitoring data file found. Predictions will be logged once you start making them.")
            logger.info("Monitoring predictions file not found.")
    
    st.sidebar.markdown("---")
    st.sidebar.header("About the Project")
    st.sidebar.info("""
    **Valencia Agri-Insights Pro** Advanced application for sustainable urban agricultural management  
    Integrates:  
    - Predictive models  
    - Recommendation systems  
    - Cost analysis  
    - Continuous monitoring  
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Developed by:** Yasmin Serena Diaconu, Óscar García Martínez y Víctor Mañez Poveda")
    st.sidebar.markdown("**Repository:** [GitHub](https://github.com/tuequipo)")
    st.sidebar.markdown("**Demo Video:** [YouTube](https://youtube.com/tuequipo)")
    logger.info("Streamlit app finished execution cycle.")

if __name__ == "__main__":
    main()