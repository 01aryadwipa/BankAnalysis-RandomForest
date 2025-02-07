import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Title and description
st.markdown("<h1 style='text-align: center; margin-bottom: 50px;'>Machine Learning Investment Analysis</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for title and file upload
st.sidebar.title("Welcome to Machine Learning Investment Analysis")

# File upload section
uploaded_file = st.sidebar.file_uploader("Upload your dataset (Excel or CSV)", type=["xlsx", "csv"], help="Limit 200MB per file • XLSX, CSV")

# Navigation bar for language selection
st.sidebar.markdown("### Select Language:")
language = st.sidebar.radio("", ["Indonesian", "English"], horizontal=True)

# Display usage instructions based on selected language
if language == "Indonesian":
    st.sidebar.markdown("### Cara Menggunakan Website:")
    st.sidebar.write("""
    1. **Masukkan file data keuangan** pada box file upload di atas.  
       Format dataset dan contoh-contohnya dapat diakses pada [link berikut](https://drive.google.com/drive/folders/1fJp8NOyLTMmiZQ6gpgcscn2IMf9TohuD?usp=drive_link).
    2. **Dataset Preview**: Anda dapat melihat dan memeriksa data pada bagian Dataset Preview.
    3. **Data Visualization**: Anda dapat melakukan visualisasi data pada bagian Data Visualization.
    4. **Analisis Machine Learning**:
       - Pilih rasio keuangan yang ingin digunakan untuk analisis dari daftar *Select ratios for analysis*.
       - Pilih kolom target (harga saham perusahaan) dari dropdown *Select the target column*.
       - Klik *Run Model and Predict Stock Prices* untuk menjalankan model machine learning dan melihat hasil prediksi.
    5. **Hasil Analisis Machine Learning**:
       - Website akan menampilkan metrik evaluasi model, seperti RMSE, MAE, MAPE, dan R².
       - Anda dapat melihat tabel perbandingan antara *Actual vs Predicted Stock Prices*.
       - Prediksi harga saham untuk periode berikutnya akan ditampilkan di bagian *Predicted Stock Price for the Next Period*.
       - Anda juga dapat melihat *Feature Importance* jika model mendukung.
    """)
elif language == "English":
    st.sidebar.markdown("### How to Use the Website:")
    st.sidebar.write("""
    1. **Upload your financial dataset** in the file upload box above.  
       You can access dataset format and examples at [this link](https://drive.google.com/drive/folders/1fJp8NOyLTMmiZQ6gpgcscn2IMf9TohuD?usp=drive_link).
    2. **Dataset Preview**: You can view and check your data in the Dataset Preview section.
    3. **Data Visualization**: You can visualize the data in the Data Visualization section.
    4. **Machine Learning Analysis**:
       - Select the financial ratios to use for analysis from the *Select ratios for analysis* list.
       - Select the target column (company's stock price) from the *Select the target column* dropdown.
       - Click *Run Model and Predict Stock Prices* to run the machine learning model and see the predictions.
    5. **Machine Learning Analysis Results**:
       - The website will display model evaluation metrics such as RMSE, MAE, MAPE, and R².
       - You can view a comparison table between *Actual vs Predicted Stock Prices*.
       - The stock price prediction for the next period will be displayed in the *Predicted Stock Price for the Next Period* section.
       - You can also view *Feature Importance* if the model supports it.
    """)

if uploaded_file:
    try:
        # Load the data
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        st.write("Data preview:", data.head())

        # Ensure the dataset has a 'quarter' column
        if 'quarter' not in data.columns:
            st.error("The dataset must include a 'quarter' column for proper visualization.")
        else:
            # Ensure 'quarter' is treated as a categorical variable with proper ordering
            data['quarter'] = pd.Categorical(
                data['quarter'],
                categories=sorted(data['quarter'].unique(), key=lambda x: (int(x.split('_')[1]), x.split('_')[0])),
                ordered=True
            )

            # Visualization Section
            st.subheader("Data Visualization")

            # Dynamically identify metrics and companies
            all_columns = data.columns.tolist()
            available_metrics = [col.split('_')[0] for col in all_columns if '_' in col and col.split('_')[0] != 'price']
            available_metrics = sorted(list(set(available_metrics)))
            companies = [col.split('_')[1] for col in all_columns if '_' in col and col.split('_')[0] == 'price']
            companies = sorted(list(set(companies)))

            selected_metric = st.selectbox("Choose a metric to visualize", available_metrics)

            if selected_metric:
                metric_data = pd.DataFrame({
                    company.upper(): data[f'{selected_metric}_{company.lower()}'] for company in companies if f'{selected_metric}_{company.lower()}' in data.columns
                })

                if not metric_data.empty:
                    metric_data['Quarter'] = data['quarter']
                    metric_data.set_index('Quarter', inplace=True)
                    st.write(f"Comparison of {selected_metric.upper()} across companies:")
                    st.line_chart(metric_data)
                else:
                    st.warning(f"No data available for {selected_metric}. Check your dataset.")

            # Analysis Section
            analysis_type = st.selectbox("Choose analysis type", ["Per Company Analysis", "Combined Analysis"])

            results_summary = []

            if analysis_type == "Per Company Analysis":
                st.subheader("Per Company Analysis")
                selected_companies = st.multiselect("Select company for analysis", companies)

                if selected_companies:
                    for company in selected_companies:
                        ratios = [col for col in all_columns if f'_{company.lower()}' in col and col.split('_')[0] != 'price']
                        target = f'price_{company.lower()}'

                        if target in data.columns:
                            X = data[ratios]
                            y = data[target]

                            # Train-test split
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                            # Hyperparameter tuning using GridSearchCV
                            param_grid = {
                                'n_estimators': [50, 100, 200],
                                'max_depth': [None, 10, 20, 30],
                                'min_samples_split': [2, 5, 10],
                                'min_samples_leaf': [1, 2, 4]
                            }

                            grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                                                       param_grid=param_grid,
                                                       cv=3,
                                                       scoring='neg_mean_squared_error',
                                                       verbose=1,
                                                       n_jobs=-1)
                            grid_search.fit(X_train, y_train)
                            best_model = grid_search.best_estimator_

                            # Predictions and metrics
                            y_pred = best_model.predict(X_test)
                            rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
                            mae = round(mean_absolute_error(y_test, y_pred), 2)
                            mape = round(np.mean(np.abs((y_test - y_pred) / y_test)) * 100, 2)
                            r2 = round(r2_score(y_test, y_pred), 2)

                            st.write(f"**Results for {company.upper()}:**")
                            st.write(f"Best Parameters: {grid_search.best_params_}")
                            st.write(f"Root Mean Squared Error (RMSE): {rmse}")
                            st.write(f"Mean Absolute Error (MAE): {mae}")
                            st.write(f"Mean Absolute Percentage Error (MAPE): {mape}%")
                            st.write(f"R-squared (R²): {r2}")

                            # Feature importance
                            feature_importances = pd.DataFrame({
                                "Feature": X.columns,
                                "Importance": [round(imp, 2) for imp in best_model.feature_importances_]
                            }).sort_values(by="Importance", ascending=False)
                            st.write(f"Feature Importances for {company.upper()}", feature_importances)

                            # Store results
                            results_summary.append({
                                'Company': company.upper(),
                                'RMSE': rmse,
                                'MAE': mae,
                                'MAPE': mape,
                                'R2': r2
                            })

            elif analysis_type == "Combined Analysis":
                st.subheader("Combined Analysis")

                # Combine data for all companies dynamically
                combined_data = pd.DataFrame()
                for company in companies:
                    ratios = [col for col in all_columns if f'_{company.lower()}' in col and col.split('_')[0] != 'price']
                    target = f'price_{company.lower()}'

                    if target in data.columns:
                        company_data = data[ratios + [target]].copy()
                        company_data.columns = [col.split('_')[0].upper() for col in ratios] + ['Stock_Price']
                        combined_data = pd.concat([combined_data, company_data], ignore_index=True)

                if not combined_data.empty:
                    combined_data = combined_data.dropna()
                    X_combined = combined_data.drop(columns=['Stock_Price'])
                    y_combined = combined_data['Stock_Price']

                    # Train-test split
                    X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
                        X_combined, y_combined, test_size=0.2, random_state=42)

                    # Hyperparameter tuning
                    param_grid_combined = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }

                    grid_search_combined = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                                                        param_grid=param_grid_combined,
                                                        cv=3,
                                                        scoring='neg_mean_squared_error',
                                                        verbose=1,
                                                        n_jobs=-1)
                    grid_search_combined.fit(X_train_combined, y_train_combined)
                    best_combined_model = grid_search_combined.best_estimator_

                    # Predictions and metrics
                    y_pred_combined = best_combined_model.predict(X_test_combined)
                    rmse_combined = round(np.sqrt(mean_squared_error(y_test_combined, y_pred_combined)), 2)
                    mae_combined = round(mean_absolute_error(y_test_combined, y_pred_combined), 2)
                    mape_combined = round(np.mean(np.abs((y_test_combined - y_pred_combined) / y_test_combined)) * 100, 2)
                    r2_combined = round(r2_score(y_test_combined, y_pred_combined), 2)

                    st.write("**Combined Analysis Results:**")
                    st.write(f"Best Parameters: {grid_search_combined.best_params_}")
                    st.write(f"Root Mean Squared Error (RMSE): {rmse_combined}")
                    st.write(f"Mean Absolute Error (MAE): {mae_combined}")
                    st.write(f"Mean Absolute Percentage Error (MAPE): {mape_combined}%")
                    st.write(f"R-squared (R²): {r2_combined}")

                    # Feature importance
                    combined_feature_importances = pd.DataFrame({
                        "Feature": X_combined.columns,
                        "Importance": [round(imp, 2) for imp in best_combined_model.feature_importances_]
                    }).sort_values(by="Importance", ascending=False)
                    st.write("Feature Importances for Combined Analysis:", combined_feature_importances)

                    # Store combined results
                    results_summary.append({
                        'Companies': 'Combined',
                        'RMSE': rmse_combined,
                        'MAE': mae_combined,
                        'MAPE': mape_combined,
                        'R2': r2_combined
                    })

            # Display results summary
            if results_summary:
                results_table = pd.DataFrame(results_summary)
                st.write("**Summary of Results Across All Analyses:**")
                st.write(results_table)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add a footer for clarity
st.markdown("---")
