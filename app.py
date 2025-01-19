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
st.sidebar.write("Dataset dapat diakses pada [link berikut](https://drive.google.com/drive/folders/1Z2gNUGtqRYHcvtl5pTmtPQpu0cuzNLCn?usp=sharing)")
if uploaded_file:
    try:
        # Load the data
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        st.write("Data preview:", data.head())

        # Check if 'quarter' column exists
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
            available_metrics = [
                'roe', 'roa', 'nim', 'npl', 'ldr', 'car', 'price'
            ]
            selected_metric = st.selectbox("Choose a metric to visualize", available_metrics)

            if selected_metric:
                # Extract data for the selected metric across banks
                metric_data = pd.DataFrame({
                    bank.upper(): data[f'{selected_metric}_{bank.lower()}'] for bank in ['bbca', 'bbri', 'bmri', 'bbni'] if f'{selected_metric}_{bank.lower()}' in data.columns
                })

                # Add the properly ordered quarter column for the x-axis
                if not metric_data.empty:
                    metric_data['Quarter'] = data['quarter']
                    metric_data.set_index('Quarter', inplace=True)
                    st.write(f"Comparison of {selected_metric.upper()} across banks:")
                    st.line_chart(metric_data)
                else:
                    st.warning(f"No data available for {selected_metric}. Check your dataset.")

            # Choose analysis type
            analysis_type = st.selectbox("Choose analysis type", ["Per Bank Analysis", "Combined Analysis"])

            # Initialize results storage
            results_summary = []

            if analysis_type == "Per Bank Analysis":
                st.subheader("Per Bank Analysis")
                banks = st.multiselect("Select banks for analysis", ['BBCA', 'BBRI', 'BMRI', 'BBNI'])

                if banks:
                    for bank in banks:
                        # Extract columns for the specified bank
                        ratios = [
                            f'roe_{bank.lower()}',
                            f'roa_{bank.lower()}',
                            f'nim_{bank.lower()}',
                            f'npl_{bank.lower()}',
                            f'ldr_{bank.lower()}',
                            f'car_{bank.lower()}'
                        ]
                        target = f'price_{bank.lower()}'

                        if all(col in data.columns for col in ratios + [target]):
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
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                            r2 = r2_score(y_test, y_pred)

                            st.write(f"**Results for {bank}:**")
                            st.write(f"Best Parameters: {grid_search.best_params_}")
                            st.write(f"Root Mean Squared Error (RMSE): {rmse}")
                            st.write(f"Mean Squared Error (MSE): {mse}")
                            st.write(f"Mean Absolute Error (MAE): {mae}")
                            st.write(f"Mean Absolute Percentage Error (MAPE): {mape}%")
                            st.write(f"R-squared (R²): {r2}")

                            # Feature importance
                            feature_importances = pd.DataFrame({
                                "Feature": X.columns,
                                "Importance": best_model.feature_importances_
                            }).sort_values(by="Importance", ascending=False)
                            st.write(f"Feature Importances for {bank}:", feature_importances)

                            # Plot actual vs predicted
                            fig, ax = plt.subplots()
                            ax.scatter(y_test, y_pred, alpha=0.5)
                            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                            ax.set_xlabel("Actual Values")
                            ax.set_ylabel("Predicted Values")
                            ax.set_title(f"Actual vs Predicted for {bank}")
                            st.pyplot(fig)

                            # Plot actual and predicted over time if 'quarter' column exists
                            if 'quarter' in data.columns:
                                fig_time, ax_time = plt.subplots()
                                time_index = X_test.index
                                full_quarters = data['quarter']
                                ax_time.plot(full_quarters, y, label="Actual", marker='o', alpha=0.5)
                                ax_time.scatter(full_quarters.loc[time_index], y_pred, label="Predicted", marker='x', color='red')
                                ax_time.set_xlabel("Quarter")
                                ax_time.set_ylabel("Stock Price")
                                ax_time.set_title(f"Actual vs Predicted Stock Price for {bank}")
                                ax_time.legend()
                                plt.xticks(rotation=45)
                                st.pyplot(fig_time)

                            # Store results
                            results_summary.append({
                                'Bank': bank,
                                'MSE': mse,
                                'RMSE': rmse,
                                'MAE': mae,
                                'MAPE': mape,
                                'R2': r2
                            })
                        else:
                            st.warning(f"Missing data for {bank}. Skipping.")

            elif analysis_type == "Combined Analysis":
                st.subheader("Combined Analysis")

                # Combine data for all banks
                combined_data = pd.DataFrame()
                for bank in ['bbca', 'bbri', 'bmri', 'bbni']:
                    if all(f'{col}_{bank}' in data.columns for col in ['roe', 'roa', 'nim', 'npl', 'ldr', 'car', 'price']):
                        bank_data = data[[f'roe_{bank}',
                                          f'roa_{bank}',
                                          f'nim_{bank}',
                                          f'npl_{bank}',
                                          f'ldr_{bank}',
                                          f'car_{bank}',
                                          f'price_{bank}']].copy()
                        bank_data.columns = ['ROE', 'ROA', 'NIM', 'NPL', 'LDR', 'CAR', 'Stock_Price']
                        combined_data = pd.concat([combined_data, bank_data], ignore_index=True)

                if not combined_data.empty:
                    combined_data = combined_data.dropna()
                    X_combined = combined_data[['ROE', 'ROA', 'NIM', 'NPL', 'LDR', 'CAR']]
                    y_combined = combined_data['Stock_Price']

                    # Train-test split
                    X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
                        X_combined, y_combined, test_size=0.2, random_state=42)

                    # Hyperparameter tuning using GridSearchCV
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
                    mse_combined = mean_squared_error(y_test_combined, y_pred_combined)
                    rmse_combined = np.sqrt(mse_combined)
                    mae_combined = mean_absolute_error(y_test_combined, y_pred_combined)
                    mape_combined = np.mean(np.abs((y_test_combined - y_pred_combined) / y_test_combined)) * 100
                    r2_combined = r2_score(y_test_combined, y_pred_combined)

                    st.write("**Combined Analysis Results:**")
                    st.write(f"Best Parameters: {grid_search_combined.best_params_}")
                    st.write(f"Mean Squared Error (MSE): {mse_combined}")
                    st.write(f"Root Mean Squared Error (RMSE): {rmse_combined}")
                    st.write(f"Mean Absolute Error (MAE): {mae_combined}")
                    st.write(f"Mean Absolute Percentage Error (MAPE): {mape_combined}%")
                    st.write(f"R-squared (R²): {r2_combined}")

                    # Feature importance
                    combined_feature_importances = pd.DataFrame({
                        "Feature": X_combined.columns,
                        "Importance": best_combined_model.feature_importances_
                    }).sort_values(by="Importance", ascending=False)
                    st.write("Feature Importances for Combined Analysis:", combined_feature_importances)

                    # Plot actual vs predicted
                    fig_combined, ax_combined = plt.subplots()
                    ax_combined.scatter(y_test_combined, y_pred_combined, alpha=0.5)
                    ax_combined.plot([y_test_combined.min(), y_test_combined.max()], [y_test_combined.min(), y_test_combined.max()], 'r--')
                    ax_combined.set_xlabel("Actual Values")
                    ax_combined.set_ylabel("Predicted Values")
                    ax_combined.set_title("Actual vs Predicted for Combined Analysis")
                    st.pyplot(fig_combined)

                    # Plot actual and predicted over time if 'quarter' column exists
                    if 'quarter' in data.columns:
                        fig_time_combined, ax_time_combined = plt.subplots()
                        full_quarters_combined = data['quarter']
                        ax_time_combined.plot(full_quarters_combined, y_combined, label="Actual", marker='o', alpha=0.5)
                        ax_time_combined.scatter(full_quarters_combined.loc[X_test_combined.index], y_pred_combined, label="Predicted", marker='x', color='red')
                        ax_time_combined.set_xlabel("Quarter")
                        ax_time_combined.set_ylabel("Stock Price")
                        ax_time_combined.set_title("Actual vs Predicted Stock Price for Combined Analysis")
                        ax_time_combined.legend()
                        plt.xticks(rotation=45)
                        st.pyplot(fig_time_combined)

                    # Store combined results
                    results_summary.append({
                        'Bank': 'Combined',
                        'MSE': mse_combined,
                        'RMSE': rmse_combined,
                        'MAE': mae_combined,
                        'MAPE': mape_combined,
                        'R2': r2_combined
                    })
                else:
                    st.warning("No data available for combined analysis.")

            # Display all results in a summary table
            if results_summary:
                results_table = pd.DataFrame(results_summary)
                st.write("**Summary of Results Across All Analyses:**")
                st.write(results_table)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add a footer for clarity
st.markdown("---")
