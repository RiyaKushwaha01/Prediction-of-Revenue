
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostRegressor

# Load trained model and preprocessors (mock for now)
# In production, replace with: pickle.load(open("model.pkl", "rb"))

# --- Login Section ---
def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "Riya" and password == "Riya@123":
            st.session_state["authenticated"] = True
        else:
            st.error("Invalid username or password")

# Session authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
else:
    
    # --- Revenue Prediction App ---
    st.title("Revenue Prediction App")
    st.header("Enter Feature Values")


    st.title("Revenue Prediction App")

    st.header("Enter Feature Values")

    # Input fields for each feature
    land_class = st.selectbox("Land Class", ["Federal", "Native American"])
    land_category = st.selectbox("Land Category", ["Onshore", "Offshore","Not Tied to a Lease"])
    state = st.selectbox("State", ["Texas", "Alaska", "California","Georgia","New York", "New Mexico", "Indiana","Florida","Washington"])  # example states
    revenue_type = st.selectbox("Royalties", ["Royalty", "Bonus", "Rent","Inspection fees","Civil penalties", "Other revenue"])
    lease_type = st.selectbox("Mineral Lease Type", ["Limestone", "Gold","Coal", "Silver","Oil & Gas","Sulfur","Gilsonite","Gypsum", "Sodium","Phosphate","Gemstones"])
    commodity = st.selectbox("Commodity", ["Oil", "Gas", "Coal", "Copper", "Hardrock", "Natural gas liquids", "Gilsonite", "Phosphate", "Oil & gas (pre-production)", "Geothermal"])
    county = st.selectbox("County", ["Carbon", "Eddy", "Sweet Water","Bannock", "Goshen", "Iron","Cleveland", "Franklin", "Washington", "Chambers"])  # example counties
    product = st.selectbox("Product", ["Nitrogen", "Oil", "Coal Bed Methane", "Coal", "Gas Plant Products", "Calcium Oxide", "Carbon Dioxide Gas (CO2)"
                                  ,"Fuel Gas", "Fuel Oil", "Helium"])

    # Collect input in a DataFrame
    input_data = pd.DataFrame([{
        "Land Class": land_class,
        "Land Category": land_category,
        "State": state,
        "Revenue Type": revenue_type,
        "Mineral Lease Type": lease_type,
        "Commodity": commodity,
        "County": county,
        "Product": product
    }])

    # Dummy prediction function for CatBoostRegressor
    def dummy_catboost_predict(df):
        # Define and train a dummy CatBoost model
        model = CatBoostRegressor(verbose=0)
    
    # Save the Trained Model 
    model.save_model("catboost_model.cbm")

    # Load the Model for Prediction:
    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm")

    # Example usage with Streamlit
    if st.button("Predict Revenue"):
    prediction = model.predict(input_data)  # input_data must match X's structure
    st.success(f"Estimated Revenue: ${prediction[0]:,.2f}")
    
    st.markdown("""
    <hr>
    <small>Developed with ❤️ using Streamlit</small>
    """, unsafe_allow_html=True)
