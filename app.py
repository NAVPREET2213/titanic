import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Apply dark theme to seaborn/matplotlib
plt.style.use("dark_background")
sns.set_theme(style="darkgrid")

# Page Config
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Custom CSS for dark background and white text
st.markdown("""
    <style>
        .main {
            background-color: #111111;
            color: white;
        }
        .block-container {
            padding-top: 2rem;
        }
        .st-bx {
            background-color: #222222 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Load Data
df = pd.read_csv("cleaned_titanic.csv")

# Sidebar Filters
st.sidebar.header("🎚️ Filter Options")

# Gender filter
gender_options = df["Sex"].dropna().unique()
gender = st.sidebar.multiselect("Select Gender", options=gender_options, default=gender_options)

# Pclass filter
pclass_options = sorted(df["Pclass"].dropna().unique())
pclass = st.sidebar.multiselect("Select Passenger Class", options=pclass_options, default=pclass_options)

# Embarked filter
embarked_options = df["Embarked"].dropna().unique()
embarked = st.sidebar.multiselect("Select Embarked Location", options=embarked_options, default=embarked_options)

# Age slider
min_age = int(df["Age"].min())
max_age = int(df["Age"].max())
age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

# Apply all filters
filtered_df = df[
    (df["Sex"].isin(gender)) &
    (df["Pclass"].isin(pclass)) &
    (df["Embarked"].isin(embarked)) &
    (df["Age"].between(age_range[0], age_range[1]))
]

# Raw data toggle
if st.checkbox("📂 Show Filtered Raw Data"):
    st.dataframe(filtered_df)

# Data preview
st.subheader("📌 Filtered Data Preview")
st.write(filtered_df.head())

# ROW 1 — 3 Charts
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🧍‍♂️ Survival Count by Gender")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax1)
    ax1.set_xticklabels(["Not Survived", "Survived"])
    st.pyplot(fig1)

with col2:
    st.markdown("### 📊 Age Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=filtered_df, x="Age", bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

with col3:
    st.markdown("### 🎓 Survival Rate by Class")
    fig3, ax3 = plt.subplots()
    sns.barplot(data=filtered_df, x="Pclass", y="Survived", hue="Sex", ax=ax3)
    st.pyplot(fig3)

# ROW 2 — 3 More Charts
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("### 💰 Fare Distribution by Class")
    fig4, ax4 = plt.subplots()
    sns.boxplot(data=filtered_df, x="Pclass", y="Fare", ax=ax4)
    st.pyplot(fig4)

with col5:
    st.markdown("### 🧠 Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        fig5, ax5 = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax5)
        st.pyplot(fig5)
    else:
        st.warning("No numeric data to display correlation heatmap.")

with col6:
    st.markdown("### 🚉 Embarked Passenger Count")
    fig6, ax6 = plt.subplots()
    sns.countplot(data=filtered_df, x="Embarked", hue="Sex", ax=ax6)
    ax6.set_title("Passengers by Embarked Location")
    st.pyplot(fig6)

# Footer
st.markdown("---")
st.markdown("**Made with ❤️ using Streamlit**", unsafe_allow_html=True)
