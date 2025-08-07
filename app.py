import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set dark theme
plt.style.use("dark_background")
sns.set_theme(style="darkgrid")

# Streamlit page config
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Dark background CSS
st.markdown("""
    <style>
        .main {
            background-color: #111111;
            color: white;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Load Data
df = pd.read_csv("cleaned_titanic.csv")

# Sidebar Filters
st.sidebar.header("🎚️ Filter Options")

gender_options = df["Sex"].dropna().unique()
gender = st.sidebar.multiselect("Select Gender", options=gender_options, default=gender_options)

pclass_options = sorted(df["Pclass"].dropna().unique())
pclass = st.sidebar.multiselect("Select Passenger Class", options=pclass_options, default=pclass_options)

embarked_options = df["Embarked"].dropna().unique()
embarked = st.sidebar.multiselect("Select Embarked Location", options=embarked_options, default=embarked_options)

min_age = int(df["Age"].min())
max_age = int(df["Age"].max())
age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

# Apply filters
filtered_df = df[
    (df["Sex"].isin(gender)) &
    (df["Pclass"].isin(pclass)) &
    (df["Embarked"].isin(embarked)) &
    (df["Age"].between(age_range[0], age_range[1]))
]

# Show data
if st.checkbox("📂 Show Filtered Raw Data"):
    st.dataframe(filtered_df)

st.subheader("📌 Filtered Data Preview")
st.write(filtered_df.head())

# ============ ROW 1 ============ #
st.markdown("### 🔍 Visual Analysis (Row 1)")
row1_col1, row1_col2, row1_col3 = st.columns([1, 1, 1])

with row1_col1:
    st.markdown("#### 🧍‍♂️ Survival Count by Gender")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax)
    ax.set_xticklabels(["Not Survived", "Survived"])
    ax.set_ylabel("Count")
    st.pyplot(fig)

with row1_col2:
    st.markdown("#### 📊 Age Distribution")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(data=filtered_df, x="Age", bins=30, kde=True, ax=ax)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

with row1_col3:
    st.markdown("#### 🎓 Survival Rate by Class")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=filtered_df, x="Pclass", y="Survived", hue="Sex", ax=ax)
    ax.set_ylabel("Survival Rate")
    st.pyplot(fig)

# ============ ROW 2 ============ #
st.markdown("### 📈 Visual Analysis (Row 2)")
row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])

with row2_col1:
    st.markdown("#### 💰 Fare Distribution by Class")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=filtered_df, x="Pclass", y="Fare", ax=ax)
    st.pyplot(fig)

with row2_col2:
    st.markdown("#### 🧠 Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric data available to show correlation heatmap.")

with row2_col3:
    st.markdown("#### 🚉 Embarked Passenger Count")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=filtered_df, x="Embarked", hue="Sex", ax=ax)
    ax.set_title("Embarked Location vs Gender")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**Made with ❤️ using Streamlit**", unsafe_allow_html=True)
