import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Load background image and convert to base64
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.75);  /* Dark overlay to improve readability */
        padding-top: 2rem;
        padding-bottom: 2rem;
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call background function
set_background("pic.jpg")

# Set dark theme for charts
plt.style.use("dark_background")
sns.set_theme(style="darkgrid")

# Streamlit config
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Load Titanic Data
df = pd.read_csv("cleaned_titanic.csv")

# Sidebar Filters
st.sidebar.header("🎚️ Filter Options")

gender = st.sidebar.multiselect("Select Gender", options=df["Sex"].dropna().unique(), default=df["Sex"].dropna().unique())
pclass = st.sidebar.multiselect("Select Passenger Class", options=sorted(df["Pclass"].dropna().unique()), default=sorted(df["Pclass"].dropna().unique()))
embarked = st.sidebar.multiselect("Select Embarked Location", options=df["Embarked"].dropna().unique(), default=df["Embarked"].dropna().unique())
age_range = st.sidebar.slider("Select Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
fare_range = st.sidebar.slider("Select Fare Range", float(df["Fare"].min()), float(df["Fare"].max()), (float(df["Fare"].min()), float(df["Fare"].max())))

# Apply Filters
filtered_df = df[
    (df["Sex"].isin(gender)) &
    (df["Pclass"].isin(pclass)) &
    (df["Embarked"].isin(embarked)) &
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["Fare"].between(fare_range[0], fare_range[1]))
]

# Optional Data Display
if st.checkbox("📂 Show Filtered Raw Data"):
    st.dataframe(filtered_df)

st.subheader("📌 Filtered Data Preview")
st.write(filtered_df.head())

# Row 1
st.markdown("### 📊 Visual Analysis (Row 1)")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🧍‍♂️ Survival Count by Gender")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax)
    ax.set_xticklabels(["Not Survived", "Survived"])
    st.pyplot(fig)

with col2:
    st.markdown("#### 📊 Age Distribution")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(data=filtered_df, x="Age", bins=30, kde=True, ax=ax)
    st.pyplot(fig)

with col3:
    st.markdown("#### 🎓 Survival Rate by Class")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=filtered_df, x="Pclass", y="Survived", hue="Sex", ax=ax)
    st.pyplot(fig)

# Row 2
st.markdown("### 📈 Visual Analysis (Row 2)")
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("#### 💰 Fare Distribution by Class")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=filtered_df, x="Pclass", y="Fare", ax=ax)
    st.pyplot(fig)

with col5:
    st.markdown("#### 🧠 Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric data available to show correlation heatmap.")

with col6:
    st.markdown("#### 🚉 Embarked Passenger Count")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=filtered_df, x="Embarked", hue="Sex", ax=ax)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**Made by Navpreet Singh(72211389)**", unsafe_allow_html=True)

