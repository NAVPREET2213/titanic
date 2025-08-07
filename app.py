import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Load Data
df = pd.read_csv("cleaned_titanic.csv")

# Show Data
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Sidebar Filters
st.sidebar.header("Filter Options")
gender = st.sidebar.selectbox("Select Gender", options=df["Sex"].unique())
pclass = st.sidebar.selectbox("Select Passenger Class", options=sorted(df["Pclass"].unique()))

# Apply filters
filtered_df = df[(df["Sex"] == gender) & (df["Pclass"] == pclass)]

st.subheader("Filtered Data Preview")
st.write(filtered_df.head())

# Visualization 1: Survival Count by Gender
st.subheader("🧍‍♂️ Survival Count by Gender")
fig1, ax1 = plt.subplots()
sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax1)
ax1.set_xticklabels(["Not Survived", "Survived"])
st.pyplot(fig1)

# Visualization 2: Age Distribution
st.subheader("📊 Age Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(data=filtered_df, x="Age", bins=30, kde=True, ax=ax2)
st.pyplot(fig2)

# Visualization 3: Survival Rate by Class
st.subheader("🎓 Survival Rate by Class")
fig3, ax3 = plt.subplots()
sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex", ax=ax3)
st.pyplot(fig3)

# Visualization 4: Fare Distribution by Class
st.subheader("💰 Fare Distribution by Passenger Class")
fig4, ax4 = plt.subplots()
sns.boxplot(data=df, x="Pclass", y="Fare", ax=ax4)
st.pyplot(fig4)

# Visualization 5: Heatmap of Correlation
st.subheader("🧠 Feature Correlation Heatmap")
numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Only numeric columns
fig5, ax5 = plt.subplots()
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax5)
st.pyplot(fig5)

# Footer
st.markdown("---")
st.markdown("**Made with ❤️ using Streamlit**")
