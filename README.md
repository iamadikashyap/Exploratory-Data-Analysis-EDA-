# 🧭 Exploratory Data Analysis (EDA) - Titanic Dataset

## 📘 Objective
To extract insights from the Titanic dataset using **visual** and **statistical exploration** techniques.

---

## 🧰 Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn

---

## 📂 Files Included
- `train.csv` → Training dataset containing passenger information and survival labels.
- `test.csv` → Test dataset for prediction (without survival labels).
- `Titanic_EDA_Report.pdf` → Summary of findings in a clean report format.
- `Titanic_EDA.ipynb` → (Optional) Jupyter Notebook for all code and visuals.

---

## 🔍 Steps Performed
1. **Data Loading:** Imported train and test CSVs using Pandas.  
2. **Data Exploration:** Used `.info()`, `.describe()`, `.value_counts()` to understand dataset structure.  
3. **Missing Value Analysis:** Identified nulls in `Age`, `Cabin`, and `Embarked`.  
4. **Univariate Analysis:** Visualized single-column distributions using histograms and boxplots.  
5. **Bivariate Analysis:** Explored survival patterns across gender, class, age, and fare using Seaborn plots.  
6. **Correlation Heatmap:** Identified relationships between numerical variables.  
7. **Pairplot:** Checked pairwise relationships visually.  
8. **Insights & Summary:** Summarized all key findings in the report.

---

## 📊 Key Insights
- Females had a much higher survival rate than males.  
- Passengers in **1st class** had better survival odds.  
- Younger passengers were more likely to survive.  
- Higher fares correlated with increased survival chances.  
- Port 'C' (Cherbourg) passengers had slightly better survival rates.  

---

## 📈 Outcome
This project improves understanding of:
- Data cleaning and visualization techniques.  
- How to interpret distributions, correlations, and categorical relationships.  
- Building data intuition before machine learning modeling.

---

## 👨‍💻 Author
**Aditya Kashyap**  
LinkedIn: [linkedin.com/in/adikashyap](https://www.linkedin.com/in/adikashyap/)

