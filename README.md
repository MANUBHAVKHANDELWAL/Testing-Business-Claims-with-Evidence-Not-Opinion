# 📊 Testing Business Claims with Evidence, Not Opinion

### Python-Based Statistical Analysis and Sales Prediction

**Project 3 | Data Analytics | Statistics | Python**

## 📌 Project Overview

RetailStat AI is a Python-based data analytics project that explores retail sales data using statistical methods, data visualization, and regression analysis.

The main aim of this project is to understand customer purchasing patterns, compare sales performance across different regions, study relationships between business variables, and evaluate business claims using statistical evidence.

The project uses a simulated retail sales dataset and demonstrates how statistical analysis can support better business decisions.

## 🎯 Problem Statement

Retail businesses generate large amounts of sales and customer data. However, having data alone is not enough to make informed decisions.

Businesses often face questions such as:

- Do sales differ significantly between regions?
- Is there a relationship between discounts and profit?
- Are customer segments associated with product preferences?
- Can sales be predicted using other available variables?

This project addresses these questions using Python and statistical techniques.

Instead of relying on assumptions, the project uses data analysis, hypothesis testing, and regression models to examine these business claims.

## 🛠️ Technologies and Libraries

**Programming Language:** Python

**Development Environment:** Visual Studio Code / Jupyter Notebook

**Libraries:**

- **Pandas:** Data cleaning and manipulation
- **NumPy:** Numerical operations
- **Matplotlib:** Graphs and visualizations
- **Seaborn:** Statistical visualizations
- **SciPy:** Hypothesis testing
- **Statsmodels:** Statistical modeling and regression
- **Scikit-learn:** Model evaluation

## 📂 Dataset Information

The project uses the provided `retail_customer_sales_dataset.csv` dataset.

- Original records: 7,084
- Unique transactions: 7,000
- Data type: Simulated retail customer and sales records
- Analysis approach: Full available dataset after cleaning, without random sampling

The dataset contains customer, transaction, product, regional, and financial information.

Data cleaning includes handling duplicate transactions, inconsistent values, and invalid entries.

## 🔍 Project Methodology

### 1. Data Cleaning and Preparation

The dataset is examined for missing values, duplicates, inconsistent categories, and invalid records. Necessary cleaning operations are performed before the statistical analysis.

### 2. Descriptive Statistical Analysis

Statistical measures are calculated to understand the distribution and characteristics of the data.

These include:

- Mean and median
- Standard deviation and variance
- Minimum and maximum
- Percentiles
- Confidence intervals

### 3. Hypothesis Testing

Four statistical tests are performed to evaluate business-related questions.

- **Independent t-test:** Comparison of sales between North and South regions
- **ANOVA:** Comparison of sales across four regions
- **Chi-square test:** Association between customer segments and product categories
- **Pearson correlation:** Relationship between discount and profit

Each test includes a hypothesis, calculated statistical results, and an interpretation.

### 4. Regression Analysis

Two regression approaches are used to study and predict sales:

**Simple Linear Regression**

Examines the relationship between sales and an individual predictor.

**Multiple Linear Regression**

Examines sales using multiple explanatory variables.

Regression results are evaluated using statistical measures and model diagnostics.

### 5. Data Visualization

The project includes 12 statistical visualizations covering:

**Univariate Analysis**
- Histograms
- Box plots

**Bivariate Analysis**
- Scatter plots
- Regression plots
- Regional comparisons

**Multivariate Analysis**
- Correlation heatmaps
- Stacked charts
- Prediction comparisons
- Residual diagnostics

A consistent visual theme is maintained across the graphs for readability and presentation.

## 📈 Key Findings

The statistical analysis highlights several findings:

- The North and South regions do not show a statistically significant difference in sales at the 5% significance level.
- The ANOVA test does not identify a statistically significant difference in sales across the four regions.
- The chi-square test does not show a statistically significant association between customer segments and product categories.
- Discount and profit have a statistically significant but weak negative correlation.

These results demonstrate that business assumptions should be evaluated using evidence rather than accepted without testing.

The regression analysis also helps explain relationships between sales and selected variables, although these relationships should not be interpreted as proof of causation.

## 📁 Project Structure

```text
Testing Business Claims with Evidence, Not Opinion/
│
├── analysis.py
├── data/
│   ├── retail_customer_sales_dataset.csv
│   └── retail_sales_cleaned.csv
│
├── outputs/
│   ├── statistical_results.txt
│   ├── results.json
│   ├── descriptive_statistics.csv
│   ├── segment_category_contingency.csv
│   └── charts/
│       └── Generated statistical graphs


## ▶️ How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RetailStat-AI.git
cd RetailStat-AI
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### Step 2: Install dependencies

Make sure Python is installed.

```bash
python -m pip install -r requirements.txt
```

### Step 3: Execute the analysis

```bash
python analysis.py
```

### Step 4: View the results

After execution, the generated statistical results and graph images can be found in the `outputs` directory.

### Step 5: Run test cases

```bash
python -m unittest discover -s tests -v
```

The project also includes a Jupyter Notebook that can be opened in VS Code or Jupyter Notebook.

## ✅ Project Testing

Automated test cases are included to check the reliability of the analysis workflow.

Testing focuses on:

- Data cleaning and validation
- Statistical calculation consistency
- Regression output generation
- Expected project outputs

The tests help confirm that the program behaves as expected.

## 🌍 Real-World Relevance

Retail businesses regularly analyze sales performance, customer preferences, regional differences, and discount strategies.

The problems explored in this project are relevant to real business analytics.

However, the supplied dataset is simulated. Therefore, the results demonstrate statistical techniques and should not be presented as findings from a real retail organization.

This distinction ensures that the conclusions remain accurate and transparent.

## 🚀 Future Improvements

Possible improvements include:

- Developing an interactive analytics dashboard
- Adding machine learning models for sales forecasting
- Using real-world retail datasets
- Integrating automated reporting
- Comparing additional predictive models
- Deploying the project as a web-based analytics application

## 🎓 Learning Outcomes

Through this project, I explored practical applications of:

- Data preprocessing
- Descriptive statistics
- Hypothesis testing
- Statistical visualization
- Regression modeling
- Model evaluation
- Python-based data analysis

The project provides a foundation for further learning in Data Science, Machine Learning, and AI Engineering.

## 📄 Project Information

**Project Name:** RetailStat AI

**Project Category:** Advanced Statistics and Data Analytics

**Language:** Python

**Academic Program:** Master of Computer Applications (MCA)

**Purpose:** Academic project and practical skill development

---
