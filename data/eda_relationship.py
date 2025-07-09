import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def perform_eda(data_path):
    """
    Performs Exploratory Data Analysis on the cleaned dataset and displays plots.
    """
    # Load the dataset
    df = pd.read_csv(data_path)

    print("Exploratory Data Analysis")
    print("\nData Head:")
    print(df.head())

    print("\nData Info:")
    df.info()

    print("\nSummary Statistics:")
    print(df.describe())

    # --- Univariate Analysis ---
    print("\nGenerating distribution plots...")

    # Distribution of numerical features
    numerical_cols = ['Age', 'High_School_GPA', 'SAT_Score', 'University_GPA', 'Starting_Salary', 'Career_Satisfaction']
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

    # Count plots for categorical features
    categorical_cols = ['Gender', 'Field_of_Study', 'Current_Job_Level', 'Entrepreneurship', 'Interest_Aligned']
    for col in categorical_cols:
        plt.figure(figsize=(12, 7))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Count of {col}')
        plt.tight_layout()
        plt.show()

    # --- Bivariate and Multivariate Analysis ---
    print("\nGenerating relationship plots...")

    # Correlation heatmap
    plt.figure(figsize=(16, 12))
    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64', 'int8', 'int16', 'float32'])
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()

    # Field of Study vs. Starting Salary
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Starting_Salary', y='Field_of_Study', data=df)
    plt.title('Starting Salary by Field of Study')
    plt.tight_layout()
    plt.show()

    # University GPA vs. Starting Salary
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='University_GPA', y='Starting_Salary', data=df, hue='Field_of_Study', alpha=0.7)
    plt.title('University GPA vs. Starting Salary')
    plt.tight_layout()
    plt.show()

    # Interest Aligned vs. Career Satisfaction
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Interest_Aligned', y='Career_Satisfaction', data=df)
    plt.title('Career Satisfaction by Interest Alignment')
    plt.show()
    
    # Relationship between Field of Study and Interest
    plt.figure(figsize=(15, 20))
    sns.countplot(y='Field_of_Study', hue='Interest', data=df, dodge=False)
    plt.title('Interests within Each Field of Study')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

    # --- Advanced Visualizations ---
    print("\nGenerating advanced visualizations...")

    # Violin plot for Starting Salary by Field of Study
    plt.figure(figsize=(14, 9))
    sns.violinplot(x='Starting_Salary', y='Field_of_Study', data=df, inner='quartile', palette='muted')
    plt.title('Distribution of Starting Salary by Field of Study')
    plt.tight_layout()
    plt.show()

    # Pair plot for a subset of numerical features
    print("\nGenerating pair plot")
    pair_plot_cols = ['University_GPA', 'Starting_Salary', 'Career_Satisfaction', 'Field_of_Study']
    pair_plot = sns.pairplot(df[pair_plot_cols], hue='Field_of_Study', palette='viridis', diag_kind='kde')
    pair_plot.fig.suptitle('Pairwise Relationships of Key Features', y=1.02)
    plt.show()

    # FacetGrid for GPA vs Salary across different Fields of Study
    g = sns.FacetGrid(df, col="Field_of_Study", col_wrap=3, height=4, hue='Gender')
    g.map(sns.scatterplot, "University_GPA", "Starting_Salary", alpha=0.7)
    g.add_legend()
    g.fig.suptitle('GPA vs. Salary across Fields of Study and Gender', y=1.02)
    plt.show()


    print(f"\nEDA complete. Plots were displayed.")

if __name__ == "__main__":
    # The path to the cleaned data file
    cleaned_data_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'cleaned_education_data.csv')
    
    perform_eda(cleaned_data_path)
