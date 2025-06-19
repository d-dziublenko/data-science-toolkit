# Quick Start Guide

Welcome to the Universal Data Science Toolkit! This guide will help you get started with the toolkit in just a few minutes. Think of this as your first hands-on tutorial where we'll walk through the basics together, step by step.

## 🎯 What You'll Learn

By the end of this guide, you'll understand how to:

- Set up the toolkit on your machine
- Load and explore your data
- Build your first machine learning model
- Evaluate model performance
- Make predictions on new data

Let's begin this journey together!

## 🚀 Installation (5 minutes)

First, let's get the toolkit installed on your computer. We'll start with the simplest approach and explain what's happening at each step.

### Step 1: Get the Code

Open your terminal and run these commands:

```bash
# Download the toolkit to your computer
git clone https://github.com/d-dziublenko/data-science-toolkit.git

# Navigate into the project folder
cd data-science-toolkit
```

### Step 2: Run the Installer

We've created an installation script that handles all the complex setup for you:

```bash
# Make the installer executable (Mac/Linux)
chmod +x install.sh

# Run the installer
./install.sh
```

When you run the installer, you'll see a menu with three options. Here's what each one means:

1. **Basic (Recommended for beginners)**: Installs only the essential components. This is perfect if you're just starting out or have limited disk space.

2. **Full**: Installs everything, including deep learning libraries. Choose this if you plan to work with neural networks or want all features available.

3. **Development**: Includes everything plus tools for contributing to the project. Select this if you want to modify the toolkit itself.

For now, choose option 1 (Basic) by typing `1` and pressing Enter.

### Step 3: Verify Installation

After installation completes, let's make sure everything is working:

```bash
# Activate the virtual environment
source data_science_env/bin/activate  # On Windows: data_science_env\Scripts\activate

# Run the test script
python test_installation.py
```

You should see green checkmarks (✓) next to each component. If you see any red X marks, don't worry - those are optional components that you can install later if needed.

## 🎓 Your First Machine Learning Project (10 minutes)

Now comes the exciting part - let's build your first machine learning model! We'll predict house prices using a simple dataset.

### Understanding the Workflow

Before we dive into code, let's understand what we're about to do. Think of machine learning as teaching a computer to make predictions based on examples:

1. **Load Data**: First, we need examples (houses with known prices)
2. **Prepare Data**: Clean and organize the examples
3. **Train Model**: Let the computer learn patterns from the examples
4. **Evaluate**: Check how well it learned
5. **Predict**: Use the model to predict prices for new houses

### Step-by-Step Implementation

Create a new file called `my_first_model.py` and let's build this together:

```python
# Import the toolkit - this gives us access to all the tools we need
from data_science_toolkit import TrainingPipeline

# Step 1: Create a pipeline
# Think of a pipeline as an assembly line that handles all the ML steps
pipeline = TrainingPipeline(
    task_type='regression',  # We're predicting numbers (prices)
    experiment_name='house_prices'  # Give your project a name
)

# Step 2: Run the complete pipeline
# This single command does a lot of work for us:
# - Loads the data
# - Splits it into training and testing sets
# - Preprocesses the data (handles missing values, etc.)
# - Trains multiple models
# - Finds the best one
# - Evaluates performance
results = pipeline.run(
    data_path='examples/data/house_prices.csv',  # Path to our data
    target_column='price',  # What we want to predict
    test_size=0.2,  # Use 20% of data for testing
    models=['linear_regression', 'random_forest'],  # Models to try
    tune_hyperparameters=True  # Automatically find best settings
)

# Step 3: View the results
print("\n🎉 Congratulations! You've trained your first model!")
print(f"Best model: {results['best_model']}")
print(f"Accuracy (R² score): {results['test_metrics']['r2']:.2%}")
print(f"Average prediction error: ${results['test_metrics']['mae']:,.2f}")

# Step 4: Make a prediction
# Let's predict the price of a new house
new_house = {
    'bedrooms': 3,
    'bathrooms': 2,
    'sqft': 1500,
    'location': 'suburban'
}

predicted_price = pipeline.predict_single(new_house)
print(f"\nPredicted price for the new house: ${predicted_price:,.2f}")
```

Run your script:

```bash
python my_first_model.py
```

### What Just Happened?

Let me explain what the toolkit did behind the scenes:

1. **Data Loading**: It read the CSV file and automatically detected data types
2. **Data Splitting**: It separated the data into training (80%) and testing (20%) sets
3. **Preprocessing**: It handled missing values, encoded categorical variables, and scaled numerical features
4. **Model Training**: It trained both Linear Regression and Random Forest models
5. **Hyperparameter Tuning**: It tested different settings to find the best configuration
6. **Model Selection**: It chose the model with the best performance
7. **Evaluation**: It calculated various metrics to measure accuracy

All of this happened with just a few lines of code!

## 🔍 Exploring Your Data (5 minutes)

Understanding your data is crucial for building good models. Let's learn how to explore data using the toolkit:

```python
from data_science_toolkit.core import TabularDataLoader, DataProfiler

# Load your data
loader = TabularDataLoader()
data = loader.load('examples/data/house_prices.csv')

# Create a data profile
profiler = DataProfiler()
profile = profiler.profile(data)

# Display key insights
print("📊 Data Overview")
print(f"Number of houses: {profile['basic_info']['n_rows']:,}")
print(f"Number of features: {profile['basic_info']['n_columns']}")
print(f"Missing values: {profile['missing_values']['total_missing']:,}")
print(f"Memory usage: {profile['basic_info']['memory_usage']}")

# Check for data quality issues
print("\n🔍 Data Quality Check")
if profile['missing_values']['total_missing'] > 0:
    print("⚠️  Found missing values in:")
    for col, count in profile['missing_values']['by_column'].items():
        if count > 0:
            print(f"   - {col}: {count} missing ({count/len(data)*100:.1f}%)")

# View feature statistics
print("\n📈 Numerical Features Summary")
for feature, stats in profile['numerical_features'].items():
    print(f"\n{feature}:")
    print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
    print(f"  Average: {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
```

## 🎯 Different Types of Problems

The toolkit can handle various types of machine learning problems. Let's understand the main categories:

### 1. Regression (Predicting Numbers)

Use this when you want to predict continuous values like:

- House prices
- Temperature
- Sales revenue
- Stock prices

```python
pipeline = TrainingPipeline(task_type='regression')
```

### 2. Classification (Predicting Categories)

Use this when you want to predict discrete categories like:

- Customer will buy (Yes/No)
- Email is spam (Spam/Not Spam)
- Image contains (Cat/Dog/Bird)

```python
pipeline = TrainingPipeline(task_type='classification')
```

### 3. Time Series (Predicting Future Values)

Use this for data that changes over time:

- Daily sales forecasting
- Weather prediction
- Stock market analysis

```python
pipeline = TrainingPipeline(
    task_type='regression',
    time_series=True
)
```

## 🛠️ Customizing Your Pipeline

As you become more comfortable, you might want more control over the process. Here's how to customize the pipeline:

```python
from data_science_toolkit import TrainingPipeline, TrainingConfig

# Create a custom configuration
config = TrainingConfig(
    task_type='classification',
    models=['random_forest', 'xgboost', 'lightgbm'],  # Specific models
    ensemble=True,  # Combine models for better accuracy
    handle_imbalance=True,  # Handle unequal class distribution
    n_trials=100,  # Number of hyperparameter combinations to try
    cv_folds=10,  # Cross-validation folds
)

# Use the custom configuration
pipeline = TrainingPipeline(config)
```

## 📊 Understanding Model Performance

After training, you'll see various metrics. Here's what they mean:

### For Regression:

- **R² Score**: How well the model fits (1.0 = perfect, 0.0 = random)
- **RMSE**: Average prediction error (lower is better)
- **MAE**: Average absolute error (easier to interpret)

### For Classification:

- **Accuracy**: Percentage of correct predictions
- **Precision**: When model says "Yes", how often is it right?
- **Recall**: Of all actual "Yes" cases, how many did it find?
- **F1 Score**: Balance between precision and recall

## 🚧 Troubleshooting Common Issues

Here are solutions to common problems you might encounter:

### Issue 1: "Module not found" Error

**Solution**: Make sure your virtual environment is activated:

```bash
source data_science_env/bin/activate
```

### Issue 2: "Out of Memory" Error

**Solution**: Reduce the data size or use sampling:

```python
# Load only a sample
data = loader.load('large_file.csv', sample_size=10000)
```

### Issue 3: Poor Model Performance

**Solution**: Try these approaches:

```python
# 1. Use more models
models=['random_forest', 'xgboost', 'lightgbm', 'catboost']

# 2. Enable feature engineering
pipeline.enable_feature_engineering(
    polynomial_features=True,
    interaction_features=True
)

# 3. Increase hyperparameter tuning
config.n_trials = 200  # Try more combinations
```

## 🎓 Next Steps

Congratulations on completing the quick start guide! You've learned the fundamentals of using the Universal Data Science Toolkit. Here's what to explore next:

1. **Read the Examples**: Check out the `examples/` folder for more complex scenarios
2. **Explore Advanced Features**: Try deep learning models, custom preprocessing, or ensemble methods
3. **Join the Community**: Share your projects and get help from other users
4. **Contribute**: Found a bug or have an idea? See CONTRIBUTING.md

Remember, machine learning is a journey of continuous learning. Each dataset brings new challenges and opportunities to grow. Keep experimenting, and don't hesitate to ask questions!

## 💡 Pro Tips

Before you go, here are some professional tips to accelerate your learning:

1. **Start Simple**: Always begin with simple models before trying complex ones
2. **Understand Your Data**: Spend time exploring before modeling
3. **Validate Results**: Always check if results make business sense
4. **Iterate Quickly**: Try many approaches rapidly rather than perfecting one
5. **Document Everything**: Keep notes on what works and what doesn't

Happy modeling! 🚀
