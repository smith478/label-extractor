import pandas as pd

df = pd.read_csv('embedding_model_results.csv')

# Get average position for each model
model_performance = df.groupby('model')['position'].mean().sort_values()
print("Average position by model:")
print(model_performance)

# Find examples where models disagree
disagreements = df.groupby(['report', 'label'])['position'].nunique() > 1
disagreeing_examples = df[df.set_index(['report', 'label']).index.isin(disagreements[disagreements].index)]
print("\nExamples where models disagree:")
print(disagreeing_examples)