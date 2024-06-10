# Evaluate the model
results = trainer.evaluate()

print(f"Test Loss: {results['eval_loss']}")
print(f"Test Accuracy: {results['eval_accuracy']}")
