import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset

'''
Script to test Fine-Tuned Bert Models' Performance After Fine-Tuning 
'''

# Load and assert the correctness of the dataset.
def load_dataset(filepath, country_code):
    data = pd.read_csv(filepath, sep='\t')
    data = data[data['id'].str.startswith(country_code)]
    return data

# Separate dataset into training and test sets.
def preprocess_dataset(data, text_column, label_column):
    data = data.dropna(subset=[text_column, label_column])
    train_data, test_data = train_test_split(data, stratify=data[label_column], test_size=0.1, random_state=42)
    return train_data, test_data

# Tokenize data.
def tokenize_test_data(data, tokenizer, text_column):
    encodings = tokenizer(data[text_column].tolist(), padding=True, truncation=True, max_length=512)
    dataset = Dataset.from_dict({'input_ids': encodings['input_ids'],
                                 'attention_mask': encodings['attention_mask'],
                                 'labels': data['label'].tolist()})
    return dataset

# Evaluate the fine-tuned model after training.
def evaluate_model(model, tokenizer, test_data, text_column):
    test_dataset = tokenize_test_data(test_data, tokenizer, text_column)

    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(test_dataset)

    preds = predictions.predictions.argmax(axis=-1)
    labels = test_data['label'].tolist()

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')

    report = classification_report(labels, preds, target_names=['Class 0', 'Class 1'], output_dict=True)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': pd.DataFrame(report).transpose()
    }

# Main script to handle the testing flow.
if __name__ == "__main__":
    ideology_train_path = "trainingset-ideology-power/orientation/orientation-tr-train.tsv"
    power_train_path = "trainingset-ideology-power/power/power-tr-train.tsv"
    country_code = "tr"

    print("Loading the Datasets...")
    ideology_data = load_dataset(ideology_train_path, country_code)
    power_data = load_dataset(power_train_path, country_code)

    print("Splitting into Training and Testing Sets...")
    _, ideology_test_data = preprocess_dataset(ideology_data, text_column='text_en', label_column='label')
    _, power_test_data = preprocess_dataset(power_data, text_column='text_en', label_column='label')

    ideology_model_path = "./fine_tuned_model"
    power_model_path = "./fine_tuned_model_power"

    print("Loading Models...")
    ideology_tokenizer, ideology_model = AutoTokenizer.from_pretrained(ideology_model_path), AutoModelForSequenceClassification.from_pretrained(ideology_model_path)
    power_tokenizer, power_model = AutoTokenizer.from_pretrained(power_model_path), AutoModelForSequenceClassification.from_pretrained(power_model_path)

    print("Evaluating Models...")
    ideology_metrics = evaluate_model(ideology_model, ideology_tokenizer, ideology_test_data, text_column='text_en')
    power_metrics = evaluate_model(power_model, power_tokenizer, power_test_data, text_column='text_en')

    print("\n--- Ideology Classification Results ---")
    print(f"Accuracy: {ideology_metrics['accuracy']:.2f}")
    print(f"Precision: {ideology_metrics['precision']:.2f}")
    print(f"Recall: {ideology_metrics['recall']:.2f}")
    print(f"F1 Score: {ideology_metrics['f1_score']:.2f}")
    print("\nDetailed Classification Report (Ideology):")
    print(ideology_metrics['classification_report'])

    print("\n--- Power Classification Results ---")
    print(f"Accuracy: {power_metrics['accuracy']:.2f}")
    print(f"Precision: {power_metrics['precision']:.2f}")
    print(f"Recall: {power_metrics['recall']:.2f}")
    print(f"F1 Score: {power_metrics['f1_score']:.2f}")
    print("\nDetailed Classification Report (Power):")
    print(power_metrics['classification_report'])

    ideology_metrics['classification_report'].to_csv("ideology_classification_report.csv", index=True)
    power_metrics['classification_report'].to_csv("power_classification_report.csv", index=True)
    print("\nReports exported as 'ideology_classification_report.csv' and 'power_classification_report.csv'")
