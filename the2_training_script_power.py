import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

'''
Script to Fine-Tune a BERT Multilingual Model for Politician State of Power Classification
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
def tokenize_data(data, tokenizer, text_column):
    return tokenizer(data[text_column].tolist(), padding=True, truncation=True, max_length=512)

# Compute evaluation metrics.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Finetuning setup.
def setup_trainer(model_name, train_dataset, eval_dataset, label_column):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenize_data(train_dataset, tokenizer, text_column='text_en')
    eval_encodings = tokenize_data(eval_dataset, tokenizer, text_column='text_en')

    train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'],
                                       'attention_mask': train_encodings['attention_mask'],
                                       'labels': train_dataset[label_column].tolist()})
    eval_dataset = Dataset.from_dict({'input_ids': eval_encodings['input_ids'],
                                      'attention_mask': eval_encodings['attention_mask'],
                                      'labels': eval_dataset[label_column].tolist()})

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results_power",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs_power',
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer

# Main script to handle the fine-tuning flow.
if __name__ == "__main__":
    dataset_path = "trainingset-ideology-power/power/power-tr-train.tsv"
    country_code = "tr"

    print("Starting preprocessing...")
    data = load_dataset(dataset_path, country_code)
    train_data, test_data = preprocess_dataset(data, text_column='text_en', label_column='label')

    model_name = "bert-base-multilingual-cased"
    print("Setting up the trainer...")
    trainer = setup_trainer(model_name, train_data, test_data, label_column='label')

    print("Starting training...")
    trainer.train()

    trainer.save_model("./fine_tuned_model_power")
    print("Model saved.")

    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
