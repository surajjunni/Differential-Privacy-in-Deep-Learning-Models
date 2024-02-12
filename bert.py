import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import glue_convert_examples_to_features
from transformers import InputExample, InputFeatures
from transformers.data.processors.glue import glue_processors, glue_output_modes
from datasets import load_dataset
from opacus import PrivacyEngine

print("Load the dataset and tokenizer")
task = "mnli"
processor = glue_processors[task]()
output_mode = glue_output_modes[task]
label_list = processor.get_labels()
train_dataset = load_dataset("glue", task, split="train")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Move the model and optimizer to the device")

# Define the model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(label_list), output_attentions=False, output_hidden_states=False
)

print("Define hyperparameters")
batch_size = 32
epsilon = 1.0
delta = 1e-5

print("Convert examples to features")
train_examples = train_dataset["train"]
train_features = glue_convert_examples_to_features(train_examples, tokenizer, max_length=128, task=task)
train_labels = [f.label for f in train_features]
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor([f.input_ids for f in train_features], dtype=torch.long),
    torch.tensor([f.attention_mask for f in train_features], dtype=torch.long),
    torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long),
    torch.tensor(train_labels, dtype=torch.long),
)

privacy_engine = PrivacyEngine()
model.to(device)

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=MAX_GRAD_NORM,
)
print("Define loss function and optimizer")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print("Train the model")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for epoch in range(50):
    for step, batch in enumerate(train_loader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        labels = batch[3].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

print("Save the model")
torch.save(model.state_dict(), "bert_model.pt")

