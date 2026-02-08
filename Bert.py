import shutil
import torch
import transformers
from params import MODEL_NAME_OR_PATH, HIDDEN_LAYER_SIZE,BASE###
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn.functional as F

UNK, PAD = '[UNK]', '[PAD]'


def data_loader(train_list_tweet_source, pad_size=64):#输入是一个list
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    train_list_tweet_source = [str(tweet) for tweet in train_list_tweet_source if pd.notna(tweet)]
    if not train_list_tweet_source:
        return
    try:
        tokens = tokenizer(train_list_tweet_source,
                           max_length=pad_size,
                           padding='max_length',
                           truncation=True,
                           return_tensors='pt')
    except Exception as e:
        print(f"Error: {train_list_tweet_source}, Exception: {e}")
    return tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids']



class BERTClass(torch.nn.Module):
    def __init__(self,args=None,num_labels=4):
        super(BERTClass, self).__init__()
        self.args = args
        self.device = args.device
        self.bert = transformers.BertModel.from_pretrained(MODEL_NAME_OR_PATH, num_labels=num_labels)
        self.fc = torch.nn.Linear(HIDDEN_LAYER_SIZE, num_labels)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = output_1.pooler_output
        output_3 = self.fc(output_2)
        output = self.dropout(output_3)
        return output

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--device', type=str, default=device)
    args = parser.parse_args()

    df = pd.read_csv("/home/zf1/jcx/ai/test/archive (1)/emotions.csv", encoding="utf-8")
    content = df.loc[0:320000]['text'].tolist()
    sentiment = df.loc[0:320000]['label'].tolist()

    model = BERTClass(num_labels=6,args=args).to(device)

    ids,attention_masks,token_type_ids=data_loader(content, pad_size=32)
    print(
        f"ids length: {len(ids)}, attention_masks length: {len(attention_masks)}, token_type_ids length: {len(token_type_ids)}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch1 in range(19):
            labels = torch.tensor(sentiment[300000 + batch1 * 1000:301000 + batch1 * 1000]).to(device)
            inputs = torch.tensor(ids[300000 + batch1 * 1000:301000 + batch1 * 1000]).to(device)
            attention_masks_tst = torch.tensor(attention_masks[300000 + batch1 * 1000:301000 + batch1 * 1000]).to(device)
            token_type_ids_tst = torch.tensor(token_type_ids[300000 + batch1 * 1000:301000 + batch1 * 1000]).to(device)

            outputs = model(inputs, attention_masks_tst,token_type_ids_tst)
            probs = F.softmax(outputs, dim=1)  # 转换为概率
            _, predicted = torch.max(probs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f"Initial Accuracy:{accuracy}")

    for batch in range(600):
        model.train()
        labels = torch.tensor(sentiment[0 + batch * 500:500 + batch * 500]).to(device)
        inputs = torch.tensor(ids[0 + batch * 500:500 + batch * 500]).to(device)
        attention_masks_tst = torch.tensor(attention_masks[0 + batch * 500:500 + batch * 500]).to(device)
        token_type_ids_tst = torch.tensor(token_type_ids[0 + batch * 500:500 + batch * 500]).to(device)

        outputs = model(inputs, attention_masks_tst,token_type_ids_tst)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            model.eval()
            print(f"batch:{batch},loss:{loss}")
            correct = 0
            total = 0

            with torch.no_grad():
                for batch1 in range(19):
                    labels = torch.tensor(
                        sentiment[300000 + batch1 * 1000:301000 + batch1 * 1000]).to(
                        device)
                    inputs = torch.tensor(ids[300000 + batch1 * 1000:301000 + batch1 * 1000]).to(device)
                    attention_masks_tst = torch.tensor(attention_masks[300000 + batch1 * 1000:301000 + batch1 * 1000]).to(
                        device)
                    token_type_ids_tst = torch.tensor(token_type_ids[300000 + batch1 * 1000:301000 + batch1 * 1000]).to(
                        device)

                    outputs = model(inputs, attention_masks_tst,token_type_ids_tst)
                    probs = F.softmax(outputs, dim=1)  # 转换为概率
                    _, predicted = torch.max(probs, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f"Accuracy:{accuracy}")
