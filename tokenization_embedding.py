import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
import tqdm

def tokenize_embed(result: pd.DataFrame, BATCH_SIZE: int=32):
    # Load model and config
    config = AutoConfig.from_pretrained("karina-zadorozhny/ume", trust_remote_code=True)
    model = AutoModel.from_pretrained("karina-zadorozhny/ume", trust_remote_code=True, config=config)

    # Load SMILES tokenizer
    tokenizer_smiles = AutoTokenizer.from_pretrained(
        "karina-zadorozhny/ume",
        subfolder="tokenizer_smiles",
        trust_remote_code=True
    )

    #Tokenizing
    # smiles_can_list = result['smiles_can'].tolist()

    #tokenize and embed
    # inputs = tokenizer_smiles(smiles_can_list, return_tensors="pt", padding=True, truncation=True)

    smiles_can_list = result['smiles_can'].tolist()

    all_embeddings = []

    for i in tqdm(range(0, len(smiles_can_list), BATCH_SIZE)):
        batch_smiles = smiles_can_list[i:i+BATCH_SIZE]

        # tokenize the batch only
        inputs = tokenizer_smiles(batch_smiles, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            # move tensors to device if using GPU
            input_ids = inputs["input_ids"].unsqueeze(1)
            attention_mask = inputs["attention_mask"].unsqueeze(1)

            batch_embeddings = model(input_ids, attention_mask)
            # If the model returns a tuple (last_hidden_state, pooler_output)
            if isinstance(batch_embeddings, tuple):
                batch_embeddings = batch_embeddings[0]  # take first element
            all_embeddings.append(batch_embeddings.cpu())

        if i % 500 == 0:
            print(f'completed {(i/len(smiles_can_list)*100):.2f}% of embeddings')

    # concatenate all batches
    embeddings = torch.cat(all_embeddings, dim=0)
    #print(embeddings.shape)
    np.savetxt('embeddings.csv', embeddings, delimiter=',')
    print('embeddings saved successfully!')
