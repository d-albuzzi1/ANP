import os
import numpy as np
import torch
import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from recbole.model.general_recommender.simplex import SimpleX
from recbole.data.interaction import Interaction
from tqdm import tqdm
import logging

# Disabilita i messaggi di logging meno importanti
logging.getLogger().setLevel(logging.ERROR)

# Percorso al modello salvato
MODEL_PATH = None

# Configurazione
config_dict = {
    'model': 'SimpleX',
    'dataset': 'academic_dataset',
    'data_path': '/home/albuzzi/academic_network_project/anp_nn/anp_data/data_path',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'field_separator': '\t',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'TIME_FIELD': 'year',
    'USER_FEATURE_FIELD': ['num_papers'],
    'ITEM_FEATURE_FIELD': ['num_citations', 'year', 'journal_id', 'topic_id'],
    'load_col': {
        'inter': ['user_id', 'item_id', 'interaction_type', 'year'],
        'user': ['user_id', 'num_papers'],
        'item': ['item_id', 'num_citations','year', 'journal_id', 'topic_id']
    },
    'field_type': {
        'user_id': 'token',
        'item_id': 'token',
        'year': 'float',
        'num_papers': 'float',
        'num_citations': 'float',
        'journal_id': 'token',
        'topic_id': 'token',
        'interaction_type': 'token'
    },
    'log_level': 'ERROR',
}

config = Config(model='SimpleX', dataset='academic_dataset', config_dict=config_dict)
init_seed(config['seed'], config['reproducibility'])

dataset = create_dataset(config)

train_data, valid_data, test_data = data_preparation(config, dataset)

# Inizializza il modello
model = SimpleX(config, train_data.dataset).to(config['device'])

if MODEL_PATH and os.path.exists(MODEL_PATH):
    print(f"Caricamento del modello dal percorso: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=config['device'])
    model_state_dict = checkpoint['state_dict']
    model.load_state_dict(model_state_dict)
else:
    print("Nessun modello salvato trovato, si procede senza il caricamento di un checkpoint.")

model.eval()


user_ids = list(dataset.field2token_id[config['USER_ID_FIELD']].values())
item_ids = list(dataset.field2token_id[config['ITEM_ID_FIELD']].values())

# Filtra eventuali ID non validi (ID 0)
user_ids = [uid for uid in user_ids if uid != 0]
item_ids = [iid for iid in item_ids if iid != 0]

# Carica le informazioni sugli articoli (citazioni, anno e journal_id)
item_features = dataset.get_item_feature()
item_ids_array = item_features['item_id'].numpy()
num_citations = item_features['num_citations'].numpy()
years = item_features['year'].numpy()
journal_ids = item_features['journal_id'].numpy()

items_df = pd.DataFrame({
    'item_id': item_ids_array,
    'num_citations': num_citations,
    'year': years,
    'journal_id': journal_ids
}).set_index('item_id')


publication_df = pd.read_csv('/home/albuzzi/academic_network_project/anp_data/raw/publication.csv')
publication_df = publication_df.rename(columns={'paper_id': 'item_id', 'journal_id': 'pub_journal_id'})

# Ensure item_id is of the same type
items_df = items_df.reset_index()
items_df['item_id'] = items_df['item_id'].astype(int)
publication_df['item_id'] = publication_df['item_id'].astype(int)

# Merge without overwriting journal_id
items_df = pd.merge(items_df, publication_df[['item_id', 'pub_journal_id']], on='item_id', how='left')
items_df = items_df.set_index('item_id')


items_df['journal_id'] = items_df['journal_id'].fillna(items_df['pub_journal_id'])

missing_journal_ids = items_df['journal_id'].isnull().sum()
if missing_journal_ids > 0:
    print(f"Warning: {missing_journal_ids} items have missing journal_id.")


journal_counts = items_df.groupby('journal_id').size().reset_index(name='paper_count')

# Ordina i journal_id per il numero di articoli e prendi le top 10 venue (journal_id)
top_journals = journal_counts.sort_values(by='paper_count', ascending=False).head(10)


top_journal_ids = set(top_journals['journal_id'])
# Aggiungi una colonna 'is_top_journal' per indicare se un articolo proviene da una top venue
items_df['is_top_journal'] = items_df['journal_id'].apply(lambda x: 1 if x in top_journal_ids else 0)


num_users_to_predict = 100  
user_ids_to_predict = user_ids[:num_users_to_predict]

# Parametri per il batch processing
batch_size = 10  
top_n = 10       

all_user_ids = []
all_item_ids = []
all_scores = []

# Processa gli utenti in batch
print("Inizio delle predizioni...")
for i in tqdm(range(0, len(user_ids_to_predict), batch_size)):
    user_ids_batch = user_ids_to_predict[i:i+batch_size]
    interaction = Interaction({config['USER_ID_FIELD']: torch.tensor(user_ids_batch)})
    interaction = interaction.to(config['device'])
    
    with torch.no_grad():
        scores = model.full_sort_predict(interaction)
        scores = scores.cpu().numpy()
        num_users_in_batch = len(user_ids_batch)
        scores = scores.reshape(num_users_in_batch, -1)
    
    # Per ogni utente nel batch
    for idx, user_id in enumerate(user_ids_batch):
        user_scores = scores[idx]
        
        user_items_df = items_df.copy()
        user_items_df['score'] = user_scores[:len(items_df)]
        
        # Filtra gli articoli che non provengono dai top journal
        user_items_df = user_items_df[user_items_df['is_top_journal'] == 1]
        
        # Se non ci sono articoli dai top journal, salta l'utente
        if user_items_df.empty:
            continue
        
        # Ordina gli articoli per punteggio in ordine decrescente
        user_items_df = user_items_df.sort_values(by='score', ascending=False)
        
        # Prendi i top N articoli
        top_items = user_items_df.head(top_n)
        
        all_user_ids.extend([user_id] * len(top_items))
        all_item_ids.extend(top_items.index.values)
        all_scores.extend(top_items['score'].values)

# Creare un DataFrame con i risultati
results_df = pd.DataFrame({
    'user_id': all_user_ids,
    'item_id': all_item_ids,
    'score': all_scores
})


results_df['user_id_token'] = dataset.id2token(config['USER_ID_FIELD'], results_df['user_id'])
results_df['item_id_token'] = dataset.id2token(config['ITEM_ID_FIELD'], results_df['item_id'])

# Ordina i risultati per utente e punteggio
results_df = results_df.sort_values(by=['user_id', 'score'], ascending=[True, False])

# Salviamo i risultati in un file CSV
output_file = 'reccomended_paper.csv'
results_df.to_csv(output_file, index=False)

print(f"Risultati salvati nel file {output_file}")