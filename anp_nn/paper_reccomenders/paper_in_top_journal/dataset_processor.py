# dataset_generation.py

import pandas as pd
import os

about_csv = '/home/albuzzi/academic_network_project/anp_data/raw/about.csv'
cites_csv = '/home/albuzzi/academic_network_project/anp_data/raw/cites.csv'
papers_csv = '/home/albuzzi/academic_network_project/anp_data/raw/papers.csv'
publication_csv = '/home/albuzzi/academic_network_project/anp_data/raw/publication.csv'
authors_topic_2019_csv = '/home/albuzzi/academic_network_project/anp_data/raw/sorted_authors_topics_2019.csv'
papers_about_csv = '/home/albuzzi/academic_network_project/anp_data/raw/sorted_papers_about.csv'
writes_csv = '/home/albuzzi/academic_network_project/anp_data/raw/writes.csv'

about_df = pd.read_csv(about_csv)
cites_df = pd.read_csv(cites_csv)
papers_df = pd.read_csv(papers_csv)
publication_df = pd.read_csv(publication_csv)
authors_topic_2019_df = pd.read_csv(authors_topic_2019_csv)
papers_about_df = pd.read_csv(papers_about_csv)
writes_df = pd.read_csv(writes_csv)


csv_folder_path = '/home/albuzzi/academic_network_project/anp_data/split'
author_ids_dfs = []
column_names = ['author_id']
for i in range(0, 5):
    filename = f'authors_{i}.csv'
    filepath = os.path.join(csv_folder_path, filename)
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, header=None, names=column_names)
        author_ids_dfs.append(df)
    else:
        print(f"File {filepath} non esiste.")


if author_ids_dfs:
    author_ids_df = pd.concat(author_ids_dfs, ignore_index=True)
    all_author_ids = set(author_ids_df['author_id'].unique())
else:
    print("non sono stati salvati correttamente gli autori")

all_author_ids = all_author_ids | set(writes_df['author_id'].unique())

# Funzione per convertire 'topic_id' in interi e rimuovere NaN
def ensure_int_topic_id(df, column='topic_id'):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df.dropna(subset=[column], inplace=True)
    df[column] = df[column].astype(int)


ensure_int_topic_id(about_df)
ensure_int_topic_id(authors_topic_2019_df)
ensure_int_topic_id(papers_about_df)

# Funzione per convertire 'journal_id' in interi e rimuovere NaN
def ensure_int_journal_id(df, column='journal_id'):
    # Converti la colonna in valori numerici, sostituendo i valori non validi con NaN
    df[column] = pd.to_numeric(df[column], errors='coerce')
    # Rimuovi le righe con NaN nella colonna specificata
    df.dropna(subset=[column], inplace=True)
    # Converti la colonna in interi
    df[column] = df[column].astype(int)

ensure_int_journal_id(publication_df)


##DATASET .INTER

author_citing_paper_df = pd.merge(
    writes_df,
    papers_df[['id', 'citations', 'year']],
    left_on='paper_id',
    right_on='id' 
)


author_citing_paper_df.rename(columns={
    'author_id': 'user_id',
    'id': 'item_id',
    'citations': 'paper_citations',
    'year': 'paper_year'
}, inplace=True)


author_citing_paper_df['interaction_type'] = 'write'

author_citing_paper_df = author_citing_paper_df[['user_id', 'item_id', 'interaction_type', 'paper_year']]

author_citing_paper_df.rename(columns={'paper_year': 'year'}, inplace=True)

#cites_df con papers_df per ottenere l'anno di pubblicazione del paper citato
author_citing_df = pd.merge(
    cites_df, 
    papers_df[['id', 'year']], 
    left_on='paper2_id', 
    right_on='id'
)

author_citing_df.rename(columns={
    'paper1_id': 'user_id',  # paper1_id rappresenta l'utente che cita
    'paper2_id': 'item_id',  # paper2_id rappresenta l'articolo citato
    'year': 'paper_year'
}, inplace=True)

author_citing_df['interaction_type'] = 'cite'

author_citing_df = author_citing_df[['user_id', 'item_id', 'interaction_type', 'paper_year']]

author_citing_df.rename(columns={'paper_year': 'year'}, inplace=True)

combined_interactions_df = pd.concat([author_citing_paper_df, author_citing_df])

# Salva il file your_dataset_name.inter
data_path = '/home/albuzzi/academic_network_project/anp_nn/anp_data/data_path'  
if not os.path.exists(data_path):
    os.makedirs(data_path)
author_citing_paper_df.to_csv(os.path.join(data_path, 'academic_dataset_inter.csv'), index=False)
print("academic_dataset_inter.csv creato correttamente.")

print("inizio conversione .csv to .inter")

csv_inter_path = '/home/albuzzi/academic_network_project/anp_nn/anp_data/data_path/academic_dataset_inter.csv'

interactions = pd.read_csv(csv_inter_path)

# Assicurati che i tipi di dati siano corretti per il formato .inter
interactions['user_id'] = interactions['user_id'].astype(int)
interactions['item_id'] = interactions['item_id'].astype(int)
interactions['interaction_type'] = interactions['interaction_type'].astype(str)
interactions['year'] = interactions['year'].astype(int)

# Definisci i tipi di campo per il file .inter
inter_field_types = {
    'user_id': 'token',
    'item_id': 'token',
    'interaction_type': 'token',
    'year': 'float'
}

# Crea l'header per il file .inter usando le colonne corrette
header_line = '\t'.join([f"{col}:{inter_field_types[col]}" for col in interactions.columns])

# Salva il file .inter
inter_path = '/home/albuzzi/academic_network_project/anp_nn/anp_data/data_path/academic_dataset/academic_dataset.inter'
with open(inter_path, 'w') as f:
    f.write(header_line + '\n')
    interactions.to_csv(f, sep='\t', index=False, header=False)

print("Conversione .inter completata.")

##   DATASET.USER

# Numero di articoli scritti da ogni autore
author_paper_counts = writes_df.groupby('author_id').size().reset_index(name='num_papers')

author_paper_counts.rename(columns={'author_id': 'user_id'}, inplace=True)

# Salva il file your_dataset_name.user
author_paper_counts.to_csv(os.path.join(data_path, 'academic_dataset_user.csv'), index=False)
print("academic_dataset_user.csv creato correttamente.")

print("inizio conversione .csv to .user")

csv_user_path = '/home/albuzzi/academic_network_project/anp_nn/anp_data/data_path/academic_dataset_user.csv'

users = pd.read_csv(csv_user_path)

users['user_id'] = users['user_id'].astype(int)
users['num_papers'] = users['num_papers'].astype(int)

users_field_types = {
    'user_id': 'token',
    'num_papers': 'float'
}

header_line = '\t'.join([f"{col}:{users_field_types[col]}" for col in users.columns])

user_path = '/home/albuzzi/academic_network_project/anp_nn/anp_data/data_path/academic_dataset/academic_dataset.user'
with open(user_path, 'w') as f:
    f.write(header_line + '\n')
    users.to_csv(f, sep='\t', index=False, header=False)

print("Conversion .user completata.")


### DATASET.ITEM

paper_citations = papers_df[['id', 'citations', 'year']].copy()

paper_citations.rename(columns={'id': 'item_id', 'citations': 'num_citations'}, inplace=True)

paper_citations.fillna(0, inplace=True)

publication_df = publication_df[['paper_id', 'journal_id']].copy()

paper_citations = pd.merge(
    paper_citations, 
    publication_df, 
    left_on='item_id', 
    right_on='paper_id', 
    how='left'
)


paper_citations.drop(columns=['paper_id'], inplace=True)

about_df = about_df[['paper_id', 'topic_id']].copy()

paper_citations = pd.merge(
    paper_citations, 
    about_df, 
    left_on='item_id', 
    right_on='paper_id', 
    how='left'
)


paper_citations.drop(columns=['paper_id'], inplace=True)

paper_citations.fillna({'journal_id': 0, 'topic_id': 0}, inplace=True)

# Salva il file your_dataset_name.item
paper_citations.to_csv(os.path.join(data_path, 'academic_dataset_item.csv'), index=False)
print("academic_dataset.item creato correttamente.")


print("inizio conversione .csv to .item")

csv_item_path = '/home/albuzzi/academic_network_project/anp_nn/anp_data/data_path/academic_dataset_item.csv'

items = pd.read_csv(csv_item_path)

items['item_id'] = items['item_id'].astype(int)
items['num_citations'] = items['num_citations'].astype(int)
items['year'] = items['year'].astype(int)
items['journal_id'] = items['journal_id'].astype(int)
items['topic_id'] = items['topic_id'].astype(int)

items_field_types = {
    'item_id': 'token',
    'num_citations': 'float',
    'year': 'float',
    'journal_id' : 'token',
    'topic_id' : 'token'
}

header_line = '\t'.join([f"{col}:{items_field_types[col]}" for col in items.columns])

item_path = '/home/albuzzi/academic_network_project/anp_nn/anp_data/data_path/academic_dataset/academic_dataset.item'
with open(item_path, 'w') as f:
    f.write(header_line + '\n')
    items.to_csv(f, sep='\t', index=False, header=False)

print("Conversion .item completata.")