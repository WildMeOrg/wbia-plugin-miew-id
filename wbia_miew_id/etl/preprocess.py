import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder



def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def load_json_to_df(anno_path):
    data = load_json(anno_path)

    dfa = pd.DataFrame(data['annotations'])
    dfi = pd.DataFrame(data['images'])

    dfi = dfi.drop_duplicates(subset=['uuid'])

    merge_on_uuid = 'image_uuid' in dfa.columns and 'uuid' in dfi.columns
    if merge_on_uuid:
        print('Merging on image uuid')
        df = dfa.merge(dfi, left_on='image_uuid', right_on='uuid')
    else:
        df = dfa.merge(dfi, left_on='image_id', right_on='id') 

    print(f'** Loaded {anno_path} **')
    print('     ', f'Found {len(df)} annotations')

    return df

def load_to_df(anno_path):
    df = pd.read_csv(anno_path)
    df['bbox'] = df[['x', 'y', 'w', 'h']].values.tolist()

    return df

def filter_viewpoint_df(df, viewpoint_list):
    df = df[df['viewpoint'].isin(viewpoint_list)]
    print('     ', len(df), 'annotations remain after filtering by viewpoint list', viewpoint_list)
    return df

def filter_min_names_df(df, n_filter_min, filter_key='name_species'):
    df = df.groupby(filter_key).filter(lambda g: len(g)>=n_filter_min)
    print('     ', len(df), 'annotations remain after filtering by min', n_filter_min, 'per', filter_key)
    return df

def subsample_max_df(df, n_subsample_max, filter_key='name_species'):
    df = df.groupby(filter_key, as_index=False).apply(lambda g: g.sample(frac=1, random_state=0).head(n_subsample_max)).droplevel(level=0)
    print('     ', len(df), 'annotations remain after subsampling by max', n_subsample_max, 'per', filter_key)
    return df

def convert_name_to_id(names):
    le = LabelEncoder()
    names_id = le.fit_transform(names)
    return names_id

def filter_df(df, viewpoint_list, n_filter_min, n_subsample_max, filter_key='name_species'):
    if viewpoint_list:
        df = filter_viewpoint_df(df, viewpoint_list)
    
    if n_filter_min:
        df = filter_min_names_df(df, n_filter_min, filter_key=filter_key)

    if not len(df):
        raise Exception("No samples remain after filtering.")
        
    if n_subsample_max:
        df = subsample_max_df(df, n_subsample_max, filter_key=filter_key)

    return df

def preprocess_data(anno_path, name_keys=['name'], convert_names_to_ids=True, viewpoint_list=None, n_filter_min=None, n_subsample_max=None, use_full_image_path=False, images_dir=None):

    if anno_path.lower().endswith('json'):
        df = load_json_to_df(anno_path)
    elif anno_path.lower().endswith('csv'):
        df = load_to_df(anno_path)
    else:
        raise NotImplementedError("Annotation file extension not supported.")

    df['name'] = df[name_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    # df.loc[df['name_species'].isna(), 'name_species'] = df.loc[df['name_species'].isna(), 'name'].astype(str) + '_' + df.loc[df['name_species'].isna(), 'species']
    if 'species' in df.columns:
        df['name_species'] = df['name'] + '_' + df['species']
        filter_key = 'name_species'
    else:
        df['species'] = 'default_species'
        filter_key = 'name'

    df['name_orig'] = df['name'].copy()

    df = filter_df(df, viewpoint_list, n_filter_min, n_subsample_max, filter_key=filter_key)

    if convert_names_to_ids:
        names = df['name'].values
        names_id = convert_name_to_id(names)
        df['name'] = names_id

    if not use_full_image_path:
        df['file_path'] = df['file_name'].apply(lambda x: os.path.join(images_dir, x))

    df = df.reset_index(drop=True)

    return df

if __name__ == "__main__":
    pass