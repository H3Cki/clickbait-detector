import pandas
from text_preprocessing import *
from tensorflow import keras
from tensorflow.keras import layers
import wandb
from wandb.keras import WandbCallback
import pathlib
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import os
import sys
if getattr(sys, 'frozen', False):
    PATH = os.path.dirname(sys.executable)
elif __file__:
    PATH = os.path.dirname(__file__)


def converter(x):
    return json.loads(x)

class EncoderHandler:
    def __init__(self, encoders_raw):
        self.encoders = [list(map(Encoder.load, encoder)) for encoder in [encoder_layer for encoder_layer in encoders_raw]]
        self.encoders_raw = []
        for layer in self.encoders:
            for e in layer:
                self.encoders_raw.append(e)
        self.encoder_dict = {}
        for encoder_layer in self.encoders:
            for encoder in encoder_layer:
                self.encoder_dict[encoder.name] = encoder
        
    def bulk_encode(self, text):
        results = []
        for encoder_layer in self.encoders:
            encoder_results = []
            for encoder in encoder_layer:
                encoder_results += encoder.encode(text)
            results.append(np.asarray(encoder_results).reshape((1,-1)))
        return results if len(results) > 1 else results[0]
        
    
class Package:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.model = keras.models.load_model('models/'+model_name) if kwargs.get('load_model', False) else None
        self.config = kwargs.get('config')
        self.encoder_configs = self.config['encoder_configs']
        self.handler = EncoderHandler(self.encoder_configs)

    def tokenize_dataset(self, fname):
        #   tokenize text dataset or load if already exists
        #   to format: encoder1;encoder2;clickbait
        fpath = f'datasets/{fname}.csv'
        df = pandas.read_csv(fpath, sep=';')
        titles = df['title']
        df_dict = dict(clickbait=df['clickbait'])
        for layer in self.handler.encoders:
            for e in layer:
                print(e.name)
                tokenized_titles = list(map(e.tokenize, titles))
                df_dict[e.name] = tokenized_titles
        
        output_df = pandas.DataFrame(data=df_dict)
        output_df.to_csv(f'datasets/{fname}_tokenized.csv', index=False, sep=';')
        
        return output_df
    
    def predict(self, text):
        if isinstance(text, str):
            encoded = self.handler.bulk_encode(text)
            pred = self.model.predict_on_batch(encoded)
        else:
            pred = [self.model.predict(self.handler.bulk_encode(t)) for t in text]
        return pred
    
    def encode_dataset(self, fname, dir_path='datasets', encoded_suffix='_encoded'):
        #   tokenize text dataset or load if already exists
        #   to format: encoder1;encoder2;clickbait
        encoded_name = f'{fname}{encoded_suffix}'
        
        # try:
        e_names = [e.name for e in self.handler.encoders_raw]
        datasets_path = os.path.join(PATH, dir_path)
        files = [os.path.join(datasets_path, f) for f in os.listdir(datasets_path) if os.path.isfile(os.path.join(datasets_path, f))]
        for f in files:
            df = pandas.read_csv(f, sep=';', converters={e_name : converter for e_name in e_names})
            if set(e_names).issubset(set(df.columns.to_list())):
                logging.info(f'Loaded {f}')
                return df
            else:
                print(f, 'Not a subset of encoders')
        print('Creating new encoded dataset')
        encoded_name = f'{fname}{encoded_suffix}_1'

        logging.info(f'Creating encoded dataset of {fname}.csv')
        
        df = read_dataset(fname, dir_path=dir_path)
        
        titles = df['title']
        df_dict = dict(clickbait=df['clickbait'])
        for j, layer in enumerate(self.handler.encoders):
            for i, e in enumerate(layer):
                logging.info(f'[Layer {j+1}/{len(self.handler.encoders)}] Encoder {i+1}/{len(layer)}')
                tokenized_titles = list(map(e.encode, titles))
                df_dict[e.name] = tokenized_titles
        
        output_df = pandas.DataFrame(data=df_dict)
        output_df.to_csv(f'{dir_path}/{encoded_name}.csv', index=False, sep=';')
        
        logging.info(f'Exported dataset {encoded_name}')
        
        return output_df

    def save(self):
        out = dict(
            encoder_configs = self.encoder_configs,
            model_name = self.model_name
        )
        
        with open(f'packs/{self.model_name}.json', 'w') as f:
            json.dump(out, f)
            
    @classmethod
    def load(cls, pack_name=None):
        if not pack_name:
            print('Pack name not provided.')
            dirname = os.path.dirname(__file__)
            found_models = [(os.path.join(dirname, 'packs', d), d) for d in os.listdir(os.path.join(dirname, 'packs')) if os.path.isfile(os.path.join(dirname, 'packs', d))]
            found_models = list(sorted(found_models, key=lambda f: pathlib.Path(f[0]).stat().st_mtime, reverse=True))
            print(found_models)
            if found_models:
                pack_name = found_models[0][1].split('.')[0]
                print(f'Loading recent pack: {pack_name}')
            else:
                print('No pack found')

        try:
            with open(f'packs/{pack_name}.json', 'r') as f:
                config = json.load(f)
        except:
            return None
        return cls(pack_name, load_model=True, config=config)
    

def read_dataset(fname, dir_path='datasets'):
    fpath = f'{dir_path}/{fname}.csv'
    df = pandas.read_csv(fpath, sep=';')
    return df
    
model_name = None
    
def train():
    dataset_name = 'mixed_dataset'
    
    dataset_df = read_dataset(dataset_name)
    titles = dataset_df['title']
    
    
    main_table = table(titles, 'tokenize', fname='main_table', tokenizer_args=dict(preserve_case=False, preserve_symbols=True, preserve_stopwords=True, preserve_contractions=True, preserve_numbers=False))#, preserve_numbers=False
    
    pos_table = table(titles, 'pos_tags', tokenizer_args=dict(tag_only=True, tokenizer_args=dict(preserve_case=False, preserve_symbols=True, preserve_stopwords=True, preserve_contractions=True, preserve_numbers=False)), fname='pos_table')#, preserve_numbers=False
    
    gram2_table = table(titles, 'ngram', tokenizer_args=dict(n=2, as_text=True, preserve_numbers=False), fname='gram2_table')
    
    #gram3_table = table(titles, 'ngram', tokenizer_args=dict(n=3, as_text=True), fname='gram3_table')
    
    pos_2gram_table = table(titles, 'ngram', tokenizer_args=dict(n=2, as_text=True, tokenizer='pos_tags', preserve_numbers=False), fname='pos_2gram_table')
    
    #pos_3gram_table = table(titles, 'ngram', tokenizer_args=dict(n=3, as_text=True, tokenizer='pos_tags'), fname='pos_3gram_table')
    
    merged_table = merge_tables([main_table, pos_table, gram2_table, pos_2gram_table], fname='merged_table')
    
    normal_encoder = dict(
                    name='normal_encoder',
                    type='TableEncoder',
                    _tokenizer='tokenize',
                    tokenizer_args=dict(preserve_numbers=False),
                    max_len=main_table['max_len'],
                    table='merged_table'
                )
    
    pos_encoder = dict(
                    name='pos_encoder',
                    type='TableEncoder',
                    _tokenizer='pos_tags',
                    tokenizer_args=dict(tag_only=True, preserve_numbers=False),
                    max_len=pos_table['max_len'],
                    table='merged_table'
                )
    
    capital_encoder = dict(
                    name='capital_encoder',
                    type='TableEncoder',
                    _tokenizer='capital',
                    tokenizer_args=dict(lower=True, full_capital=True, preserve_numbers=False),
                    max_len=main_table['max_len'],
                    table='merged_table'
                )

    capital_postags_encoder = dict(
                    name='capital_encoder',
                    type='TableEncoder',
                    _tokenizer='capital',
                    tokenizer_args=dict(lower=False, full_capital=True, postags=True, preserve_numbers=False),
                    max_len=pos_table['max_len'],
                    table='merged_table'
                )

    gram2_encoder = dict(
                    name='2gram_encoder',
                    type='TableEncoder',
                    _tokenizer='ngram',
                    tokenizer_args=dict(n=2, as_text=True, preserve_numbers=False),
                    max_len=gram2_table['max_len'],
                    table='merged_table'
                )
    
    # gram3_encoder = dict(
    #                 name='3gram_encoder',
    #                 type='TableEncoder',
    #                 _tokenizer='ngram',
    #                 tokenizer_args=dict(n=3, as_text=True),
    #                 max_len=gram3_table['max_len'],
    #                 table='merged_table'
    #             )
    
    pos_2gram_encoder = dict(
                    name='pos_2gram_encoder',
                    type='TableEncoder',
                    _tokenizer='ngram',
                    tokenizer_args=dict(n=2, as_text=True, tokenizer='pos_tags', preserve_numbers=False),
                    max_len=gram2_table['max_len'],
                    table='merged_table'
                )
    
    # pos_3gram_encoder = dict(
    #                 name='pos_3gram_encoder',
    #                 type='TableEncoder',
    #                 _tokenizer='ngram',
    #                 tokenizer_args=dict(n=3, as_text=True, tokenizer='pos_tags'),
    #                 max_len=gram3_table['max_len'],
    #                 table='merged_table'
    #             )
    
    
    numeric_encoder = dict(
        name='numeric_encoder',
        type='CustomEncoder',
        _tokenizer='numeric',
        tokenizer_args=dict(),
        max_len=5
    )

    tables = [
        merged_table
    ]
    
    config = dict(
        encoder_configs = [
            [normal_encoder, pos_encoder, capital_encoder, capital_postags_encoder],
            [numeric_encoder]
        ]
    )

    pack = Package(model_name, config=config)
    encoded = pack.encode_dataset(dataset_name, encoded_suffix='_encoded')
    
    wandb.init(project="clickbait_detector")

    train_Y = encoded['clickbait'].values
    train_X = [[] for _ in range(len(pack.handler.encoders))]
    
    for i, layer in enumerate(pack.handler.encoders):
        print([e.name for e in layer])
        cols = list(encoded[[e.name for e in layer]].values)
        for row in cols:
            concat = []
            for r in row:
                concat += r
            train_X[i].append(np.asarray(concat))
        train_X[i] = np.asarray(train_X[i])
        
    # print(f'Train x list shape: ({len(train_X)},{len(train_X[0])},{len(train_X[0][0])}, {len(train_X[1])},{len(train_X[1][0])})')
    # print(f'Train col lens: {[len(x) for x in train_X]}')

    # print('Train x shape: ', train_X.shape)
    # print('Train y shape:', train_Y.shape)

    max_length = sum([enc['max_len'] for enc in config['encoder_configs'][0]])
    max_length1 = sum([enc['max_len'] for enc in config['encoder_configs'][1]])
    total_vocab_size = sum([table['vocab_size'] for table in tables])
    print('vocab_size', total_vocab_size)
    print('max_len', max_length)

    #EMBEDDINGS
    inpt = keras.Input(shape=(max_length,))
    x = layers.Embedding(total_vocab_size, 100, input_length=max_length)(inpt)
    x = layers.Conv1D(32, (3,), padding='same', activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling1D(pool_size=3, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    embed_model = keras.Model(inputs=inpt, outputs=x)
    
    #NUMBERS
    inpt1 = keras.Input(shape=(max_length1,))
    x = layers.Dense(max_length1, activation='relu')(inpt1)
    vec_model = keras.Model(inputs=inpt1, outputs=x)
    
    #SHARE
    x = layers.Concatenate()([embed_model.outputs[0], vec_model.outputs[0]])
    otpt = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=[embed_model.inputs[0], vec_model.inputs[0]], outputs=otpt, name=model_name)
    
    logging.info('COMPILING')
    optimizer = keras.optimizers.SGD(lr=0.001)
    #optimizer = keras.optimizers.Adam(lr=0.001)
    #metrics
    # binary_accuracy = keras.metrics.BinaryAccuracy(
    #     name="binary_accuracy", threshold=0.5
    # )
    
    
    model.compile(
        loss        =   keras.losses.BinaryCrossentropy(),
        optimizer   =   optimizer,
        metrics     =   ["accuracy"]
    )
    
    model.summary()
    
    logging.info('CREATING CALLBACKS')
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        'models/'+model_name, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='auto', save_freq='epoch',
        options=None,
    )
    
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=1,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )
    
    rdc = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_delta=0.001, patience=2, verbose=2)
    
    
    logging.info('SAVING PACK')
    pack.save()
    model.summary()
    logging.info('STARTING TRAINING')
    model.fit(x=train_X, y=train_Y, validation_split=0.1, epochs=15, batch_size=2, callbacks=[checkpoint, rdc, WandbCallback()])



model_name = 'clickbait_detector_model'
if __name__ == '__main__':
    train()
    
    pack = Package.load(model_name)
    encoded = pack.encode_dataset('test_dataset', dir_path='datasets')
    train_Y = encoded['clickbait'].values
    train_X = [[] for _ in range(len(pack.handler.encoders))]
    
    for i, layer in enumerate(pack.handler.encoders):
        print([e.name for e in layer])
        cols = list(encoded[[e.name for e in layer]].values)
        for row in cols:
            concat = []
            for r in row:
                concat += r
            train_X[i].append(np.asarray(concat))
        train_X[i] = np.asarray(train_X[i])
    
    loss, acc = pack.model.evaluate(train_X, train_Y, batch_size=2)
    
    print('Test loss:', loss)
    print('Test accuracy:', acc)

    df = pandas.read_csv('datasets/mixed_dataset.csv', sep=';')
    titles = df['title'].values
    for title in titles:
        pass