

def goenc(val, encoders):
    return [encoder(val).reshape(1,-1) for encoder in encoders]

def predict(s, model, encoders):
    x = goenc(s, encoders)
    p = model.predict([x, ])[0][0]
    l = 20
    prog = int(round(l*p))
    t = "█"*prog + "—"*(20-prog)
    if p > .66:
        col = '\033[31m'
    elif p > .33:
        col = '\033[33m'
    else:
        col = '\033[32m'
    end = '\033[0m'
    print(f'I think this title is {col}[{t}]{end} {int(round(p*100))}% clickbait')

def run_model():
    model_name = input('Model name: ')
    from tensorflow import keras
    import os
    from text_preprocessing import Encoder

    model = keras.models.load_model('models/' + model_name)
    
    
    encoders = Encoder.load(model_name=model_name)
    os.system('cls')

    while True:
        s = input('> ')
        predict(s, model, encoders)
        
        
        
if __name__ == '__main__':
    run_model()