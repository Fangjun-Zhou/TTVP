# Add word2vec to python path.
import sys
sys.path.append("external/word2vec")

from title_predictor_model import TitlePredictor

import torch
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer

# Word2Vec.
from external.word2vec.utils.helper import (
    load_vocab,
)

print("Loading title predictor model...")

# Load vocab.
vocab: Vocab = load_vocab("models/title-predictor/vocab.pt")
# Load tokenizer.
tokenizer = get_tokenizer("basic_english")
# Load model.
model = torch.load("models/title-predictor/best_val_2.59.pt")
predictor = TitlePredictor(
    vocabSize=model["vocabSize"],
    embeddingSize=model["embeddingSize"],
    hiddenSize=model["hiddenSize"],
    numLayers=model["numLayers"],
    dropout=model["dropout"],
)
# Load model weights.
predictor.load_state_dict(model["model"])

print("Title predictor model loaded.")

def title_predict_view(
    title: str
):
    # Tokenize title.
    tokens = tokenizer(title)
    # Add <sos> and <eos> tokens.
    tokens = ["<SOS>"] + tokens + ["<EOS>"]
    # Convert tokens to indices.
    indices = [vocab[token] for token in tokens]
    # Convert indices to tensor.
    indices = torch.tensor(indices)
    # Add batch dimension.
    indices = indices.unsqueeze(1)
    # Predict.
    with torch.no_grad():
        prediction = predictor(indices)
    # Convert prediction back to exponentiated value.
    prediction = torch.exp(prediction)
    # Convert prediction to float.
    prediction = prediction.item()
    # Return prediction.
    return prediction