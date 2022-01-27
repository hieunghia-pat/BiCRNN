import torch
from torch.utils.data.dataloader import DataLoader

from data_utils.dataloader import OCRDataset, Batch
from metric_utils.metrics import Metrics
from metric_utils.tracker import Tracker

from model.transformer import make_model
import config

import pickle
from tqdm import tqdm

vocab = pickle.load(open("saved_models/line-syn-character-level/vocab_character.pkl", "rb"))
dataset = OCRDataset(config.test_image_dirs, image_size=(1024, 64), out_level=config.out_level, vocab=vocab)

saved_info = torch.load("saved_models/line-syn-character-level/best_model.pth", "cuda")

loader = DataLoader(
        dataset,
        batch_size=config.batch_test,
        shuffle=False
)

model = make_model(len(vocab.stoi), N=config.num_layers, d_model=config.d_model, d_ff=config.dff,
                            h=config.heads, dropout=config.dropout).cuda()
metrics = Metrics(vocab)

tracker = Tracker()
cer_tracker = tracker.track("cer_tracker", tracker.MeanMonitor())
wer_tracker = tracker.track("wer_tracker", tracker.MeanMonitor())

model.load_state_dict(saved_info["state_dict"])

with open("results_best_model.txt", "w+") as file:
    for images, tokens, shifted_right_tokens in tqdm(loader):
        batch = Batch(images, tokens, shifted_right_tokens, vocab.padding_idx)
        outs = model.get_predictions(batch.imgs, batch.src_mask, vocab, loader.dataset.max_len)
        file.write(vocab.decode_sentence(outs.to("cpu"))[0] + "\n")
        file.write(vocab.decode_sentence(tokens.to("cpu"))[0] + "\n")
        scores = metrics.get_scores(vocab.decode_sentence(outs.to("cpu")), vocab.decode_sentence(tokens.to("cpu")))
        file.write(f"{scores['cer']} \n")
        file.write(f"{scores['wer']} \n")
        file.write("*"*13 + "\n")

        cer_tracker.append(scores["cer"])
        wer_tracker.append(scores["wer"])

print(f"CER: {cer_tracker.mean.value}")
print(f"WER: {wer_tracker.mean.value}")