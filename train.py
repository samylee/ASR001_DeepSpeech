import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset

from models import DeepSpeech
from utils import data_processing


def train(model, train_dataloader, criterion, optimizer, epoch, epochs, device):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        spectrograms, labels, input_lengths, label_lengths = batch
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        y_hat = model(spectrograms)
        output = F.log_softmax(y_hat, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)
        loss = criterion(output, labels, input_lengths, label_lengths)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("epoch: [%d/%d], iter: [%d/%d], loss: %.4f, avg_loss: %.4f" % (
                epoch, epochs, i, len(train_dataloader), loss.item(), total_loss / (i + 1)))
    return total_loss / (i+1)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_root = "../datasets"
    data_train = ["train-clean-100", "train-clean-360", "train-other-500"]
    # data_test = ["test-clean"]

    learning_rate = 0.0005
    epochs = 100
    num_workers = 8
    batch_size = 32

    n_cnn_layers = 3
    n_rnn_layers = 5
    rnn_dim = 512
    n_class = 29
    n_feats = 128

    train_dataset = ConcatDataset(
        [
            torchaudio.datasets.LIBRISPEECH(data_root, url=path, download=False)
            for path in data_train
        ]
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: data_processing(x, "train"),
        num_workers=num_workers,
    )

    model = DeepSpeech(n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate / 10)
    criterion = nn.CTCLoss(blank=28)

    for epoch in range(epochs):
        loss = train(model, train_dataloader, criterion, optimizer, epoch, epochs, device)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'checkpoints/deepspeech_' + str(epoch) + '_' + str(loss) + '.pth')
