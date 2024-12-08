import torch
import torch.nn as nn
import torchaudio

sos_id = 0
eos_id = 1


class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        char_map_str = """
        sos_id 0
        eos_id 1
        ' 2
        <SPACE> 3
        a 4
        b 5
        c 6
        d 7
        e 8
        f 9
        g 10
        h 11
        i 12
        j 13
        k 14
        l 15
        m 16
        n 17
        o 18
        p 19
        q 20
        r 21
        s 22
        t 23
        u 24
        v 25
        w 26
        x 27
        y 28
        z 29
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split("\n"):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[3] = " "

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == " ":
                ch = self.char_map["<SPACE>"]
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return "".join(string).replace("<SPACE>", " ")


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100),
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()


def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == "train":
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == "valid":
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception("data_type should be train or valid")
        spectrograms.append(spec)
        label = torch.Tensor(
            [sos_id] + text_transform.text_to_int(utterance.lower()) + [eos_id]
        )
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = (
        nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        .unsqueeze(1)
        .transpose(2, 3)
    )
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths