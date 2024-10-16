from typing import Dict, List

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: List[Dict]) -> Dict[str, Tensor]:
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    audios = [item["audio"].squeeze(0) for item in dataset_items]
    # Each audio: [1, num_samples] -> [num_samples]
    # audios list: [num_samples_1, num_samples_2, ..., num_samples_N]

    # [1, freq_bins, time_steps] -> [freq_bins, time_steps]
    spectrograms = [
        item["spectrogram"].squeeze(0).transpose(0, 1) for item in dataset_items
    ]
    # [time_steps_1, freq_bins], [time_steps_2, freq_bins], ..., [time_steps_N, freq_bins]

    text_encoded = [item["text_encoded"].squeeze(0) for item in dataset_items]
    # [1, sequence_length] -> [sequence_length]
    # [sequence_length_1, sequence_length_2, ..., sequence_length_N]

    texts = [item["text"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]

    audio_padded = pad_sequence(audios, batch_first=True, padding_value=0)
    # audio_padded shape: [batch_size, max_audio_length]

    spectrogram_padded = pad_sequence(spectrograms, batch_first=True, padding_value=0)
    # spectrogram_padded shape (after padding): [batch_size, max_time_steps, freq_bins]

    # Transpose back to [freq_bins, max_time_steps]
    spectrogram_padded = spectrogram_padded.transpose(1, 2)
    # spectrogram_padded shape: [batch_size, freq_bins, max_time_steps]

    # Pad the text sequences to the length of the longest text_encoded sequence
    text_encoded_padded = pad_sequence(text_encoded, batch_first=True, padding_value=0)
    # text_encoded_padded shape: [batch_size, max_sequence_length]

    audio_lengths = torch.tensor([x.size(0) for x in audios], dtype=torch.long)
    # audio_lengths shape: [batch_size]

    spectrogram_lengths = torch.tensor(
        [x.size(0) for x in spectrograms], dtype=torch.long
    )
    # spectrogram_lengths shape: [batch_size] (time_steps)

    text_lengths = torch.tensor([x.size(0) for x in text_encoded], dtype=torch.long)
    # text_lengths shape: [batch_size]

    # [batch_size, max_audio_length]
    audio_attention_mask = (audio_padded != 0).float()

    # [batch_size, max_time_steps]
    # we check if any freq_bin is non-zero at each time_step
    spectrogram_attention_mask = (spectrogram_padded.sum(dim=1) != 0).float()

    text_attention_mask = (text_encoded_padded != 0).float()
    # [batch_size, max_sequence_length]

    result_batch = {
        "audio": audio_padded,  # [batch_size, max_audio_length]
        "audio_length": audio_lengths,  # [batch_size]
        "audio_attention_mask": audio_attention_mask,  # [batch_size, max_audio_length]
        "spectrogram": spectrogram_padded,  # [batch_size, freq_bins, max_time_steps]
        "spectrogram_length": spectrogram_lengths,  # [batch_size]
        "spectrogram_attention_mask": spectrogram_attention_mask,  # [batch_size, max_time_steps]
        "text_encoded": text_encoded_padded,  # [batch_size, max_sequence_length]
        "text_encoded_length": text_lengths,  # [batch_size]
        "text_attention_mask": text_attention_mask,  # [batch_size, max_sequence_length]
        "text": texts,
        "audio_path": audio_paths,
    }

    return result_batch
