import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from jiwer import wer, cer
from typing import List
from tqdm.auto import tqdm


import librosa
from pesq import pesq
from scipy.io import wavfile
import numpy as np

from pystoi import stoi


class WhisperWERCalculator:
    # def __init__(self, model_id: str = "openai/whisper-large-v3-turbo"):
    def __init__(self, model_id: str = "openai/whisper-small.en"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.batch_size = 64
        
        self.real_hyps = None

    def transcribe_batch(
        self,
        waveforms
    ):
        
        # Convert all waveforms to CPU numpy arrays
        numpy_waveforms = [wav.squeeze().detach().cpu().numpy() for wav in waveforms]

        # Tokenize and pad batch
        inputs = self.processor(
            audio=numpy_waveforms,
            return_tensors="pt",
            sampling_rate=16000,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)
            transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Clean and uppercase
        return [t.strip().upper() for t in transcriptions]

    def _chunkify(self, items: List, chunk_size: int) -> List[List]:
        """
        Splits a list into successive chunks of given size.
        """
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def compute_wer(
        self,
        pred_waveforms,
        real_waveforms,
        transcripts
    ):

        pred_hyps, self.real_hyps = [], []

        # Batchify transcription
        for chunk in tqdm(self._chunkify(pred_waveforms, self.batch_size)):
            pred_hyps.extend(self.transcribe_batch(chunk))
        if self.real_hyps is None:
            for chunk in tqdm(self._chunkify(real_waveforms, self.batch_size)):
                self.real_hyps.extend(self.transcribe_batch(chunk))

        wer_pred = wer(transcripts, pred_hyps)
        cer_pred = cer(transcripts, pred_hyps)
        wer_real = wer(transcripts, self.real_hyps)        
        cer_real = cer(transcripts, self.real_hyps)

        print(f"CER for Predicted Waveforms: {cer_pred:.3f}")
        print(f"CER for Real Waveforms:      {cer_real:.3f}")
        print(f"WER for Predicted Waveforms: {wer_pred:.3f}")
        print(f"WER for Real Waveforms:      {wer_real:.3f}")

        return cer_pred, cer_real, wer_pred, wer_real, pred_hyps, self.real_hyps


def compute_pesq(reference_waveforms, synthesized_waveforms):
    pesq_scores = []

    for ref_waveform, synth_waveform in tqdm(zip(reference_waveforms, synthesized_waveforms)):
        # Ensure the waveforms are numpy arrays and convert them to 1D
        ref_waveform = ref_waveform.squeeze().cpu().numpy()
        synth_waveform = synth_waveform.squeeze().cpu().numpy()

        # Ensure the lengths of the waveforms match
        min_len = min(len(ref_waveform), len(synth_waveform))
        ref_waveform = ref_waveform[:min_len]
        synth_waveform = synth_waveform[:min_len]

        # Compute PESQ score
        pesq_score = pesq(16000, ref_waveform, synth_waveform)  # fs=16000 for 16kHz
        pesq_scores.append(pesq_score)

    return sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0


def compute_stoi(reference_waveforms, synthesized_waveforms, sample_rate=16000):
    stoi_scores = []

    for ref_waveform, synth_waveform in tqdm(zip(reference_waveforms, synthesized_waveforms)):
        # Ensure the waveforms are numpy arrays and convert them to 1D
        ref_waveform = ref_waveform.squeeze().cpu().numpy()
        synth_waveform = synth_waveform.squeeze().cpu().numpy()
        
        # Ensure the lengths of the waveforms match
        min_len = min(len(ref_waveform), len(synth_waveform))
        ref_waveform = ref_waveform[:min_len]
        synth_waveform = synth_waveform[:min_len]

        # Compute STOI score
        stoi_score = stoi(ref_waveform, synth_waveform, sample_rate)
        stoi_scores.append(stoi_score)

    return sum(stoi_scores) / len(stoi_scores) if stoi_scores else 0
