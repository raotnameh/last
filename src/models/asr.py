import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from jiwer import wer
from typing import List
from tqdm.auto import tqdm

class WhisperWERCalculator:
    # def __init__(self, model_id: str = "openai/whisper-large-v3-turbo"):
    def __init__(self, model_id: str = "openai/openai/whisper-small.en"):
    # def __init__(self, model_id: str = "openai/whisper-large-v3-turbo"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def transcribe(self, waveform: torch.Tensor) -> str:
        """
        Expects waveform as a torch.Tensor with shape (1, T), already on CUDA.
        """
        waveform = waveform.squeeze().detach().cpu().numpy()  # Whisper expects CPU numpy input
        inputs = self.processor(audio=waveform, return_tensors="pt", sampling_rate=16000).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return transcription.strip().upper()

    def compute_wer(
        self,
        pred_waveforms: List[torch.Tensor],
        real_waveforms: List[torch.Tensor],
        transcripts: List[str]
    ):
        pred_hyps = [self.transcribe(wav) for wav in tqdm(pred_waveforms)]
        real_hyps = [self.transcribe(wav) for wav in tqdm(real_waveforms)]

        wer_pred = wer(transcripts, pred_hyps)
        wer_real = wer(transcripts, real_hyps)

        print(f"WER for Predicted Waveforms: {wer_pred:.3f}")
        print(f"WER for Real Waveforms:      {wer_real:.3f}")

        return wer_pred, wer_real
    
    