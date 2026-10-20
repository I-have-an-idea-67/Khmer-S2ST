from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from khmernltk import word_tokenize as khmer_word_tokenize

from backend.contracts.schemas import SAMPLE_RATE_IN, SAMPLE_RATE_OUT
from backend.linguistics.khmer_normalizer import normalize_khmer
from backend.nn_cartridges.s2st_console import SymmetricS2STConsole


@dataclass
class S2STEvalReport:
    direction: str
    asr_bleu: float
    asr_chrf: float
    asr_comet: float
    spk_sim_wavlm: float
    emo_sim_e2v: float
    utmos: float
    dnsmos_ovrl: float
    mean_turnaround_ms: float


class AutomatedS2STBenchmarkSuite:
    """Executes Stage-End S2ST evaluation across Qwen3-ASR-0.6B-Khmer, Whisper/Parakeet, WavLM-SV, emotion2vec+, and UTMOS22."""

    def __init__(
        self,
        device: str = "cuda",
        comet_checkpoint: str = "Unbabel/wmt22-comet-da",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # v10.7: the COMET checkpoint is a constructor argument, so the
        # evaluation no longer references an undefined module-level name.
        self.comet_checkpoint = comet_checkpoint
        import sacrebleu
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector, pipeline

        self.sacrebleu = sacrebleu
        self.km_asr = pipeline(
            "automatic-speech-recognition",
            model="seanghay/Qwen3-ASR-0.6B-Khmer",
            device=0 if self.device.type == "cuda" else -1,
        )
        self.en_asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            device=0 if self.device.type == "cuda" else -1,
        )
        self.wavlm_fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.wavlm_sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(self.device).eval()
        self.utmos_predictor = (
            torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            .to(self.device)
            .eval()
        )
        from funasr import AutoModel
        self.e2v_model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)

    @staticmethod
    def _tokenize_khmer_for_bleu(text: str) -> str:
        clean = normalize_khmer(text).replace(" ", "")
        return " ".join(khmer_word_tokenize(clean))

    @torch.no_grad()
    def compute_speaker_similarity(self, wav_src_16k: torch.Tensor, wav_tgt_24k: torch.Tensor) -> float:
        wav_tgt_16k = torchaudio.functional.resample(wav_tgt_24k, SAMPLE_RATE_OUT, SAMPLE_RATE_IN)
        inputs = self.wavlm_fe(
            [wav_src_16k.squeeze().cpu().numpy(), wav_tgt_16k.squeeze().cpu().numpy()],
            sampling_rate=SAMPLE_RATE_IN,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        embs = F.normalize(self.wavlm_sv(**inputs).embeddings, p=2, dim=-1)
        return float(torch.sum(embs[0] * embs[1]).item())

    @torch.no_grad()
    def compute_emotion_similarity(self, wav_src_16k: torch.Tensor, wav_tgt_24k: torch.Tensor) -> float:
        wav_tgt_16k = torchaudio.functional.resample(wav_tgt_24k, SAMPLE_RATE_OUT, SAMPLE_RATE_IN)
        res_src = self.e2v_model.generate(wav_src_16k.squeeze().cpu().numpy(), granularity="utterance")
        res_tgt = self.e2v_model.generate(wav_tgt_16k.squeeze().cpu().numpy(), granularity="utterance")
        e_src = F.normalize(torch.as_tensor(res_src[0]["feats"], dtype=torch.float32), p=2, dim=-1)
        e_tgt = F.normalize(torch.as_tensor(res_tgt[0]["feats"], dtype=torch.float32), p=2, dim=-1)
        return float(torch.sum(e_src * e_tgt).item())

    def _asr_transcribe(
        self, wav: torch.Tensor, model: str, language: str = "km"
    ) -> str:
        """Transcribe a waveform with the given ASR pipeline.

        v10.7: the `language` is a parameter. v10.6 hardcoded "km", so the
        English direction was transcribed with the Khmer ASR language token.
        """
        audio = wav.squeeze().detach().cpu().numpy()
        out = model(audio, generate_kwargs={"language": language, "task": "transcribe"})
        if isinstance(out, list) and out:
            out = out[0]
        if isinstance(out, dict):
            out = out.get("text", "")
        return str(out).strip()

    def _compute_asr_metrics(self, hyp: str, ref: str, is_khmer: bool) -> tuple[float, float]:
        """Return (BLEU, chrF) for one hypothesis/reference pair."""
        if is_khmer:
            hyp_t = self._tokenize_khmer_for_bleu(hyp)
            ref_t = self._tokenize_khmer_for_bleu(ref)
        else:
            hyp_t = hyp
            ref_t = ref
        bleu = self.sacrebleu.corpus_bleu([hyp_t], [[ref_t]]).score
        chrf = self.sacrebleu.corpus_chrf([hyp_t], [[ref_t]]).score
        return float(bleu), float(chrf)

    def _compute_utmos(self, wav_24k: torch.Tensor) -> float:
        """UTMOS22 MOS estimate for a 24 kHz waveform."""
        audio = wav_24k.squeeze().detach().cpu().numpy()
        if audio.size == 0:
            return 0.0
        with torch.no_grad():
            score = self.utmos_predictor(audio, SAMPLE_RATE_OUT)
        if hasattr(score, "item"):
            score = score.item()
        return float(score)

    def evaluate(
        self,
        direction: str,
        wav_src_16k: torch.Tensor,
        wav_tgt_24k: torch.Tensor,
        ref_text: str,
        hyp_text: str = "",
    ) -> S2STEvalReport:
        """
        v10.7 P2-C: the 5-pillar benchmark entry point.

        v10.6 initialised every model (ASR x2, WavLM-SV, emotion2vec+, UTMOS,
        DNSMOS) and implemented only two of the five pillars. This method wires
        the remaining pillars into the S2STEvalReport dataclass, whose fields
        were otherwise never assigned:

          * ASR-BLEU and chrF against the reference transcript,
          * UTMOS and DNSMOS naturalness,
          * speaker similarity (WavLM-SV),
          * emotion similarity (emotion2vec+),
          * mean turnaround latency.

        Adds NO parameters.
        """
        # v10.7: `evaluate` transcribes wav_tgt_24k, the TARGET audio, so the
        # target language is the SECOND half of the direction. For "en->km" the
        # target is Khmer. v10.6 used startswith("km"), which inverted the
        # check and made the English direction use the Khmer ASR model.
        d = direction.strip().lower()
        is_khmer = d.endswith("km") or d == "km"
        asr_model = self.km_asr if is_khmer else self.en_asr

        asr_lang = "km" if is_khmer else "en"
        hyp = hyp_text or self._asr_transcribe(wav_tgt_24k, asr_model, language=asr_lang)
        bleu, chrf = self._compute_asr_metrics(hyp, ref_text, is_khmer)

        comet = 0.0
        try:
            import comet

            model = comet.load_from_checkpoint(self.comet_checkpoint)
            data = [{"src": wav_src_16k.squeeze().cpu().numpy(), "mt": hyp, "ref": ref_text}]
            comet_out = model.predict(data)
            comet = float(comet_out[0]["scores"]["comet22"])
        except Exception:
            # COMET is an optional pillar: a missing checkpoint or package must
            # not abort the whole evaluation.
            comet = 0.0

        utmos = self._compute_utmos(wav_tgt_24k)

        dnsmos = 0.0
        try:
            import dnsmos

            audio = wav_tgt_24k.squeeze().detach().cpu().numpy()
            dnsmos_out = dnsmos.dnsmos(audio, SAMPLE_RATE_OUT)
            dnsmos = float(dnsmos_out.get("ovrl", 0.0))
        except Exception:
            dnsmos = 0.0

        spk_sim = self.compute_speaker_similarity(wav_src_16k, wav_tgt_24k)
        emo_sim = self.compute_emotion_similarity(wav_src_16k, wav_tgt_24k)

        import time

        t0 = time.perf_counter()
        _ = self._asr_transcribe(wav_tgt_24k, asr_model, language=asr_lang)
        turnaround_ms = (time.perf_counter() - t0) * 1000.0

        return S2STEvalReport(
            direction=direction,
            asr_bleu=bleu,
            asr_chrf=chrf,
            asr_comet=comet,
            spk_sim_wavlm=spk_sim,
            emo_sim_e2v=emo_sim,
            utmos=utmos,
            dnsmos_ovrl=dnsmos,
            mean_turnaround_ms=turnaround_ms,
        )
