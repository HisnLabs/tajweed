# Daghia hifz student: Tarteel's Quran-tuned Whisper-base ENCODER (no decoder) + multi-level CTC heads that write the
# Quran phonetic script (madd length = repeated symbol, ghunnah, qalqalah...) and the 10 sifat, like the Muaalem teacher.
# ~20M parameters (≈ 21 MB in 8-bit). Unlike Whisper, it takes the real length of the audio (no 30 s padding) and
# outputs every frame at once (no word-by-word decoder): a 4 s phrase is one small pass.
import torch, torch.nn as nn, torch.nn.functional as F

# the teacher's alphabet (obadx/muaalem-model-v3_2 vocab.json), index 0 = CTC blank
PHONEMES = ['ء', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك',
            'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ا', 'ۦ', 'ۥ', 'َ', 'ُ', 'ِ', '۪', 'ـ', 'ٲ', 'ڇ', 'ں', '۾', 'ۜ', 'ؙ']
SIFAT = {
    'hams_or_jahr': ['hams', 'jahr'],
    'shidda_or_rakhawa': ['shadeed', 'between', 'rikhw'],
    'tafkheem_or_taqeeq': ['mofakham', 'moraqaq', 'low_mofakham'],
    'itbaq': ['monfateh', 'motbaq'],
    'safeer': ['safeer', 'no_safeer'],
    'qalqla': ['moqalqal', 'not_moqalqal'],
    'tikraar': ['mokarar', 'not_mokarar'],
    'tafashie': ['motafashie', 'not_motafashie'],
    'istitala': ['mostateel', 'not_mostateel'],
    'ghonna': ['maghnoon', 'not_maghnoon'],
}
LEVELS = ['phonemes'] + list(SIFAT)
RIWAYAT = ['hafs', 'warsh']  # the reading the page's button selects: an input of the model (v2)
VOCAB = {'phonemes': PHONEMES, **SIFAT}


class HifzStudent(nn.Module):
    def __init__(self, whisper_encoder):
        super().__init__()
        self.enc = whisper_encoder  # transformers WhisperEncoder (conv1, conv2, embed_positions, layers, layer_norm)
        d = self.enc.config.d_model
        self.heads = nn.ModuleDict({lv: nn.Linear(d, len(VOCAB[lv]) + 1) for lv in LEVELS})
        # Hafs / Warsh: one learned vector added to every frame, so the model knows which reading to expect (Warsh's
        # 6-count madds, taqlil, naql...). Zero at the start: a v1 checkpoint loads and behaves exactly as before.
        self.riwaya = nn.Embedding(len(RIWAYAT), d)
        nn.init.zeros_(self.riwaya.weight)

    def encode(self, mel, frames=None, riwaya=None):
        """mel: [B, 80, T] log-mel (Whisper's), any T ≤ 3000 (even); frames: [B] real lengths (mel frames) or None;
        riwaya: [B] index in RIWAYAT (None = Hafs)"""
        e = self.enc
        x = F.gelu(e.conv1(mel))
        x = F.gelu(e.conv2(x)).transpose(1, 2)  # [B, T/2, d], 50 frames per second
        t = x.shape[1]
        x = x + e.embed_positions.weight[:t].to(x.dtype)
        if riwaya is not None:
            x = x + self.riwaya(riwaya).to(x.dtype)[:, None, :]
        mask = None
        if frames is not None:
            valid = torch.arange(t, device=x.device)[None, :] < ((frames + 1) // 2)[:, None]
            mask = torch.zeros(valid.shape, dtype=x.dtype, device=x.device).masked_fill(~valid, torch.finfo(x.dtype).min)
            mask = mask[:, None, None, :]
        for layer in e.layers:
            o = layer(x, attention_mask=mask, layer_head_mask=None)
            x = o[0] if isinstance(o, tuple) else o
        return e.layer_norm(x)

    def forward(self, mel, frames=None, riwaya=None):
        h = self.encode(mel, frames, riwaya)
        return {lv: head(h) for lv, head in self.heads.items()}  # logits [B, T/2, V+1]


def greedy(logits, vocab):
    """CTC best path → list of labels (blank = 0, repeats collapsed)"""
    ids = logits.argmax(-1).tolist()
    out, prev = [], 0
    for i in ids:
        if i != prev and i != 0: out.append(vocab[i - 1])
        prev = i
    return out
