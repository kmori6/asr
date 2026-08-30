import argparse
from pathlib import Path

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Lowercase, Replace, Sequence, Strip
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

BLANK_TOKEN = "[BLANK]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SPECIAL_TOKENS = [BLANK_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    text_path = args.text_path.expanduser().resolve()
    if not text_path.is_file():
        parser.error(f"--text_path does not exist: {text_path}")
    if args.vocab_size <= len(SPECIAL_TOKENS):
        parser.error(f"--vocab_size must be greater than {len(SPECIAL_TOKENS)}")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use BPE for subword learning and reserve [UNK] for symbols outside the
    # learned LibriSpeech alphabet.
    # https://huggingface.co/docs/tokenizers/v0.22.2/api/models
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    # Serialize the transcript normalization with the tokenizer so training
    # and inference use the same NFKC, lowercase, and whitespace contract.
    # Lowercase is equivalent to casefold for the ASCII LibriSpeech transcripts.
    # https://huggingface.co/docs/tokenizers/v0.22.2/api/normalizers
    tokenizer.normalizer = Sequence(
        [
            NFKC(),
            Lowercase(),
            Replace(Regex(r"\s+"), " "),
            Strip(),
        ]
    )

    # ByteLevel maps input bytes to visible symbols before BPE. Do not add an
    # artificial leading space because ASR decoding should reproduce the
    # normalized transcript exactly. The 128-token experiment intentionally
    # learns only bytes observed in English LibriSpeech; seeding all 256 symbols
    # via ByteLevel.alphabet() would require at least 258 vocabulary entries.
    # https://huggingface.co/docs/tokenizers/v0.22.2/api/pre-tokenizers
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Reverse ByteLevel's visible-symbol mapping when token IDs are decoded.
    # https://huggingface.co/docs/tokenizers/v0.22.2/api/decoders
    tokenizer.decoder = ByteLevelDecoder()

    # vocab_size includes the observed alphabet and all special tokens.
    # https://huggingface.co/docs/tokenizers/v0.22.2/api/trainers
    tokenizer.train(
        files=[str(text_path)],
        trainer=BpeTrainer(
            vocab_size=args.vocab_size,
            special_tokens=SPECIAL_TOKENS,
        ),
    )

    bos_token_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
    if bos_token_id is None or eos_token_id is None:
        raise RuntimeError("Tokenizer training did not create BOS and EOS tokens")
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, bos_token_id), (EOS_TOKEN, eos_token_id)],
    )

    # Register [UNK] with its standard role and [BLANK] as an additional special
    # token so both survive save/load and can be excluded by skip_special_tokens.
    # https://huggingface.co/docs/transformers/v5.1.0/en/main_classes/tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        additional_special_tokens=[BLANK_TOKEN],
    )
    fast_tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
