import argparse
import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    def __init__(self, model_name: str, batch_size: int = 16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.batch_size = batch_size

    def _read_file(self, filepath: str) -> list[str]:
        with open(filepath, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]

    def _write_file(self, lines: list[str], filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

    def _translate_batch(self, texts: list[str]) -> list[str]:
        outputs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                generated_tokens = self.model.generate(**inputs)
            
            decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            outputs.extend(decoded)
            
        return outputs

    def evaluate_model(self, input_path: str, ref_path: str, output_path: str) -> float:
        source_texts = self._read_file(input_path)
        reference_texts = self._read_file(ref_path)
        
        translated_texts = self._translate_batch(source_texts)
        self._write_file(translated_texts, output_path)
        
        bleu = sacrebleu.corpus_bleu(translated_texts, [reference_texts])
        return bleu.score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Helsinki-NLP/opus-mt-<src_lang>-<tgt_lang>")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.txt")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    translator = Translator(args.model, args.batch_size)
    bleu_score = translator.evaluate_model(args.input, args.reference, args.output)
    
    print(f"BLEU:{bleu_score:.2f}")

if __name__ == "__main__":
    main()