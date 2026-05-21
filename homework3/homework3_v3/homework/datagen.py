import json
import math
from homework.cot import CoTModel
from homework.data import Dataset


def generate_dataset(
    output_json: str,
    oversample: int = 3,
    temperature: float = 0.6,
    max_examples: int = 50,
):
    model = CoTModel()
    raw = list(Dataset("train"))[:max_examples]

    out = []
    stats = {"pass1": 0, "pass2": 0, "skipped": 0}

    print(f"Starting datagen on {len(raw)} examples")
    print(f"Saving to: {output_json}")
    print(f"oversample={oversample}, temperature={temperature}")

    for idx, (question, true_answer) in enumerate(raw, start=1):
        prompt = model.format_prompt(question)

        chosen = None

        # pass 1: greedy
        try:
            gen0 = model.batched_generate(
                [prompt],
                num_return_sequences=1,
                temperature=0.0,
            )[0][0]

            pred0 = model.parse_answer(gen0)
            if (not math.isnan(pred0)) and abs(pred0 - true_answer) < 1e-3:
                chosen = gen0
                stats["pass1"] += 1
        except Exception as e:
            print(f"[{idx}] greedy generation failed: {e}")

        # pass 2: sampled
        if chosen is None:
            try:
                gens = model.batched_generate(
                    [prompt],
                    num_return_sequences=oversample,
                    temperature=temperature,
                )[0]

                for gen in gens:
                    pred = model.parse_answer(gen)
                    if (not math.isnan(pred)) and abs(pred - true_answer) < 1e-3:
                        chosen = gen
                        stats["pass2"] += 1
                        break
            except Exception as e:
                print(f"[{idx}] sampled generation failed: {e}")

        if chosen is not None:
            out.append([question, true_answer, chosen])
        else:
            stats["skipped"] += 1

        # progress print every 5 examples
        if idx % 5 == 0 or idx == len(raw):
            print(
                f"Processed {idx}/{len(raw)} | kept={len(out)} "
                f"(pass1={stats['pass1']}, pass2={stats['pass2']}, skipped={stats['skipped']})"
            )

            # save partial progress every 5 examples
            with open(output_json, "w") as f:
                json.dump(out, f, indent=2)

    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"Done. Wrote {len(out)} examples "
        f"(pass1={stats['pass1']}, pass2={stats['pass2']}, skipped={stats['skipped']}) "
        f"to {output_json}"
    )


if __name__ == "__main__":
    from fire import Fire
    Fire(generate_dataset)