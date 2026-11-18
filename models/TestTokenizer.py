from data.data_loader import PaligemmaTokenizer
import numpy as np

if __name__ == "__main__":

    max_len = 40
    tokenizer_path = r"/media/sunhao/T71/Pi0.5/paligemma_tokenizer/paligemma_tokenizer.model"
    tok = PaligemmaTokenizer(tokenizer_path, max_len=max_len)

    def _show(label: str, tokens: np.ndarray, mask: np.ndarray) -> None:
        n = int(mask.sum())
        decoded = tok._sp.decode([int(x) for x in tokens[:n].tolist()])
        print(f"[{label}] shape={tuple(tokens.shape)} valid_len={n}/{len(tokens)}")
        print(f"         decoded={decoded!r}")
        print(f"         first_20_tokens={tokens[:20].tolist()}")
        print(f"         first_20_mask=  {mask[:20].tolist()}")

    print("=== 1. Pi0 mode (no state) ===")
    t0, m0 = tok.tokenize("pick up the")
    _show("pi0", t0, m0)
    assert t0.shape == (max_len,) and t0.dtype == np.int32
    assert m0.shape == (max_len,) and m0.dtype == np.bool_
    n0 = int(m0.sum())
    assert n0 < max_len, "short prompt should leave padding"
    assert bool((t0[n0:] == 0).all()), "padding region must be zero"
    assert bool((~m0[n0:]).all()), "padding region of mask must be False"

    print()
    print("=== 2. Pi0.5 mode (discrete state in prompt) ===")
    state = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.75], dtype=np.float32)
    t1, m1 = tok.tokenize("pick up the", state=state)
    _show("pi05", t1, m1)
    assert t1.shape == (max_len,)
    assert not np.array_equal(t0, t1), "pi0 and pi05 outputs must differ"
    n1 = int(m1.sum())
    assert n1 >= n0, "pi05 prompt (Task+State+Action prefix) is at least as long as pi0"

    print()
    print("=== 3. Truncation (very long prompt) ===")
    long_prompt = ("very long prompt with many words " * 20).strip()
    t2, m2 = tok.tokenize(long_prompt)
    _show("long", t2, m2)
    assert t2.shape == (max_len,)
    assert bool(m2.all()), "truncated case should have all-True mask"

    print()
    print("=== 4. Determinism ===")
    t3a, _ = tok.tokenize("identical prompt")
    t3b, _ = tok.tokenize("identical prompt")
    assert np.array_equal(t3a, t3b), "tokenize must be deterministic"
    print("OK — identical prompts produce identical tokens")

    print()
    print("=== 5. Underscore / newline cleanup ===")
    t4, _ = tok.tokenize("pick_up\nthe red\nblock")
    t5, _ = tok.tokenize("pick up the red block")
    assert np.array_equal(t4, t5), "_ and \\n should be normalized to spaces"
    print("OK — '_' and '\\n' are normalized")

    print()
    print("All checks passed.")
