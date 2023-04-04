if __name__ == "__main__":
    out = 'data/nlsyms.txt'
    langs = ['de', 'ja', 'zh']
    max_sec = 30
    resolution = 0.02

    specials = [
        "<endoftext>",
        "<startoftranscript>",
        *[f"<translate{l}>" for l in langs],
        "<transcribe>",
        "<startofprev>",
        "<notimestamps>",
        *[f"<{i * resolution:.2f}>" for i in range(int(max_sec / resolution) + 1)],
    ]

    with open(out, 'w') as fp:
        for l in specials:
            fp.write(f"{l}\n")
