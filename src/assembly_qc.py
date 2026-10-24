"""
Phase 2: check every assembly before it goes any further.

    python -m src.assembly_qc

Like a quality inspector at the end of an assembly line: each rebuilt genome
is measured (total size, number of pieces, N50, GC content) and anything that
doesn't look like one clean E. coli genome is set aside. Annotating a broken
or contaminated assembly would only feed junk genes into the pangenome.

Writes moradigaravand/assembly_qc.csv (every isolate, with its measurements)
and moradigaravand/passed.txt (run accessions that passed, one per line).
"""
import json

import numpy as np
import pandas as pd

from .data import load_config


def read_fasta_lengths(path):
    """Contig lengths and overall GC fraction of one assembly."""
    lengths, gc, current = [], 0, 0
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if current:
                    lengths.append(current)
                current = 0
            else:
                seq = line.strip().upper()
                current += len(seq)
                gc += seq.count("G") + seq.count("C")
    if current:
        lengths.append(current)
    return np.array(lengths), gc / max(sum(lengths), 1)


def n50(lengths):
    """Sort contigs longest first; N50 is the length at which half the genome is covered."""
    ordered = np.sort(lengths)[::-1]
    return int(ordered[np.searchsorted(np.cumsum(ordered), ordered.sum() / 2)])


def main():
    cfg = load_config()
    qc = cfg["assembly_qc"]
    morad = cfg["data_dir"] / "moradigaravand"
    runs = (morad / "runs.txt").read_text().split()

    rows = []
    for run in runs:
        row = {"run": run, "assembled": False}
        fasta = morad / "assemblies" / f"{run}.fasta"
        if fasta.exists():
            lengths, gc = read_fasta_lengths(fasta)
            row.update(assembled=True, length=int(lengths.sum()), contigs=len(lengths),
                       n50=n50(lengths), gc=round(gc, 4))
        report = morad / "qc" / f"{run}.fastp.json"
        if report.exists():
            after = json.loads(report.read_text())["summary"]["after_filtering"]
            row.update(read_pairs=after["total_reads"] // 2, q30=round(after["q30_rate"], 3))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["passed"] = (df["assembled"]
                    & df["length"].between(qc["min_length"], qc["max_length"])
                    & (df["contigs"] <= qc["max_contigs"])
                    & (df["n50"] >= qc["min_n50"]))
    df.to_csv(morad / "assembly_qc.csv", index=False)
    (morad / "passed.txt").write_text("".join(f"{r}\n" for r in df.loc[df["passed"], "run"]))

    done = df["assembled"].sum()
    print(f"{done}/{len(df)} assembled, {df['passed'].sum()} passed QC")
    if done:
        a = df[df["assembled"]]
        print(f"  length {a['length'].median() / 1e6:.2f} Mb median, contigs {a['contigs'].median():.0f} median, "
              f"N50 {a['n50'].median() / 1e3:.0f} kb median")
    failed = df[df["assembled"] & ~df["passed"]]
    if len(failed):
        print(f"  {len(failed)} set aside, e.g.:\n{failed.head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()
