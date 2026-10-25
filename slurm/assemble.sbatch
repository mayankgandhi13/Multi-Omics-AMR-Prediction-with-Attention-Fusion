#!/bin/bash
#SBATCH --job-name=amr-assemble
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/assemble/%x_%A_%a.out
# One isolate per array task: download reads -> clean (fastp) -> assemble (Shovill).
#
# Like rebuilding a shredded book: fetch the strips, throw out the smudged ones,
# then stitch the rest back together by their overlaps.
# Restartable: an isolate that already has an assembly is skipped, and reads
# already on disk are not downloaded again. Submitted by slurm/run_assembly.sh.
set -eo pipefail
source slurm/common.sh

line=$(( ${OFFSET:-0} + SLURM_ARRAY_TASK_ID + 1 ))   # SLURM caps arrays at 1,001 tasks, so big runs come in chunks
RUN=$(sed -n "${line}p" "$MORAD_DIR/runs.txt")
OUT="$MORAD_DIR/assemblies/$RUN.fasta"
READS="$MORAD_DIR/reads/$RUN"
echo "Isolate $line: $RUN on $(hostname)"
if [ -s "$OUT" ]; then echo "Already assembled, skipping."; exit 0; fi
mkdir -p "$READS" "$MORAD_DIR/assemblies" "$MORAD_DIR/qc"

# 1. Download the paired reads from ENA (kept on /scratch so reruns skip this).
# ENA pushes back ("503 busy", dropped connections) when 50 downloads arrive at
# once, so we knock politely: start at a random moment, wait longer after each
# refusal, resume half-finished files, and only accept a file that unzips cleanly.
patient_curl() {   # patient_curl <output> <url>
    for attempt in 1 2 3 4 5 6 7 8; do
        if curl -sS --fail -C - --retry 3 --retry-all-errors --retry-delay 15 -o "$1" "$2"; then return 0; fi
        sleep $(( attempt * 30 + RANDOM % 30 ))
    done
    return 1
}
sleep $(( RANDOM % 90 ))

urls=""
for attempt in 1 2 3 4 5; do
    urls=$(curl -sS --fail "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$RUN&result=read_run&fields=fastq_ftp" \
           | tail -n 1 | cut -f2 | tr ';' ' ') || true
    [ -n "$urls" ] && break
    sleep $(( attempt * 30 ))
done
[ -n "$urls" ] || { echo "Could not get read URLs for $RUN from ENA"; exit 1; }

for url in $urls; do
    file="$READS/$(basename "$url")"
    # A file already on disk gets the same check; a corrupt one is fetched again.
    if [ -s "$file" ] && gzip -t "$file" 2>/dev/null; then continue; fi
    rm -f "$file"
    patient_curl "$file.partial" "https://$url"
    if ! gzip -t "$file.partial" 2>/dev/null; then   # complete-looking but corrupt: start over next time
        rm -f "$file.partial"; echo "Corrupt download: $url"; exit 1
    fi
    mv "$file.partial" "$file"
done
# Some isolates are listed by sample ID (ERS...) but ENA names the files by run
# ID (ERR...), so find the pair by its _1/_2 ending rather than by name.
R1=$(ls "$READS"/*_1.fastq.gz | head -1); R2=$(ls "$READS"/*_2.fastq.gz | head -1)

# 2 + 3 work on the node's local disk: fast, and nothing half-done lands on /scratch.
WORK="/tmp/$USER-$RUN"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"

# 2. fastp: trim adapters and low-quality ends; keep its quality report.
tool fastp fastp -i "$R1" -I "$R2" -o "$WORK/clean_1.fq.gz" -O "$WORK/clean_2.fq.gz" \
    --thread "$SLURM_CPUS_PER_TASK" --json "$MORAD_DIR/qc/$RUN.fastp.json" --html /dev/null

# 3. Shovill: SPAdes assembly with read correction, polishing and junk-contig removal.
# --gsize 5M: we already know an E. coli genome is ~5 Mb, so skip Shovill's own
# size estimate (kmc), which hung on 13 isolates in the first full run.
tool shovill shovill --R1 "$WORK/clean_1.fq.gz" --R2 "$WORK/clean_2.fq.gz" --outdir "$WORK/shovill" \
    --gsize 5M --cpus "$SLURM_CPUS_PER_TASK" --ram 14 --tmpdir "$WORK"

cp "$WORK/shovill/contigs.fa" "$OUT.partial"
mv "$OUT.partial" "$OUT"   # only a finished assembly gets the real name
echo "Done: $(grep -c '>' "$OUT") contigs"
