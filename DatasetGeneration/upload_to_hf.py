import os
import zipfile
import json
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

# HF Audio column format: struct with 'bytes' (binary) and 'path' (string).
# pyarrow writes this natively — no torchcodec / datasets encoding needed.
SCHEMA = pa.schema([
    pa.field("id",       pa.string()),
    pa.field("tier",     pa.string()),
    pa.field("text",     pa.string()),
    pa.field("audio",    pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path",  pa.string()),
    ])),
    pa.field("emotions", pa.string()),
    pa.field("styles",   pa.string()),
])


def chunked_example_generator(main_zip_path, chunk_size=2000):
    """Yields (tier, list_of_rows) batches from the tiered zip."""
    current_chunk = []
    current_tier  = None

    print(f"Opening {main_zip_path}...")
    with zipfile.ZipFile(main_zip_path, "r") as main_zip:
        inner_zips = sorted(
            [info for info in main_zip.infolist() if info.filename.endswith(".zip")],
            key=lambda z: z.filename,
        )
        print(f"Found {len(inner_zips)} inner batch zips.")

        for i, inner_zip_info in enumerate(inner_zips):
            if i % 200 == 0:
                print(f"  [{i}/{len(inner_zips)}] chunk={len(current_chunk)}")

            # Extract tier: KionTTS_Dataset_v2/Tier1/BatchXX/batch.zip → "tier1"
            tier = "unknown"
            for part in inner_zip_info.filename.split("/"):
                if part.lower().startswith("tier"):
                    tier = part.lower()
                    break

            try:
                with main_zip.open(inner_zip_info) as inner_zip_file:
                    with zipfile.ZipFile(inner_zip_file) as inner_zip:
                        if "metadata.json" not in inner_zip.namelist():
                            continue
                        metadata = json.loads(inner_zip.read("metadata.json"))
                        if not isinstance(metadata, list):
                            metadata = list(metadata.values())

                        for item in metadata:
                            wav_id       = item["id"]
                            wav_path     = f"wavs/{wav_id}.wav"
                            if wav_path not in inner_zip.namelist():
                                print(f"  WARN: {wav_path} missing, skipping.")
                                continue

                            audio_bytes  = inner_zip.read(wav_path)
                            segments     = item.get("segments", [{}])
                            first_seg    = segments[0] if segments else {}
                            emotions     = first_seg.get("emotions", {})
                            styles       = first_seg.get("styles", {})

                            row = {
                                "id":       wav_id,
                                "tier":     tier,
                                "text":     item.get("text", ""),
                                "audio":    {"bytes": audio_bytes, "path": wav_path},
                                "emotions": json.dumps(emotions),
                                "styles":   json.dumps(styles),
                            }

                            # Flush when tier switches or chunk is full
                            if current_tier is not None and (
                                tier != current_tier or len(current_chunk) >= chunk_size
                            ):
                                yield current_tier, current_chunk
                                current_chunk = []

                            current_chunk.append(row)
                            current_tier = tier

            except Exception as e:
                print(f"  ERROR reading {inner_zip_info.filename}: {e}")

        if current_chunk:
            yield current_tier, current_chunk


def rows_to_parquet(rows: list, path: str):
    """Write a list of row dicts to a parquet file using the fixed schema."""
    ids       = [r["id"]       for r in rows]
    tiers     = [r["tier"]     for r in rows]
    texts     = [r["text"]     for r in rows]
    audio_b   = [r["audio"]["bytes"] for r in rows]
    audio_p   = [r["audio"]["path"]  for r in rows]
    emotions  = [r["emotions"] for r in rows]
    styles    = [r["styles"]   for r in rows]

    audio_col = pa.StructArray.from_arrays(
        [pa.array(audio_b, type=pa.binary()), pa.array(audio_p, type=pa.string())],
        names=["bytes", "path"],
    )

    table = pa.table({
        "id":       pa.array(ids,      type=pa.string()),
        "tier":     pa.array(tiers,    type=pa.string()),
        "text":     pa.array(texts,    type=pa.string()),
        "audio":    audio_col,
        "emotions": pa.array(emotions, type=pa.string()),
        "styles":   pa.array(styles,   type=pa.string()),
    }, schema=SCHEMA)

    pq.write_table(
        table,
        path,
        row_group_size=100,          # small row groups = HF random access without full scan
        write_page_index=True,       # page index lets HF skip rows without reading full groups
        compression="snappy",
    )


def main():
    parser = argparse.ArgumentParser(description="Upload KionTTS v2 to Hugging Face")
    parser.add_argument("--zip_path",  required=True,        help="Path to KionTTS_Dataset_v2.zip")
    parser.add_argument("--repo_id",   required=True,        help="HF repo ID, e.g. nate0001/KionTTS")
    parser.add_argument("--token",     required=True,        help="HF write token")
    parser.add_argument("--overwrite", action="store_true",  help="Delete existing data/ on HF first")
    parser.add_argument("--chunk_size",type=int, default=500)   # ~100–150 MB per shard for audio
    args = parser.parse_args()

    print("Authenticating with Hugging Face...")
    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    if args.overwrite:
        print("[Overwrite] Deleting existing data/ folder on HF...")
        try:
            api.delete_folder(
                repo_id=args.repo_id, path_in_repo="data", repo_type="dataset"
            )
            print("  Deleted.")
        except Exception as e:
            print(f"  Note: {e}")

    staging = Path(__file__).resolve().parent / "hf_upload_staging"
    staging.mkdir(exist_ok=True)

    shard_counters: dict[str, int] = {}

    for tier, chunk in chunked_example_generator(args.zip_path, chunk_size=args.chunk_size):
        shard_counters[tier] = shard_counters.get(tier, 0) + 1
        idx = shard_counters[tier]

        print(f"\n--- [{tier}] Shard {idx:05d} — {len(chunk)} examples ---")

        shard_file = staging / f"{tier}-{idx:05d}.parquet"
        rows_to_parquet(chunk, str(shard_file))
        print(f"  Written: {shard_file.name}  ({shard_file.stat().st_size / 1e6:.1f} MB)")

        hf_path = f"data/{tier}/train-{idx:05d}.parquet"
        print(f"  Uploading → {hf_path}")
        api.upload_file(
            path_or_fileobj=str(shard_file),
            path_in_repo=hf_path,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        shard_file.unlink()
        print(f"  Done. ({sum(shard_counters.values())} total shards uploaded)")

    print("\n" + "=" * 60)
    print("Upload complete!")
    for t, count in sorted(shard_counters.items()):
        print(f"  {t}: {count} shards  →  data/{t}/train-XXXXX.parquet")


if __name__ == "__main__":
    main()
