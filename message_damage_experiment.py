from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embedding import to_u8_array
from read_changes import (
    iter_record_paths,
    load_change_record,
    resolve_probe_image_path,
    sample_pixel_indices,
)


# Input records produced by embedding.py.
INPUT_PATH = Path("output")

# Where to find original covers if PROBE_IMAGE_SOURCE == "cover".
COVER_ROOT = Path("ALASKA_v2_TIFF_512_GrayScale_50")

# The current HUGO/MiPOD simulator does not expose a decoder, so this script
# builds a reproducible coded payload on top of the saved stego images.
# Payload positions are chosen from simulator-changed pixels.
PAYLOAD_POSITION_SOURCE = "changed_pixels"
PAYLOAD_POSITION_FRACTION = 0.5
MESSAGE_SEED = 20260428

# Simple error-control layer for a more truthful destruction benchmark.
# Supported values: uncoded, repetition
ECC_MODE = "repetition"
REPETITION_FACTOR = 3
INTERLEAVE_CODEWORD = True

# Probe/attack policy.
PROBE_DISTRIBUTION = "texture_energy"
PROBE_DISTRIBUTION_PARAMS = {
    "gamma": 1.0,
    "floor": 1e-6,
}
PROBE_IMAGE_SOURCE = "stego"
PROBE_SEED = 41

# Attack budgets in number of altered pixels.
ATTACK_BUDGETS = [100, 250, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 40000]

# Attack model applied to probed pixels.
# Supported values: lsb_flip, plus_minus_one
ATTACK_MODE = "plus_minus_one"
ATTACK_SEED = 314159

# "Destroyed" means the decoded message is not exactly recovered.
# "High damage" means the decoded BER crosses a practical threshold.
HIGH_DAMAGE_DECODED_BER_THRESHOLD = 0.10

# CSV outputs for Excel.
DETAIL_CSV_PATH = Path("output/message_damage_detail.csv")
SUMMARY_CSV_PATH = Path("output/message_damage_summary.csv")


SUPPORTED_PAYLOAD_POSITION_SOURCES = {"changed_pixels"}
SUPPORTED_ECC_MODES = {"uncoded", "repetition"}
SUPPORTED_ATTACK_MODES = {"lsb_flip", "plus_minus_one"}


@dataclass
class PayloadLayout:
    payload_positions: np.ndarray
    payload_lookup: np.ndarray
    raw_message_bits: np.ndarray
    codeword_bits: np.ndarray
    inverse_interleave: np.ndarray
    encoded_image: np.ndarray
    candidate_pool_count: int

    @property
    def raw_message_length_bits(self) -> int:
        return int(self.raw_message_bits.size)

    @property
    def codeword_length_bits(self) -> int:
        return int(self.codeword_bits.size)

    @property
    def code_rate(self) -> float:
        if self.codeword_length_bits == 0:
            return 0.0
        return self.raw_message_length_bits / self.codeword_length_bits


def derive_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big")
    return (int(base_seed) + offset) % (2**32)


def get_codeword_block_size() -> int:
    if ECC_MODE not in SUPPORTED_ECC_MODES:
        raise ValueError(f"Unsupported ECC mode '{ECC_MODE}'. Supported: {sorted(SUPPORTED_ECC_MODES)}")
    if ECC_MODE == "uncoded":
        return 1
    if REPETITION_FACTOR < 3 or REPETITION_FACTOR % 2 == 0:
        raise ValueError("REPETITION_FACTOR must be an odd integer >= 3 for majority decoding.")
    return REPETITION_FACTOR


def select_payload_positions(record: dict) -> tuple[np.ndarray, int]:
    if PAYLOAD_POSITION_SOURCE not in SUPPORTED_PAYLOAD_POSITION_SOURCES:
        raise ValueError(
            "Unsupported PAYLOAD_POSITION_SOURCE "
            f"'{PAYLOAD_POSITION_SOURCE}'. Supported: {sorted(SUPPORTED_PAYLOAD_POSITION_SOURCES)}"
        )
    if not 0 < PAYLOAD_POSITION_FRACTION <= 1:
        raise ValueError(
            f"PAYLOAD_POSITION_FRACTION must be in (0, 1], got {PAYLOAD_POSITION_FRACTION}."
        )

    changed_positions = record["changed_flat_indices"].astype(np.int64, copy=False)
    if changed_positions.size == 0:
        raise ValueError(f"Record has no changed pixels: {record['record_path']}")

    candidate_pool_count = max(1, int(round(PAYLOAD_POSITION_FRACTION * changed_positions.size)))
    block_size = get_codeword_block_size()
    raw_message_length = candidate_pool_count // block_size
    codeword_length = raw_message_length * block_size
    if codeword_length <= 0:
        raise ValueError(
            "Payload pool is too small for the configured ECC. "
            f"candidate_pool_count={candidate_pool_count}, block_size={block_size}"
        )

    selection_seed = derive_seed(MESSAGE_SEED, record["record_path"].name, "payload_positions")
    rng = np.random.default_rng(selection_seed)
    permutation = rng.permutation(changed_positions.size)
    payload_positions = changed_positions[permutation[:codeword_length]]
    return payload_positions, candidate_pool_count


def generate_raw_message_bits(length: int, record_name: str) -> np.ndarray:
    if length <= 0:
        raise ValueError(f"Raw message length must be positive, got {length}.")
    seed = derive_seed(MESSAGE_SEED, record_name, "raw_message_bits")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=length, dtype=np.uint8)


def encode_message_bits(raw_message_bits: np.ndarray) -> np.ndarray:
    if ECC_MODE == "uncoded":
        return raw_message_bits.astype(np.uint8, copy=True)
    return np.repeat(raw_message_bits.astype(np.uint8, copy=False), REPETITION_FACTOR)


def decode_message_bits(received_codeword_bits: np.ndarray) -> np.ndarray:
    if ECC_MODE == "uncoded":
        return received_codeword_bits.astype(np.uint8, copy=True)

    groups = received_codeword_bits.reshape(-1, REPETITION_FACTOR)
    threshold = REPETITION_FACTOR // 2 + 1
    return (groups.sum(axis=1) >= threshold).astype(np.uint8)


def build_interleave(length: int, record_name: str) -> tuple[np.ndarray, np.ndarray]:
    if not INTERLEAVE_CODEWORD:
        identity = np.arange(length, dtype=np.int64)
        return identity, identity

    seed = derive_seed(MESSAGE_SEED, record_name, "codeword_interleave")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(length).astype(np.int64, copy=False)
    inverse_permutation = np.empty_like(permutation)
    inverse_permutation[permutation] = np.arange(length, dtype=np.int64)
    return permutation, inverse_permutation


def write_bits_to_lsb(image: np.ndarray, positions: np.ndarray, bits: np.ndarray) -> np.ndarray:
    if positions.size != bits.size:
        raise ValueError("positions and bits must have the same length.")
    encoded = image.copy()
    flat = encoded.reshape(-1)
    current_bits = flat[positions] & 1
    mismatch = current_bits != bits
    if np.any(mismatch):
        flat[positions[mismatch]] ^= 1
    return encoded


def read_bits_from_lsb(image: np.ndarray, positions: np.ndarray) -> np.ndarray:
    flat = image.reshape(-1)
    return (flat[positions] & 1).astype(np.uint8, copy=False)


def apply_attack(
    image: np.ndarray,
    attack_positions: np.ndarray,
    attack_seed: int,
    attack_mode: str | None = None,
) -> np.ndarray:
    mode = ATTACK_MODE if attack_mode is None else str(attack_mode)
    if mode not in SUPPORTED_ATTACK_MODES:
        raise ValueError(
            f"Unsupported attack mode '{mode}'. Supported: {sorted(SUPPORTED_ATTACK_MODES)}"
        )

    attacked = image.copy()
    flat = attacked.reshape(-1)

    if mode == "lsb_flip":
        flat[attack_positions] ^= 1
        return attacked

    rng = np.random.default_rng(attack_seed)
    deltas = rng.choice(np.array([-1, 1], dtype=np.int16), size=attack_positions.size)
    values = flat[attack_positions].astype(np.int16, copy=False)
    proposed = values + deltas
    low = proposed < 0
    high = proposed > 255
    proposed[low] = values[low] + 1
    proposed[high] = values[high] - 1
    flat[attack_positions] = proposed.astype(np.uint8)
    return attacked


def compute_psnr(reference: np.ndarray, attacked: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float32) - attacked.astype(np.float32)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / mse)


def prepare_payload(record: dict, stego_image: np.ndarray) -> PayloadLayout:
    payload_positions, candidate_pool_count = select_payload_positions(record)
    raw_message_length = payload_positions.size // get_codeword_block_size()
    raw_message_bits = generate_raw_message_bits(raw_message_length, record["record_path"].name)
    codeword_bits = encode_message_bits(raw_message_bits)
    interleave, inverse_interleave = build_interleave(codeword_bits.size, record["record_path"].name)
    stored_bits = codeword_bits[interleave]
    encoded_image = write_bits_to_lsb(stego_image, payload_positions, stored_bits)

    payload_lookup = np.zeros(record["image_shape"][0] * record["image_shape"][1], dtype=bool)
    payload_lookup[payload_positions] = True

    return PayloadLayout(
        payload_positions=payload_positions,
        payload_lookup=payload_lookup,
        raw_message_bits=raw_message_bits,
        codeword_bits=codeword_bits,
        inverse_interleave=inverse_interleave,
        encoded_image=encoded_image,
        candidate_pool_count=candidate_pool_count,
    )


def extract_received_codeword_bits(attacked_image: np.ndarray, payload: PayloadLayout) -> np.ndarray:
    stored_bits = read_bits_from_lsb(attacked_image, payload.payload_positions)
    return stored_bits[payload.inverse_interleave]


def sample_attack_order(record: dict, guide_image: np.ndarray, max_budget: int) -> np.ndarray:
    return sample_pixel_indices(
        image_shape=record["image_shape"],
        sample_count=max_budget,
        seed=derive_seed(PROBE_SEED, record["record_path"].name, PROBE_DISTRIBUTION),
        distribution=PROBE_DISTRIBUTION,
        distribution_params=PROBE_DISTRIBUTION_PARAMS,
        sampling_image=guide_image if PROBE_DISTRIBUTION != "uniform" else None,
    )


def evaluate_attack_budget(
    *,
    record: dict,
    payload: PayloadLayout,
    attack_budget: int,
    attack_positions: np.ndarray,
    attacked_image: np.ndarray,
) -> dict:
    received_codeword_bits = extract_received_codeword_bits(attacked_image, payload)
    decoded_message_bits = decode_message_bits(received_codeword_bits)

    attacked_payload_positions = int(np.count_nonzero(payload.payload_lookup[attack_positions]))
    channel_bit_errors = int(np.count_nonzero(received_codeword_bits != payload.codeword_bits))
    decoded_bit_errors = int(np.count_nonzero(decoded_message_bits != payload.raw_message_bits))

    channel_ber = channel_bit_errors / payload.codeword_length_bits
    decoded_ber = decoded_bit_errors / payload.raw_message_length_bits
    decode_failure_flag = int(decoded_bit_errors > 0)
    high_damage_flag = int(decoded_ber >= HIGH_DAMAGE_DECODED_BER_THRESHOLD)

    return {
        "record_name": record["record_path"].name,
        "record_path": str(record["record_path"]),
        "cover_relative_path": record["cover_relative_path"],
        "stego_relative_path": record["stego_relative_path"],
        "method": record["method"],
        "alpha": record["alpha"],
        "record_seed": int(record["seed"]),
        "message_seed": derive_seed(MESSAGE_SEED, record["record_path"].name, "raw_message_bits"),
        "image_height": int(record["image_shape"][0]),
        "image_width": int(record["image_shape"][1]),
        "used_pixels": int(record["used_pixels"]),
        "used_pixel_fraction": float(record["used_pixel_fraction"]),
        "payload_position_source": PAYLOAD_POSITION_SOURCE,
        "payload_position_fraction": float(PAYLOAD_POSITION_FRACTION),
        "candidate_pool_count": int(payload.candidate_pool_count),
        "payload_position_count": int(payload.payload_positions.size),
        "ecc_mode": ECC_MODE,
        "repetition_factor": int(get_codeword_block_size()),
        "interleave_codeword": int(INTERLEAVE_CODEWORD),
        "raw_message_length_bits": int(payload.raw_message_length_bits),
        "codeword_length_bits": int(payload.codeword_length_bits),
        "code_rate": float(payload.code_rate),
        "probe_distribution": PROBE_DISTRIBUTION,
        "probe_image_source": PROBE_IMAGE_SOURCE,
        "probe_seed": int(PROBE_SEED),
        "attack_mode": ATTACK_MODE,
        "attack_seed": int(ATTACK_SEED),
        "attack_budget": int(attack_budget),
        "attacked_payload_positions": attacked_payload_positions,
        "attack_hit_rate": attacked_payload_positions / attack_budget if attack_budget else 0.0,
        "payload_hit_fraction": attacked_payload_positions / payload.codeword_length_bits,
        "channel_bit_errors": channel_bit_errors,
        "channel_ber": channel_ber,
        "decoded_bit_errors": decoded_bit_errors,
        "decoded_ber": decoded_ber,
        "decode_failure_flag": decode_failure_flag,
        "high_damage_threshold": float(HIGH_DAMAGE_DECODED_BER_THRESHOLD),
        "high_damage_flag": high_damage_flag,
        "psnr_db_vs_payload_stego": compute_psnr(payload.encoded_image, attacked_image),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_by_budget(detail_rows: list[dict]) -> list[dict]:
    budgets = sorted({int(row["attack_budget"]) for row in detail_rows})
    summary_rows: list[dict] = []
    for budget in budgets:
        rows = [row for row in detail_rows if int(row["attack_budget"]) == budget]
        summary_rows.append(
            {
                "attack_budget": budget,
                "num_records": len(rows),
                "mean_attack_hit_rate": float(np.mean([row["attack_hit_rate"] for row in rows])),
                "mean_payload_hit_fraction": float(np.mean([row["payload_hit_fraction"] for row in rows])),
                "mean_channel_ber": float(np.mean([row["channel_ber"] for row in rows])),
                "mean_decoded_ber": float(np.mean([row["decoded_ber"] for row in rows])),
                "mean_decode_failure_rate": float(np.mean([row["decode_failure_flag"] for row in rows])),
                "mean_high_damage_rate": float(np.mean([row["high_damage_flag"] for row in rows])),
                "mean_psnr_db_vs_payload_stego": float(
                    np.mean([row["psnr_db_vs_payload_stego"] for row in rows])
                ),
            }
        )
    return summary_rows


def run_experiment() -> tuple[list[dict], list[dict]]:
    input_path = Path(INPUT_PATH).expanduser().resolve()
    input_base = input_path if input_path.is_dir() else input_path.parent
    budgets = sorted(set(int(budget) for budget in ATTACK_BUDGETS))
    max_budget = max(budgets)

    detail_rows: list[dict] = []
    for record_path in iter_record_paths(input_path):
        record = load_change_record(record_path)
        stego_image_path = resolve_probe_image_path(
            record=record,
            input_base=input_base,
            image_source="stego",
            cover_root=COVER_ROOT,
        )
        guide_image_path = resolve_probe_image_path(
            record=record,
            input_base=input_base,
            image_source=PROBE_IMAGE_SOURCE,
            cover_root=COVER_ROOT,
        )

        stego_image = to_u8_array(stego_image_path)
        guide_image = to_u8_array(guide_image_path)
        payload = prepare_payload(record, stego_image)
        attack_order = sample_attack_order(record, guide_image, max_budget)

        for attack_budget in budgets:
            attack_positions = attack_order[:attack_budget]
            attacked_image = apply_attack(
                image=payload.encoded_image,
                attack_positions=attack_positions,
                attack_seed=derive_seed(ATTACK_SEED, record["record_path"].name, str(attack_budget)),
            )
            detail_rows.append(
                evaluate_attack_budget(
                    record=record,
                    payload=payload,
                    attack_budget=attack_budget,
                    attack_positions=attack_positions,
                    attacked_image=attacked_image,
                )
            )

    summary_rows = summarize_by_budget(detail_rows)
    return detail_rows, summary_rows


if __name__ == "__main__":
    detail_rows, summary_rows = run_experiment()
    write_csv(Path(DETAIL_CSV_PATH).expanduser().resolve(), detail_rows)
    write_csv(Path(SUMMARY_CSV_PATH).expanduser().resolve(), summary_rows)
    print(f"Wrote {len(detail_rows)} detail rows to {Path(DETAIL_CSV_PATH).expanduser().resolve()}")
    print(f"Wrote {len(summary_rows)} summary rows to {Path(SUMMARY_CSV_PATH).expanduser().resolve()}")
