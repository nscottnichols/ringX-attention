import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

batch_seqlen_pattern = re.compile(r'Batch size: (\d+).*?Sequence length: (\d+)', re.DOTALL)
perf_pattern = re.compile(r'# (\w+)_func\s+ngpus: (\d+).*?batch: (\d+), seqlen: (\d+), num_heads: (\d+), head_dim: (\d+).*?\n.*? ([0-9.]+) sec')


def _parse_result_lines(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.startswith('BENCHMARK_RESULT '):
                continue
            payload = json.loads(line[len('BENCHMARK_RESULT '):])
            results.append(payload)
    return results


def parse_file(file_path):
    json_results = _parse_result_lines(file_path)
    if json_results:
        parsed = []
        for payload in json_results:
            if payload.get('status') != 'ok':
                continue
            parsed.append(
                (
                    int(payload['batch']),
                    int(payload['seqlen']),
                    payload['impl'].removesuffix('_func'),
                    int(payload['ngpus']),
                    int(payload['num_heads']),
                    int(payload['head_dim']),
                    float(payload['total_sec']),
                    payload.get('mode', 'fwd_bwd'),
                    payload.get('requested_backend', ''),
                    payload.get('forward_backend', ''),
                    payload.get('backward_backend', ''),
                    payload.get('dtype', ''),
                )
            )
        return parsed

    with open(file_path, 'r') as f:
        content = f.read()

    match = batch_seqlen_pattern.findall(content)
    if not match:
        return []

    results = []
    for func, ngpus, batch, seq, num_heads, head_dim, sec in perf_pattern.findall(content):
        results.append((int(batch), int(seq), func, int(ngpus), int(num_heads), int(head_dim), float(sec), 'fwd_bwd', '', '', '', ''))

    return results


def process_files(file_list):
    grouped_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file_path in file_list:
        results = parse_file(file_path)
        for batch_size, seqlen, func, ngpus, num_heads, head_dim, sec, mode, requested_backend, forward_backend, backward_backend, dtype in results:
            grouped_results[(batch_size, seqlen, mode, dtype)][func][ngpus].append((sec, num_heads, head_dim, requested_backend, forward_backend, backward_backend))

    for key in grouped_results:
        for func in grouped_results[key]:
            grouped_results[key][func] = dict(sorted(grouped_results[key][func].items()))

    return grouped_results


# Calculate FLOPS

def calculate_flops(batch_size, seqlen, ngpus, num_heads, head_dim):
    s = seqlen * ngpus
    h = num_heads * head_dim
    return 8 * batch_size * s * h**2 + 4 * batch_size * s**2 * h


if __name__ == "__main__":
    summary_csv = Path('summary.csv')
    if summary_csv.exists():
        with summary_csv.open('r', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get('status') != 'ok':
                    continue
                print(
                    f"algo={row['algo']} impl={row['impl']} mode={row['mode']} "
                    f"requested={row['requested_backend']} forward={row['forward_backend']} "
                    f"backward={row['backward_backend']} total_sec={row['total_sec']}"
                )
    else:
        file_list = [f for f in os.listdir('.') if f.startswith('log.algo')]
        results = process_files(file_list)
        for (batch, seqlen, mode, dtype), funcs in results.items():
            print(f"Batch size: {batch}, Sequence length: {seqlen}, Mode: {mode}, Dtype: {dtype}")
            for func, perf_data in funcs.items():
                print(f"  Function: {func}")
                for ngpus, sec_list in perf_data.items():
                    for sec, num_heads, head_dim, requested_backend, forward_backend, backward_backend in sec_list:
                        forward_flops = calculate_flops(batch, seqlen, ngpus, num_heads, head_dim)
                        iters = 5
                        tflops = 3 * forward_flops / (sec / iters) / 1e12
                        print(
                            f"    GPUs: {ngpus}, Time: {sec} sec, TFLOPS: {tflops / ngpus:.1f}, "
                            f"requested={requested_backend}, forward={forward_backend}, backward={backward_backend}"
                        )
