#!/usr/bin/env python3
"""
Block 0 weight comparison analysis tool
Only the weight differences of the blocks.0 layer are analyzed
"""

import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
import argparse
import json


KEY_MAPPING = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # Hack to swap the layer names
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # For the I2V model
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    # for the FLF2V model
    "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
    # Add attention component mappings
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.attn_op.local_attn.proj_l": "attn1.attn_op.local_attn.proj_l",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
}


def load_model(model_path: str) -> Dict[str, torch.Tensor]:
    """loadTheModelFile"""
    print(f"loading: {model_path}")

    if model_path.endswith('.safetensors'):
        from safetensors import safe_open
        model_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                model_dict[key] = f.get_tensor(key)
    else:
        # .pth or .pt files
        loaded = torch.load(model_path, map_location='cpu')
        # Handle different formats
        if isinstance(loaded, dict):
            if 'state_dict' in loaded:
                model_dict = loaded['state_dict']
            elif 'model' in loaded:
                model_dict = loaded['model']
            else:
                model_dict = loaded
        else:
            raise ValueError("unsupported model")

    # Filter only tensor parameters and blocks.0
    filtered_dict = {}
    for k, v in model_dict.items():
        if isinstance(v, torch.Tensor) and k.startswith('blocks.0.'):
            filtered_dict[k] = v

    return filtered_dict


def map_key(old_key: str, key_mapping: Dict[str, str]) -> str:
    """Convert the key according to the mapping relationship"""
    if old_key.startswith('blocks.0.'):
        remaining = old_key[9:]  #Remove the 'blocks.0.' prefix

        for old_pattern, new_pattern in key_mapping.items():
            if remaining.startswith(old_pattern + '.') or remaining == old_pattern:
                new_remaining = remaining.replace(old_pattern, new_pattern, 1)
                return f'blocks.0.{new_remaining}'

    return old_key


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor,
                    rtol: float = 1e-5, atol: float = 1e-8) -> Tuple[bool, float, float, float]:
    """
    Compare whether the two tensors are equal
    Return: (Approximate Equality, Maximum Absolute Difference, Relative Difference, Mean Square Error)
    """
    if tensor1.shape != tensor2.shape:
        return False, float('inf'), float('inf'), float('inf')

    # Convert to float64 for precise comparison
    t1 = tensor1.to(torch.float64).flatten()
    t2 = tensor2.to(torch.float64).flatten()

    # Calculate differences
    diff = torch.abs(t1 - t2)
    max_diff = torch.max(diff).item()
    mse = torch.mean(diff ** 2).item()

    # Relative difference
    max_val = torch.max(torch.abs(t1), torch.abs(t2)).max().item()
    rel_diff = max_diff / max_val if max_val > 0 else 0.0

    # Check if close
    is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)

    return is_close, max_diff, rel_diff, mse


def analyze_block0_weights(model1_path: str, model2_path: str,
                           key_mapping: Dict[str, str] = None,
                           rtol: float = 1e-5, atol: float = 1e-8) -> Dict:
    """Only the weights of blocks.0 are analyzed"""
    if key_mapping is None:
        key_mapping = KEY_MAPPING

    model1 = load_model(model1_path)
    model2 = load_model(model2_path)

    print(f"The number of blocks.0 parameters in model 1: {len(model1)}")
    print(f"The number of blocks.0 parameters in Model 2: {len(model2)}")
    print()

    results = {
        "model1_path": model1_path,
        "model2_path": model2_path,
        "total_params": len(model1),
        "exact_matches": 0,
        "close_matches": 0,
        "value_mismatches": 0,
        "missing_mappings": [],
        "detailed_results": [],
        "summary": {}
    }

    for key1 in sorted(model1.keys()):
        tensor1 = model1[key1]

        key2 = map_key(key1, key_mapping)

        if key2 not in model2:
            results["missing_mappings"].append({
                "source_key": key1,
                "mapped_key": key2,
                "problem": f"can't found: {key2}"
            })
            continue

        tensor2 = model2[key2]

        # Compare weights
        is_close, max_diff, rel_diff, mse = compare_tensors(
            tensor1, tensor2, rtol=rtol, atol=atol
        )

        param_result = {
            "key1": key1,
            "key2": key2,
            "shape": list(tensor1.shape),
            "max_diff": max_diff,
            "rel_diff": rel_diff,
            "mse": mse,
            "is_close": is_close,
            "status": ""
        }

        if max_diff == 0:
            param_result["status"] = "EXACT_MATCH"
            results["exact_matches"] += 1
        elif is_close:
            param_result["status"] = "CLOSE_MATCH"
            results["close_matches"] += 1
        else:
            param_result["status"] = "MISMATCH"
            results["value_mismatches"] += 1

        results["detailed_results"].append(param_result)

    # generateASummary
    total = results["total_params"]
    results["summary"] = {
        "exact_matches": results["exact_matches"],
        "exact_match_rate": results["exact_matches"] / total * 100 if total > 0 else 0,
        "close_matches": results["close_matches"],
        "close_match_rate": results["close_matches"] / total * 100 if total > 0 else 0,
        "mismatches": results["value_mismatches"],
        "mismatch_rate": results["value_mismatches"] / total * 100 if total > 0 else 0,
        "missing_mappings": len(results["missing_mappings"])
    }

    return results


def print_block0_report(results: Dict, verbose: bool = False):
    """Print the blocks.0 analysis report"""
    print("=" * 80)
    print("Block 0 weight comparison analysis")
    print("=" * 80)
    print(f"model1: {results['model1_path']}")
    print(f"model2: {results['model2_path']}")
    print()

    summary = results["summary"]
    print(f"Total number of parameters: {results['total_params']}")
    print(f"✓ Exact match: {summary['exact_matches']} ({summary['exact_match_rate']:.2f}%)")
    print(f"✓ Close match: {summary['close_matches']} ({summary['close_match_rate']:.2f}%)")
    print(f"✗ Miss match: {summary['mismatches']} ({summary['mismatch_rate']:.2f}%)")

    if results["missing_mappings"]:
        print(f"✗ Mapping failed: {summary['missing_mappings']} 个")

    print()

    # 显示不匹配详情
    if results["value_mismatches"] > 0:
        print("-" * 80)
        print("Numeric mismatch details (sorted by relative difference):")
        print("-" * 80)

        mismatches = [r for r in results["detailed_results"] if r["status"] == "MISMATCH"]
        mismatches.sort(key=lambda x: x["rel_diff"], reverse=True)

        for i, item in enumerate(mismatches[:10], 1):
            print(f"\n[{i}] {item['key1']}")
            print(f"     -> {item['key2']}")
            print(f"     shape: {item['shape']}")
            print(f"     Maximum absolute difference: {item['max_diff']:.6e}")
            print(f"     Relative difference: {item['rel_diff']:.6e}")
            print(f"     MEAN SQUARE ERROR: {item['mse']:.6e}")

    # Displays mapping failure details
    if results["missing_mappings"]:
        print("\n" + "-" * 80)
        print("Mapping failure details:")
        print("-" * 80)
        for i, item in enumerate(results["missing_mappings"][:10], 1):
            print(f"\n[{i}] Source key: {item['source_key']}")
            print(f"     mapped key: {item['mapped_key']}")
            print(f"     problem: {item['problem']}")

    # If verbose mode is turned on, all matches are displayed
    if verbose:
        print("\n" + "-" * 80)
        print("Detailed matching results:")
        print("-" * 80)
        for item in results["detailed_results"]:
            status_icon = "✓" if item["status"] in ["EXACT_MATCH", "CLOSE_MATCH"] else "✗"
            print(f"{status_icon} {item['key1']} -> {item['key2']}")
            if item["status"] == "CLOSE_MATCH":
                print(f"  Relative difference: {item['rel_diff']:.2e}")

    print("\n" + "=" * 80)


def save_results(results: Dict, output_path: str = "block0_comparison.json"):
    """Save the analysis results to a JSON file"""
    # Convert tensors to lists for JSON serialization
    json_results = results.copy()

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\nThe detailed results are saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Only the weights of blocks.0 are compared')
    parser.add_argument('model1', help='The first model file path')
    parser.add_argument('model2', help='The second model file path')
    parser.add_argument('--rtol', type=float, default=1e-5, help='Relative tolerance')
    parser.add_argument('--atol', type=float, default=1e-8, help='Absolute tolerance')
    parser.add_argument('--verbose', action='store_true', help='Displays detailed results for all parameters')
    parser.add_argument('--output', type=str, default='block0_comparison.json',
                        help='Output the JSON file path')

    args = parser.parse_args()

    print("=" * 80)
    print("Block 0 weight comparison analysis")
    print("=" * 80)
    print()

    results = analyze_block0_weights(
        args.model1,
        args.model2,
        rtol=args.rtol,
        atol=args.atol
    )

    print_block0_report(results, verbose=args.verbose)

    save_results(results, args.output)


if __name__ == "__main__":
    main()

# demo:
"""
# 
python block0_analyzer.py model1.pth model2.pth

python block0_analyzer.py model1.pth model2.pth --rtol 1e-4 --atol 1e-6

python block0_analyzer.py model1.pth model2.pth --verbose

python block0_analyzer.py model1.pth model2.pth --output my_results.json
"""