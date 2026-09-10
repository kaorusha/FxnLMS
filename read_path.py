#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
REW Impulse Response to C Header Converter for ANC FxNLMS
===============================================================================
This script parses REW (Room EQ Wizard) exported text impulse response files,
extracts the specified number of filter taps around the main peak, applies a
tail taper (Half-Hann window) to avoid spectral leakage, and exports the
coefficients as a C header file (compile-time constant array) ready for
embedded DSP real-time execution.

By default, it automatically outputs the generated header files to the
EV-SC598-SOM CCES project directory:
  ../EV-SC598-SOM/Audio_Talkthrough_I2S/second_path.h
  ../EV-SC598-SOM/Audio_Talkthrough_I2S/feedback_path.h

Usage:
  Default batch mode (converts both secondary path and feedback path):
    python read_path.py

  Custom file conversion:
    python read_path.py -i "R Jul 7 secondary path.txt" -o "second_path.h"
    python read_path.py -i "input.txt" -o "second_path.h" -a SECOND_PATH -t 2048 -m 20
===============================================================================
"""

import os
import sys
import math
import argparse
from typing import Dict, Any, List, Optional


def parse_rew_header(lines: List[str]) -> Dict[str, Any]:
    """
    Extracts metadata from the REW text file header before '* Data start'.
    """
    meta = {
        "peak_index": 0,
        "peak_value": 0.0,
        "sample_interval": 0.0,
        "sample_rate": 48000.0,
        "response_length": 0,
        "start_time": 0.0,
        "data_start_line": -1
    }

    for idx, line in enumerate(lines):
        line_clean = line.strip()
        if "* Data start" in line_clean:
            meta["data_start_line"] = idx + 1
            break

        if "Peak value before normalisation" in line_clean:
            try:
                meta["peak_value"] = float(line_clean.split()[0])
            except (ValueError, IndexError):
                pass
        elif "Peak index" in line_clean:
            try:
                meta["peak_index"] = int(line_clean.split()[0])
            except (ValueError, IndexError):
                pass
        elif "Response length" in line_clean:
            try:
                meta["response_length"] = int(line_clean.split()[0])
            except (ValueError, IndexError):
                pass
        elif "Sample interval (seconds)" in line_clean:
            try:
                interval = float(line_clean.split()[0])
                meta["sample_interval"] = interval
                if interval > 0.0:
                    meta["sample_rate"] = 1.0 / interval
            except (ValueError, IndexError):
                pass
        elif "Start time (seconds)" in line_clean:
            try:
                meta["start_time"] = float(line_clean.split()[0])
            except (ValueError, IndexError):
                pass

    return meta


def load_rew_ir(filepath: str,
                target_taps: int = 2048,
                pre_peak_margin: int = 20,
                taper_ratio: float = 0.10) -> tuple:
    """
    Loads REW impulse response, truncates around the peak, and applies tail tapering.

    Args:
        filepath: Path to the REW text file.
        target_taps: Desired FIR filter tap count (e.g. 2048).
        pre_peak_margin: Samples to preserve before the peak index (e.g. 20).
        taper_ratio: Ratio of the tail to taper with Half-Hann window (e.g. 0.10 for 10%).

    Returns:
        (ir_tapered, meta): Processed float list of length target_taps, and metadata dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    meta = parse_rew_header(lines)
    if meta["data_start_line"] < 0:
        raise ValueError(f"Could not find '* Data start' marker in {filepath}")

    # Parse float samples from data lines
    raw_samples = []
    for line in lines[meta["data_start_line"]:]:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("*"):
            continue
        try:
            raw_samples.append(float(line_clean))
        except ValueError:
            continue

    total_samples = len(raw_samples)
    if total_samples == 0:
        raise ValueError(f"No numeric samples found in {filepath}")

    # If peak index was not in header or out of range, calculate from raw samples
    if meta["peak_index"] <= 0 or meta["peak_index"] >= total_samples:
        max_abs = -1.0
        best_idx = 0
        for i, val in enumerate(raw_samples):
            abs_val = abs(val)
            if abs_val > max_abs:
                max_abs = abs_val
                best_idx = i
        meta["peak_index"] = best_idx
        meta["peak_value"] = raw_samples[best_idx]

    # Determine start sample index
    start_idx = max(0, meta["peak_index"] - pre_peak_margin)
    meta["start_index"] = start_idx
    meta["pre_peak_margin"] = pre_peak_margin
    meta["taper_ratio"] = taper_ratio

    # Extract target_taps samples
    end_idx = start_idx + target_taps
    extracted = raw_samples[start_idx:end_idx]

    # Zero-pad if file ended prematurely
    if len(extracted) < target_taps:
        extracted.extend([0.0] * (target_taps - len(extracted)))

    # 保存加窗前的截斷數據 (長度為 target_taps)，供視覺化前後對比使用
    meta["truncated_ir"] = extracted.copy()

    # Apply tail tapering (Half-Hann window on the last taper_ratio portion)
    if 0.0 < taper_ratio < 1.0:
        taper_len = int(target_taps * taper_ratio)
        taper_start = target_taps - taper_len
        for i in range(taper_len):
            # Half-Hann: starts at 1.0 when i=0, smoothly decays to 0.0 at i=taper_len
            w = 0.5 * (1.0 + math.cos(math.pi * i / taper_len))
            extracted[taper_start + i] *= w

    return extracted, meta


def export_to_c_header(output_h_path: str,
                       ir_data: List[float],
                       array_name: str,
                       macro_name: Optional[str] = None,
                       meta: Optional[Dict[str, Any]] = None,
                       source_filepath: Optional[str] = None) -> None:
    """
    Exports impulse response coefficients to a C header file formatted as a static const float array.
    """
    if not macro_name:
        macro_name = f"{array_name}_TAPS"

    # Derive include guard name from output file name
    base_name = os.path.basename(output_h_path)
    guard_name = "__" + "".join([c if c.isalnum() else "_" for c in base_name]).upper() + "__"

    num_taps = len(ir_data)
    mem_bytes = num_taps * 4
    mem_kb = mem_bytes / 1024.0

    lines = []
    lines.append("/******************************************************************************")
    lines.append(f" * @file {base_name}")
    lines.append(" * @brief FIR filter coefficients for ANC FxNLMS algorithm.")
    lines.append(" *")
    if source_filepath:
        # Convert to relative path or filename to prevent exposing absolute personal folder names
        script_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            rel_source = os.path.relpath(source_filepath, script_dir).replace(os.sep, '/')
        except Exception:
            rel_source = os.path.basename(source_filepath)
        lines.append(f" * Source File:       {rel_source}")
    if meta:
        lines.append(f" * REW Peak Value:    {meta.get('peak_value', 0.0):.8e}")
        p_idx = meta.get('peak_index', 0)
        p_time = p_idx * meta.get('sample_interval', 0.0) + meta.get('start_time', 0.0)
        lines.append(f" * REW Peak Index:    {p_idx} (Time: {p_time:.6f} s)")
        lines.append(f" * Pre-Peak Margin:   {meta.get('pre_peak_margin', 0)} samples")
        taper_pct = meta.get('taper_ratio', 0.0) * 100.0
        lines.append(f" * Tail Taper Ratio:  {taper_pct:.1f}% (Half-Hann window)")
        lines.append(f" * Response Length:   {meta.get('response_length', 0)} samples")
        lines.append(f" * Sample Rate:       {meta.get('sample_rate', 48000.0):.1f} Hz (Interval: {meta.get('sample_interval', 0.0):.8e} s)")
        lines.append(f" * Extracted Start:   Sample {meta.get('start_index', 0)} (Peak - PreMargin)")
    lines.append(f" * Filter Taps:       {num_taps}")
    lines.append(f" * Memory Footprint:  {mem_bytes} bytes ({mem_kb:.2f} KB)")
    lines.append(" ******************************************************************************/\n")

    lines.append(f"#ifndef {guard_name}")
    lines.append(f"#define {guard_name}\n")
    lines.append("#include <stdint.h>\n")
    lines.append("/* Number of filter taps */")
    lines.append(f"#define {macro_name} ({num_taps})\n")

    if meta and meta.get("sample_rate", 0.0) > 0:
        fs = int(round(meta["sample_rate"]))
        lines.append("/* Sampling rate in Hz */")
        lines.append(f"#define {macro_name}_FS ((uint32_t){fs})\n")

    lines.append("/* FIR filter coefficients (IEEE 754 32-bit float) */")
    lines.append(f"static const float {array_name}[{macro_name}] = {{")

    # Format 8 floats per line
    for i in range(0, num_taps, 8):
        chunk = ir_data[i:i + 8]
        items = []
        for idx, val in enumerate(chunk):
            sign = "+" if val >= 0 else "-"
            formatted = f"{sign}{abs(val):.8e}f"
            items.append(formatted)
        line_str = "    " + ", ".join(items)
        if i + 8 < num_taps:
            line_str += ","
        lines.append(line_str)

    lines.append("};\n")
    lines.append(f"#endif /* {guard_name} */\n")

    # Ensure destination folder exists
    out_dir = os.path.dirname(output_h_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_h_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_sc598_dir = os.path.normpath(os.path.join(script_dir, "..", "EV-SC598-SOM", "Audio_Talkthrough_I2S"))

    parser = argparse.ArgumentParser(
        description="Convert REW Impulse Response text export to C header for ANC FxNLMS."
    )
    parser.add_argument("-i", "--input", type=str, default=None,
                        help="Input REW TXT file path.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output C header (.h) file path. If only filename is given, outputs to SC598 directory.")
    parser.add_argument("-a", "--array", type=str, default=None,
                        help="C array name (e.g. SECOND_PATH). Defaults to uppercase filename.")
    parser.add_argument("-t", "--taps", type=int, default=2048,
                        help="Number of filter taps (default: 2048).")
    parser.add_argument("-m", "--margin", type=int, default=20,
                        help="Pre-peak margin in samples (default: 20).")
    parser.add_argument("-w", "--taper", type=float, default=0.10,
                        help="Tail tapering ratio (default: 0.10 = 10%%).")

    args = parser.parse_args()

    print("===================================================================")
    print("  REW Impulse Response to C Header Converter (Python Script)      ")
    print("===================================================================\n")

    # Default paths in FxNLMS and SC598
    default_sec_in = os.path.join(script_dir, "R Aug 3 secondary path.txt")
    default_sec_out = os.path.join(default_sc598_dir, "second_path.h")

    default_fb_in  = os.path.join(script_dir, "R Aug 3 feedback path.txt")
    default_fb_out  = os.path.join(default_sc598_dir, "feedback_path.h")

    if args.input:
        in_path = args.input
        if not os.path.isabs(in_path) and not os.path.exists(in_path):
            # check in script_dir
            candidate = os.path.join(script_dir, in_path)
            if os.path.exists(candidate):
                in_path = candidate

        out_path = args.output if args.output else "second_path.h"
        # If output path is just a file name without dir, route it to SC598 project dir
        if not os.path.dirname(out_path):
            out_path = os.path.join(default_sc598_dir, out_path)

        if args.array:
            arr_name = args.array
        else:
            base = os.path.splitext(os.path.basename(out_path))[0]
            arr_name = "".join([c if c.isalnum() else "_" for c in base]).upper()

        print(f"[Single Conversion]")
        print(f"  Input:       '{in_path}'")
        print(f"  Output:      '{out_path}'")
        print(f"  Array Name:  '{arr_name}'")
        print(f"  Taps:        {args.taps}, Pre-Margin: {args.margin}, Taper: {args.taper * 100:.0f}%\n")

        data, meta = load_rew_ir(in_path, target_taps=args.taps, pre_peak_margin=args.margin, taper_ratio=args.taper)
        export_to_c_header(out_path, data, arr_name, meta=meta, source_filepath=in_path)
        print(f"  -> SUCCESS: '{out_path}' generated!\n")

    else:
        # Default batch mode: Convert both secondary path and feedback path into SC598 folder
        print(f"Target SC598 Project Directory:\n  {default_sc598_dir}\n")

        print("[1/2] Converting Secondary Path:")
        print(f"      Input:   '{default_sec_in}'")
        print(f"      Output:  '{default_sec_out}'")
        print(f"      Array:   'SECOND_PATH'")
        print(f"      Taps:    {args.taps}, Pre-Margin: {args.margin}, Taper: {args.taper * 100:.0f}%")

        data_sec, meta_sec = load_rew_ir(default_sec_in, target_taps=args.taps, pre_peak_margin=args.margin, taper_ratio=args.taper)
        export_to_c_header(default_sec_out, data_sec, "SECOND_PATH", meta=meta_sec, source_filepath=default_sec_in)
        print(f"      -> SUCCESS: '{default_sec_out}' generated!\n")

        print("[2/2] Converting Feedback Path:")
        print(f"      Input:   '{default_fb_in}'")
        print(f"      Output:  '{default_fb_out}'")
        print(f"      Array:   'FEEDBACK_PATH'")
        print(f"      Taps:    {args.taps}, Pre-Margin: {args.margin}, Taper: {args.taper * 100:.0f}%")

        data_fb, meta_fb = load_rew_ir(default_fb_in, target_taps=args.taps, pre_peak_margin=args.margin, taper_ratio=args.taper)
        export_to_c_header(default_fb_out, data_fb, "FEEDBACK_PATH", meta=meta_fb, source_filepath=default_fb_in)
        print(f"      -> SUCCESS: '{default_fb_out}' generated!\n")

    print("All conversions completed successfully!")


if __name__ == "__main__":
    main()
