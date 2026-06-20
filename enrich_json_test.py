import os
import json
import numpy as np

RESULTS_DIR = "results"
DRY_RUN = True
DRY_RUN_LIMIT = 400


def enrich_json(json_path, dry_run=False):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -------------------
    # Yes / No decision
    # -------------------
    decision = data["logprobs"][2]
    decision_scores = {
        item["token"].strip(): item["logprob"]
        for item in decision["top_logprobs"]
    }
    prob_yes = decision_scores.get("Yes", np.nan)
    prob_no  = decision_scores.get("No", np.nan)
    answer   = decision["token"].strip()

    # -------------------
    # High / Low dosage
    # -------------------
    prob_high = np.nan
    prob_low  = np.nan
    dosage    = np.nan
    dosage_token = data["logprobs"][8]["token"].strip()
    if dosage_token in ("High", "Low"):
        dosage = dosage_token
        dosage_scores = {
            item["token"].strip(): item["logprob"]
            for item in data["logprobs"][8]["top_logprobs"]
        }
        prob_high = dosage_scores.get("High", np.nan)
        prob_low  = dosage_scores.get("Low", np.nan)

    # -------------------
    # Explanation: parse from generated_text (3rd line, after Answer/Dosage)
    # -------------------
    gen_text = data.get("generated_text", "")
    lines = gen_text.split("\n")
    explanation = lines[2].strip() if len(lines) > 2 else np.nan

    new_fields = {
        "prob_yes":   prob_yes,
        "prob_no":    prob_no,
        "prob_high":  prob_high,
        "prob_low":   prob_low,
        "answer":     answer,
        "dosage":     dosage,
        "explanation": explanation,
    }

    if dry_run:
        print(f"\n--- {json_path} ---")
        print(f"  decision_token (idx 2): {decision['token'].strip()!r}")
        print(f"  dosage_token   (idx 8): {dosage_token!r}")
        for k, v in new_fields.items():
            print(f"  {k:12s} = {v}")
        return new_fields

    data.update(new_fields)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return new_fields


def enrich_all_results(results_dir, dry_run=False, dry_run_limit=3):
    count = 0
    previewed = 0

    for root, _, files in os.walk(results_dir):
        for filename in sorted(files):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(root, filename)

            if dry_run and previewed >= dry_run_limit:
                continue

            try:
                enrich_json(filepath, dry_run=dry_run)
                count += 1
                if dry_run:
                    previewed += 1
            except (KeyError, IndexError) as e:
                print(f"  WARNING: skipped {filepath} ({e})")

    if dry_run:
        print(f"\n[DRY RUN] Previewed {previewed} files. No files were modified.")
        print("Set DRY_RUN = False to actually write changes to all files.")
    else:
        print(f"\nEnriched {count} files in place.")


enrich_all_results(RESULTS_DIR, dry_run=DRY_RUN, dry_run_limit=DRY_RUN_LIMIT)




import os
import json
import numpy as np
import pandas as pd

RESULTS_DIR = "results"


def check_record(json_path):
    """Extract decision info and flag whether prob_yes/prob_no are NaN."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    decision = data["logprobs"][2]
    decision_scores = {
        item["token"].strip(): item["logprob"]
        for item in decision["top_logprobs"]
    }
    prob_yes = decision_scores.get("Yes", np.nan)
    prob_no  = decision_scores.get("No", np.nan)
    answer   = decision["token"].strip()

    return {
        "filepath": json_path,
        "context":  data.get("context"),
        "vignette_idx": data.get("vignette_idx"),
        "race":     data.get("race"),
        "gender":   data.get("gender"),
        "answer_token": answer,
        "prob_yes_nan": np.isnan(prob_yes),
        "prob_no_nan":  np.isnan(prob_no),
        "any_nan":      np.isnan(prob_yes) or np.isnan(prob_no),
    }


def scan_all_results(results_dir):
    rows = []
    for root, _, files in os.walk(results_dir):
        for filename in sorted(files):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(root, filename)
            try:
                rows.append(check_record(filepath))
            except (KeyError, IndexError) as e:
                rows.append({
                    "filepath": filepath, "context": None, "vignette_idx": None,
                    "race": None, "gender": None, "answer_token": None,
                    "prob_yes_nan": True, "prob_no_nan": True, "any_nan": True,
                })
                print(f"  WARNING: {filepath} raised {e} — counted as NaN")

    return pd.DataFrame(rows)


# ── Run the scan ────────────────────────────────────────────────────────────
check_df = scan_all_results(RESULTS_DIR)

print(f"Total files scanned: {len(check_df)}")
print(f"Total with any NaN:  {check_df['any_nan'].sum()}  ({check_df['any_nan'].mean():.1%})")

print("\n--- NaN rate by context ---")
print(check_df.groupby("context")["any_nan"].agg(["sum", "count", "mean"]).round(3))

print("\n--- NaN rate by race × gender ---")
print(check_df.groupby(["race", "gender"])["any_nan"].agg(["sum", "count", "mean"]).round(3))

print("\n--- NaN rate by context × race × gender (full breakdown) ---")
print(check_df.groupby(["context", "race", "gender"])["any_nan"].agg(["sum", "count"]))