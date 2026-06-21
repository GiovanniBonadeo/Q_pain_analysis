"""
run_experiment.py
Q-Pain experiment runner — calls the local inference server and saves
one JSON file per pain context into ./results/.

Usage:
    python run_experiment.py

Output:
    results/results_acute_cancer.json
    results/results_acute_non_cancer.json
    results/results_chronic_cancer.json
    results/results_chronic_non_cancer.json
    results/results_post_operative.json
"""

import json
import os
import random
import time

import pandas as pd
import requests

# ── Configuration ──────────────────────────────────────────────────────────────

URL     = "http://10.70.13.33:11434/v1/chat/completions"
API_KEY = "sk-RZSBTkuZYOeXULKBTKupkA"
MODEL   = "deepseek-32k"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

TEMPERATURE  = 0.0
MAX_TOKENS   = 150
STOP         = ["##"]
TOP_LOGPROBS = 5
MAX_RETRIES  = 3
RETRY_DELAY  = 2          # seconds between retries

OUTPUT_DIR   = "results"  # folder where JSON files are saved

# ── Dataset constants ──────────────────────────────────────────────────────────

NAMES = {
    "Black": {
        "man":   ["Roosevelt", "Jermaine", "Darnell", "Willie", "Mattie",
                  "Reginald", "Cedric", "Sylvester", "Tyrone", "Errol"],
        "woman": ["Latonya", "Tamika", "Ebony", "Latasha", "Keisha",
                  "Lillie", "Minnie", "Gwendolyn", "Bessie", "Marva"],
    },
    "White": {
        "man":   ["Bradley", "Brett", "Scott", "Kurt", "Todd",
                  "Chad", "Matthew", "Dustin", "Shane", "Douglas"],
        "woman": ["Beth", "Megan", "Kristin", "Jill", "Erin",
                  "Colleen", "Kathleen", "Heather", "Holly", "Laurie"],
    },
    "Asian": {
        "man":   ["Viet", "Thong", "Qiang", "Kwok", "Hao",
                  "Yang", "Nam", "Huy", "Yuan", "Ho"],
        "woman": ["Zhen", "Nga", "Lien", "Lam", "Hui",
                  "Wing", "Hoa", "Wai", "Min", "Huong"],
    },
    "Hispanic": {
        "man":   ["Rigoberto", "Santos", "Javier", "Efrain", "Juan",
                  "Ramiro", "Jesus", "Humberto", "Gonzalo", "Hector"],
        "woman": ["Guadalupe", "Marisela", "Guillermina", "Rocio", "Yesenia",
                  "Blanca", "Rosalba", "Elvia", "Alejandra", "Mayra"],
    },
}

PRONOUNS = {
    "subject":    {"man": "he",  "woman": "she"},
    "possessive": {"man": "his", "woman": "her"},
}

RACES   = ["Black", "White", "Asian", "Hispanic"]
GENDERS = ["man", "woman"]

DOSE_LOW  = "Dosage: Low (0.5 mg)"
DOSE_HIGH = "Dosage: High (1 mg)"

CONTEXT_CSV = {
    "Acute Cancer Pain":        "data_acute_cancer.csv",
    "Acute Non Cancer Pain":    "data_acute_non_cancer.csv",
    "Chronic Cancer Pain":      "data_chronic_cancer.csv",
    "Chronic Non Cancer Pain":  "data_chronic_non_cancer.csv",
    "Post Operative Pain":      "data_post_op.csv"
}

CONTEXT_FOLDER = {
    "Acute Cancer Pain":        "acute_cancer",
    "Acute Non Cancer Pain":    "acute_non_cancer",
    "Chronic Cancer Pain":      "chronic_cancer",
    "Chronic Non Cancer Pain":  "chronic_non_cancer",
    "Post Operative Pain":      "post_operative"
}

def save_result(result, output_dir):
    context_folder  = CONTEXT_FOLDER[result["context"]]
    vignette_folder = f"vignette_{result['vignette_idx']:02d}"
    filename        = f"{result['race'].lower()}_{result['gender']}.json"
    folder   = os.path.join(output_dir, context_folder, vignette_folder)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

# ── Prompt builders ────────────────────────────────────────────────────────────

def create_closed(data, i, dose):
    """Build a raw closed-prompt string from row i of the open vignettes df."""
    return (
        "Vignette: " + data.Vignette.iloc[i] + "\n"
        "Question: " + data.Question.iloc[i] + "\n"
        "Answer: "   + data.Answer.iloc[i]   + "\n"
        + dose + "\n"
        "Explanation: " + data.Explanation.iloc[i] + "##\n\n"
    )


def standardize_closed(p, patient_label):
    """Remove demographic placeholders from a closed prompt."""
    p = p.replace("[gender] ", "")
    p = p.replace("[race] ",   "")
    p = p.replace("[possessive]", "their")
    p = p.replace("[subject]",    "they")   # pronoun, not the patient name
    p = p.replace("Patient D",   patient_label)
    return p


def build_closed_promptNo(closed_df):
    """Build the fixed 'No' closed prompt from the No-answer row."""
    row = closed_df.iloc[0]
    return (
        "Vignette: "     + row.Vignette    + "\n"
        "Question: "     + row.Question    + "\n"
        "Answer: "       + row.Answer      + "\n"
        "Dosage: "       + str(row.Dosage) + "\n"
        "Explanation: "  + row.Explanation + "##\n\n"
    )


def create_open_standard(data, q):
    """Build the demographic-free open prompt for vignette q."""
    return (
        "Vignette: " + data.Vignette.iloc[q] + "\n"
        "Question: " + data.Question.iloc[q] + "\n"
        "Answer:"
    )


def genderize_open(p, g):
    """Inject gender/pronoun placeholders into an open prompt."""
    p = p.replace("[gender]",     g)
    p = p.replace("[possessive]", PRONOUNS["possessive"][g])
    p = p.replace("[subject]",    PRONOUNS["subject"][g])
    return p


def race_name_open(p, r, g, q, shuffled_names):
    """Inject race and the pre-shuffled name into an open prompt."""
    p = p.replace("[race]",    r)
    p = p.replace("Patient D", shuffled_names[r][g][q])
    return p

# ── API call ───────────────────────────────────────────────────────────────────

def my_completion(prompt):
    """
    Call the inference server and return a dict with:
        generated_text  – the model's reply
        logprobs        – list of per-token logprob dicts (may be empty)
        finish_reason   – why generation stopped
    Retries up to MAX_RETRIES times on transient errors.
    """
    payload = {
        "model":       MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        "stop":        STOP,
        "logprobs":    True,
        "top_logprobs": TOP_LOGPROBS,
        "extra_body":  {"think": False},
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(URL, headers=HEADERS, json=payload, timeout=60)
            response.raise_for_status()
            data   = response.json()
            choice = data["choices"][0]
            return {
                "generated_text": choice["message"]["content"],
                "logprobs":       choice.get("logprobs", {}).get("content", []),
                "finish_reason":  choice.get("finish_reason"),
            }
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"    Attempt {attempt + 1} failed ({e}), retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

# ── Context runner ─────────────────────────────────────────────────────────────

def run_context(context_label, csv_path, shuffled_names):
    """
    Run all 80 prompts for one pain context (10 vignettes × 8 profiles).
    Returns a list of result dicts ready to be saved as JSON.
    """
    vignettes  = pd.read_csv(csv_path)
    open_data  = vignettes[vignettes.Answer == "Yes."].reset_index(drop=True)
    closed_df  = vignettes[vignettes.Answer == "No."].reset_index(drop=True)

    closed_promptNo = build_closed_promptNo(closed_df)

    #print(closed_promptNo)
    #print(open_data)

    n_open  = len(open_data)

    for q in range(n_open):
        print(f"  --- Vignette {q + 1}/{n_open} ---")

        open_prompt_standard = create_open_standard(open_data, q)

        #print("open prompt:", open_prompt_standard)

        # Sample 2 different positive vignettes for the closed examples
        available          = [i for i in range(n_open) if i != q]
        sample             = random.sample(available, 2)
        closed_prompt_high = create_closed(open_data, sample[0], DOSE_HIGH)
        closed_prompt_low  = create_closed(open_data, sample[1], DOSE_LOW)

        # Standardize closed prompts (remove demographics, rename patients)
        closed_prompt_high = standardize_closed(closed_prompt_high, "Patient A")
        closed_prompt_low  = standardize_closed(closed_prompt_low,  "Patient C")
        closed_prompt      = closed_prompt_low + closed_promptNo + closed_prompt_high 

        #print("closed prompt: ", closed_prompt)

        for g in GENDERS:
            open_prompt_gendered = genderize_open(open_prompt_standard, g)

            for r in RACES:
                open_prompt  = race_name_open(open_prompt_gendered, r, g, q, shuffled_names)
                final_prompt = closed_prompt + open_prompt

                #print("FINAL PROMPT: ", final_prompt)

                response = my_completion(final_prompt)

                # Attach metadata
                response["context"]     = context_label
                response["closed_prompt"] = closed_prompt
                response["open_prompt"] = open_prompt
                response["vignette_idx"] = q
                response["race"]        = r
                response["gender"]      = g
                response["name"]        = shuffled_names[r][g][q]

                save_result(response, OUTPUT_DIR)   # ← save immediately
                print(f"    [{r} {g}] → {response['generated_text'][:60].strip()!r}")


# ── Main ───────────────────────────────────────────────────────────────────────

def sanity_check_csvs():
    """Validate all CSVs before running any API calls."""
    print("Running sanity checks on all CSV files...\n")
    all_ok = True

    for context_label, csv_file in CONTEXT_CSV.items():
        print(f"--- {context_label} ({csv_file}) ---")

        if not os.path.exists(csv_file):
            print(f"  ❌ File not found")
            all_ok = False
            continue

        vignettes = pd.read_csv(csv_file)

        # Check required columns exist
        required_cols = {"Vignette", "Question", "Answer", "Dosage", "Explanation"}
        missing_cols = required_cols - set(vignettes.columns)
        if missing_cols:
            print(f"  ❌ Missing columns: {missing_cols}")
            all_ok = False
            continue

        # Check Yes/No filtering works
        open_data = vignettes[vignettes.Answer.str.strip().str.lower().str.startswith("yes")]
        no_data   = vignettes[vignettes.Answer.str.strip().str.lower().str.startswith("no")]

        print(f"  Total rows: {len(vignettes)}")
        print(f"  Yes rows:   {len(open_data)}")
        print(f"  No rows:    {len(no_data)}")

        if len(open_data) == 0:
            print(f"  ❌ No 'Yes' vignettes found — check Answer column values")
            print(f"     Unique Answer values: {vignettes.Answer.unique().tolist()}")
            all_ok = False

        if len(no_data) == 0:
            print(f"  ❌ No 'No' vignette found — check Answer column values")
            all_ok = False

        # Check High/Low dosage rows exist among Yes vignettes
        high_rows = open_data[open_data.Dosage.str.strip().str.lower().str.startswith("high", na=False)]
        low_rows  = open_data[open_data.Dosage.str.strip().str.lower().str.startswith("low",  na=False)]
        print(f"  High dosage rows: {len(high_rows)}")
        print(f"  Low dosage rows:  {len(low_rows)}")

        if len(high_rows) == 0:
            print(f" No 'High' dosage vignette found")
        if len(low_rows) == 0:
            print(f"  ❌ No 'Low' dosage vignette found")
            all_ok = False

        # Check placeholders exist in Yes vignettes (needed for demographic injection)
        sample_vignette = open_data.Vignette.iloc[0] if len(open_data) > 0 else ""
        for placeholder in ["[race]", "[gender]", "Patient D"]:
            if placeholder not in sample_vignette:
                print(f"  ⚠️  Warning: '{placeholder}' not found in first Yes vignette")

        print()

    if not all_ok:
        print("=" * 60)
        print("SANITY CHECK FAILED — fix the issues above before running.")
        print("=" * 60)
        raise SystemExit(1)
    else:
        print("=" * 60)
        print("All sanity checks passed. Proceeding to run experiment.")
        print("=" * 60)

def main():

    #sanity_check_csvs()

    random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Shuffle names once, shared across all contexts (matches paper approach)
    shuffled_names = {
        r: {g: random.sample(NAMES[r][g], len(NAMES[r][g])) for g in GENDERS}
        for r in RACES
    }

    for context_label, csv_file in CONTEXT_CSV.items():
        if not os.path.exists(csv_file):
            print(f"\nWARNING: {csv_file} not found — skipping '{context_label}'")
            continue

        print(f"\n{'=' * 60}")
        print(f"CONTEXT: {context_label}")
        print(f"{'=' * 60}")

        results = run_context(context_label, csv_file, shuffled_names)

    print("\nAll contexts done.")


if __name__ == "__main__":
    main()