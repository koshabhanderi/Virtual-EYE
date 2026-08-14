import csv
import os
import json


# ============================================================
# FAKESHIELD - PHASE 3I
# RESULTS ANALYSIS
# ============================================================

normal_csv = "phase3g_test/results/phase3g_results.csv"
aligned_csv = "phase3g_test/results/phase3g_aligned_results.csv"
benchmark_json = "phase3h_test/results/phase3h_submission.json"

output_file = "phase3i_results.txt"


# ============================================================
# FUNCTION TO ANALYZE CSV
# ============================================================

def analyze_csv(path):

    total = 0

    correct = 0

    real_total = 0
    fake_total = 0

    real_correct = 0
    fake_correct = 0

    total_time = 0.0

    with open(path, "r", newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            actual = row["actual"].strip().upper()
            predicted = row["predicted"].strip().upper()

            total += 1

            if actual == "REAL":
                real_total += 1

            elif actual == "FAKE":
                fake_total += 1

            if actual == predicted:

                correct += 1

                if actual == "REAL":
                    real_correct += 1

                elif actual == "FAKE":
                    fake_correct += 1

            total_time += float(
                row["time_seconds"]
            )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = (
        correct / total * 100
        if total > 0 else 0
    )

    # Positive class = FAKE
    #
    # False Positive:
    # REAL predicted as FAKE
    #
    # False Negative:
    # FAKE predicted as REAL

    false_positive = real_total - real_correct
    false_negative = fake_total - fake_correct

    fpr = (
        false_positive / real_total * 100
        if real_total > 0 else 0
    )

    fnr = (
        false_negative / fake_total * 100
        if fake_total > 0 else 0
    )

    average_time = (
        total_time / total
        if total > 0 else 0
    )

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "real_total": real_total,
        "fake_total": fake_total,
        "real_correct": real_correct,
        "fake_correct": fake_correct,
        "accuracy": accuracy,
        "fpr": fpr,
        "fnr": fnr,
        "average_time": average_time
    }


# ============================================================
# ANALYZE NORMAL RESULTS
# ============================================================

print()
print("==============================================")
print("         FAKESHIELD - PHASE 3I")
print("             RESULTS ANALYSIS")
print("==============================================")

print()
print("Analyzing normal Phase 3G results...")

normal = analyze_csv(normal_csv)


# ============================================================
# ANALYZE ALIGNED RESULTS
# ============================================================

print("Analyzing aligned Phase 3G results...")

aligned = analyze_csv(aligned_csv)


# ============================================================
# ANALYZE PHASE 3H
# ============================================================

print("Analyzing Phase 3H benchmark predictions...")

with open(
    benchmark_json,
    "r",
    encoding="utf-8"
) as file:

    benchmark = json.load(file)


benchmark_total = len(benchmark)

benchmark_real = sum(
    1
    for value in benchmark.values()
    if value == "real"
)

benchmark_fake = sum(
    1
    for value in benchmark.values()
    if value == "fake"
)


# ============================================================
# CREATE REPORT
# ============================================================

lines = []

lines.append(
    "=============================================="
)

lines.append(
    "        FAKESHIELD - PHASE 3I REPORT"
)

lines.append(
    "=============================================="
)

lines.append("")


# ============================================================
# NORMAL
# ============================================================

lines.append(
    "PHASE 3G - NORMAL PREPROCESSING"
)

lines.append(
    "----------------------------------------------"
)

lines.append(
    f"Total images: {normal['total']}"
)

lines.append(
    f"Correct predictions: {normal['correct']}"
)

lines.append(
    f"Incorrect predictions: {normal['incorrect']}"
)

lines.append(
    f"REAL images: {normal['real_total']}"
)

lines.append(
    f"FAKE images: {normal['fake_total']}"
)

lines.append(
    f"REAL correctly detected: {normal['real_correct']}"
)

lines.append(
    f"FAKE correctly detected: {normal['fake_correct']}"
)

lines.append(
    f"Accuracy: {normal['accuracy']:.2f}%"
)

lines.append(
    f"False Positive Rate: {normal['fpr']:.2f}%"
)

lines.append(
    f"False Negative Rate: {normal['fnr']:.2f}%"
)

lines.append(
    f"Average inference time: "
    f"{normal['average_time'] * 1000:.2f} ms"
)

lines.append("")


# ============================================================
# ALIGNED
# ============================================================

lines.append(
    "PHASE 3G - ALIGNED PREPROCESSING"
)

lines.append(
    "----------------------------------------------"
)

lines.append(
    f"Total images: {aligned['total']}"
)

lines.append(
    f"Correct predictions: {aligned['correct']}"
)

lines.append(
    f"Incorrect predictions: {aligned['incorrect']}"
)

lines.append(
    f"REAL images: {aligned['real_total']}"
)

lines.append(
    f"FAKE images: {aligned['fake_total']}"
)

lines.append(
    f"REAL correctly detected: {aligned['real_correct']}"
)

lines.append(
    f"FAKE correctly detected: {aligned['fake_correct']}"
)

lines.append(
    f"Accuracy: {aligned['accuracy']:.2f}%"
)

lines.append(
    f"False Positive Rate: {aligned['fpr']:.2f}%"
)

lines.append(
    f"False Negative Rate: {aligned['fnr']:.2f}%"
)

lines.append(
    f"Average inference time: "
    f"{aligned['average_time'] * 1000:.2f} ms"
)

lines.append("")


# ============================================================
# PHASE 3H
# ============================================================

lines.append(
    "PHASE 3H - OFFICIAL BENCHMARK PREDICTIONS"
)

lines.append(
    "----------------------------------------------"
)

lines.append(
    f"Benchmark images processed: {benchmark_total}"
)

lines.append(
    f"REAL predictions: {benchmark_real}"
)

lines.append(
    f"FAKE predictions: {benchmark_fake}"
)

lines.append(
    "Accuracy: NOT AVAILABLE"
)

lines.append(
    "Reason: benchmark ground-truth labels are hidden."
)

lines.append("")


# ============================================================
# COMPARISON
# ============================================================

lines.append(
    "PHASE 3G COMPARISON"
)

lines.append(
    "----------------------------------------------"
)

if normal["accuracy"] > aligned["accuracy"]:

    lines.append(
        "Better accuracy: Normal preprocessing"
    )

elif aligned["accuracy"] > normal["accuracy"]:

    lines.append(
        "Better accuracy: Aligned preprocessing"
    )

else:

    lines.append(
        "Both preprocessing methods have equal accuracy."
    )


lines.append("")


# ============================================================
# LIMITATIONS
# ============================================================

lines.append(
    "OBSERVATIONS"
)

lines.append(
    "----------------------------------------------"
)

lines.append(
    "1. The model performs substantially better on REAL"
)

lines.append(
    "   images than on FAKE images in the Phase 3G test."
)

lines.append(
    "2. The model shows a strong tendency to predict REAL."
)

lines.append(
    "3. False-negative errors are high for FAKE images."
)

lines.append(
    "4. Phase 3H cannot be assigned an accuracy score locally"
)

lines.append(
    "   because the benchmark ground-truth labels are hidden."
)

lines.append(
    "5. Phase 3H predictions were successfully generated"
)

lines.append(
    "   for all 1000 benchmark images."
)

lines.append("")


lines.append(
    "=============================================="
)

lines.append(
    "              END OF REPORT"
)

lines.append(
    "=============================================="
)


# ============================================================
# PRINT REPORT
# ============================================================

report = "\n".join(lines)

print()
print(report)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


print()
print(
    "Report saved to:",
    output_file
)