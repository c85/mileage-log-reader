# SCRUM-19: Quantify reimbursement impact by digit position
# ISM 6642 Final Group Project

REIMBURSEMENT_RATE = 0.62

digit_positions = {
    "Ones": 1,
    "Tens": 10,
    "Hundreds": 100,
    "Thousands": 1000,
    "Ten-thousands": 10000,
    "Hundred-thousands": 100000
}

print("Reimbursement Exposure by Digit Position")
print("-" * 50)

for position, miles_error in digit_positions.items():
    cost = miles_error * REIMBURSEMENT_RATE
    print(f"{position:20} {miles_error:>7,} miles  ->  ${cost:,.2f}")

    import random

print("\nSynthetic Digit Error Simulation")
print("-" * 70)

random.seed(42)

num_simulations = 1000
results = {position: [] for position in digit_positions}

for position, place_value in digit_positions.items():
    for _ in range(num_simulations):

        # Simulate changing a digit by 1-9 at this position
        digit_change = random.randint(1, 9)

        miles_difference = digit_change * place_value
        reimbursement_difference = miles_difference * REIMBURSEMENT_RATE

        results[position].append(reimbursement_difference)

    average_exposure = sum(results[position]) / num_simulations
    maximum_exposure = max(results[position])

    print(
        f"{position:20} "
        f"Avg exposure: ${average_exposure:>10,.2f} | "
        f"Max exposure: ${maximum_exposure:>10,.2f}"
    )

    print("\nNeeds-Review Policy")
print("-" * 70)

REVIEW_THRESHOLD = 100.00

for position, exposures in results.items():
    average_exposure = sum(exposures) / len(exposures)
    maximum_exposure = max(exposures)

    if maximum_exposure >= REVIEW_THRESHOLD:
        decision = "NEEDS REVIEW"
    else:
        decision = "LOWER RISK"

    print(
        f"{position:20} "
        f"Max exposure: ${maximum_exposure:>10,.2f} | "
        f"{decision}"
    )