import random
import bisect

# Letters for third-letter choices (include "-" as "no match")
letters_with_dash = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","-"]
letters_AZ = [c for c in letters_with_dash if c != "-"]  # for generating Input

factor = 0.01

threeLetterWords = [
    "the", "and", "for", "but", "nor", "yet", "all", "any",
    "can", "man", "dog", "cat", "car", "bus", "sun", "sky",
    "boy", "wet", "mom", "dad", "run", "fun", "win", "top",
    "red", "lie", "new", "old", "big", "hot", "doc", "wet",
    "dry", "eat", "sit", "get", "let", "put", "say", "see",
    "too", "two", "one", "out", "off", "end", "use", "try"
]

# Build prefix -> set of third letters
prefix_map = {}
for w in threeLetterWords:
    prefix = w[:2].upper()
    third = w[2].upper()
    prefix_map.setdefault(prefix, set()).add(third)

# Weights: (first, second, third) -> float
weights = {(i, j, k): 1.0 for i in letters_AZ for j in letters_AZ for k in letters_with_dash}

# Probability dict for current Input
probsDict = {l: 0.0 for l in letters_with_dash}

Input = "NI"  # always two uppercase letters

def checkProb(letter):
    # Safe access—weights exist for all (A-Z, A-Z, letter) including "-"
    return weights[(Input[0], Input[1], letter)]

def checkProbsAndAddToDict():
    # Fill probsDict from weights for current Input
    for l in letters_with_dash:
        probsDict[l] = checkProb(l)

def choiceSelection():
    # Normalize probabilities; guard against zero sum
    probsSum = sum(probsDict.values())
    if probsSum <= 0.0:
        # fallback to uniform distribution
        uniform = 1.0 / len(letters_with_dash)
        for l in letters_with_dash:
            probsDict[l] = uniform
    else:
        for l in letters_with_dash:
            probsDict[l] /= probsSum

    # Build cumulative distribution
    cum = []
    total = 0.0
    for l in letters_with_dash:
        total += probsDict[l]
        cum.append(total)
    # Numerical safety: ensure last is exactly 1.0
    cum[-1] = 1.0

    # Sample via binary search
    r = random.random()
    idx = bisect.bisect_left(cum, r)
    return letters_with_dash[idx]

def get_target_letter(two_letters):
    # Return a valid third letter if any exists; otherwise "-"
    two = two_letters.upper()
    if two in prefix_map and prefix_map[two]:
        # Choose deterministically (sorted) or randomly; here deterministic
        return sorted(prefix_map[two])[0]
    return "-"

def updateWeights(choice):
    correct_letter = get_target_letter(Input)
    key = (Input[0], Input[1], correct_letter)
    if correct_letter == choice:
        weights[key] *= (1000 + 1000000000000*factor)
    else:
        weights[key] *= (1 - 100*factor)
    # Optional: clamp to avoid underflow
    if weights[key] < 1e-12:
        weights[key] = 1e-12

# ---------------- Training loop ----------------
i = 0
running = True
while running:
    i += 1
    # Generate a valid two-letter Input (A-Z only)
    Input = random.choice(letters_AZ) + random.choice(letters_AZ)

    checkProbsAndAddToDict()
    choice = choiceSelection()
    updateWeights(choice)

    if i % 1000 == 0:
        factor -= 0.000005
        print(f"given {Input}, i chose {choice}, target was {get_target_letter(Input)}")
        print(i / 100000, "% of the way there")
    if factor < 0:
        running = False
        break

# ---------------- Interactive loop ----------------
running = True
while running:
    user_in = input("Input two letters: ").strip().upper()
    if len(user_in) != 2 or not all(c in letters_AZ for c in user_in):
        print("Please enter exactly two letters A–Z.")
        continue

    Input = user_in
    checkProbsAndAddToDict()
    choice = choiceSelection()
    print("The letter after that is:", choice)

    again = input("Do you want another letter?: ").strip()
    if again.upper() == "NO":
        running = False

# Optional: print a subset of weights for inspection
# print({k: v for k, v in weights.items() if k[0] == "C" and k[1] == "A"})