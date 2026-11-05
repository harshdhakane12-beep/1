facts = {"fever", "cough", "body_pain"}

rules = [
    (["fever", "cough"], "flu"),
    (["fever", "body_pain"], "viral_infection"),
    (["cough", "sore_throat"], "cold"),
    (["flu"], "weakness"),
    (["viral_infection"], "weakness"),
    (["cold"], "mild_illness")
]

def forward_chaining(facts, rules):
    known_facts = set(facts)

    print("Initial Facts:", known_facts)
    print("\nForward Chaining Process:")

    changed = True

    while changed:
        changed = False

        for conditions, conclusion in rules:
            if all(c in known_facts for c in conditions):
                if conclusion not in known_facts:
                    known_facts.add(conclusion)
                    print("Rule Fired:", conditions, "->", conclusion)
                    changed = True

    return known_facts

def backward_chaining(goal, facts, rules, visited=None):

    if visited is None:
        visited = set()

    if goal in facts:
        print("Fact found:", goal)
        return True

    if goal in visited:
        return False

    visited.add(goal)

    print("\nTrying to prove:", goal)

    for conditions, conclusion in rules:
        if conclusion == goal:

            print("Using Rule:", conditions, "->", conclusion)

            if all(
                backward_chaining(c, facts, rules, visited.copy())
                for c in conditions
            ):
                print("Goal proved:", goal)
                return True

    print("Cannot prove:", goal)
    return False

goal = "weakness"

print("\nBackward Chaining Result:")

if backward_chaining(goal, facts, rules):
    print("\nConclusion:", goal, "is TRUE")
else:
    print("\nConclusion:", goal, "is FALSE")

def expert_system():

    print("\n======================================")
    print("RULE-BASED EXPERT SYSTEM")
    print("======================================")

    print("\nAvailable symptoms:")
    print("fever, cough, body_pain, sore_throat")

    user_input = input("\nEnter symptoms separated by comma: ")

    user_facts = {
        s.strip().lower()
        for s in user_input.split(",")
        if s.strip()
    }

    print("\nYour Facts:", user_facts)

    print("\nChoose Inference Method:")
    print("1. Forward Chaining")
    print("2. Backward Chaining")

    choice = input("\nEnter choice (1/2): ")

    if choice == "1":

        print("\n--- Forward Chaining ---")

        result = forward_chaining(user_facts, rules)

        print("\nDerived Knowledge:")

        for fact in sorted(result):
            print("-", fact)

    elif choice == "2":

        goal = input(
            "\nEnter goal to verify "
            "(flu / viral_infection / cold / weakness): "
        ).strip().lower()

        print("\n--- Backward Chaining ---")

        result = backward_chaining(goal, user_facts, rules)

        print("\nFinal Result:")

        if result:
            print(goal, "can be proved.")
        else:
            print(goal, "cannot be proved.")

    else:
        print("Invalid choice!")

expert_system()
