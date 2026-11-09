import config
from ape_pipeline import writer_call, parse_list

# These are the three tasks we want to test
tasks = [
    "word_sorting",
    "formal_fallacies",
    "logical_deduction"
]

# Test the writer on each task
for task in tasks:

    print(f"\nTesting writer for: {task}")

    # Ask the writer to generate 3 candidate prompts
    prompt = (
        f"Generate exactly 3 different instructions "
        f"for the task: {task}. "
        "Return ONLY a JSON array of 3 strings."
    )

    # Send the request to Claude
    response = writer_call(
        prompt,
        f"{task}-writer-test"
    )

    # Check that the response contains exactly 3 prompts
    candidates = parse_list(response, expected=3)

    # Display the results
    print(f"Writer model: {config.WRITER_MODEL}")

    for i, candidate in enumerate(candidates, 1):
        print(f"\nCandidate {i}:")
        print(candidate)

    print(f"\n{task}: TEST PASSED!")

print("\nALL THREE WRITER TESTS PASSED!")
