*This project has been created as part of the 42 curriculum by jpaim-so.*

# Call Me Maybe


## Description

**Call Me Maybe** is a Python project that explores constrained text generation
for function calling with a small language model that is run locally.

The goal is to generate structured function calls from natural-language prompts
while constraining the model so that generated values follow the available
function definitions. The project uses a **trie** to constrain function-name
and string generation to known valid values, while numbers and booleans are
generated under type-specific constraints.

The project includes:

- JSON parsing and validation using Pydantic.
- Token encoding for fixed JSON syntax and parameter types.
- Trie-based constrained generation for function and string values.
- Extraction of potential string values from user prompts.
- A generation pipeline that combines these components into structured `Output`
  objects.
- Tests covering the parsers, trie, token setup, constrained generation, and
  pipeline.


## Instructions

### Requirements

- Python 3.10+;
- `uv`;
- The required `llm_sdk` package and model environment.

#### NOTE

Disk quota / cache configuration

The project is configured in pyproject.toml to store the uv cache locally in
.uv-cache. This is intentional because 42 machines have strict disk-quota
limits on the home partition.

The project can therefore be cloned to any location and installed with:

`make install`

No additional configuration is required on a personal computer.


### Installation

Install the project dependencies and set up the virtual environment with:

`make`

or:

`make install`

### Run

Run the function-calling pipeline with:

`make run`

This reads the function definitions from `data/input/functions_definition.json`
and the prompts from `data/input/function_calling_tests.json`, then writes the
generated results to:

`data/output/function_calling_results.json`

### Debugging

Run the application with Python's debugger using:

`make debug`

### Testing

Run the complete pytest test suite with:

`make test`

Additional test commands are available:

- `make test-silent` — run tests with minimal output;
- `make test-verbose` — run tests with verbose output (lists all tests);
- `make test-errors` — run tests while displaying custom test output
  (displaying error messages).

Note that several tests load the real language model and perform genuine
generation, so running the full suite is significantly slower than typical
unit tests and requires network access on first run to download the model.

### Linting and Type Checking

Run the standard code-quality checks with:

`make lint`

Executes the commands flake8 . and mypy . --warn-return-any
--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs
--check-untyped-defs

For stricter type checking:

`make lint-strict`

Which executes the commands flake8 . and mypy . --strict.

### Cleaning

Remove generated caches and Python bytecode with:

`make clean`

To additionally remove the virtual environment and generated output files:

`make fclean`

### Project structure

```
src/
├── __init__.py
├── __main__.py
├── cli.py
├── constrained_generator.py
├── json_builder.py
├── json_parser.py
├── main.py
├── pipeline.py
├── schemas.py
├── token_setup.py
└── trie.py

tests/
├── data/
├── test_cli.py
├── test_constrained_generator.py
├── test_json_parser.py
├── test_main.py
├── test_pipeline.py
├── test_token_setup.py
└── test_trie.py

data/
├── input/
│   ├── function_calling_tests.json
│   └── functions_definition.json
└── output/
    └── function_calling_results.json  # Created when the program is run
```


## Resources

### Documentation

- [Pydantic documentation](https://docs.pydantic.dev/) — used throughout for
  schema validation.
- [Karpathy, "Deep Dive into LLMs like ChatGPT"](https://www.youtube.com/watch?v=7xTGNNLPyMI)
  — background on tokenization, inference, and tool use, which motivated the overall approach.
- [A. Cooper, "Constrained Decoding"](https://www.aidancooper.co.uk/constrained-decoding/)
  — background reading on the general constrained decoding technique.
- [uv documentation](https://docs.astral.sh/uv/) — used for dependency
  management, virtual environment creation, and project execution.
- [Python `re` documentation](https://docs.python.org/3/library/re.html) —
  reference for the regular-expression functionality used during candidate
  extraction.

### AI Usage

AI was used throughout development as a conversational aid for:

- Understanding the underlying concepts (tokenization, logits, softmax,
constrained decoding, trie data structures) before implementing them.
- Debugging specific runtime issues (e.g. token-boundary/concatenation bugs,
disk-quota errors on school machines, degenerate generation loops) through
iterative, evidence-based investigation (building small throwaway scripts to
isolate and confirm each hypothesis before changing production code).
- Reviewing code, docstrings, and tests for correctness and completeness
against the actual current behavior of each module.
- Discussing design trade-offs (e.g. the regex-keyword fallback, the
main-content-parameter heuristic) and their limitations, including where they
diverge from a fully general LLM-driven solution.

All code was written, tested, and understood by me; AI assistance was used for
guidance, debugging, and review.


## Additional sections:

### Algorithm explanation

The project uses constrained decoding to guide the language model toward valid
function-call outputs while still allowing the model to choose values based on
the input prompt.

The generation process first encodes the fixed parts of the expected JSON
structure, such as brackets, keys, commas, and closing characters. Function
names are stored in a trie, allowing the model to select only token sequences
that correspond to valid function names. The same approach is used for string
values extracted from the prompt: candidate strings are encoded and stored in a
trie, restricting generation to those candidates.

Numbers are generated by restricting the model's choices to the token IDs
representing digits and numeric symbols. Generation continues while the model
selects a valid numeric token and stops when a permitted terminator is
selected. Boolean values are generated by comparing the model's scores for the
first token of the true and false token sequences, then emitting the
corresponding token sequence.

Two deliberate exceptions exist to this candidate/type-driven approach. First,
parameters whose name suggests a "main content" role (e.g. `source_string`,
`text`) are directly assigned the longest remaining candidate rather than left
to model selection, since the model proved unreliable at this specific
disambiguation when a much shorter, more salient candidate (e.g. an all-caps
keyword) was also available. Second, parameters whose name suggests a
regex/pattern field are checked against a small, fixed keyword-to-pattern
lookup before falling back to candidate extraction, since a regex pattern
implied by a keyword (e.g. "vowels") never appears literally in the prompt for
extraction to find. Both are documented, narrow heuristics rather than general
solutions — see Challenges Faced.

The pipeline therefore combines model predictions with deterministic
constraints. The model decides which valid option is most appropriate, while
the constraints prevent it from producing values outside the allowed token
space.

### Design decisions

#### - Pydantic and Class Design

Pydantic is used for classes that represent structured data and require
validation. Behavioural classes retain appropriate standard Python base
classes, such as `argparse.ArgumentParser` for the CLI parser and `Exception`
for CLI errors, since Pydantic would not provide useful validation for their
roles.

#### - Trie-based constrained generation

A trie is used for function names and extracted string candidates because it
represents valid token sequences directly. At each generation step, only the
tokens corresponding to children of the current trie node are allowed. This
avoids generating an arbitrary string and checking it afterwards.

#### - Token-level constraints

The implementation operates on the model's token IDs rather than individual
characters. This is important because the model tokenizer may represent
multiple characters as a single token. The constraints therefore operate on the
same representation used by the model itself.

#### - Separation of generation by parameter type

Numbers, strings, booleans, and function names use different generation
strategies because their valid outputs have different constraints. This keeps
the constrained generation logic specific to the requirements of each type
rather than applying one generic mechanism to all values.

#### - Candidate extraction for strings

String candidates are extracted from the user's prompt before generation. This
provides a finite set of values that the model can choose from and prevents
unconstrained string generation from producing arbitrary values that were not
present in the input.

#### - Separation of preparation and generation

Token encoding and trie construction are separated from the actual generation
pipeline. The generation context can therefore be prepared once for a set of
function definitions and reused while processing multiple prompts.

### Performance analysis

On the provided dataset, the implementation correctly selects the function for
all 11 prompts (100% function-selection accuracy) and correctly extracts 18
of 19 individual argument values (approximately 94.7% argument-extraction
accuracy). The remaining error is the `replacement` value in the
vowel-substitution example, where the correct value ("asterisks") is
implied by context rather than present as a literal or keyword-mapped
pattern — see Challenges Faced.

The implementation uses constrained decoding rather than relying on the model
to produce valid JSON by itself. Fixed JSON structure, function names, strings,
numbers, and booleans are constrained during generation, providing reliable
schema-compliant output.

Execution times to run the script are well under the five minute limit when ran
in the schools computers (consistently around one minute to process the data
given in the projects intra page).

### Challenges faced

One of the main challenges was constraining different parameter types using the
model's token-level output. Function names and strings can be represented as
finite candidate sets and constrained with tries, while numbers require 
token-level numeric constraints and booleans require a comparison between the
valid boolean choices.

Another challenge was extracting useful string candidates from natural-language
prompts. The implementation uses quoted strings, likely proper nouns, and a
fallback candidate to provide the constrained string generator with possible
values.

A closely related challenge was disambiguating multiple string-typed parameters
on the same function (e.g. `source_string`, `regex`, `replacement`), where
several candidates could plausibly fit more than one parameter. An initial
attempt using open-ended, unconstrained token generation for string values
proved unreliable: the model would frequently fall into repeating short,
degenerate token sequences rather than terminating cleanly, regardless of the
surrounding prompt phrasing, context length, or masking strategy tried
(confirmed through isolated, minimal-context testing that ruled out the
surrounding pipeline as the cause). Switching to trie-constrained selection
among extracted candidates resolved this reliably, but introduced a new
problem: nothing initially prevented the same candidate being assigned to more
than one parameter, or a short, salient candidate (e.g. an all-caps keyword)
from being assigned to a parameter that should hold the longest available
candidate instead. These were addressed with, respectively, a "used" set that
removes already-assigned candidates from consideration for subsequent
parameters, and a narrow, explicit heuristic (`is_main_content_param`) that
deterministically assigns the longest remaining candidate to parameters
conventionally named for "main content" roles. Both are documented as
deliberate, narrow compromises rather than general solutions — earlier attempts
to fix this through richer prompt hints or parameter-processing order proved
unstable, fixing one case while breaking a previously-correct one.

A separate, related limitation is that some parameter values (e.g. a regex
pattern implied by the keyword "vowels") never appear literally anywhere in the
prompt, and so cannot be produced by any extraction-based strategy regardless
of how it is tuned. A small, explicit keyword-to-pattern lookup 
(`REGEX_KEYWORD_MAP`) was added as a narrow, deliberate fallback for this
specific case. This is knowingly not a general solution — an attempt to have
the model construct a regex pattern from scratch under a broader (but still
constrained) character-level alphabet was tested and abandoned, since it
reproduced the same degenerate-generation failure seen with open-ended string
generation, even though the same model reliably produced correct regex syntax
when generating completely unconstrained in a minimal test context. This
suggests the limitation lies in how well masked, peek-based token selection
cooperates with this specific small model, rather than in the model's
underlying knowledge.

Testing also revealed that a small language model can still make incorrect
choices among valid constrained values, so prompting and candidate preparation
were kept separate from the actual structural constraints.

### Testing strategy

Tests fall into two categories. Pure-logic functions (e.g.
`extract_string_candidates`, schema validation, trie insertion) are tested
deterministically, with exact expected outputs. Functions that call the real
language model are tested with real model instances and, where the expected
output is deterministic given a controlled minimal context, assert on the exact
expected value rather than only checking the return type — this was found
necessary during development, since type-only assertions passed even when the
actual generated content was substantively wrong.

The provided function-calling dataset was also run through the complete
pipeline. The resulting output was inspected to verify function selection,
argument extraction, parameter types, and JSON structure.

### Example usage

Set up the project and install its dependencies:

`uv sync`

or:

`make install`

Run the project:

`uv run python -m src`

The program reads the function definitions and test prompts from the default input files and produces:

data/output/function_calling_results.json


## Bonus Features

### Comprehensive Test Suite

The project includes a comprehensive pytest test suite covering JSON parsing,
Pydantic schemas, trie construction, token preparation, constrained generation,
pipeline and CLI behaviour.
