*This project has been created as part of the 42 curriculum by jpaim-so.*

# Readme Requirements

A README.md file must be provided at the root of your Git repository. Its purpose is
to allow anyone unfamiliar with the project (peers, staff, recruiters, etc.) to quickly
understand what the project is about, how to run it, and where to find more information
on the topic.

## The README.md must include at least:

• The very first line must be italicized and read: This project has been created as part
of the 42 curriculum by <login1>[, <login2>[, <login3>[...]]].

• A “Description” section that clearly presents the project, including its goal and a
brief overview.

• An “Instructions” section containing any relevant information about compilation,
installation, and/or execution.

• A “Resources” section listing classic references related to the topic (documentation, articles, tutorials, etc.), as well as a description of how AI was used —
specifying for which tasks and which parts of the project.


## Additional sections may be required depending on the project (e.g., usage examples, feature list, technical choices, etc.).

---

# Call Me Maybe

## Description

## Instructions

### Requirements

- Python 3.10+

### Installation

`make`

### Run

`make run`

or

`python3 main.py <map_file>`

Example:

`python3 main.py maps/simple.txt`

---

## Resources

### Documentation

- to fill in
- 
https://www.aidancooper.co.uk/constrained-decoding/

https://www.youtube.com/watch?v=7xTGNNLPyMI

### AI Usage

---

## Additional sections:

### Algorithm explanation
• Algorithm explanation: Describe your constrained decoding approach in detail

### Design decisions
• Design decisions: Explain key choices in your implementation

Pydantic and Class Design

Pydantic is used for classes that represent structured data and require validation. Behavioural classes retain appropriate standard Python base classes, such as argparse.ArgumentParser for the CLI parser and Exception for CLI errors, since Pydantic would not provide useful validation for their roles.

### Performance analysis
• Performance analysis: Discuss accuracy, speed, and reliability of your solution

### Challenges faced
• Challenges faced: Document difficulties encountered and how you solved them

### Testing strategy
• Testing strategy: Describe how you validated your implementation

### Example usage
• Example usage: Provide clear examples of running your program


# TO MENTION?: Handle function names where one valid name is a prefix of another.
# Currently, generation stops as soon as a finished trie node is reached,
# so a shorter name such as "fn_get" can be selected instead of
# "fn_get_current_time".