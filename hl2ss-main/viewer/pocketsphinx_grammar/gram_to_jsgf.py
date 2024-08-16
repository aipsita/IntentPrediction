from jsgf import Grammar, PublicRule, Literal, AlternativeSet

# Create a grammar
grammar = Grammar("keyword")

# Define the rule for your keywords
keywords = [
    "left", "right", "above", "below", "front", "back", "disappear", "submit"
]

# Create a list of Literal objects
keyword_literals = [Literal(keyword) for keyword in keywords]

# Wrap the list of Literals in an AlternativeSet by unpacking the list
alternative_set = AlternativeSet(*keyword_literals)

# Create a PublicRule using the AlternativeSet
keyword_rule = PublicRule("keyword", alternative_set)

# Add the rule to the grammar
grammar.add_rule(keyword_rule)

# Save the grammar to a .jsgf file
with open("keyword.jsgf", "w") as f:
    f.write(grammar.compile())

print("Grammar saved to keyword.jsgf")
