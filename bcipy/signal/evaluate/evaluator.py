

class Evaluator:
    """Evaluator.

    Takes in raw data and tests them against given
    rules, which elicit the rejection of a inquiry when broken.
    Feeds warnings to artifact rejector as suggestions.

    Add rules given in parameters to evaluator's ruleset and set
    keys for broken_rules. One heading per rule.

    rules (list of rule objects, defined in rules.py)
    """

    def __init__(self, rules):
        self.rules = rules

    def evaluate(self, data):
        """Evaluate.

        Evaluates inquiry data using selected rules from parameters file.
        """

        for rule in self.rules:

            if rule.is_broken(data):

                return False

        return True

    def __str__(self):
        rules = [str(rule) for rule in self.rules]
        return f'Evaluator with {rules}'
