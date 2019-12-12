import pandas

class Policy:
    def get_best_action(self, data, probas):
        placebo_utility = 0*probas[0][0][0] + 1*probas[0][0][1]
        treatment_utility = -0.1*probas[1][0][0] + 0.9*probas[1][0][1]

        if placebo_utility > treatment_utility:
            return 0
        else:
            return 1
