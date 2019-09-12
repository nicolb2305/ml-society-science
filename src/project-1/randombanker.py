from banker import BankerBase, run
from random import choice


class RandomBanker(BankerBase):
    """Example Banker implementation. To implement your own, you need to take
    care of a number of details.

    1. Your class should inherit from BankerBase in `banker.py`. If so, it
       will be automatically discovered and scored when you call the run
       function from the same file.
    1. Your class needs to have a class member with the same name for each
       constructor argument (to be sklearn compliant).
       """

    def __init__(self, interest_rate):
        self.interest_rate = interest_rate

    def fit(self, X, y):
        pass

    def get_best_action(self, X):
        return choice((1, 2))


if __name__ == '__main__':
    run()
