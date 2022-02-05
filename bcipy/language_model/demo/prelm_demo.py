"""Demo for the PRELM language model"""
import sys
from bcipy.helpers.language_model import norm_domain
from bcipy.language_model.prelm_language_model import PrelmLanguageModel
sys.path.append('.')


def main():
    """Runs the demo"""
    # init LMWrapper
    lmodel = PrelmLanguageModel()
    print('\nNo History\n')
    # get initial priors
    print(lmodel.recent_priors())
    # get priors
    print('\nHistory: T\n')
    lmodel.state_update(['T'])
    # display priors
    print(lmodel.recent_priors())
    print('\nHistory: TH\n')
    lmodel.state_update(['H'])
    print(lmodel.recent_priors())
    print('\nHistory: THE\n')
    lmodel.state_update(['E'])
    print(lmodel.recent_priors())
    # reset history al together
    lmodel.reset()
    print("\n--------------RESET-------------\n")
    print('\nHistory: THE (fed as a single string)\n')
    lmodel.state_update(list('THE'))
    print(lmodel.recent_priors())

    print('\n\nLetters in the probability domain:\n')
    print(norm_domain(lmodel.recent_priors()['letter']))


if __name__ == "__main__":
    main()
