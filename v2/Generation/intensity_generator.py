from v2.Domain.intensity_combinator import IntensityCombinator


class IntensityGenerator:
    def __init__(self):
        self.ig = IntensityCombinator().get_data(completed = False)
        print(self.ig)





ig = IntensityGenerator()