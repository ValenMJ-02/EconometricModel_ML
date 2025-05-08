class PredictedPrices:
    def __init__(self, city: str, prices: str):
        self.city: str = city
        self.prices: str = prices

    def isEqual(self, other):
        assert(self.city == other.city)
        assert(self.prices == other.prices)