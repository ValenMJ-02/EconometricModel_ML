import json
from typing import Any, List, Dict, Optional

class PredictedPrices:
    def __init__(self, city: Optional[str], prices_data: Any):
        self.city = city
        if isinstance(prices_data, str):
            self.prices: List[Dict] = json.loads(prices_data)
        else:
            self.prices = prices_data or []

    def isEqual(self, other: Any) -> bool:
        # duck-typing: comprobamos que tenga los atributos necesarios
        if not hasattr(other, "city") or not hasattr(other, "prices"):
            return False
        if self.city != other.city:
            return False

        # ordenamos por 'yearbuilt'
        sorted_self = sorted(self.prices or [], key=lambda r: r.get('yearbuilt', 0))
        sorted_other = sorted(other.prices or [], key=lambda r: r.get('yearbuilt', 0))

        # serializamos con sort_keys para que la comparación sea determinista
        s1 = json.dumps(sorted_self, sort_keys=True)
        s2 = json.dumps(sorted_other, sort_keys=True)
        return s1 == s2
    
    def toDict(self):
        return {
            "city": self.city,
            "prices": self.prices
        }
