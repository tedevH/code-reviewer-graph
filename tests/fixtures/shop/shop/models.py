def parse_price(text):
    return round(float(text.strip("$")), 2)


class Order:
    def __init__(self, items):
        self.items = items

    def total(self):
        return sum(parse_price(i["price"]) for i in self.items)
