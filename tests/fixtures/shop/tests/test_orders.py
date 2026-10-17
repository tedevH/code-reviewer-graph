from shop.orders import place_order


def test_place_order():
    assert place_order([{"price": "$10"}])
