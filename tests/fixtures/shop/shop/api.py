import shop


def create_order_endpoint(request):
    return shop.place_order(request["items"])
