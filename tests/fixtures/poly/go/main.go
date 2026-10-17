package main

import (
	"fmt"

	shop "example.com/shop/order"
)

func main() {
	o := shop.New([]float64{1})
	fmt.Println(o.Total())
}
