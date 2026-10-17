package util

import "fmt"

func Fmt(n float64) string {
	return fmt.Sprintf("$%.2f", n)
}
