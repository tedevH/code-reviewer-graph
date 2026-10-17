<?php
namespace App;

class Util {
    public static function fmt(float $n): string {
        return sprintf('$%.2f', $n);
    }
}
